"""Tests for the SIP asyncio protocol handler."""

import asyncio
import dataclasses
import errno
import hashlib
from unittest.mock import MagicMock

import pytest
from voip.call import Call
from voip.rtp import RealtimeTransportProtocol
from voip.sdp.messages import SessionDescription
from voip.sdp.types import Timing
from voip.sip.messages import Message, Request, Response
from voip.sip.protocol import (
    SIP,
    RegistrationError,
    SessionInitiationProtocol,
    _mask_caller,
)
from voip.sip.types import CallerID
from voip.types import DigestAlgorithm

INVITE_WITH_PCMA = (
    b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
    b"Via: SIP/2.0/UDP pc33.atlanta.com\r\n"
    b"From: sip:alice@atlanta.com\r\n"
    b"To: sip:bob@biloxi.com\r\n"
    b"Call-ID: test-call-id-1\r\n"
    b"CSeq: 1 INVITE\r\n"
    b"Content-Type: application/sdp\r\n"
    b"Content-Length: 72\r\n"
    b"\r\n"
    b"v=0\r\n"
    b"o=- 0 0 IN IP4 192.0.2.1\r\n"
    b"s=-\r\n"
    b"c=IN IP4 192.0.2.1\r\n"
    b"t=0 0\r\n"
    b"m=audio 49170 RTP/AVP 8 0\r\n"
)

INVITE_WITH_PCMU_ONLY = (
    b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
    b"Via: SIP/2.0/UDP pc33.atlanta.com\r\n"
    b"From: sip:alice@atlanta.com\r\n"
    b"To: sip:bob@biloxi.com\r\n"
    b"Call-ID: test-call-id-2\r\n"
    b"CSeq: 1 INVITE\r\n"
    b"Content-Type: application/sdp\r\n"
    b"Content-Length: 68\r\n"
    b"\r\n"
    b"v=0\r\n"
    b"o=- 0 0 IN IP4 192.0.2.1\r\n"
    b"s=-\r\n"
    b"c=IN IP4 192.0.2.1\r\n"
    b"t=0 0\r\n"
    b"m=audio 49170 RTP/AVP 0\r\n"
)

INVITE_WITH_UNKNOWN_CODEC = (
    b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
    b"Via: SIP/2.0/UDP pc33.atlanta.com\r\n"
    b"From: sip:alice@atlanta.com\r\n"
    b"To: sip:bob@biloxi.com\r\n"
    b"Call-ID: test-call-id-3\r\n"
    b"CSeq: 1 INVITE\r\n"
    b"Content-Type: application/sdp\r\n"
    b"Content-Length: 100\r\n"
    b"\r\n"
    b"v=0\r\n"
    b"o=- 0 0 IN IP4 192.0.2.1\r\n"
    b"s=-\r\n"
    b"c=IN IP4 192.0.2.1\r\n"
    b"t=0 0\r\n"
    b"m=audio 49170 RTP/AVP 126\r\n"
    b"a=rtpmap:126 telephone-event/8000\r\n"
)

INVITE_NO_SDP = (
    b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
    b"Via: SIP/2.0/UDP pc33.atlanta.com\r\n"
    b"From: sip:alice@atlanta.com\r\n"
    b"To: sip:bob@biloxi.com\r\n"
    b"Call-ID: test-call-id-4\r\n"
    b"CSeq: 1 INVITE\r\n"
    b"\r\n"
)


class FakeTransport:
    """Fake TCP/TLS transport for testing."""

    def __init__(
        self,
        local_addr: tuple[str, int] = ("127.0.0.1", 5061),
        peer_addr: tuple[str, int] = ("192.0.2.1", 5061),
    ):
        self._local_addr = local_addr
        self._peer_addr = peer_addr
        self.sent: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.sent.append(data)

    def get_extra_info(self, key, default=None):
        if key == "sockname":
            return self._local_addr
        if key == "peername":
            return self._peer_addr
        if key == "ssl_object":
            return object()  # non-None signals TLS
        return default


class FakeProtocol(SessionInitiationProtocol):
    """Fake SIP protocol that captures sent messages."""

    def __init__(self):
        super().__init__(outbound_proxy=("127.0.0.1", 5061), aor="sip:test@example.com")
        self.connection_made(FakeTransport())
        self._sent_responses: list[tuple[Response, None]] = []

    def send(self, message):
        if isinstance(message, Response):
            self._sent_responses.append((message, None))
        super().send(message)


class ConcreteProtocol(SessionInitiationProtocol):
    """Concrete subclass for testing that records received messages."""

    def __init__(self):
        super().__init__(outbound_proxy=("127.0.0.1", 5061), aor="sip:test@example.com")
        self.requests = []
        self.responses = []

    def request_received(self, request, addr):
        self.requests.append((request, addr))

    def response_received(self, response, addr):
        self.responses.append((response, addr))


class TestMaskCaller:
    def test_full_from_header_with_display_name(self):
        """Mask all but the last 4 chars of a 12-digit display name (8 asterisks)."""
        header = '"015114455910" <sip:015114455910@telefonica.de>;tag=abc123'
        assert _mask_caller(header) == "********5910"

    def test_bare_sip_uri(self):
        """Extract user part from a bare SIP URI and mask all but the last 4 chars."""
        assert _mask_caller("sip:alice@example.com") == "*lice"

    def test_short_caller__no_masking(self):
        """Identifiers with 4 or fewer characters are returned as-is."""
        assert _mask_caller("<sip:bob@example.com>") == "bob"

    def test_strips_tag_parameter(self):
        """The tag= and any subsequent parameters are stripped before masking."""
        header = '"015114455910" <sip:015114455910@example.com>;tag=xyz;other=1'
        result = _mask_caller(header)
        assert "tag" not in result
        assert result.endswith("5910")

    def test_angle_bracket_uri_without_display_name(self):
        """Parse <sip:user@host> style without a display name."""
        assert _mask_caller("<sip:alice@example.com>") == "*lice"


class TestCallerID:
    def test_str__returns_raw_header(self):
        """str() returns the original SIP header value unchanged."""
        raw = '"015114455910" <sip:015114455910@telefonica.de>;tag=abc'
        assert str(CallerID(raw)) == raw

    def test_repr__masks_display_name_and_includes_domain(self):
        """repr() shows last 4 chars of display name and the carrier domain."""
        caller = CallerID('"015114455910" <sip:015114455910@telefonica.de>;tag=abc')
        assert repr(caller) == "********5910@telefonica.de"

    def test_repr__bare_sip_uri(self):
        """repr() masks the user part of a bare SIP URI and includes the domain."""
        assert repr(CallerID("sip:alice@example.com")) == "*lice@example.com"

    def test_repr__angle_bracket_uri(self):
        """repr() handles <sip:user@host> without a display name."""
        assert repr(CallerID("<sip:bob@biloxi.com>")) == "bob@biloxi.com"

    def test_user__phone_number(self):
        """User property extracts the SIP user part from a phone number URI."""
        caller = CallerID('"015114455910" <sip:015114455910@telefonica.de>')
        assert caller.user == "015114455910"

    def test_user__bare_uri(self):
        """User property extracts the username from a bare SIP URI."""
        assert CallerID("sip:alice@example.com").user == "alice"

    def test_host__returns_carrier_domain(self):
        """Host property returns the domain part of the SIP URI."""
        assert CallerID("sip:alice@carrier.example.com").host == "carrier.example.com"

    def test_display_name__quoted(self):
        """display_name returns the quoted display name."""
        assert CallerID('"Alice" <sip:alice@example.com>').display_name == "Alice"

    def test_display_name__absent(self):
        """display_name is None when no display name is present."""
        assert CallerID("sip:alice@example.com").display_name is None

    def test_tag__present(self):
        """Tag property extracts the tag parameter value."""
        assert CallerID("sip:alice@example.com;tag=abc123").tag == "abc123"

    def test_tag__absent(self):
        """Tag is None when no tag parameter is present."""
        assert CallerID("sip:alice@example.com").tag is None

    def test_is_str_subclass(self):
        """CallerID is a str, so it passes isinstance checks transparently."""
        assert isinstance(CallerID("sip:alice@example.com"), str)

    def test_equality_with_plain_string(self):
        """CallerID compares equal to a plain str with the same value."""
        assert CallerID("sip:alice@example.com") == "sip:alice@example.com"

    def test_data_received__request(self):
        """Dispatch a received SIP request to request_received via TCP stream."""
        protocol = ConcreteProtocol()
        transport = FakeTransport(peer_addr=("192.0.2.1", 5060))
        protocol.transport = transport
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS pc33.atlanta.com\r\n"
            b"\r\n"
        )
        protocol.data_received(data)
        assert len(protocol.requests) == 1
        request, called_addr = protocol.requests[0]
        assert isinstance(request, Request)
        assert request.method == "INVITE"
        assert called_addr == ("192.0.2.1", 5060)

    def test_data_received__response(self):
        """Dispatch a received SIP response to response_received via TCP stream."""
        protocol = ConcreteProtocol()
        transport = FakeTransport(peer_addr=("192.0.2.1", 5060))
        protocol.transport = transport
        data = b"SIP/2.0 200 OK\r\nVia: SIP/2.0/TLS pc33.atlanta.com\r\n\r\n"
        protocol.data_received(data)
        assert len(protocol.responses) == 1
        response, called_addr = protocol.responses[0]
        assert isinstance(response, Response)
        assert response.status_code == 200
        assert called_addr == ("192.0.2.1", 5060)

    def test_error_received__blocking_io(self):
        """Log blocking IO errors without re-raising."""
        protocol = SessionInitiationProtocol(
            outbound_proxy=("127.0.0.1", 5061), aor="sip:test@example.com"
        )
        exc = OSError(errno.EAGAIN, "Resource temporarily unavailable")
        protocol.error_received(exc)  # should not raise

    def test_error_received__reraises(self):
        """Log unexpected transport errors without re-raising (delegates to base)."""
        protocol = SessionInitiationProtocol(
            outbound_proxy=("127.0.0.1", 5061), aor="sip:test@example.com"
        )
        exc = OSError("Unexpected error")
        protocol.error_received(exc)  # should not raise

    def test_connection_lost__no_exception(self):
        """Handle a clean connection close without raising."""
        protocol = SessionInitiationProtocol(
            outbound_proxy=("127.0.0.1", 5061), aor="sip:test@example.com"
        )
        protocol.connection_lost(None)  # should not raise

    def test_connection_lost__with_exception(self):
        """Log an exception on connection lost without re-raising."""
        protocol = SessionInitiationProtocol(
            outbound_proxy=("127.0.0.1", 5061), aor="sip:test@example.com"
        )
        protocol.connection_lost(Exception("Connection reset"))  # should not raise


class TestWithToTag:
    def test__with_to_tag__adds_tag(self):
        """Append the To tag to the To header for a known Call-ID."""
        protocol = SessionInitiationProtocol(
            outbound_proxy=("127.0.0.1", 5061), aor="sip:test@example.com"
        )
        protocol._to_tags["call-1"] = "abc123"
        result = protocol._with_to_tag({"To": "sip:bob@biloxi.com"}, "call-1")
        assert result["To"] == "sip:bob@biloxi.com;tag=abc123"

    def test__with_to_tag__unknown_call_id(self):
        """Leave the To header unchanged when the Call-ID has no stored tag."""
        protocol = SessionInitiationProtocol(
            outbound_proxy=("127.0.0.1", 5061), aor="sip:test@example.com"
        )
        result = protocol._with_to_tag({"To": "sip:bob@biloxi.com"}, "unknown")
        assert result["To"] == "sip:bob@biloxi.com"

    def test__with_to_tag__missing_to_header(self):
        """Return an empty To header when none is present and no tag exists."""
        protocol = SessionInitiationProtocol(
            outbound_proxy=("127.0.0.1", 5061), aor="sip:test@example.com"
        )
        result = protocol._with_to_tag({}, "unknown")
        assert result["To"] == ""


class TestRinging:
    def test_ringing__includes_to_tag(self):
        """Include the To tag in a 180 Ringing response (RFC 3261 §8.2.6.2)."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        request = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "ring-test-1",
                "CSeq": "1 INVITE",
            },
        )
        protocol.request_received(request, addr)
        protocol.ringing(request)
        assert len(protocol._sent_responses) == 1
        response, _ = protocol._sent_responses[0]
        assert response.status_code == 180
        to_header = response.headers.get("To", "")
        assert ";tag=" in to_header

    def test_ringing__no_address(self, caplog):
        """Log an error and send nothing when no pending INVITE is stored for the Call-ID."""
        protocol = FakeProtocol()
        request = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={"Call-ID": "nonexistent"},
        )
        with caplog.at_level("ERROR"):
            protocol.ringing(request)
        assert not protocol._sent_responses
        assert "No pending INVITE found" in caplog.text


class TestReject:
    def test_reject__includes_to_tag(self):
        """Include the To tag in a reject response (RFC 3261 §8.2.6.2)."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        request = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "reject-test-1",
                "CSeq": "1 INVITE",
            },
        )
        protocol.request_received(request, addr)
        protocol.reject(request)
        assert len(protocol._sent_responses) == 1
        response, _ = protocol._sent_responses[0]
        assert response.status_code == 486
        assert ";tag=" in response.headers.get("To", "")

    def test_reject__cleans_up_to_tag(self):
        """Remove the To tag after rejecting (no lingering state)."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        request = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={"Call-ID": "reject-cleanup-1", "To": "sip:bob@biloxi.com"},
        )
        protocol.request_received(request, addr)
        assert "reject-cleanup-1" in protocol._to_tags
        protocol.reject(request)
        assert "reject-cleanup-1" not in protocol._to_tags

    def test_reject__no_address(self, caplog):
        """Log an error and send nothing when no pending INVITE is stored for the Call-ID."""
        protocol = FakeProtocol()
        request = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={"Call-ID": "nonexistent"},
        )
        with caplog.at_level("ERROR"):
            protocol.reject(request)
        assert not protocol._sent_responses
        assert "No pending INVITE found" in caplog.text


class TestBYEHandler:
    def test_bye__includes_to_tag_when_present(self):
        """Include the stored To tag in a 200 OK BYE response."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        invite = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "bye-tag-test-1",
                "CSeq": "1 INVITE",
            },
        )
        protocol.request_received(invite, addr)
        tag = protocol._to_tags["bye-tag-test-1"]
        bye = Request(
            method="BYE",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "bye-tag-test-1",
                "CSeq": "2 BYE",
            },
        )
        protocol.request_received(bye, addr)
        assert len(protocol._sent_responses) == 1
        response, _ = protocol._sent_responses[0]
        assert response.status_code == 200
        assert f";tag={tag}" in response.headers.get("To", "")

    def test_bye__cleans_up_to_tag(self):
        """Remove the To tag from state after processing BYE."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        invite = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={"Call-ID": "bye-cleanup-1", "To": "sip:bob@biloxi.com"},
        )
        protocol.request_received(invite, addr)
        assert "bye-cleanup-1" in protocol._to_tags
        bye = Request(
            method="BYE",
            uri="sip:bob@biloxi.com",
            headers={"Call-ID": "bye-cleanup-1", "To": "sip:bob@biloxi.com"},
        )
        protocol.request_received(bye, addr)
        assert "bye-cleanup-1" not in protocol._to_tags

    def test_bye__without_prior_to_tag(self):
        """Send a 200 OK BYE response without tag when no To tag is stored."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        bye = Request(
            method="BYE",
            uri="sip:bob@biloxi.com",
            headers={
                "To": "sip:bob@biloxi.com",
                "Call-ID": "bye-no-tag-1",
                "CSeq": "2 BYE",
            },
        )
        protocol.request_received(bye, addr)
        assert len(protocol._sent_responses) == 1
        response, _ = protocol._sent_responses[0]
        assert response.status_code == 200
        assert ";tag=" not in response.headers.get("To", "")


class TestAnswer:
    @pytest.fixture()
    def fake_rtp_transport(self):
        """Provide a fake RTP transport with a fixed local address."""
        return FakeTransport(("127.0.0.1", 12000))

    def _make_invite(
        self,
        call_id: str,
        sdp_body: SessionDescription | None = None,
        *,
        record_route: str | None = None,
    ) -> Request:
        """Build a minimal INVITE request."""
        headers = {
            "Via": "SIP/2.0/UDP pc33.atlanta.com",
            "From": "sip:alice@atlanta.com",
            "To": "sip:bob@biloxi.com",
            "Call-ID": call_id,
            "CSeq": "1 INVITE",
        }
        if sdp_body:
            headers["Content-Type"] = "application/sdp"
        if record_route:
            headers["Record-Route"] = record_route
        return Request(
            method="INVITE", uri="sip:bob@biloxi.com", headers=headers, body=sdp_body
        )

    async def _run_answer(self, protocol, invite, fake_rtp_transport):
        """Run _answer coroutine with a pre-populated shared RTP mux."""
        loop = asyncio.get_running_loop()
        # Pre-populate the shared RTP mux so _answer() skips socket creation.
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        protocol._rtp_protocol = mux
        protocol._rtp_transport = fake_rtp_transport
        # Resolve the SIP protocol's own local address (for Contact header).
        protocol.local_address = ("127.0.0.1", 5061)
        await protocol._answer(invite, _CodecAwareCall)

    @pytest.mark.asyncio
    async def test_answer__selects_pcma_from_offer(self, fake_rtp_transport):
        """Select PCMA (8) when the remote SDP offers both PCMA and PCMU."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        sdp_body = SessionDescription.parse(
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 192.0.2.1\r\n"
            b"s=-\r\n"
            b"c=IN IP4 192.0.2.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 8 0\r\n"
        )
        invite = self._make_invite("answer-pcma-1", sdp_body)
        protocol.request_received(invite, addr)
        await self._run_answer(protocol, invite, fake_rtp_transport)
        assert protocol._sent_responses
        response, _ = protocol._sent_responses[-1]
        assert response.status_code == 200
        assert response.body.origin is not None
        assert response.body.timings == [Timing(start_time=0, stop_time=0)]
        assert response.body.media[0].fmt[0].payload_type == 8
        assert any(a.name == "sendrecv" for a in response.body.media[0].attributes)
        assert response.body.media[0].fmt[0].encoding_name.startswith("PCMA")

    @pytest.mark.asyncio
    async def test_answer__selects_pcmu_when_only_option(self, fake_rtp_transport):
        """Select PCMU (0) when the remote SDP offers only PCMU."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        sdp_body = SessionDescription.parse(
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 192.0.2.1\r\n"
            b"s=-\r\n"
            b"c=IN IP4 192.0.2.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 0\r\n"
        )
        invite = self._make_invite("answer-pcmu-1", sdp_body)
        protocol.request_received(invite, addr)
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert response.body.media[0].fmt[0].payload_type == 0  # PCMU when only option

    @pytest.mark.asyncio
    async def test_answer__selects_opus_from_offer(self, fake_rtp_transport):
        """Select Opus (111) when the remote SDP offers Opus alongside PCMA and PCMU."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        sdp_body = SessionDescription.parse(
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 192.0.2.1\r\n"
            b"s=-\r\n"
            b"c=IN IP4 192.0.2.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 111 8 0\r\n"
            b"a=rtpmap:111 opus/48000/2\r\n"
        )
        invite = self._make_invite("answer-opus-1", sdp_body)
        protocol.request_received(invite, addr)
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert response.body.media[0].fmt[0].payload_type == 111
        assert response.body.media[0].fmt[0].encoding_name.lower().startswith("opus")

    @pytest.mark.asyncio
    async def test_answer__selects_g722_when_no_opus(self, fake_rtp_transport):
        """Select G.722 (9) when the remote SDP offers G.722 and PCMA but not Opus."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        sdp_body = SessionDescription.parse(
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 192.0.2.1\r\n"
            b"s=-\r\n"
            b"c=IN IP4 192.0.2.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 9 8\r\n"
        )
        invite = self._make_invite("answer-g722-1", sdp_body)
        protocol.request_received(invite, addr)
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert response.body.media[0].fmt[0].payload_type == 9  # G.722

    @pytest.mark.asyncio
    async def test_answer__selects_opus_by_name_match_with_different_pt(
        self, fake_rtp_transport
    ):
        """Select Opus by codec name match when remote uses a non-standard payload type."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        sdp_body = SessionDescription.parse(
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 192.0.2.1\r\n"
            b"s=-\r\n"
            b"c=IN IP4 192.0.2.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 100 8\r\n"
            b"a=rtpmap:100 opus/48000/2\r\n"
        )
        invite = self._make_invite("answer-opus-name-1", sdp_body)
        protocol.request_received(invite, addr)
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert (
            response.body.media[0].fmt[0].payload_type == 100
        )  # Opus at non-standard PT

    @pytest.mark.asyncio
    async def test_answer__unsupported_codec__raises(self, fake_rtp_transport):
        """Raise NotImplementedError when the INVITE offers only unsupported codecs."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        sdp_body = SessionDescription.parse(
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 192.0.2.1\r\n"
            b"s=-\r\n"
            b"c=IN IP4 192.0.2.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 126\r\n"
            b"a=rtpmap:126 telephone-event/8000\r\n"
        )
        invite = self._make_invite("answer-fallback-1", sdp_body)
        protocol.request_received(invite, addr)
        with pytest.raises(NotImplementedError):
            await self._run_answer(protocol, invite, fake_rtp_transport)

    @pytest.mark.asyncio
    async def test_answer__no_sdp_falls_back_to_default(self, fake_rtp_transport):
        """Use payload type 0 (PCMU) with SAVP when the INVITE has no SDP body."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        invite = self._make_invite("answer-no-sdp-1")
        protocol.request_received(invite, addr)
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert response.body.media[0].fmt[0].payload_type == 0  # PCMU default

    @pytest.mark.asyncio
    async def test_answer__includes_to_tag_in_200_ok(self, fake_rtp_transport):
        """Include the locally generated To tag in the 200 OK response."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        sdp_body = SessionDescription.parse(
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 192.0.2.1\r\n"
            b"s=-\r\n"
            b"c=IN IP4 192.0.2.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 8\r\n"
        )
        invite = self._make_invite("answer-tag-1", sdp_body)
        protocol.request_received(invite, addr)
        stored_tag = protocol._to_tags["answer-tag-1"]
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert f";tag={stored_tag}" in response.headers.get("To", "")

    @pytest.mark.asyncio
    async def test_answer__no_address_logs_error(self, caplog):
        """Log an error and return early when no pending INVITE is tracked for the Call-ID."""
        protocol = FakeProtocol()
        invite = self._make_invite("no-addr-answer-1")

        with caplog.at_level("ERROR"):
            await protocol._answer(invite, Call)
        assert "No pending INVITE found" in caplog.text
        assert not protocol._sent_responses

    @pytest.mark.asyncio
    async def test_answer__includes_contact_header(self, fake_rtp_transport):
        """Include a Contact header with the local SIP address in 200 OK."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        invite = self._make_invite("answer-contact-1")
        protocol.request_received(invite, addr)
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert "Contact" in response.headers
        # FakeProtocol AOR is sip:test@example.com → sip: Contact.
        assert response.headers["Contact"].startswith("<sip:")
        # FakeTransport is TLS-wrapped, so transport=tls param should be present.
        assert ";transport=tls" in response.headers["Contact"]

    @pytest.mark.asyncio
    async def test_answer__includes_allow_header(self, fake_rtp_transport):
        """Include an Allow header listing supported SIP methods in 200 OK."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        invite = self._make_invite("answer-allow-1")
        protocol.request_received(invite, addr)
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert "Allow" in response.headers
        assert "INVITE" in response.headers["Allow"]
        assert "BYE" in response.headers["Allow"]

    @pytest.mark.asyncio
    async def test_answer__includes_supported_header(self, fake_rtp_transport):
        """Include a Supported header in the 200 OK response."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        invite = self._make_invite("answer-supported-1")
        protocol.request_received(invite, addr)
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert "Supported" in response.headers

    @pytest.mark.asyncio
    async def test_answer__echoes_record_route(self, fake_rtp_transport):
        """Echo the Record-Route header from the INVITE in the 200 OK."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        route = "<sip:proxy.example.com;lr>"
        invite = self._make_invite("answer-rr-1", record_route=route)
        protocol.request_received(invite, addr)
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert response.headers.get("Record-Route") == route

    @pytest.mark.asyncio
    async def test_answer__omits_record_route_when_absent(self, fake_rtp_transport):
        """Omit the Record-Route header when the INVITE contains none."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        invite = self._make_invite("answer-no-rr-1")
        protocol.request_received(invite, addr)
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert "Record-Route" not in response.headers


class TestCANCELHandler:
    def test_cancel__sends_200_ok_for_cancel(self):
        """Send a 200 OK response to the CANCEL request (RFC 3261 §9.2)."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        invite = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "cancel-test-1",
                "CSeq": "1 INVITE",
            },
        )
        protocol.request_received(invite, addr)
        cancel = Request(
            method="CANCEL",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "cancel-test-1",
                "CSeq": "1 CANCEL",
            },
        )
        protocol.request_received(cancel, addr)
        ok_response = next(
            (r for r, _ in protocol._sent_responses if r.status_code == 200), None
        )
        assert ok_response is not None

    def test_cancel__sends_487_request_terminated_for_invite(self):
        """Send a 487 Request Terminated for the pending INVITE (RFC 3261 §9.2)."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        invite = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "cancel-test-2",
                "CSeq": "1 INVITE",
            },
        )
        protocol.request_received(invite, addr)
        cancel = Request(
            method="CANCEL",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "cancel-test-2",
                "CSeq": "1 CANCEL",
            },
        )
        protocol.request_received(cancel, addr)
        terminated = next(
            (r for r, _ in protocol._sent_responses if r.status_code == 487), None
        )
        assert terminated is not None
        assert terminated.reason == "Request Terminated"

    def test_cancel__487_includes_to_tag(self):
        """Include the stored To tag in the 487 Request Terminated response."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        invite = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "cancel-tag-1",
                "CSeq": "1 INVITE",
            },
        )
        protocol.request_received(invite, addr)
        tag = protocol._to_tags["cancel-tag-1"]
        cancel = Request(
            method="CANCEL",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "cancel-tag-1",
                "CSeq": "1 CANCEL",
            },
        )
        protocol.request_received(cancel, addr)
        terminated = next(
            (r for r, _ in protocol._sent_responses if r.status_code == 487), None
        )
        assert terminated is not None
        assert f";tag={tag}" in terminated.headers.get("To", "")

    def test_cancel__cleans_up_state(self):
        """Remove Call-ID from _answered_calls, _pending_invites, and _to_tags."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        invite = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "cancel-cleanup-1",
                "CSeq": "1 INVITE",
            },
        )
        protocol.request_received(invite, addr)
        assert "cancel-cleanup-1" in protocol._answered_calls
        assert "cancel-cleanup-1" in protocol._to_tags
        cancel = Request(
            method="CANCEL",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "cancel-cleanup-1",
                "CSeq": "1 CANCEL",
            },
        )
        protocol.request_received(cancel, addr)
        assert "cancel-cleanup-1" not in protocol._answered_calls
        assert "cancel-cleanup-1" not in protocol._pending_invites
        assert "cancel-cleanup-1" not in protocol._to_tags

    def test_cancel__no_pending_invite_skips_487(self):
        """Skip sending 487 when no pending INVITE address is found."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        cancel = Request(
            method="CANCEL",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "cancel-no-invite-1",
                "CSeq": "1 CANCEL",
            },
        )
        protocol.request_received(cancel, addr)
        assert all(r.status_code != 487 for r, _ in protocol._sent_responses)
        ok_responses = [r for r, _ in protocol._sent_responses if r.status_code == 200]
        assert len(ok_responses) == 1

    def test_cancel__calls_cancel_received_hook(self):
        """Invoke cancel_received hook after handling a CANCEL request."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        received = []
        protocol.cancel_received = lambda req: received.append(req)
        cancel = Request(
            method="CANCEL",
            uri="sip:bob@biloxi.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "From": "sip:alice@atlanta.com",
                "To": "sip:bob@biloxi.com",
                "Call-ID": "cancel-hook-1",
                "CSeq": "1 CANCEL",
            },
        )
        protocol.request_received(cancel, addr)
        assert len(received) == 1
        assert received[0].method == "CANCEL"


# ---------------------------------------------------------------------------
# Helpers shared by TestSIPProtocol and TestRegistration below
# ---------------------------------------------------------------------------


def make_invite(headers: dict | None = None) -> Request:
    """Return an INVITE request with default dialog headers and no SDP body."""
    return Request(
        method="INVITE",
        uri="sip:alice@atlanta.com",
        headers={
            "Via": "SIP/2.0/UDP pc33.atlanta.com",
            "To": "sip:alice@atlanta.com",
            "From": "sip:bob@biloxi.com",
            "Call-ID": "1234@pc33",
            "CSeq": "1 INVITE",
            **(headers or {}),
        },
    )


def make_register_session(
    server_addr=("192.0.2.2", 5060),
    aor="sip:alice@example.com",
    username="alice",
    password="secret",  # noqa: S107
) -> SessionInitiationProtocol:
    """Return a SessionInitiationProtocol session without triggering connection_made."""
    return SessionInitiationProtocol(
        outbound_proxy=server_addr,
        aor=aor,
        username=username,
        password=password,
    )


def make_mock_transport(
    host: str = "127.0.0.1",
    port: int = 5061,
    peer: tuple[str, int] = ("192.0.2.1", 5061),
):
    """Return a MagicMock transport with get_extra_info configured for TLS."""
    from unittest.mock import MagicMock  # noqa: PLC0415

    transport = MagicMock()

    def _get_extra_info(key, default=None):
        if key == "sockname":
            return (host, port)
        if key == "peername":
            return peer
        if key == "ssl_object":
            return object()  # non-None signals TLS
        return default

    transport.get_extra_info.side_effect = _get_extra_info
    return transport


@dataclasses.dataclass
class _MinimalCall(Call):
    """Minimal Call subclass for SIP protocol tests that require codec negotiation."""

    @classmethod
    def negotiate_codec(cls, remote_media):
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        return MediaDescription(
            media="audio",
            port=0,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat.from_pt(0)],
        )


@dataclasses.dataclass
class _CodecAwareCall(Call):
    """Call subclass that performs real codec negotiation for SIP answer tests.

    Mirrors AudioCall.PREFERRED_CODECS without importing voip.audio.
    """

    @classmethod
    def negotiate_codec(cls, remote_media):
        from voip.rtp import RTPPayloadType  # noqa: PLC0415
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        preferred = [
            RTPPayloadFormat(
                payload_type=RTPPayloadType.OPUS,
                encoding_name="opus",
                sample_rate=48000,
                channels=2,
            ),
            RTPPayloadFormat(payload_type=RTPPayloadType.G722),
            RTPPayloadFormat(payload_type=RTPPayloadType.PCMA),
            RTPPayloadFormat(payload_type=RTPPayloadType.PCMU),
        ]
        if not remote_media.fmt:
            raise NotImplementedError("Remote SDP offer contains no audio formats")
        remote_pts = {f.payload_type for f in remote_media.fmt}
        for codec in preferred:
            if codec.payload_type in remote_pts:
                remote_fmt = remote_media.get_format(codec.payload_type)
                chosen = (
                    remote_fmt if remote_fmt and remote_fmt.encoding_name else codec
                )
                return MediaDescription(
                    media="audio", port=0, proto="RTP/AVP", fmt=[chosen]
                )
            for rfmt in remote_media.fmt:
                if (
                    rfmt.encoding_name
                    and rfmt.encoding_name.lower()
                    == (codec.encoding_name or "").lower()
                ):
                    return MediaDescription(
                        media="audio", port=0, proto="RTP/AVP", fmt=[rfmt]
                    )
        raise NotImplementedError(
            f"No supported codec in {[f.payload_type for f in remote_media.fmt]!r}"
        )


# ---------------------------------------------------------------------------
# Tests for the SIP protocol's call answering / rejection / transport layer
# ---------------------------------------------------------------------------


class TestSIPProtocol:
    """Tests for SIP protocol connection, dispatching, answer and reject."""

    class _CapturingSIP(SIP):
        """SIP subclass that captures sent messages without monkey-patching slots."""

        def __init__(self):
            super().__init__(
                outbound_proxy=("127.0.0.1", 5061),
                aor="sip:test@example.com",
            )
            self._sent: list[tuple] = []

        def send(self, message):
            self._sent.append((message, None))

    async def test_connection_made__stores_transport(self):
        """Store the transport when a connection is established."""
        protocol = SIP(
            outbound_proxy=("127.0.0.1", 5061),
            aor="sip:test@example.com",
            rtp_stun_server_address=None,
        )
        transport = make_mock_transport()
        protocol.connection_made(transport)
        assert protocol.transport is transport

    async def test_send__serializes_and_forwards_to_transport(self):
        """Serialize the message and forward it to the underlying TCP transport."""
        protocol = SIP(
            outbound_proxy=("127.0.0.1", 5061),
            aor="sip:test@example.com",
            rtp_stun_server_address=None,
        )
        transport = make_mock_transport()
        protocol.connection_made(transport)
        transport.write.reset_mock()  # clear any calls made during connection_made
        response = Response(status_code=200, reason="OK")
        protocol.send(response)
        protocol.transport.write.assert_called_once_with(bytes(response))

    async def test_request_received__invite__tracks_pending_call(self):
        """Dispatch an INVITE to call_received and track the Call-ID as pending."""
        received = []

        class MySIP(SIP):
            def call_received(self, request):
                received.append(request)

        protocol = MySIP(outbound_proxy=("127.0.0.1", 5060), aor="sip:test@example.com")
        protocol.connection_made(MagicMock())
        request = make_invite()
        addr = ("192.0.2.1", 5060)
        protocol.request_received(request, addr)
        assert len(received) == 1
        assert received[0] is request
        assert request.headers["Call-ID"] in protocol._pending_invites

    async def test_call_received__noop_by_default(self):
        """call_received is a no-op in the base class."""
        protocol = SIP(outbound_proxy=("127.0.0.1", 5060), aor="sip:test@example.com")
        protocol.connection_made(MagicMock())
        protocol.call_received(make_invite())  # must not raise

    async def test_answer__sends_200_ok(self):
        """Send a 200 OK response with an SDP body when answering."""
        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.local_address = ("127.0.0.1", 5061)
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        await protocol._answer(request, Call)
        assert len(protocol._sent) == 1
        response, _ = protocol._sent[0]
        assert response.status_code == 200
        assert response.reason == "OK"

    async def test_answer__sdp_contains_opus_audio_line(self):
        """Include an audio media line in the SDP body of the 200 OK."""
        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.local_address = ("127.0.0.1", 5061)
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        await protocol._answer(request, Call)
        response, _ = protocol._sent[0]
        assert b"m=audio" in bytes(response.body)
        assert b"RTP/SAVP 0" in bytes(response.body)

    async def _setup_answer_protocol(self):
        """Return a _CapturingSIP with a live mux, ready to answer an INVITE."""
        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.local_address = ("127.0.0.1", 5061)
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        return protocol

    async def test_answer__rtp_avp_offer_returns_rtp_avp(self):
        """When the remote offers RTP/AVP, respond with RTP/AVP (no SRTP)."""
        protocol = await self._setup_answer_protocol()

        # Build an INVITE with a plain RTP/AVP offer (no crypto).
        invite_bytes = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com\r\n"
            b"From: sip:alice@atlanta.com\r\n"
            b"To: sip:bob@biloxi.com\r\n"
            b"Call-ID: avp-offer-1\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"Content-Type: application/sdp\r\n"
            b"Content-Length: 68\r\n"
            b"\r\n"
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 192.0.2.1\r\n"
            b"s=-\r\n"
            b"c=IN IP4 192.0.2.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/AVP 0\r\n"
        )
        from voip.sip.messages import Request  # noqa: PLC0415

        request = Request.parse(invite_bytes)
        protocol._pending_invites.add(request.headers["Call-ID"])
        AudioCall = pytest.importorskip("voip.audio.AudioCall")

        await protocol._answer(request, AudioCall)
        response, _ = protocol._sent[0]
        body = bytes(response.body)
        assert b"RTP/AVP" in body
        assert b"RTP/SAVP" not in body
        assert b"crypto" not in body

        # The registered call handler must not have an SRTP session.
        handler = next(iter(protocol._rtp_protocol.calls.values()))
        assert handler.srtp is None

    async def test_answer__rtp_savp_offer_returns_rtp_savp(self):
        """When the remote offers RTP/SAVP, respond with RTP/SAVP (with SRTP)."""
        protocol = await self._setup_answer_protocol()

        invite_bytes = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com\r\n"
            b"From: sip:alice@atlanta.com\r\n"
            b"To: sip:bob@biloxi.com\r\n"
            b"Call-ID: savp-offer-1\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"Content-Type: application/sdp\r\n"
            b"Content-Length: 69\r\n"
            b"\r\n"
            b"v=0\r\n"
            b"o=- 0 0 IN IP4 192.0.2.1\r\n"
            b"s=-\r\n"
            b"c=IN IP4 192.0.2.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 49170 RTP/SAVP 0\r\n"
        )
        from voip.sip.messages import Request  # noqa: PLC0415

        request = Request.parse(invite_bytes)
        protocol._pending_invites.add(request.headers["Call-ID"])
        AudioCall = pytest.importorskip("voip.audio.AudioCall")

        await protocol._answer(request, AudioCall)
        response, _ = protocol._sent[0]
        body = bytes(response.body)
        assert b"RTP/SAVP" in body
        assert b"crypto" in body

        handler = next(iter(protocol._rtp_protocol.calls.values()))
        assert handler.srtp is not None

    async def test_answer__copies_dialog_headers(self):
        """Copy Via, To, From, Call-ID, and CSeq headers into the 200 OK."""
        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.local_address = ("127.0.0.1", 5061)
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        await protocol._answer(request, Call)
        response, _ = protocol._sent[0]
        assert response.headers["Via"] == "SIP/2.0/UDP pc33.atlanta.com"
        assert response.headers["To"] == "sip:alice@atlanta.com"
        assert response.headers["From"] == "sip:bob@biloxi.com"
        assert response.headers["Call-ID"] == "1234@pc33"
        assert response.headers["CSeq"] == "1 INVITE"

    async def test_answer__instantiates_call_class_with_caller(self):
        """The call_class is instantiated with the caller from the From header."""
        created: list[str] = []

        @dataclasses.dataclass
        class MyCall(Call):
            def __post_init__(self) -> None:
                created.append(str(self.caller))

        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.local_address = ("127.0.0.1", 5061)
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        await protocol._answer(request, MyCall)
        assert created == ["sip:bob@biloxi.com"]

    async def test_answer__rtp_receives_audio(self):
        """Deliver SRTP payloads to the call handler's datagram_received (decrypted)."""
        received_payloads: list[bytes] = []

        @dataclasses.dataclass
        class DatagramCapture(Call):
            def datagram_received(self, data: bytes, addr) -> None:
                # RTP fixed header is 12 bytes; extract raw payload.
                received_payloads.append(data[12:])

        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.local_address = ("127.0.0.1", 5061)
        mux = RealtimeTransportProtocol(stun_server_address=None)
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: mux, local_addr=("127.0.0.1", 0)
        )
        protocol._rtp_protocol = mux
        protocol._rtp_transport = rtp_transport
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        try:
            await protocol._answer(request, DatagramCapture)
            response, _ = protocol._sent[0]
            sdp_line = next(
                line
                for line in bytes(response.body).decode().splitlines()
                if line.startswith("m=audio")
            )
            rtp_port = int(sdp_line.split()[1])

            send_transport, _ = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol,
                remote_addr=("127.0.0.1", rtp_port),
            )
            # Get the SRTP session from the registered call handler and encrypt.
            call_handler = mux.calls.get(None)
            rtp_packet = b"\x80\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00audio"
            srtp_packet = call_handler.srtp.encrypt(rtp_packet)
            send_transport.sendto(srtp_packet)
            await asyncio.sleep(0.05)
            send_transport.close()
            assert received_payloads == [b"audio"]
        finally:
            rtp_transport.close()

    async def test_answer__rtp_receives_multiple_packets(self):
        """Call datagram_received for each SRTP packet that arrives (decrypted)."""
        received_payloads: list[bytes] = []

        @dataclasses.dataclass
        class DatagramCapture(Call):
            def datagram_received(self, data: bytes, addr) -> None:
                received_payloads.append(data[12:])

        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.local_address = ("127.0.0.1", 5061)
        mux = RealtimeTransportProtocol(stun_server_address=None)
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: mux, local_addr=("127.0.0.1", 0)
        )
        protocol._rtp_protocol = mux
        protocol._rtp_transport = rtp_transport
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        try:
            await protocol._answer(request, DatagramCapture)
            response, _ = protocol._sent[0]
            sdp_line = next(
                line
                for line in bytes(response.body).decode().splitlines()
                if line.startswith("m=audio")
            )
            rtp_port = int(sdp_line.split()[1])

            send_transport, _ = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol,
                remote_addr=("127.0.0.1", rtp_port),
            )
            # Get the SRTP session from the registered call handler and encrypt.
            call_handler = mux.calls.get(None)
            header = b"\x80\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00"
            send_transport.sendto(call_handler.srtp.encrypt(header + b"chunk1"))
            send_transport.sendto(call_handler.srtp.encrypt(header + b"chunk2"))
            await asyncio.sleep(0.05)
            send_transport.close()
            assert received_payloads == [b"chunk1", b"chunk2"]
        finally:
            rtp_transport.close()

    async def test_answer__content_length_serialized(self):
        """Content-Length is automatically included when the response is serialized."""
        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.local_address = ("127.0.0.1", 5061)
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        await protocol._answer(request, Call)
        response, _ = protocol._sent[0]
        serialized = bytes(response)
        parsed = Message.parse(serialized)
        assert "Content-Length" in parsed.headers

    async def test_answer__reuses_shared_rtp_socket_for_second_call(self):
        """A second _answer() reuses the same shared RTP socket (one port for all calls)."""
        from voip.sdp.messages import SessionDescription  # noqa: PLC0415
        from voip.sip.messages import Request as SIPRequest  # noqa: PLC0415

        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.local_address = ("127.0.0.1", 5061)
        mux = RealtimeTransportProtocol(stun_server_address=None)
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: mux, local_addr=("127.0.0.1", 0)
        )
        protocol._rtp_protocol = mux
        protocol._rtp_transport = rtp_transport

        sdp_body1 = SessionDescription.parse(
            b"v=0\r\no=- 0 0 IN IP4 1.2.3.4\r\ns=-\r\nc=IN IP4 1.2.3.4\r\nt=0 0\r\nm=audio 5000 RTP/AVP 0\r\n"
        )
        invite1 = SIPRequest(
            method="INVITE",
            uri="sip:alice@atlanta.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "To": "sip:alice@atlanta.com",
                "From": "sip:bob@biloxi.com",
                "Call-ID": "call-1@test",
                "CSeq": "1 INVITE",
            },
            body=sdp_body1,
        )
        protocol._pending_invites.add("call-1@test")
        await protocol._answer(invite1, _MinimalCall)
        rtp_proto_1 = protocol._rtp_protocol
        rtp_transport_1 = protocol._rtp_transport

        sdp_body2 = SessionDescription.parse(
            b"v=0\r\no=- 0 0 IN IP4 5.6.7.8\r\ns=-\r\nc=IN IP4 5.6.7.8\r\nt=0 0\r\nm=audio 6000 RTP/AVP 0\r\n"
        )
        invite2 = SIPRequest(
            method="INVITE",
            uri="sip:alice@atlanta.com",
            headers={
                "Via": "SIP/2.0/UDP pc33.atlanta.com",
                "To": "sip:alice@atlanta.com",
                "From": "sip:charlie@biloxi.com",
                "Call-ID": "call-2@test",
                "CSeq": "1 INVITE",
            },
            body=sdp_body2,
        )
        protocol._pending_invites.add("call-2@test")
        await protocol._answer(invite2, _MinimalCall)

        assert protocol._rtp_protocol is rtp_proto_1
        assert protocol._rtp_transport is rtp_transport_1
        assert ("1.2.3.4", 5000) in rtp_proto_1.calls
        assert ("5.6.7.8", 6000) in rtp_proto_1.calls

        rtp_transport.close()

    async def test_answer__bye_unregisters_call_from_rtp_mux(self):
        """BYE for an active call removes its handler from the shared RTP mux."""
        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.local_address = ("127.0.0.1", 5061)
        mux = RealtimeTransportProtocol(stun_server_address=None)
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: mux, local_addr=("127.0.0.1", 0)
        )
        protocol._rtp_protocol = mux
        protocol._rtp_transport = rtp_transport
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        try:
            await protocol._answer(request, Call)
            assert None in mux.calls

            bye = Request(
                method="BYE",
                uri="sip:alice@atlanta.com",
                headers={
                    "Via": "SIP/2.0/UDP pc33.atlanta.com",
                    "To": "sip:alice@atlanta.com",
                    "From": "sip:bob@biloxi.com",
                    "Call-ID": request.headers["Call-ID"],
                    "CSeq": "2 BYE",
                },
            )
            protocol.request_received(bye, ("192.0.2.1", 5060))
            assert None not in mux.calls
        finally:
            rtp_transport.close()

    async def test_answer__logs_info(self, caplog):
        """Log an info message when answering a call."""
        import logging

        loop = asyncio.get_running_loop()
        protocol = self._CapturingSIP()
        protocol.transport = make_mock_transport()
        protocol.local_address = ("127.0.0.1", 5061)
        mux = RealtimeTransportProtocol()
        mux.public_address = loop.create_future()
        mux.public_address.set_result(("127.0.0.1", 12000))
        mock_rtp_transport = MagicMock()
        mock_rtp_transport.get_extra_info.return_value = ("127.0.0.1", 12000)
        protocol._rtp_protocol = mux
        protocol._rtp_transport = mock_rtp_transport
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        with caplog.at_level(logging.INFO, logger="voip.sip"):
            await protocol._answer(request, Call)
        assert any("call_answered" in r.message for r in caplog.records)

    def test_reject__sends_busy_here_by_default(self):
        """Send a 486 Busy Here response when no status code is given."""
        protocol = self._CapturingSIP()
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        protocol.reject(request)
        assert len(protocol._sent) == 1
        response, _ = protocol._sent[0]
        assert isinstance(response, Response)
        assert response.status_code == 486
        assert response.reason == "Busy Here"

    def test_reject__custom_status(self):
        """Send the specified status code and reason."""
        protocol = self._CapturingSIP()
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        protocol.reject(request, status_code=603, reason="Decline")
        response, _ = protocol._sent[0]
        assert response.status_code == 603
        assert response.reason == "Decline"

    def test_reject__copies_dialog_headers(self):
        """Copy Via, To, From, Call-ID, and CSeq headers into the response."""
        protocol = self._CapturingSIP()
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        protocol.reject(request)
        response, _ = protocol._sent[0]
        assert response.headers["Via"] == "SIP/2.0/UDP pc33.atlanta.com"
        assert response.headers["To"] == "sip:alice@atlanta.com"
        assert response.headers["From"] == "sip:bob@biloxi.com"
        assert response.headers["Call-ID"] == "1234@pc33"
        assert response.headers["CSeq"] == "1 INVITE"

    @pytest.mark.parametrize("extra_header", ["X-Custom"])
    def test_reject__excludes_extra_headers(self, extra_header):
        """Exclude non-dialog headers from the reject response."""
        protocol = self._CapturingSIP()
        request = make_invite({extra_header: "value"})
        protocol._pending_invites.add(request.headers["Call-ID"])
        protocol.reject(request)
        response, _ = protocol._sent[0]
        assert extra_header not in response.headers

    def test_reject__logs_info(self, caplog):
        """Log an info message when rejecting a call."""
        import logging

        with caplog.at_level(logging.INFO, logger="voip.sip"):
            protocol = self._CapturingSIP()
            request = make_invite()
            protocol._pending_invites.add(request.headers["Call-ID"])
            protocol.reject(request)
        assert any("call_rejected" in r.message for r in caplog.records)

    async def test_data_received__keepalive__sends_pong(self):
        """Double-CRLF keepalive (RFC 5626 §4.4.1) is answered with a single-CRLF pong."""
        protocol = SIP(
            outbound_proxy=("127.0.0.1", 5061),
            aor="sip:test@example.com",
            rtp_stun_server_address=None,
        )
        transport = make_mock_transport()
        protocol.connection_made(transport)
        transport.write.reset_mock()
        protocol.data_received(b"\r\n\r\n")
        transport.write.assert_called_once_with(b"\r\n")

    async def test_request_received__unsupported_method__raises(self):
        """Raise NotImplementedError for any non-INVITE SIP request method."""
        protocol = SIP(outbound_proxy=("127.0.0.1", 5060), aor="sip:test@example.com")
        protocol.connection_made(MagicMock())
        request = Request(method="OPTIONS", uri="sip:alice@atlanta.com")
        with pytest.raises(NotImplementedError, match="OPTIONS"):
            protocol.request_received(request, ("192.0.2.1", 5060))

    async def test_answer__via_call_received__schedules_answer(self):
        """answer() is async; wrapping it in create_task from call_received works."""
        answered = []

        class MySIP(SIP):
            def call_received(self, request):
                asyncio.create_task(
                    self.answer(request=request, call_class=_MinimalCall)
                )

            async def _answer(self, request, call_class):
                answered.append((request, call_class))

        protocol = MySIP(outbound_proxy=("127.0.0.1", 5060), aor="sip:test@example.com")
        protocol.connection_made(MagicMock())
        request = make_invite()
        protocol._pending_invites.add(request.headers["Call-ID"])
        protocol.call_received(request)

        await asyncio.sleep(0.01)
        assert len(answered) == 1
        assert answered[0][1] is _MinimalCall


# ---------------------------------------------------------------------------
# Tests for SIP REGISTER / digest-auth / response handling
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registrar_uri__strips_user_from_aor(self):
        """Derive registrar URI from AOR by stripping the user part."""
        p = make_register_session(aor="sip:alice@example.com")
        assert p.registrar_uri == "sip:example.com"

    def test_registrar_uri__preserves_port(self):
        """Preserve a non-default port in the derived registrar URI."""
        p = make_register_session(aor="sip:alice@example.com:5080")
        assert p.registrar_uri == "sip:example.com:5080"

    def test_registrar_uri__preserves_sips_scheme(self):
        """sips: AOR produces sips: registrar URI (RFC 3261 §10.2)."""
        p = make_register_session(aor="sips:alice@example.com")
        assert p.registrar_uri == "sips:example.com"

    def test_registrar_uri__preserves_sip_scheme(self):
        """sip: AOR produces sip: registrar URI regardless of transport."""
        p = make_register_session(aor="sip:alice@example.com")
        p._is_tls = True  # TLS transport should not change the scheme
        assert p.registrar_uri == "sip:example.com"

    async def test_connection_made__sends_register(self):
        """Send a REGISTER request when the connection is established."""

        class _SessionNoRTP(SessionInitiationProtocol):
            async def _initialize(self):
                # Skip real socket creation and just register directly.
                await self.register()

        p = _SessionNoRTP(
            outbound_proxy=("192.0.2.2", 5061),
            aor="sip:alice@example.com",
            username="alice",
            password="secret",  # noqa: S106
        )
        transport = make_mock_transport()
        p.connection_made(transport)
        await asyncio.sleep(0.05)
        transport.write.assert_called()
        (data,) = transport.write.call_args[0]
        # sip: AOR → sip: registrar URI even over TLS.
        assert b"REGISTER sip:example.com SIP/2.0" in data

    async def test_register__includes_required_headers(self):
        """REGISTER request includes From, To, Call-ID, CSeq, Contact and Expires."""
        p = make_register_session()
        transport = make_mock_transport()
        p.transport = transport
        p.local_address = ("127.0.0.1", 5061)
        p._is_tls = True
        await p.register()
        (data,) = transport.write.call_args[0]
        assert b"From: sip:alice@example.com" in data
        assert b"To: sip:alice@example.com" in data
        # sip: AOR over TLS → sip:;transport=tls Contact
        assert b"Contact: <sip:alice@127.0.0.1:5061;transport=tls>" in data
        assert b"Expires: 3600" in data

    async def test_register__increments_cseq(self):
        """CSeq increments with each REGISTER sent."""
        p = make_register_session()
        p.local_address = ("127.0.0.1", 5061)
        p.transport = make_mock_transport()
        await p.register()
        assert p.cseq == 1
        await p.register()
        assert p.cseq == 2

    @pytest.mark.asyncio
    async def test_register__with_authorization(self):
        """Authorization header is included when credentials are provided."""
        p = make_register_session()
        transport = make_mock_transport()
        p.transport = transport
        p.local_address = "127.0.0.1", 5061
        await p.register(authorization='Digest username="alice"')
        (data,) = transport.write.call_args[0]
        assert b'Authorization: Digest username="alice"' in data

    async def test_register__with_proxy_authorization(self):
        """Proxy-Authorization header is included for proxy challenges."""
        p = make_register_session()
        transport = make_mock_transport()
        p.transport = transport
        p.local_address = "127.0.0.1", 5061
        await p.register(proxy_authorization='Digest username="alice"')
        (data,) = transport.write.call_args[0]
        assert b'Proxy-Authorization: Digest username="alice"' in data

    async def test_response_received__200_ok_calls_registered(self):
        """Receiving 200 OK for REGISTER triggers registered()."""
        calls = []

        class ConcreteSession(SessionInitiationProtocol):
            def registered(self):
                calls.append(True)

        p = ConcreteSession(
            outbound_proxy=("192.0.2.2", 5060),
            aor="sip:alice@example.com",
            username="a",
            password="b",  # noqa: S106
        )
        p.connection_made(make_mock_transport())
        p.response_received(
            Response(status_code=200, reason="OK", headers={"CSeq": "1 REGISTER"}),
            ("192.0.2.2", 5060),
        )
        assert calls == [True]

    async def test_response_received__200_non_register_raises(self):
        """Receiving 200 OK for a non-REGISTER method raises RegistrationError."""
        p = make_register_session()
        p.connection_made(make_mock_transport())
        with pytest.raises(RegistrationError):
            p.response_received(
                Response(status_code=200, reason="OK", headers={"CSeq": "1 INVITE"}),
                ("192.0.2.2", 5060),
            )

    async def test_response_received__401_retries_with_authorization(self):
        """Receiving 401 triggers a re-REGISTER with an Authorization header."""
        p = make_register_session(username="alice", password="secret")  # noqa: S106
        transport = make_mock_transport()
        p.transport = transport
        p.local_address = ("127.0.0.1", 5061)
        challenge = 'Digest realm="example.com", nonce="abc123"'
        p.response_received(
            Response(
                status_code=401,
                reason="Unauthorized",
                headers={"WWW-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5061),
        )
        await asyncio.sleep(0.05)
        (data,) = transport.write.call_args[0]
        assert b"Authorization: Digest" in data
        assert b'username="alice"' in data
        assert b'realm="example.com"' in data
        assert b'nonce="abc123"' in data
        assert b'algorithm="SHA-256"' in data

    async def test_response_received__407_retries_with_proxy_authorization(self):
        """Receiving 407 triggers a re-REGISTER with a Proxy-Authorization header."""
        p = make_register_session(username="alice", password="secret")  # noqa: S106
        transport = make_mock_transport()
        p.transport = transport
        p.local_address = ("127.0.0.1", 5061)
        challenge = 'Digest realm="example.com", nonce="xyz"'
        p.response_received(
            Response(
                status_code=407,
                reason="Proxy Auth Required",
                headers={"Proxy-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5061),
        )
        await asyncio.sleep(0.05)
        (data,) = transport.write.call_args[0]
        assert b"Proxy-Authorization: Digest" in data
        assert b'username="alice"' in data

    async def test_response_received__401_with_qop_auth_includes_nc_cnonce(self):
        """401 with qop=auth causes the retry to include nc and cnonce fields."""
        p = make_register_session()
        transport = make_mock_transport()
        p.transport = transport
        p.local_address = ("127.0.0.1", 5061)
        challenge = 'Digest realm="example.com", nonce="n", qop="auth"'
        p.response_received(
            Response(
                status_code=401,
                reason="Unauthorized",
                headers={"WWW-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5061),
        )
        await asyncio.sleep(0.05)
        (data,) = transport.write.call_args[0]
        assert b"qop=auth" in data
        assert b"nc=00000001" in data
        assert b"cnonce=" in data

    async def test_response_received__401_with_opaque_echoes_opaque(self):
        """The opaque field from the challenge is echoed back in the Authorization."""
        p = make_register_session()
        transport = make_mock_transport()
        p.transport = transport
        p.local_address = ("127.0.0.1", 5061)
        challenge = 'Digest realm="example.com", nonce="n", opaque="secret-opaque"'
        p.response_received(
            Response(
                status_code=401,
                reason="Unauthorized",
                headers={"WWW-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5061),
        )
        await asyncio.sleep(0.05)
        (data,) = transport.write.call_args[0]
        assert b'opaque="secret-opaque"' in data

    async def test_register__via_header_has_rport(self):
        """REGISTER request includes a Via header with the rport parameter."""
        import re

        p = make_register_session()
        p.local_address = ("192.0.2.10", 5061)
        transport = make_mock_transport("192.0.2.10", 5061)
        p.transport = transport
        p._is_tls = True
        await p.register()
        (data,) = transport.write.call_args[0]
        assert b"Via: SIP/2.0/TLS 192.0.2.10:5061;rport;branch=z9hG4bK" in data
        assert re.search(rb"branch=z9hG4bK[0-9a-f]{32}", data)

    async def test_register__via_branch_is_unique_per_request(self):
        """Each REGISTER generates a unique Via branch."""
        import re

        p = make_register_session()
        p.local_address = "127.0.0.1", 5061
        transport = make_mock_transport()
        p.transport = transport
        await p.register()
        (data1,) = transport.write.call_args[0]
        transport.reset_mock()
        await p.register()
        (data2,) = transport.write.call_args[0]
        branch1 = re.search(rb"branch=(z9hG4bK[0-9a-f]{32})", data1).group(1)
        branch2 = re.search(rb"branch=(z9hG4bK[0-9a-f]{32})", data2).group(1)
        assert branch1 != branch2

    async def test_register__contact_uses_local_addr(self):
        """Contact header uses sip:;transport=tls when AOR is sip: over TLS."""
        p = make_register_session()
        p.local_address = "10.0.0.5", 5061
        transport = make_mock_transport("10.0.0.5", 5061)
        p.transport = transport
        p._is_tls = True
        await p.register()
        (data,) = transport.write.call_args[0]
        assert b"Contact: <sip:alice@10.0.0.5:5061;transport=tls>" in data

    async def test_register__contact_uses_sips_when_aor_is_sips(self):
        """Contact header uses sips: when AOR scheme is sips:."""
        p = make_register_session(aor="sips:alice@example.com")
        p.local_address = "10.0.0.5", 5061
        transport = make_mock_transport("10.0.0.5", 5061)
        p.transport = transport
        p._is_tls = True
        await p.register()
        (data,) = transport.write.call_args[0]
        assert b"Contact: <sips:alice@10.0.0.5:5061>" in data

    async def test_response_received__403_raises_registration_error(self):
        """403 Forbidden for REGISTER raises RegistrationError with the response message."""
        p = make_register_session()
        p.local_address = ("127.0.0.1", 5061)
        p.transport = make_mock_transport()
        with pytest.raises(RegistrationError, match="403 Forbidden"):
            p.response_received(
                Response(
                    status_code=403,
                    reason="Forbidden",
                    headers={"CSeq": "1 REGISTER"},
                ),
                ("192.0.2.2", 5061),
            )

    async def test_response_received__unexpected_raises_registration_error(self):
        """Any unexpected REGISTER response raises RegistrationError."""
        p = make_register_session()
        p.local_address = ("127.0.0.1", 5061)
        p.transport = make_mock_transport()
        with pytest.raises(RegistrationError, match="500 Server Error"):
            p.response_received(
                Response(
                    status_code=500,
                    reason="Server Error",
                    headers={"CSeq": "1 REGISTER"},
                ),
                ("192.0.2.2", 5061),
            )

    async def test_data_received__sip_response__calls_response_received(self):
        """data_received routes SIP messages to response_received via TCP stream."""
        received = []

        class ConcreteSession(SessionInitiationProtocol):
            def response_received(self, response, addr):
                received.append(response)

        p = ConcreteSession(
            outbound_proxy=("192.0.2.2", 5061),
            aor="sip:alice@example.com",
            username="a",
            password="b",  # noqa: S106
        )
        p.connection_made(make_mock_transport())
        sip_data = b"SIP/2.0 200 OK\r\nCSeq: 1 REGISTER\r\n\r\n"
        p.data_received(sip_data)
        assert len(received) == 1
        assert received[0].status_code == 200

    async def test_invite_received_after_register(self):
        """INVITE dispatching still works after registration."""
        received = []

        class ConcreteSession(SessionInitiationProtocol):
            def call_received(self, request):
                received.append(request)

        p = ConcreteSession(
            outbound_proxy=("192.0.2.2", 5060),
            aor="sip:alice@example.com",
            username="a",
            password="b",  # noqa: S106
        )
        p.connection_made(make_mock_transport())
        request = Request(
            method="INVITE",
            uri="sip:alice@example.com",
            headers={"From": "sip:bob@example.com", "Call-ID": "test@pc"},
        )
        p.request_received(request, ("192.0.2.1", 5060))
        assert len(received) == 1
        assert received[0] is request

    async def test_response_received__200_ok__logs_info(self, caplog):
        """Receiving 200 OK logs an info message."""
        import logging

        p = make_register_session()
        p.connection_made(make_mock_transport())
        with caplog.at_level(logging.INFO, logger="voip.sip"):
            p.response_received(
                Response(status_code=200, reason="OK", headers={"CSeq": "1 REGISTER"}),
                ("192.0.2.2", 5060),
            )
        assert any("Registration successful" in r.message for r in caplog.records)

    async def test_response_received__unexpected_status__raises_registration_error(
        self, caplog
    ):
        """An unhandled status code raises RegistrationError with status and reason."""
        import logging

        p = make_register_session()
        p.connection_made(make_mock_transport())
        with caplog.at_level(logging.WARNING, logger="voip.sip"):
            with pytest.raises(RegistrationError, match="500 Server Error"):
                p.response_received(
                    Response(
                        status_code=500,
                        reason="Server Error",
                        headers={"CSeq": "1 REGISTER"},
                    ),
                    ("192.0.2.2", 5060),
                )


# ---------------------------------------------------------------------------
# Tests for digest_response (RFC 3261 §22, RFC 8760)
# ---------------------------------------------------------------------------


class TestDigestResponse:
    """Unit tests for SessionInitiationProtocol.digest_response."""

    def test_default_algorithm_is_sha256(self):
        """digest_response uses SHA-256 by default (RFC 8760)."""
        result = SessionInitiationProtocol.digest_response(
            username="alice",
            password="secret",  # noqa: S106
            realm="example.com",
            nonce="abc123",
            method="REGISTER",
            uri="sip:example.com",
        )
        # SHA-256 produces a 64-character hex digest
        assert len(result) == 64

    def test_sha256_response_is_correct(self):
        """SHA-256 digest is computed from the correct input per RFC 8760."""
        username, password, realm, nonce, method, uri = (
            "alice",
            "secret",
            "example.com",
            "abc123",
            "REGISTER",
            "sip:example.com",
        )
        ha1 = hashlib.sha256(f"{username}:{realm}:{password}".encode()).hexdigest()
        ha2 = hashlib.sha256(f"{method}:{uri}".encode()).hexdigest()
        expected = hashlib.sha256(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()

        result = SessionInitiationProtocol.digest_response(
            username=username,
            password=password,  # noqa: S106
            realm=realm,
            nonce=nonce,
            method=method,
            uri=uri,
            algorithm=DigestAlgorithm.SHA_256,
        )
        assert result == expected

    def test_md5_response_is_32_hex_chars(self):
        """MD5 algorithm still produces a valid 32-character hex digest."""
        result = SessionInitiationProtocol.digest_response(
            username="alice",
            password="secret",  # noqa: S106
            realm="example.com",
            nonce="abc123",
            method="REGISTER",
            uri="sip:example.com",
            algorithm=DigestAlgorithm.MD5,
        )
        assert len(result) == 32

    def test_sha512_256_response_is_64_hex_chars(self):
        """SHA-512-256 produces a 64-character hex digest."""
        result = SessionInitiationProtocol.digest_response(
            username="alice",
            password="secret",  # noqa: S106
            realm="example.com",
            nonce="abc123",
            method="REGISTER",
            uri="sip:example.com",
            algorithm=DigestAlgorithm.SHA_512_256,
        )
        assert len(result) == 64

    def test_algorithms_produce_distinct_responses(self):
        """Different algorithms produce distinct digest values."""
        digest_params = {
            "username": "alice",
            "password": "secret",  # noqa: S106
            "realm": "example.com",
            "nonce": "abc123",
            "method": "REGISTER",
            "uri": "sip:example.com",
        }
        r_md5 = SessionInitiationProtocol.digest_response(
            **digest_params, algorithm=DigestAlgorithm.MD5
        )
        r_sha256 = SessionInitiationProtocol.digest_response(
            **digest_params, algorithm=DigestAlgorithm.SHA_256
        )
        r_sha512 = SessionInitiationProtocol.digest_response(
            **digest_params, algorithm=DigestAlgorithm.SHA_512_256
        )
        assert r_md5 != r_sha256
        assert r_sha256 != r_sha512

    async def test_response_received__401_uses_server_algorithm(self):
        """401 challenge with algorithm=SHA-256 causes Authorization to echo SHA-256."""
        p = make_register_session(username="alice", password="secret")  # noqa: S106
        transport = make_mock_transport()
        p.transport = transport
        p.local_address = ("127.0.0.1", 5061)
        challenge = 'Digest realm="example.com", nonce="abc123", algorithm="SHA-256"'
        p.response_received(
            Response(
                status_code=401,
                reason="Unauthorized",
                headers={"WWW-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5061),
        )
        await asyncio.sleep(0.05)
        (data,) = transport.write.call_args[0]
        assert b'algorithm="SHA-256"' in data
        assert b'algorithm="MD5"' not in data

    def test_sess_algorithm_incorporates_cnonce(self):
        """SHA-256-sess and MD5-sess include cnonce in HA1, changing the result."""
        base = {
            "username": "alice",
            "password": "secret",  # noqa: S106
            "realm": "example.com",
            "nonce": "abc123",
            "method": "REGISTER",
            "uri": "sip:example.com",
        }
        r1 = SessionInitiationProtocol.digest_response(
            **base, algorithm=DigestAlgorithm.SHA_256_SESS, cnonce="cnonce-A"
        )
        r2 = SessionInitiationProtocol.digest_response(
            **base, algorithm=DigestAlgorithm.SHA_256_SESS, cnonce="cnonce-B"
        )
        # Different cnonce values must yield different responses
        assert r1 != r2
        # A -sess result must differ from the non-sess result for the same inputs
        r_plain = SessionInitiationProtocol.digest_response(
            **base, algorithm=DigestAlgorithm.SHA_256
        )
        assert r1 != r_plain

    def test_md5_sess_incorporates_cnonce(self):
        """MD5-sess includes cnonce in HA1."""
        base = {
            "username": "alice",
            "password": "secret",  # noqa: S106
            "realm": "example.com",
            "nonce": "abc123",
            "method": "REGISTER",
            "uri": "sip:example.com",
        }
        r1 = SessionInitiationProtocol.digest_response(
            **base, algorithm=DigestAlgorithm.MD5_SESS, cnonce="cnonce-A"
        )
        r2 = SessionInitiationProtocol.digest_response(
            **base, algorithm=DigestAlgorithm.MD5_SESS, cnonce="cnonce-B"
        )
        assert r1 != r2

    def test_sess_algorithm_without_cnonce_raises(self):
        """Calling a *-sess algorithm without cnonce raises ValueError."""
        with pytest.raises(ValueError, match="cnonce"):
            SessionInitiationProtocol.digest_response(
                username="alice",
                password="secret",  # noqa: S106
                realm="example.com",
                nonce="abc123",
                method="REGISTER",
                uri="sip:example.com",
                algorithm=DigestAlgorithm.SHA_256_SESS,
                cnonce=None,
            )

    def test_unsupported_algorithm_raises(self):
        """An unrecognised algorithm string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported digest algorithm"):
            SessionInitiationProtocol.digest_response(
                username="alice",
                password="secret",  # noqa: S106
                realm="example.com",
                nonce="abc123",
                method="REGISTER",
                uri="sip:example.com",
                algorithm="BLAKE2b",
            )
