"""Tests for the SIP asyncio protocol handler."""

import asyncio
import errno
import unittest.mock

import pytest
from voip.sdp.messages import SessionDescription
from voip.sdp.types import Timing
from voip.sip.messages import Request, Response
from voip.sip.protocol import SessionInitiationProtocol

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
    """Fake UDP transport for testing."""

    def __init__(self, local_addr=("127.0.0.1", 5060)):
        self._local_addr = local_addr
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append((data, addr))

    def get_extra_info(self, key, default=None):
        return self._local_addr if key == "sockname" else default


class FakeProtocol(SessionInitiationProtocol):
    """Fake SIP protocol that captures sent messages."""

    def __init__(self):
        super().__init__()
        self._transport = FakeTransport()
        self._sent_responses: list[tuple[Response, tuple]] = []

    def send(self, message, addr):
        if isinstance(message, Response):
            self._sent_responses.append((message, addr))
        super().send(message, addr)


class ConcreteProtocol(SessionInitiationProtocol):
    """Concrete subclass for testing that records received messages."""

    def __init__(self):
        super().__init__()
        self.requests = []
        self.responses = []

    def request_received(self, request, addr):
        self.requests.append((request, addr))

    def response_received(self, response, addr):
        self.responses.append((response, addr))


class TestSessionInitiationProtocol:
    def test_datagram_received__request(self):
        """Dispatch a received SIP request datagram to request_received."""
        protocol = ConcreteProtocol()
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com\r\n"
            b"\r\n"
        )
        addr = ("192.0.2.1", 5060)
        protocol.datagram_received(data, addr)
        assert len(protocol.requests) == 1
        request, called_addr = protocol.requests[0]
        assert isinstance(request, Request)
        assert request.method == "INVITE"
        assert called_addr == addr

    def test_datagram_received__response(self):
        """Dispatch a received SIP response datagram to response_received."""
        protocol = ConcreteProtocol()
        data = b"SIP/2.0 200 OK\r\nVia: SIP/2.0/UDP pc33.atlanta.com\r\n\r\n"
        addr = ("192.0.2.1", 5060)
        protocol.datagram_received(data, addr)
        assert len(protocol.responses) == 1
        response, called_addr = protocol.responses[0]
        assert isinstance(response, Response)
        assert response.status_code == 200
        assert called_addr == addr

    def test_error_received__blocking_io(self):
        """Log blocking IO errors without re-raising."""
        protocol = SessionInitiationProtocol()
        exc = OSError(errno.EAGAIN, "Resource temporarily unavailable")
        protocol.error_received(exc)  # should not raise

    def test_error_received__reraises(self):
        """Re-raise unexpected transport errors."""
        protocol = SessionInitiationProtocol()
        exc = OSError("Unexpected error")
        with pytest.raises(OSError):
            protocol.error_received(exc)

    def test_connection_lost__no_exception(self):
        """Handle a clean connection close without raising."""
        protocol = SessionInitiationProtocol()
        protocol.connection_lost(None)  # should not raise

    def test_connection_lost__with_exception(self):
        """Log an exception on connection lost without re-raising."""
        protocol = SessionInitiationProtocol()
        protocol.connection_lost(Exception("Connection reset"))  # should not raise


class TestWithToTag:
    def test__with_to_tag__adds_tag(self):
        """Append the To tag to the To header for a known Call-ID."""
        protocol = SessionInitiationProtocol()
        protocol._to_tags["call-1"] = "abc123"
        result = protocol._with_to_tag({"To": "sip:bob@biloxi.com"}, "call-1")
        assert result["To"] == "sip:bob@biloxi.com;tag=abc123"

    def test__with_to_tag__unknown_call_id(self):
        """Leave the To header unchanged when the Call-ID has no stored tag."""
        protocol = SessionInitiationProtocol()
        result = protocol._with_to_tag({"To": "sip:bob@biloxi.com"}, "unknown")
        assert result["To"] == "sip:bob@biloxi.com"

    def test__with_to_tag__missing_to_header(self):
        """Return an empty To header when none is present and no tag exists."""
        protocol = SessionInitiationProtocol()
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
        """Log an error and send nothing when no address is stored for the Call-ID."""
        protocol = FakeProtocol()
        request = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={"Call-ID": "nonexistent"},
        )
        with caplog.at_level("ERROR"):
            protocol.ringing(request)
        assert not protocol._sent_responses
        assert "No address found" in caplog.text


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
        """Log an error and send nothing when no address is stored for the Call-ID."""
        protocol = FakeProtocol()
        request = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={"Call-ID": "nonexistent"},
        )
        with caplog.at_level("ERROR"):
            protocol.reject(request)
        assert not protocol._sent_responses
        assert "No address found" in caplog.text


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
        """Run _answer coroutine synchronously using a new event loop."""

        class FakeRTPProtocol(asyncio.DatagramProtocol):
            def __init__(self, caller, payload_type=0):
                self.caller = caller
                self.payload_type = payload_type

        async def _answer_coro():
            with unittest.mock.patch.object(
                asyncio.get_event_loop(),
                "create_datagram_endpoint",
                return_value=(fake_rtp_transport, FakeRTPProtocol(caller="")),
            ):
                await protocol._answer(invite, FakeRTPProtocol)

        await _answer_coro()

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
        assert response.body.media[0].fmt == ["8"]
        assert any(a.name == "sendrecv" for a in response.body.media[0].attributes)
        assert any(
            a.value and a.value.startswith("8 PCMA")
            for a in response.body.media[0].attributes
        )

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
        assert response.body.media[0].fmt == ["0"]

    @pytest.mark.asyncio
    async def test_answer__selects_opus_when_offered(self, fake_rtp_transport):
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
        assert response.body.media[0].fmt == ["111"]
        assert any(
            a.value and a.value.startswith("111 opus")
            for a in response.body.media[0].attributes
        )

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
        assert response.body.media[0].fmt == ["9"]

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
        assert response.body.media[0].fmt == ["100"]

    def test_preferred_codecs__class_attribute(self):
        """PREFERRED_CODECS is a class attribute with Opus first."""
        codecs = SessionInitiationProtocol.PREFERRED_CODECS
        assert isinstance(codecs, list)
        fmts = [fmt for fmt, _, _ in codecs]
        assert fmts[0] == "111"  # Opus is highest priority
        assert "8" in fmts  # PCMA present
        assert "0" in fmts  # PCMU present

    @pytest.mark.asyncio
    async def test_answer__falls_back_to_first_offered_codec(self, fake_rtp_transport):
        """Fall back to the first offered payload type when no preferred codec matches."""
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
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert response.body.media[0].fmt == ["126"]
        assert any(
            a.value == "126 telephone-event/8000"
            for a in response.body.media[0].attributes
            if a.value is not None
        )

    @pytest.mark.asyncio
    async def test_answer__no_sdp_falls_back_to_default(self, fake_rtp_transport):
        """Use payload type 0 (PCMU) when the INVITE has no SDP body."""
        protocol = FakeProtocol()
        addr = ("192.0.2.1", 5060)
        invite = self._make_invite("answer-no-sdp-1")
        protocol.request_received(invite, addr)
        await self._run_answer(protocol, invite, fake_rtp_transport)
        response, _ = protocol._sent_responses[-1]
        assert response.body.media[0].fmt == ["0"]

    @pytest.mark.asyncio
    async def test_answer__includes_to_tag(self, fake_rtp_transport):
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
        """Log an error and return early when no address is stored for the Call-ID."""
        protocol = FakeProtocol()
        invite = self._make_invite("no-addr-answer-1")

        with caplog.at_level("ERROR"):
            await protocol._answer(invite, asyncio.DatagramProtocol)
        assert "No address found" in caplog.text
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
        assert response.headers["Contact"].startswith("<sip:")

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
        """Remove Call-ID from _answered_calls, _request_addrs, and _to_tags."""
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
        assert "cancel-cleanup-1" not in protocol._request_addrs
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
