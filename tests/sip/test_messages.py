"""Tests for SIP message parsing and serialization."""

import pytest
from voip.sdp.messages import SessionDescription
from voip.sip import messages
from voip.sip.types import SipUri


class TestHeaderMap:
    def test_init(self):
        """Initialize a HeaderMap with a dictionary of headers."""
        headers = messages.SIPHeaderDict(
            {"From": "Alice", "Route": "sip:proxy.example.com"}
        )
        assert headers["From"] == "Alice"
        assert headers["Route"] == "sip:proxy.example.com"

    def test_init__empty(self):
        """Initialize an empty HeaderMap."""
        headers = messages.SIPHeaderDict()
        assert headers == {}

    def test__str__(self):
        """String representation of a HeaderMap."""
        headers = messages.SIPHeaderDict()
        headers["From"] = "Alice"
        headers.add("Route", "sip:proxy.example.com")
        headers.add("Route", "sip:example.com")
        assert str(headers) == (
            "From: Alice\r\nRoute: sip:proxy.example.com\r\nRoute: sip:example.com\r\n"
        )

    def test__bytes__(self):
        """Byte representation of a HeaderMap."""
        headers = messages.SIPHeaderDict()
        headers["From"] = "Alice"
        headers.add("Route", "sip:proxy.example.com")
        headers.add("Route", "sip:example.com")
        assert bytes(headers) == (
            b"From: Alice\r\nRoute: sip:proxy.example.com\r\nRoute: sip:example.com\r\n"
        )


class TestMessage:
    def test_parse__request(self):
        """Parse a SIP request from bytes."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds\r\n"
            b"\r\n"
        )
        result = messages.Message.parse(data)
        assert isinstance(result, messages.Request)
        assert result.method == "INVITE"
        assert result.uri == "sip:bob@biloxi.com"
        assert result.version == "SIP/2.0"
        assert result.headers == {
            "Via": "SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds"
        }
        assert result.body is None

    def test_parse__request__with_sdp_body(self):
        """Parse a SIP request with an SDP body from bytes."""
        sdp = b"v=0\r\ns=-\r\nt=0 0\r\n"
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Content-Type: application/sdp\r\n"
            b"\r\n" + sdp
        )
        result = messages.Message.parse(data)
        assert isinstance(result, messages.Request)
        assert isinstance(result.body, SessionDescription)

    def test_parse__request__without_sdp_content_type(self):
        """Return None body when Content-Type is not application/sdp."""
        data = b"INVITE sip:bob@biloxi.com SIP/2.0\r\nContent-Length: 4\r\n\r\ntest"
        result = messages.Message.parse(data)
        assert isinstance(result, messages.Request)
        assert result.body is None

    def test_parse__response(self):
        """Parse a SIP response from bytes."""
        data = (
            b"SIP/2.0 200 OK\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds\r\n"
            b"\r\n"
        )
        result = messages.Message.parse(data)
        assert isinstance(result, messages.Response)
        assert result.status_code == 200
        assert result.phrase == "OK"
        assert result.version == "SIP/2.0"
        assert result.headers == {
            "Via": "SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds"
        }
        assert result.body is None

    def test_parse__response__with_sdp_body(self):
        """Parse a SIP response with an SDP body from bytes."""
        sdp = b"v=0\r\ns=-\r\nt=0 0\r\n"
        data = b"SIP/2.0 200 OK\r\nContent-Type: application/sdp\r\n\r\n" + sdp
        result = messages.Message.parse(data)
        assert isinstance(result, messages.Response)
        assert isinstance(result.body, SessionDescription)

    def test_parse__roundtrip_request(self):
        """Round-trip a SIP request through parse and bytes."""
        request = messages.Request(
            method="REGISTER",
            uri="sip:registrar.biloxi.com",
            headers={"From": "sip:bob@biloxi.com"},
        )
        assert messages.Message.parse(bytes(request)) == request

    def test_parse__roundtrip_response(self):
        """Round-trip a SIP response through parse and bytes."""
        response = messages.Response(
            status_code=404,
            phrase="Not Found",
            headers={"From": "sip:bob@biloxi.com"},
        )
        assert messages.Message.parse(bytes(response)) == response

    def test_parse__from_header__roundtrip_preserves_raw_value(self):
        """str(CallerID) equals the original header string, so serialization is unchanged."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b'From: "08001234567" <sip:08001234567@telefonica.de>;tag=abc\r\n'
            b"\r\n"
        )
        result = messages.Message.parse(data)
        assert bytes(result) == data

    def test_parse__raises_value_error_on_invalid_first_line(self):
        """Raise ValueError when the first line cannot be parsed as a request."""
        with pytest.raises(ValueError, match="Invalid header"):
            messages.Message.parse(b"TOOSHORT\r\n\r\n")

    def test_parse__raises_value_error_on_malformed_request_line(self):
        """Raise ValueError when the request first line has too few parts."""
        with pytest.raises(ValueError, match="Invalid SIP message first line"):
            messages.Message.parse(b"INVITE sip:bob\r\nContent-Length: 0\r\n\r\n")

    def test___str____returns_decoded_bytes(self):
        """Return the string representation of a request as decoded bytes."""
        request = messages.Request(
            method="REGISTER",
            uri="sip:registrar.biloxi.com",
            headers={"From": "sip:bob@biloxi.com"},
        )
        assert str(request) == bytes(request).decode()

    def test_branch__extracts_via_branch_parameter(self):
        """Return the branch parameter from the top Via header."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bKabc\r\n"
            b"\r\n"
        )
        request = messages.Message.parse(data)
        assert request.branch == "z9hG4bKabc"

    def test_remote_tag__with_tag(self):
        """Return the To-header tag parameter."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bKabc\r\n"
            b"To: sip:bob@biloxi.com;tag=to-tag-1\r\n"
            b"\r\n"
        )
        request = messages.Message.parse(data)
        assert request.remote_tag == "to-tag-1"

    def test_local_tag__with_tag(self):
        """Return the From-header tag parameter."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bKabc\r\n"
            b"From: sip:alice@atlanta.com;tag=from-tag-1\r\n"
            b"\r\n"
        )
        request = messages.Message.parse(data)
        assert request.local_tag == "from-tag-1"

    def test_sequence__returns_cseq_number(self):
        """Return the integer sequence number from the CSeq header."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bKabc\r\n"
            b"CSeq: 42 INVITE\r\n"
            b"\r\n"
        )
        request = messages.Message.parse(data)
        assert request.sequence == 42


class TestRequest:
    def test___bytes__(self):
        """Serialize a SIP request to bytes."""
        request = messages.Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={"Via": "SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds"},
        )
        assert bytes(request) == (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds\r\n"
            b"\r\n"
        )

    def test___bytes____with_sdp_body(self):
        """Serialize a SIP request with an SDP body to bytes."""
        sdp = SessionDescription()
        request = messages.Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            body=sdp,
        )
        serialized = bytes(request)
        assert b"Content-Length:" in serialized
        assert b"v=0" in serialized

    def test_branch__with_branch(self):
        """Branch returns the branch parameter from the Via header."""
        request = messages.Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={"Via": "SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bKabc123"},
        )
        assert request.branch == "z9hG4bKabc123"

    def test_from_dialog__merges_dialog_headers(self):
        """Merge the provided headers with the dialog's headers."""
        dialog = messages.Dialog(
            uac=SipUri.parse("sips:alice@example.com"),
            local_tag="local-tag",
            remote_tag="remote-tag",
        )
        request = messages.Request.from_dialog(
            dialog=dialog,
            headers={"Via": "SIP/2.0/TLS example.com;branch=z9hG4bK123"},
            method="REGISTER",
            uri="sips:example.com",
        )
        assert "From" in request.headers
        assert "Call-ID" in request.headers
        assert "Via" in request.headers


class TestResponse:
    def test___bytes__(self):
        """Serialize a SIP response to bytes."""
        response = messages.Response(
            status_code=200,
            phrase="OK",
            headers={"Via": "SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds"},
        )
        assert bytes(response) == (
            b"SIP/2.0 200 OK\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds\r\n"
            b"\r\n"
        )

    def test___bytes____with_sdp_body(self):
        """Serialize a SIP response with an SDP body to bytes."""
        sdp = SessionDescription()
        response = messages.Response(status_code=200, phrase="OK", body=sdp)
        serialized = bytes(response)
        assert b"Content-Length:" in serialized
        assert b"v=0" in serialized

    def test___bytes____with_sdp_body__auto_content_length(self):
        """Auto-calculate Content-Length when SDP body is present and header is not set."""
        sdp = SessionDescription()
        response = messages.Response(status_code=200, phrase="OK", body=sdp)
        serialized = bytes(response)
        assert b"Content-Length:" in serialized
        parsed = messages.Message.parse(serialized)
        assert parsed.body is None

    def test_from_request__with_dialog_remote_tag(self):
        """Include dialog remote_tag in To header when dialog has a remote_tag."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bKabc\r\n"
            b"From: sip:alice@atlanta.com;tag=from-tag-1\r\n"
            b"To: sip:bob@biloxi.com\r\n"
            b"Call-ID: test-call@atlanta.com\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"\r\n"
        )
        request = messages.Message.parse(data)
        dialog = messages.Dialog(
            uac=SipUri.parse("sip:alice@atlanta.com"),
            remote_tag="server-tag",
        )
        response = messages.Response.from_request(
            request, dialog=dialog, status_code=200, phrase="OK"
        )
        assert "server-tag" in str(response.headers["To"])

    def test_from_request__without_dialog(self):
        """Copy To header verbatim from the request when no dialog is provided."""
        data = (
            b"OPTIONS sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bKxyz\r\n"
            b"From: sip:alice@atlanta.com;tag=ft1\r\n"
            b"To: sip:bob@biloxi.com\r\n"
            b"Call-ID: opts-call@atlanta.com\r\n"
            b"CSeq: 1 OPTIONS\r\n"
            b"\r\n"
        )
        request = messages.Message.parse(data)
        response = messages.Response.from_request(request, status_code=200, phrase="OK")
        assert response.headers["To"] == request.headers["To"]


class TestDialog:
    def test_from_header__contains_local_tag(self):
        """from_header includes the local_tag parameter."""
        dialog = messages.Dialog(
            uac=SipUri.parse("sips:alice@example.com"),
            local_tag="my-local-tag",
        )
        assert "my-local-tag" in dialog.from_header

    def test_to_header__without_remote_tag(self):
        """to_header omits the tag parameter when remote_tag is None."""
        dialog = messages.Dialog(
            uac=SipUri.parse("sip:bob@biloxi.com:5060"),
            remote_tag=None,
        )
        assert ";tag=" not in dialog.to_header

    def test_to_header__with_remote_tag(self):
        """to_header includes the remote_tag parameter."""
        dialog = messages.Dialog(
            uac=SipUri.parse("sip:bob@biloxi.com:5060"),
            remote_tag="their-tag",
        )
        assert "their-tag" in dialog.to_header

    def test_headers__returns_required_keys(self):
        """Headers property returns From, To, and Call-ID keys."""
        dialog = messages.Dialog(uac=SipUri.parse("sips:alice@example.com"))
        headers = dialog.headers
        assert "From" in headers
        assert "To" in headers
        assert "Call-ID" in headers

    def test_from_request__extracts_call_id_and_tags(self):
        """from_request creates a Dialog with the correct call_id and tags."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bKabc\r\n"
            b"From: sip:alice@atlanta.com;tag=from-tag-99\r\n"
            b"To: sip:bob@biloxi.com\r\n"
            b"Call-ID: call-99@atlanta.com\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"\r\n"
        )
        request = messages.Message.parse(data)
        dialog = messages.Dialog.from_request(request)
        assert dialog.call_id == "call-99@atlanta.com"
        assert dialog.local_tag == "from-tag-99"
        assert dialog.remote_tag is not None

    def test_ringing__delegates_to_invite_tx(self):
        """ringing() calls ringing() on the invite_tx when it is set."""
        from unittest.mock import MagicMock

        mock_tx = MagicMock()
        dialog = messages.Dialog(invite_tx=mock_tx)
        dialog.ringing()
        mock_tx.ringing.assert_called_once()

    def test_ringing__noop_when_no_invite_tx(self):
        """ringing() is a no-op when invite_tx is None."""
        dialog = messages.Dialog()
        dialog.ringing()  # must not raise

    def test_accept__delegates_to_invite_tx(self):
        """accept() calls answer() on the invite_tx when it is set."""
        from unittest.mock import MagicMock

        class FakeCall:
            pass

        mock_tx = MagicMock()
        dialog = messages.Dialog(invite_tx=mock_tx)
        dialog.accept(call_class=FakeCall)
        mock_tx.answer.assert_called_once_with(call_class=FakeCall)

    def test_accept__noop_when_no_invite_tx(self):
        """accept() is a no-op when invite_tx is None."""
        dialog = messages.Dialog()
        dialog.accept(call_class=object)  # must not raise

    def test_reject__delegates_to_invite_tx(self):
        """reject() calls reject() on the invite_tx when it is set."""
        from unittest.mock import MagicMock
        from voip.sip.types import SIPStatus

        mock_tx = MagicMock()
        dialog = messages.Dialog(invite_tx=mock_tx)
        dialog.reject(SIPStatus.NOT_FOUND)
        mock_tx.reject.assert_called_once_with(SIPStatus.NOT_FOUND)

    def test_reject__noop_when_no_invite_tx(self):
        """reject() is a no-op when invite_tx is None."""
        dialog = messages.Dialog()
        dialog.reject()  # must not raise

    def test_call_received__rejects_by_default(self):
        """call_received() rejects the call with 486 Busy Here by default."""
        from unittest.mock import MagicMock

        mock_tx = MagicMock()
        dialog = messages.Dialog(invite_tx=mock_tx)
        dialog.call_received()
        mock_tx.reject.assert_called_once()

    def test_hangup_received__is_noop(self):
        """hangup_received() base implementation does nothing."""
        dialog = messages.Dialog()
        dialog.hangup_received()  # must not raise

    async def test_bye__noop_when_sip_is_none(self):
        """bye() is a no-op when sip is not set."""
        dialog = messages.Dialog()
        await dialog.bye()  # must not raise

    async def test_bye__sends_bye_request(self):
        """bye() sends a BYE request via dialog.sip."""
        from unittest.mock import MagicMock

        mock_sip = MagicMock()
        mock_sip.aor.transport = "TLS"
        mock_sip.local_address = "127.0.0.1:5061"
        mock_sip.transactions = {}
        mock_sip.dialogs = {}
        dialog = messages.Dialog(
            call_id="test@example.com",
            local_party="sip:alice@example.com;tag=a",
            remote_party="sip:bob@biloxi.com;tag=b",
            remote_contact="sip:bob@192.0.2.2",
            outbound_cseq=1,
            sip=mock_sip,
        )
        import asyncio

        bye_task = asyncio.create_task(dialog.bye())
        await asyncio.sleep(0)
        (tx,) = mock_sip.transactions.values()
        tx.done.set()
        await bye_task
        mock_sip.send.assert_called_once()

    async def test_bye__noop_when_local_party_missing(self):
        """bye() is a no-op when local_party is not set."""
        from unittest.mock import MagicMock

        mock_sip = MagicMock()
        dialog = messages.Dialog(
            remote_contact="sip:bob@192.0.2.2",
            sip=mock_sip,
        )
        await dialog.bye()
        mock_sip.send.assert_not_called()

    async def test_bye__noop_when_remote_contact_missing(self):
        """bye() is a no-op when remote_contact is not set."""
        from unittest.mock import MagicMock

        mock_sip = MagicMock()
        dialog = messages.Dialog(
            local_party="sip:alice@example.com;tag=a",
            remote_party="sip:bob@biloxi.com;tag=b",
            sip=mock_sip,
        )
        await dialog.bye()
        mock_sip.send.assert_not_called()

    async def test_dial__creates_invite_transaction_and_sends(self):
        """dial() creates an InviteTransaction and sends an INVITE."""
        import ipaddress

        from voip.rtp import RealtimeTransportProtocol
        from voip.sip.protocol import SessionInitiationProtocol
        from voip.sip.types import SipUri
        from voip.types import NetworkAddress

        from tests.sip.conftest import CallFixture, FakeTransport

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = SessionInitiationProtocol(
            aor=SipUri.parse("sips:alice:secret@example.com"),
            rtp=rtp,
            dialog_class=messages.Dialog,
        )
        sip.transport = transport
        sip.local_address = NetworkAddress(ipaddress.ip_address("127.0.0.1"), 5061)
        sip.is_secure = True

        dialog = messages.Dialog(sip=sip)
        await dialog.dial("sip:bob@biloxi.com", call_class=CallFixture)
        assert any(b"INVITE" in data for data in transport.sent)
        assert dialog.uac is sip.aor
