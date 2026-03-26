"""Tests for SIP message parsing and serialization."""

import pytest
from voip.sdp.messages import SessionDescription
from voip.sip.messages import Dialog, Message, Request, Response
from voip.sip.types import CallerID, SipUri


class TestMessage:
    def test_parse__request(self):
        """Parse a SIP request from bytes."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds\r\n"
            b"\r\n"
        )
        result = Message.parse(data)
        assert isinstance(result, Request)
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
        result = Message.parse(data)
        assert isinstance(result, Request)
        assert isinstance(result.body, SessionDescription)

    def test_parse__request__without_sdp_content_type(self):
        """Return None body when Content-Type is not application/sdp."""
        data = b"INVITE sip:bob@biloxi.com SIP/2.0\r\nContent-Length: 4\r\n\r\ntest"
        result = Message.parse(data)
        assert isinstance(result, Request)
        assert result.body is None

    def test_parse__response(self):
        """Parse a SIP response from bytes."""
        data = (
            b"SIP/2.0 200 OK\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds\r\n"
            b"\r\n"
        )
        result = Message.parse(data)
        assert isinstance(result, Response)
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
        result = Message.parse(data)
        assert isinstance(result, Response)
        assert isinstance(result.body, SessionDescription)

    def test_parse__roundtrip_request(self):
        """Round-trip a SIP request through parse and bytes."""
        request = Request(
            method="REGISTER",
            uri="sip:registrar.biloxi.com",
            headers={"From": "sip:bob@biloxi.com"},
        )
        assert Message.parse(bytes(request)) == request

    def test_parse__roundtrip_response(self):
        """Round-trip a SIP response through parse and bytes."""
        response = Response(
            status_code=404,
            phrase="Not Found",
            headers={"From": "sip:bob@biloxi.com"},
        )
        assert Message.parse(bytes(response)) == response

    def test_parse__skips_header_line_without_colon(self):
        """Skip header lines that contain no colon separator."""
        data = b"REGISTER sip:example.com SIP/2.0\r\nInvalidHeaderLine\r\n\r\n"
        result = Message.parse(data)
        assert isinstance(result, Request)
        assert "InvalidHeaderLine" not in result.headers

    def test_parse__from_header__is_caller_id(self):
        """From header is parsed as a CallerID instance."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\nFrom: sip:alice@atlanta.com\r\n\r\n"
        )
        result = Message.parse(data)
        assert isinstance(result.headers["From"], CallerID)
        assert result.headers["From"] == "sip:alice@atlanta.com"

    def test_parse__to_header__is_caller_id(self):
        """To header is parsed as a CallerID instance."""
        data = b"INVITE sip:bob@biloxi.com SIP/2.0\r\nTo: sip:bob@biloxi.com\r\n\r\n"
        result = Message.parse(data)
        assert isinstance(result.headers["To"], CallerID)

    def test_parse__from_header__roundtrip_preserves_raw_value(self):
        """str(CallerID) equals the original header string, so serialization is unchanged."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b'From: "08001234567" <sip:08001234567@telefonica.de>;tag=abc\r\n'
            b"\r\n"
        )
        result = Message.parse(data)
        assert bytes(result) == data

    def test_parse__raises_value_error_on_invalid_first_line(self):
        """Raise ValueError when the first line cannot be parsed as a request."""
        with pytest.raises(ValueError, match="Invalid SIP message"):
            Message.parse(b"TOOSHORT\r\n\r\n")

    def test___str____returns_decoded_bytes(self):
        """Return the string representation of a request as decoded bytes."""
        request = Request(
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
        request = Message.parse(data)
        assert request.branch == "z9hG4bKabc"

    def test_remote_tag__with_tag(self):
        """Return the To-header tag parameter."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bKabc\r\n"
            b"To: sip:bob@biloxi.com;tag=to-tag-1\r\n"
            b"\r\n"
        )
        request = Message.parse(data)
        assert request.remote_tag == "to-tag-1"

    def test_local_tag__with_tag(self):
        """Return the From-header tag parameter."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bKabc\r\n"
            b"From: sip:alice@atlanta.com;tag=from-tag-1\r\n"
            b"\r\n"
        )
        request = Message.parse(data)
        assert request.local_tag == "from-tag-1"

    def test_sequence__returns_cseq_number(self):
        """Return the integer sequence number from the CSeq header."""
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bKabc\r\n"
            b"CSeq: 42 INVITE\r\n"
            b"\r\n"
        )
        request = Message.parse(data)
        assert request.sequence == 42


class TestRequest:
    def test___bytes__(self):
        """Serialize a SIP request to bytes."""
        request = Request(
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
        request = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            body=sdp,
        )
        serialized = bytes(request)
        assert b"Content-Length:" in serialized
        assert b"v=0" in serialized

    def test_branch__with_branch(self):
        """Branch returns the branch parameter from the Via header."""
        request = Request(
            method="INVITE",
            uri="sip:bob@biloxi.com",
            headers={"Via": "SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bKabc123"},
        )
        assert request.branch == "z9hG4bKabc123"

    def test_from_dialog__merges_dialog_headers(self):
        """Merge the provided headers with the dialog's headers."""
        dialog = Dialog(
            uac=SipUri.parse("sips:alice@example.com"),
            local_tag="local-tag",
            remote_tag="remote-tag",
        )
        request = Request.from_dialog(
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
        response = Response(
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
        response = Response(status_code=200, phrase="OK", body=sdp)
        serialized = bytes(response)
        assert b"Content-Length:" in serialized
        assert b"v=0" in serialized

    def test___bytes____with_sdp_body__auto_content_length(self):
        """Auto-calculate Content-Length when SDP body is present and header is not set."""
        sdp = SessionDescription()
        response = Response(status_code=200, phrase="OK", body=sdp)
        serialized = bytes(response)
        assert b"Content-Length:" in serialized
        parsed = Message.parse(serialized)
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
        request = Message.parse(data)
        dialog = Dialog(
            uac=SipUri.parse("sip:alice@atlanta.com"),
            remote_tag="server-tag",
        )
        response = Response.from_request(
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
        request = Message.parse(data)
        response = Response.from_request(request, status_code=200, phrase="OK")
        assert response.headers["To"] == request.headers["To"]


class TestDialog:
    def test_from_header__contains_local_tag(self):
        """from_header includes the local_tag parameter."""
        dialog = Dialog(
            uac=SipUri.parse("sips:alice@example.com"),
            local_tag="my-local-tag",
        )
        assert "my-local-tag" in dialog.from_header

    def test_to_header__without_remote_tag(self):
        """to_header omits the tag parameter when remote_tag is None."""
        dialog = Dialog(
            uac=SipUri.parse("sip:bob@biloxi.com:5060"),
            remote_tag=None,
        )
        assert ";tag=" not in dialog.to_header

    def test_to_header__with_remote_tag(self):
        """to_header includes the remote_tag parameter."""
        dialog = Dialog(
            uac=SipUri.parse("sip:bob@biloxi.com:5060"),
            remote_tag="their-tag",
        )
        assert "their-tag" in dialog.to_header

    def test_headers__returns_required_keys(self):
        """Headers property returns From, To, and Call-ID keys."""
        dialog = Dialog(uac=SipUri.parse("sips:alice@example.com"))
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
        request = Message.parse(data)
        dialog = Dialog.from_request(request)
        assert dialog.call_id == "call-99@atlanta.com"
        assert dialog.local_tag == "from-tag-99"
        assert dialog.remote_tag is not None
