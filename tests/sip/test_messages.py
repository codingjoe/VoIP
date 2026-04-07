"""Tests for SIP message parsing and serialization."""

import pytest
from voip.sdp.messages import SessionDescription
from voip.sip import messages
from voip.sip.dialog import Dialog
from voip.sip.types import SipURI


class TestSIPHeaderDict:
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

    def test_parse(self):
        """Parse headers from bytes."""
        data = b"From: Alice\r\nRoute: sip:proxy.example.com\r\nRoute: sip:example.com"
        headers = messages.SIPHeaderDict.parse(data)
        assert headers["From"] == "Alice"
        assert headers.getlist("Route") == ["sip:proxy.example.com", "sip:example.com"]

    def test_parse__empty(self):
        with pytest.raises(ValueError, match="Invalid header: ''"):
            messages.SIPHeaderDict.parse(b"")


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
        dialog = Dialog(
            uac=SipURI.parse("sips:alice@example.com"),
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
        dialog = Dialog(
            uac=SipURI.parse("sip:alice@atlanta.com"),
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
