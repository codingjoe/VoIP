"""Tests for SIP call handling."""

import asyncio
import hashlib
from unittest.mock import MagicMock

import pytest
from sip.calls import (
    IncomingCall,
    IncomingCallProtocol,
    RegisterProtocol,
    RTPProtocol,
    _digest_response,
    _parse_auth_challenge,
)
from sip.messages import Message, Request, Response


def make_invite(headers: dict | None = None) -> Request:
    """Return an INVITE request with default headers."""
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


def make_call(send: MagicMock | None = None) -> IncomingCall:
    """Return an IncomingCall with a mock send callable."""
    return IncomingCall(
        make_invite(),
        ("192.0.2.1", 5060),
        send or MagicMock(),
    )


class TestRTPProtocol:
    def test_rtp_header_size__is_class_attr(self):
        """rtp_header_size is a class attribute set to the standard 12-byte header."""
        assert RTPProtocol.rtp_header_size == 12

    def test_datagram_received__forwards_audio(self):
        """Strip RTP header and forward audio payload via audio_received."""
        received = []

        class ConcreteRTP(RTPProtocol):
            def audio_received(self, data: bytes) -> None:
                received.append(data)

        protocol = ConcreteRTP()
        rtp_packet = b"\x80\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00" + b"audio"
        protocol.datagram_received(rtp_packet, ("192.0.2.1", 5004))
        assert received == [b"audio"]

    def test_datagram_received__skips_short_packet(self):
        """Skip packets shorter than the minimum RTP header size."""
        received = []

        class ConcreteRTP(RTPProtocol):
            def audio_received(self, data: bytes) -> None:
                received.append(data)

        ConcreteRTP().datagram_received(b"\x80\x00", ("192.0.2.1", 5004))
        assert received == []

    def test_datagram_received__skips_exact_header_size(self):
        """Skip packets that contain only an RTP header with no audio payload."""
        received = []

        class ConcreteRTP(RTPProtocol):
            def audio_received(self, data: bytes) -> None:
                received.append(data)

        ConcreteRTP().datagram_received(b"\x80" * 12, ("192.0.2.1", 5004))
        assert received == []

    def test_audio_received__returns_not_implemented(self):
        """Return NotImplemented for unhandled audio data."""
        assert RTPProtocol().audio_received(b"audio") is NotImplemented


class TestIncomingCall:
    def test_caller__returns_from_header(self):
        """Return the caller's SIP address from the From header."""
        assert make_call().caller == "sip:bob@biloxi.com"

    def test_caller__missing_header(self):
        """Return an empty string when the From header is absent."""
        call = IncomingCall(
            Request(method="INVITE", uri="sip:alice@atlanta.com"),
            ("192.0.2.1", 5060),
            MagicMock(),
        )
        assert call.caller == ""

    def test_audio_received__returns_not_implemented(self):
        """Return NotImplemented for unhandled audio data."""
        assert make_call().audio_received(b"audio") is NotImplemented

    def test_reject__sends_busy_here_by_default(self):
        """Send a 486 Busy Here response when no status code is given."""
        send = MagicMock()
        make_call(send).reject()
        send.assert_called_once()
        response, addr = send.call_args[0]
        assert isinstance(response, Response)
        assert response.status_code == 486
        assert response.reason == "Busy Here"
        assert addr == ("192.0.2.1", 5060)

    def test_reject__custom_status(self):
        """Send the specified status code and reason."""
        send = MagicMock()
        make_call(send).reject(status_code=603, reason="Decline")
        response, _ = send.call_args[0]
        assert response.status_code == 603
        assert response.reason == "Decline"

    def test_reject__copies_dialog_headers(self):
        """Copy Via, To, From, Call-ID, and CSeq headers into the response."""
        send = MagicMock()
        make_call(send).reject()
        response, _ = send.call_args[0]
        assert response.headers["Via"] == "SIP/2.0/UDP pc33.atlanta.com"
        assert response.headers["To"] == "sip:alice@atlanta.com"
        assert response.headers["From"] == "sip:bob@biloxi.com"
        assert response.headers["Call-ID"] == "1234@pc33"
        assert response.headers["CSeq"] == "1 INVITE"

    def test_answer__sends_200_ok(self):
        """Send a 200 OK response with an SDP body when answering."""

        async def run() -> None:
            send = MagicMock()
            await make_call(send).answer()
            send.assert_called_once()
            response, addr = send.call_args[0]
            assert response.status_code == 200
            assert response.reason == "OK"
            assert addr == ("192.0.2.1", 5060)

        asyncio.run(run())

    def test_answer__sdp_contains_opus_audio_line(self):
        """Include an Opus audio media line in the SDP body of the 200 OK."""

        async def run() -> None:
            send = MagicMock()
            await make_call(send).answer()
            response, _ = send.call_args[0]
            assert b"m=audio" in response.body
            assert b"RTP/AVP 111" in response.body
            assert b"a=rtpmap:111 opus/48000/2" in response.body

        asyncio.run(run())

    def test_answer__copies_dialog_headers(self):
        """Copy Via, To, From, Call-ID, and CSeq headers into the 200 OK."""

        async def run() -> None:
            send = MagicMock()
            await make_call(send).answer()
            response, _ = send.call_args[0]
            assert response.headers["Via"] == "SIP/2.0/UDP pc33.atlanta.com"
            assert response.headers["To"] == "sip:alice@atlanta.com"
            assert response.headers["From"] == "sip:bob@biloxi.com"
            assert response.headers["Call-ID"] == "1234@pc33"
            assert response.headers["CSeq"] == "1 INVITE"

        asyncio.run(run())

    def test_answer__rtp_receives_audio(self):
        """Deliver audio from RTP packets to the call's audio_received via the RTP socket."""
        received_audio = []

        class AudioCapture(IncomingCall):
            def audio_received(self, data: bytes) -> None:
                received_audio.append(data)

        async def run() -> None:
            send = MagicMock()
            call = AudioCapture(make_invite(), ("192.0.2.1", 5060), send)
            await call.answer()
            response, _ = send.call_args[0]

            sdp_line = next(
                line
                for line in response.body.decode().splitlines()
                if line.startswith("m=audio")
            )
            rtp_port = int(sdp_line.split()[1])

            loop = asyncio.get_running_loop()
            send_transport, _ = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol,
                remote_addr=("127.0.0.1", rtp_port),
            )
            rtp_packet = b"\x80\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00audio"
            send_transport.sendto(rtp_packet)
            await asyncio.sleep(0.05)
            send_transport.close()

        asyncio.run(run())
        assert received_audio == [b"audio"]

    def test_answer__rtp_receives_multiple_packets(self):
        """Call audio_received with each RTP payload when multiple packets arrive."""
        received_audio = []

        class AudioCapture(IncomingCall):
            def audio_received(self, data: bytes) -> None:
                received_audio.append(data)

        async def run() -> None:
            send = MagicMock()
            call = AudioCapture(make_invite(), ("192.0.2.1", 5060), send)
            await call.answer()
            response, _ = send.call_args[0]
            sdp_line = next(
                line
                for line in response.body.decode().splitlines()
                if line.startswith("m=audio")
            )
            rtp_port = int(sdp_line.split()[1])

            loop = asyncio.get_running_loop()
            send_transport, _ = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol,
                remote_addr=("127.0.0.1", rtp_port),
            )
            header = b"\x80\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00"
            send_transport.sendto(header + b"chunk1")
            send_transport.sendto(header + b"chunk2")
            await asyncio.sleep(0.05)
            send_transport.close()

        asyncio.run(run())
        assert received_audio == [b"chunk1", b"chunk2"]

    @pytest.mark.parametrize("extra_header", ["X-Custom"])
    def test_reject__excludes_extra_headers(self, extra_header):
        """Exclude non-dialog headers from the reject response."""
        send = MagicMock()
        call = IncomingCall(
            make_invite({extra_header: "value"}),
            ("192.0.2.1", 5060),
            send,
        )
        call.reject()
        response, _ = send.call_args[0]
        assert extra_header not in response.headers

    def test_answer__content_length_serialized(self):
        """Content-Length is automatically included when the response is serialized."""

        async def run() -> None:
            send = MagicMock()
            await make_call(send).answer()
            response, addr = send.call_args[0]
            # Serialize via IncomingCallProtocol.send to confirm Content-Length is added
            serialized = bytes(response)
            parsed = Message.parse(serialized)
            assert "Content-Length" in parsed.headers

        asyncio.run(run())

    def test_answer__logs_info(self, caplog):
        """Log an info message when answering a call."""
        import logging

        async def run():
            with caplog.at_level(logging.INFO, logger="sip.calls"):
                await make_call().answer()

        asyncio.run(run())
        assert any("Answering" in r.message for r in caplog.records)

    def test_reject__logs_info(self, caplog):
        """Log an info message when rejecting a call."""
        import logging

        with caplog.at_level(logging.INFO, logger="sip.calls"):
            make_call().reject()
        assert any("Rejecting" in r.message for r in caplog.records)


class TestIncomingCallProtocol:
    def test_connection_made__stores_transport(self):
        """Store the transport when a connection is established."""
        protocol = IncomingCallProtocol()
        transport = MagicMock()
        protocol.connection_made(transport)
        assert protocol._transport is transport

    def test_send__serializes_and_forwards_to_transport(self):
        """Serialize the message and forward it to the underlying transport."""
        protocol = IncomingCallProtocol()
        protocol.connection_made(MagicMock())
        response = Response(status_code=200, reason="OK")
        addr = ("192.0.2.1", 5060)
        protocol.send(response, addr)
        protocol._transport.sendto.assert_called_once_with(bytes(response), addr)

    def test_request_received__invite__calls_invite_received(self):
        """Dispatch an INVITE request to invite_received with an IncomingCall."""
        calls = []

        class ConcreteProtocol(IncomingCallProtocol):
            def invite_received(self, call, addr):
                calls.append((call, addr))

        protocol = ConcreteProtocol()
        protocol.connection_made(MagicMock())
        request = Request(
            method="INVITE",
            uri="sip:alice@atlanta.com",
            headers={"From": "sip:bob@biloxi.com"},
        )
        addr = ("192.0.2.1", 5060)
        protocol.request_received(request, addr)
        assert len(calls) == 1
        call, called_addr = calls[0]
        assert isinstance(call, IncomingCall)
        assert call.caller == "sip:bob@biloxi.com"
        assert called_addr == addr

    def test_request_received__non_invite__returns_not_implemented(self):
        """Return NotImplemented for non-INVITE requests."""
        protocol = IncomingCallProtocol()
        request = Request(method="OPTIONS", uri="sip:alice@atlanta.com")
        assert protocol.request_received(request, ("192.0.2.1", 5060)) is NotImplemented

    def test_invite_received__returns_not_implemented(self):
        """Return NotImplemented for unhandled incoming calls."""
        protocol = IncomingCallProtocol()
        assert (
            protocol.invite_received(MagicMock(), ("192.0.2.1", 5060)) is NotImplemented
        )

    def test_create_call__returns_incoming_call(self):
        """Return an IncomingCall bound to the protocol's send method."""
        protocol = IncomingCallProtocol()
        protocol.connection_made(MagicMock())
        request = make_invite()
        addr = ("192.0.2.1", 5060)
        call = protocol.create_call(request, addr)
        assert isinstance(call, IncomingCall)
        assert call.caller == "sip:bob@biloxi.com"

    def test_create_call__custom_class(self):
        """Use the overridden create_call to produce a custom call object."""

        class CustomCall(IncomingCall):
            pass

        class CustomProtocol(IncomingCallProtocol):
            def create_call(self, request, addr) -> CustomCall:
                return CustomCall(request, addr, self.send)

        protocol = CustomProtocol()
        protocol.connection_made(MagicMock())
        call = protocol.create_call(make_invite(), ("192.0.2.1", 5060))
        assert isinstance(call, CustomCall)


def make_register_protocol(
    server_addr=("192.0.2.2", 5060),
    aor="sip:alice@example.com",
    username="alice",
    password="test-password",  # noqa: S107
) -> RegisterProtocol:
    """Return a RegisterProtocol without triggering connection_made."""
    return RegisterProtocol(server_addr, aor, username, password)


class TestParseAuthChallenge:
    def test_parses_realm_and_nonce(self):
        """Parse realm and nonce from a Digest challenge header."""
        header = 'Digest realm="example.com", nonce="abc123"'
        params = _parse_auth_challenge(header)
        assert params["realm"] == "example.com"
        assert params["nonce"] == "abc123"

    def test_parses_qop(self):
        """Parse qop value from a Digest challenge header."""
        header = 'Digest realm="example.com", nonce="abc", qop="auth"'
        params = _parse_auth_challenge(header)
        assert params["qop"] == "auth"

    def test_parses_opaque(self):
        """Parse opaque value from a Digest challenge header."""
        header = 'Digest realm="r", nonce="n", opaque="xyz"'
        params = _parse_auth_challenge(header)
        assert params["opaque"] == "xyz"

    def test_empty_header_returns_empty_dict(self):
        """Return an empty dict for an empty or missing header value."""
        assert _parse_auth_challenge("") == {}


class TestDigestResponse:
    def test_no_qop(self):
        """Compute correct MD5 digest without qop per RFC 2617."""
        ha1 = hashlib.md5(b"alice:example.com:test-password").hexdigest()  # noqa: S324
        ha2 = hashlib.md5(b"REGISTER:sip:example.com").hexdigest()  # noqa: S324
        expected = hashlib.md5(f"{ha1}:nonce123:{ha2}".encode()).hexdigest()  # noqa: S324
        result = _digest_response(
            username="alice",
            password="test-password",  # noqa: S106
            realm="example.com",
            nonce="nonce123",
            method="REGISTER",
            uri="sip:example.com",
        )
        assert result == expected

    def test_with_qop_auth(self):
        """Compute correct MD5 digest with qop=auth per RFC 2617."""
        ha1 = hashlib.md5(b"alice:example.com:test-password").hexdigest()  # noqa: S324
        ha2 = hashlib.md5(b"REGISTER:sip:example.com").hexdigest()  # noqa: S324
        expected = hashlib.md5(  # noqa: S324
            f"{ha1}:nonce123:00000001:cnonce1:auth:{ha2}".encode()
        ).hexdigest()
        result = _digest_response(
            username="alice",
            password="test-password",  # noqa: S106
            realm="example.com",
            nonce="nonce123",
            method="REGISTER",
            uri="sip:example.com",
            qop="auth",
            nc="00000001",
            cnonce="cnonce1",
        )
        assert result == expected


class TestRegisterProtocol:
    def test_registrar_uri__strips_user_from_aor(self):
        """Derive registrar URI from AOR by stripping the user part."""
        p = make_register_protocol(aor="sip:alice@example.com")
        assert p._registrar_uri == "sip:example.com"

    def test_registrar_uri__preserves_port(self):
        """Preserve a non-default port in the derived registrar URI."""
        p = make_register_protocol(aor="sip:alice@example.com:5080")
        assert p._registrar_uri == "sip:example.com:5080"

    def test_connection_made__sends_register(self):
        """Send a REGISTER request immediately after connection is made."""
        p = make_register_protocol()
        transport = MagicMock()
        p.connection_made(transport)
        transport.sendto.assert_called_once()
        data, addr = transport.sendto.call_args[0]
        assert b"REGISTER sip:example.com SIP/2.0" in data
        assert addr == ("192.0.2.2", 5060)

    def test_register__includes_required_headers(self):
        """REGISTER request includes From, To, Call-ID, CSeq, Contact and Expires."""
        p = make_register_protocol()
        transport = MagicMock()
        p.connection_made(MagicMock())
        p._transport = transport
        transport.reset_mock()
        p.register()
        data, _ = transport.sendto.call_args[0]
        assert b"From: sip:alice@example.com" in data
        assert b"To: sip:alice@example.com" in data
        assert b"Contact: <sip:alice@example.com>" in data
        assert b"Expires: 3600" in data

    def test_register__increments_cseq(self):
        """CSeq increments with each REGISTER sent."""
        p = make_register_protocol()
        p.connection_made(MagicMock())
        assert p._cseq == 1
        p.register()
        assert p._cseq == 2

    def test_register__with_authorization(self):
        """Authorization header is included when credentials are provided."""
        p = make_register_protocol()
        p.connection_made(MagicMock())
        transport = p._transport
        transport.reset_mock()
        p.register(authorization='Digest username="alice"')
        data, _ = transport.sendto.call_args[0]
        assert b'Authorization: Digest username="alice"' in data

    def test_register__with_proxy_authorization(self):
        """Proxy-Authorization header is included for proxy challenges."""
        p = make_register_protocol()
        p.connection_made(MagicMock())
        transport = p._transport
        transport.reset_mock()
        p.register(proxy_authorization='Digest username="alice"')
        data, _ = transport.sendto.call_args[0]
        assert b'Proxy-Authorization: Digest username="alice"' in data

    def test_response_received__200_ok_calls_registered(self):
        """Receiving 200 OK for REGISTER triggers registered()."""
        calls = []

        class ConcreteProtocol(RegisterProtocol):
            def registered(self):
                calls.append(True)

        p = ConcreteProtocol(("192.0.2.2", 5060), "sip:alice@example.com", "a", "b")
        p.connection_made(MagicMock())
        p.response_received(
            Response(status_code=200, reason="OK", headers={"CSeq": "1 REGISTER"}),
            ("192.0.2.2", 5060),
        )
        assert calls == [True]

    def test_response_received__200_non_register_does_not_call_registered(self):
        """Receiving 200 OK for a non-REGISTER method does not call registered()."""
        calls = []

        class ConcreteProtocol(RegisterProtocol):
            def registered(self):
                calls.append(True)

        p = ConcreteProtocol(("192.0.2.2", 5060), "sip:alice@example.com", "a", "b")
        p.connection_made(MagicMock())
        p.response_received(
            Response(status_code=200, reason="OK", headers={"CSeq": "1 INVITE"}),
            ("192.0.2.2", 5060),
        )
        assert calls == []

    def test_response_received__401_retries_with_authorization(self):
        """Receiving 401 triggers a re-REGISTER with an Authorization header."""
        p = make_register_protocol(username="alice", password="test-password")  # noqa: S106
        p.connection_made(MagicMock())
        transport = p._transport
        transport.reset_mock()
        challenge = 'Digest realm="example.com", nonce="abc123"'
        p.response_received(
            Response(
                status_code=401,
                reason="Unauthorized",
                headers={"WWW-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5060),
        )
        data, _ = transport.sendto.call_args[0]
        assert b"Authorization: Digest" in data
        assert b'username="alice"' in data
        assert b'realm="example.com"' in data
        assert b'nonce="abc123"' in data
        assert b'algorithm="MD5"' in data

    def test_response_received__407_retries_with_proxy_authorization(self):
        """Receiving 407 triggers a re-REGISTER with a Proxy-Authorization header."""
        p = make_register_protocol(username="alice", password="test-password")  # noqa: S106
        p.connection_made(MagicMock())
        transport = p._transport
        transport.reset_mock()
        challenge = 'Digest realm="example.com", nonce="xyz"'
        p.response_received(
            Response(
                status_code=407,
                reason="Proxy Auth Required",
                headers={"Proxy-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5060),
        )
        data, _ = transport.sendto.call_args[0]
        assert b"Proxy-Authorization: Digest" in data
        assert b'username="alice"' in data

    def test_response_received__401_with_qop_auth_includes_nc_cnonce(self):
        """401 with qop=auth causes the retry to include nc and cnonce fields."""
        p = make_register_protocol()
        p.connection_made(MagicMock())
        transport = p._transport
        transport.reset_mock()
        challenge = 'Digest realm="example.com", nonce="n", qop="auth"'
        p.response_received(
            Response(
                status_code=401,
                reason="Unauthorized",
                headers={"WWW-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5060),
        )
        data, _ = transport.sendto.call_args[0]
        assert b"qop=auth" in data
        assert b"nc=00000001" in data
        assert b"cnonce=" in data

    def test_response_received__401_with_opaque_echoes_opaque(self):
        """The opaque field from the challenge is echoed back in the Authorization."""
        p = make_register_protocol()
        p.connection_made(MagicMock())
        transport = p._transport
        transport.reset_mock()
        challenge = 'Digest realm="example.com", nonce="n", opaque="secret-opaque"'
        p.response_received(
            Response(
                status_code=401,
                reason="Unauthorized",
                headers={"WWW-Authenticate": challenge, "CSeq": "1 REGISTER"},
            ),
            ("192.0.2.2", 5060),
        )
        data, _ = transport.sendto.call_args[0]
        assert b'opaque="secret-opaque"' in data

    def test_response_received__other_status__returns_not_implemented(self):
        """An unhandled status code returns NotImplemented."""
        p = make_register_protocol()
        p.connection_made(MagicMock())
        result = p.response_received(
            Response(
                status_code=500, reason="Server Error", headers={"CSeq": "1 REGISTER"}
            ),
            ("192.0.2.2", 5060),
        )
        assert result is NotImplemented

    def test_registered__returns_not_implemented(self):
        """Base registered() returns NotImplemented."""
        p = make_register_protocol()
        assert p.registered() is NotImplemented

    def test_invite_received_after_register(self):
        """INVITE dispatching still works after registration (inherits IncomingCallProtocol)."""
        calls = []

        class ConcreteProtocol(RegisterProtocol):
            def invite_received(self, call, addr):
                calls.append((call, addr))

        p = ConcreteProtocol(("192.0.2.2", 5060), "sip:alice@example.com", "a", "b")
        p.connection_made(MagicMock())
        request = Request(
            method="INVITE",
            uri="sip:alice@example.com",
            headers={"From": "sip:bob@example.com"},
        )
        p.request_received(request, ("192.0.2.1", 5060))
        assert len(calls) == 1
        assert isinstance(calls[0][0], IncomingCall)

    def test_response_received__200_ok__logs_info(self, caplog):
        """Receiving 200 OK logs an info message."""
        import logging

        p = make_register_protocol()
        p.connection_made(MagicMock())
        with caplog.at_level(logging.INFO, logger="sip.calls"):
            p.response_received(
                Response(status_code=200, reason="OK", headers={"CSeq": "1 REGISTER"}),
                ("192.0.2.2", 5060),
            )
        assert any("Registration successful" in r.message for r in caplog.records)

    def test_response_received__unexpected_status__logs_warning(self, caplog):
        """An unhandled status code logs a warning."""
        import logging

        p = make_register_protocol()
        p.connection_made(MagicMock())
        with caplog.at_level(logging.WARNING, logger="sip.calls"):
            p.response_received(
                Response(status_code=500, reason="Server Error", headers={"CSeq": "1 REGISTER"}),
                ("192.0.2.2", 5060),
            )
        assert any("500" in r.message for r in caplog.records)
