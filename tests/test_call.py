import asyncio
import socket
import struct
from unittest.mock import MagicMock, patch

import pytest
from voip.call import IncomingCall, IncomingCallProtocol, RegisterProtocol
from voip.sip.messages import Message, Request, Response


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
            assert b"a=rtpmap:111 opus/48000/1" in response.body

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
            with caplog.at_level(logging.INFO, logger="voip.call"):
                await make_call().answer()

        asyncio.run(run())
        assert any("Answering" in r.message for r in caplog.records)

    def test_reject__logs_info(self, caplog):
        """Log an info message when rejecting a call."""
        import logging

        with caplog.at_level(logging.INFO, logger="voip.call"):
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

    def test_datagram_received__keepalive__sends_pong(self):
        """Double-CRLF keepalive (RFC 5626 §4.4.1) is answered with a single-CRLF pong."""
        protocol = IncomingCallProtocol()
        transport = MagicMock()
        protocol.connection_made(transport)
        addr = ("192.0.2.1", 5060)
        protocol.datagram_received(b"\r\n\r\n", addr)
        transport.sendto.assert_called_once_with(b"\r\n", addr)

    def test_request_received__unsupported_method__raises(self):
        """Raise NotImplementedError for any non-INVITE SIP request method."""
        protocol = IncomingCallProtocol()
        protocol.connection_made(MagicMock())
        request = Request(method="OPTIONS", uri="sip:alice@atlanta.com")
        with pytest.raises(NotImplementedError, match="OPTIONS"):
            protocol.request_received(request, ("192.0.2.1", 5060))


class TestRegisterProtocol:
    def test_registrar_uri__strips_user_from_aor(self):
        """Derive registrar URI from AOR by stripping the user part."""
        p = make_register_protocol(aor="sip:alice@example.com")
        assert p.registrar_uri == "sip:example.com"

    def test_registrar_uri__preserves_port(self):
        """Preserve a non-default port in the derived registrar URI."""
        p = make_register_protocol(aor="sip:alice@example.com:5080")
        assert p.registrar_uri == "sip:example.com:5080"

    def test_connection_made__sends_register(self):
        """Send a REGISTER request immediately after connection is made."""
        p = make_register_protocol()
        transport = make_mock_transport()
        p.connection_made(transport)
        transport.sendto.assert_called_once()
        data, addr = transport.sendto.call_args[0]
        assert b"REGISTER sip:example.com SIP/2.0" in data
        assert addr == ("192.0.2.2", 5060)

    def test_connection_made__with_stun__registers_after_discovery(self):
        """Send REGISTER after STUN discovery when a STUN server is configured."""

        async def run():
            p = RegisterProtocol(
                ("192.0.2.2", 5060),
                "sip:alice@example.com",
                "alice",
                "password",  # noqa: S106
                stun_server_address=("stun.example.com", 3478),
            )
            transport = make_mock_transport()
            with patch.object(
                p, "stun_discover", return_value=("203.0.113.1", 54321)
            ) as mock_discover:
                p.connection_made(transport)
                await asyncio.sleep(0.05)
            mock_discover.assert_called_once_with("stun.example.com", 3478)
            assert p.public_address == ("203.0.113.1", 54321)
            transport.sendto.assert_called()

        asyncio.run(run())

    def test_connection_made__with_stun__registers_even_if_discovery_fails(self):
        """Send REGISTER even when STUN discovery raises an error."""

        async def run():
            p = RegisterProtocol(
                ("192.0.2.2", 5060),
                "sip:alice@example.com",
                "alice",
                "password",  # noqa: S106
                stun_server_address=("stun.example.com", 3478),
            )
            transport = make_mock_transport()
            with patch.object(p, "stun_discover", side_effect=TimeoutError("timeout")):
                p.connection_made(transport)
                await asyncio.sleep(0.05)
            transport.sendto.assert_called()

        asyncio.run(run())

    def test_register__includes_required_headers(self):
        """REGISTER request includes From, To, Call-ID, CSeq, Contact and Expires."""
        p = make_register_protocol()
        transport = make_mock_transport()
        p.connection_made(make_mock_transport())
        p._transport = transport
        transport.reset_mock()
        p.register()
        data, _ = transport.sendto.call_args[0]
        assert b"From: sip:alice@example.com" in data
        assert b"To: sip:alice@example.com" in data
        assert b"Contact: <sip:alice@127.0.0.1:5060>" in data
        assert b"Expires: 3600" in data

    def test_register__increments_cseq(self):
        """CSeq increments with each REGISTER sent."""
        p = make_register_protocol()
        p.connection_made(make_mock_transport())
        assert p.cseq == 1
        p.register()
        assert p.cseq == 2

    def test_register__with_authorization(self):
        """Authorization header is included when credentials are provided."""
        p = make_register_protocol()
        p.connection_made(make_mock_transport())
        transport = p._transport
        transport.reset_mock()
        p.register(authorization='Digest username="alice"')
        data, _ = transport.sendto.call_args[0]
        assert b'Authorization: Digest username="alice"' in data

    def test_register__with_proxy_authorization(self):
        """Proxy-Authorization header is included for proxy challenges."""
        p = make_register_protocol()
        p.connection_made(make_mock_transport())
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
        p.connection_made(make_mock_transport())
        p.response_received(
            Response(status_code=200, reason="OK", headers={"CSeq": "1 REGISTER"}),
            ("192.0.2.2", 5060),
        )
        assert calls == [True]

    def test_response_received__200_non_register_raises(self):
        """Receiving 200 OK for a non-REGISTER method raises NotImplementedError."""
        p = make_register_protocol()
        p.connection_made(make_mock_transport())
        with pytest.raises(NotImplementedError):
            p.response_received(
                Response(status_code=200, reason="OK", headers={"CSeq": "1 INVITE"}),
                ("192.0.2.2", 5060),
            )

    def test_response_received__401_retries_with_authorization(self):
        """Receiving 401 triggers a re-REGISTER with an Authorization header."""
        p = make_register_protocol(username="alice", password="test-password")  # noqa: S106
        p.connection_made(make_mock_transport())
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
        p.connection_made(make_mock_transport())
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
        p.connection_made(make_mock_transport())
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
        p.connection_made(make_mock_transport())
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

    def test_register__via_header_has_rport(self):
        """REGISTER request includes a Via header with the rport parameter for NAT traversal."""
        import re

        p = make_register_protocol()
        transport = make_mock_transport("192.0.2.10", 5060)
        p.connection_made(transport)
        data, _ = transport.sendto.call_args[0]
        assert b"Via: SIP/2.0/UDP 192.0.2.10:5060;rport;branch=z9hG4bK" in data
        assert re.search(rb"branch=z9hG4bK[0-9a-f]{32}", data)

    def test_register__via_branch_is_unique_per_request(self):
        """Each REGISTER generates a unique Via branch to prevent transaction conflicts."""
        import re

        p = make_register_protocol()
        transport = make_mock_transport()
        p.connection_made(transport)
        data1, _ = transport.sendto.call_args[0]
        transport.reset_mock()
        p.register()
        data2, _ = transport.sendto.call_args[0]
        branch1 = re.search(rb"branch=(z9hG4bK[0-9a-f]{32})", data1).group(1)
        branch2 = re.search(rb"branch=(z9hG4bK[0-9a-f]{32})", data2).group(1)
        assert branch1 != branch2

    def test_register__contact_uses_local_addr_when_no_stun(self):
        """Contact header uses local socket address when STUN is not configured."""
        p = make_register_protocol()
        transport = make_mock_transport("10.0.0.5", 5060)
        p.connection_made(transport)
        data, _ = transport.sendto.call_args[0]
        assert b"Contact: <sip:alice@10.0.0.5:5060>" in data

    def test_register__contact_uses_public_addr_when_stun_discovered(self):
        """Contact header uses the STUN-discovered public address."""
        p = make_register_protocol()
        transport = make_mock_transport("10.0.0.5", 5060)
        p.connection_made(transport)
        p.public_address = ("203.0.113.1", 12345)
        transport.reset_mock()
        p.register()
        data, _ = transport.sendto.call_args[0]
        assert b"Contact: <sip:alice@203.0.113.1:12345>" in data

    def test_handle_stun__parses_xor_mapped_address(self):
        """handle_stun resolves the future with the XOR-MAPPED-ADDRESS."""

        async def run():
            p = make_register_protocol()
            p.connection_made(make_mock_transport())
            p._stun_transactions = {}
            # Build a fake STUN Binding Success Response with XOR-MAPPED-ADDRESS
            magic_cookie = 0x2112A442
            transaction_id = b"\x01" * 12
            # Public addr: 203.0.113.5:54321
            public_ip = (203 << 24) | (0 << 16) | (113 << 8) | 5
            xor_ip = public_ip ^ magic_cookie
            xor_port = 54321 ^ (magic_cookie >> 16)
            attr_val = struct.pack(">BBH I", 0x00, 0x01, xor_port, xor_ip)
            attr = struct.pack(">HH", 0x0020, len(attr_val)) + attr_val
            msg_len = len(attr)
            response = (
                struct.pack(">HHI12s", 0x0101, msg_len, magic_cookie, transaction_id)
                + attr
            )
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            p._stun_transactions[transaction_id] = future
            p.handle_stun(response, ("stun.l.google.com", 19302))
            result = future.result()
            assert result == ("203.0.113.5", 54321)

        asyncio.run(run())

    def test_handle_stun__falls_back_to_mapped_address(self):
        """handle_stun falls back to MAPPED-ADDRESS when XOR-MAPPED-ADDRESS is absent."""

        async def run():
            p = make_register_protocol()
            p.connection_made(make_mock_transport())
            p._stun_transactions = {}
            magic_cookie = 0x2112A442
            transaction_id = b"\x02" * 12
            # MAPPED-ADDRESS: 198.51.100.7:60001 (not XOR'd)
            attr_val = struct.pack(
                ">BBH4s", 0x00, 0x01, 60001, socket.inet_aton("198.51.100.7")
            )
            attr = struct.pack(">HH", 0x0001, len(attr_val)) + attr_val
            msg_len = len(attr)
            response = (
                struct.pack(">HHI12s", 0x0101, msg_len, magic_cookie, transaction_id)
                + attr
            )
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            p._stun_transactions[transaction_id] = future
            p.handle_stun(response, ("stun.example.com", 3478))
            assert future.result() == ("198.51.100.7", 60001)

        asyncio.run(run())

    def test_datagram_received__stun_routes_to_handle_stun(self):
        """datagram_received routes STUN messages (first byte 0–3) to _handle_stun."""
        p = make_register_protocol()
        p.connection_made(make_mock_transport())
        # STUN message starts with byte 0x01 (< 4)
        stun_data = b"\x01\x01" + b"\x00" * 18
        with patch.object(p, "handle_stun") as mock_handle:
            p.datagram_received(stun_data, ("stun.l.google.com", 19302))
            mock_handle.assert_called_once_with(stun_data, ("stun.l.google.com", 19302))

    def test_datagram_received__sip_routes_to_super(self):
        """datagram_received routes SIP messages (first byte >= 4) to the SIP parser."""
        p = make_register_protocol()
        p.connection_made(make_mock_transport())
        sip_data = b"SIP/2.0 200 OK\r\nCSeq: 1 REGISTER\r\n\r\n"
        with patch.object(IncomingCallProtocol, "datagram_received") as mock_super:
            p.datagram_received(sip_data, ("192.0.2.2", 5060))
            mock_super.assert_called_once_with(sip_data, ("192.0.2.2", 5060))

    def test_handle_stun__truncated_returns_early(self):
        """handle_stun silently ignores packets shorter than 20 bytes."""
        p = make_register_protocol()
        transport = make_mock_transport()
        p.connection_made(transport)
        transport.sendto.reset_mock()  # clear the REGISTER sendto call
        p.handle_stun(b"\x00" * 19, ("stun.l.google.com", 19302))
        transport.sendto.assert_not_called()

    def test_invite_received_after_register(self):
        """INVITE dispatching still works after registration (inherits IncomingCallProtocol)."""
        calls = []

        class ConcreteProtocol(RegisterProtocol):
            def invite_received(self, call, addr):
                calls.append((call, addr))

        p = ConcreteProtocol(("192.0.2.2", 5060), "sip:alice@example.com", "a", "b")
        p.connection_made(make_mock_transport())
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
        p.connection_made(make_mock_transport())
        with caplog.at_level(logging.INFO, logger="voip.call"):
            p.response_received(
                Response(status_code=200, reason="OK", headers={"CSeq": "1 REGISTER"}),
                ("192.0.2.2", 5060),
            )
        assert any("Registration successful" in r.message for r in caplog.records)

    def test_response_received__unexpected_status__logs_warning(self, caplog):
        """An unhandled status code logs a warning and raises NotImplementedError."""
        import logging

        p = make_register_protocol()
        p.connection_made(make_mock_transport())
        with caplog.at_level(logging.WARNING, logger="voip.call"):
            with pytest.raises(NotImplementedError):
                p.response_received(
                    Response(
                        status_code=500,
                        reason="Server Error",
                        headers={"CSeq": "1 REGISTER"},
                    ),
                    ("192.0.2.2", 5060),
                )
        assert any("500" in r.message for r in caplog.records)


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


def make_register_protocol(
    server_addr=("192.0.2.2", 5060),
    aor="sip:alice@example.com",
    username="alice",
    password="test-password",  # noqa: S107
) -> RegisterProtocol:
    """Return a RegisterProtocol without triggering connection_made."""
    return RegisterProtocol(server_addr, aor, username, password)


def make_mock_transport(host: str = "127.0.0.1", port: int = 5060):
    """Return a MagicMock transport with get_extra_info('sockname') configured."""
    transport = MagicMock()
    transport.get_extra_info.return_value = (host, port)
    return transport
