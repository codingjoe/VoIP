"""Tests for the SIP session and RTP call handler."""

import asyncio
import socket
import struct
from unittest.mock import MagicMock, patch

import pytest
from voip.rtp import RTP
from voip.sip import SIP, SessionInitiationProtocol
from voip.sip.messages import Message, Request, Response


class TestRTP:
    def test_caller__returns_caller_arg(self):
        """Return the caller string passed at construction."""
        call = RTP(caller="sip:bob@biloxi.com")
        assert call.caller == "sip:bob@biloxi.com"

    def test_caller__defaults_to_empty_string(self):
        """Return an empty string when no caller is given."""
        assert RTP().caller == ""

    def test_audio_received__noop_by_default(self):
        """audio_received is a no-op in the base class."""
        RTP().audio_received(b"data")  # must not raise


class TestSIP:
    def test_connection_made__stores_transport(self):
        """Store the transport when a connection is established."""
        protocol = SIP()
        transport = MagicMock()
        protocol.connection_made(transport)
        assert protocol._transport is transport

    def test_send__serializes_and_forwards_to_transport(self):
        """Serialize the message and forward it to the underlying transport."""
        protocol = SIP()
        protocol.connection_made(MagicMock())
        response = Response(status_code=200, reason="OK")
        addr = ("192.0.2.1", 5060)
        protocol.send(response, addr)
        protocol._transport.sendto.assert_called_once_with(bytes(response), addr)

    def test_request_received__invite__stores_addr_and_calls_call_received(self):
        """Dispatch an INVITE to call_received and store the addr by Call-ID."""
        received = []

        class MySIP(SIP):
            def call_received(self, request):
                received.append(request)

        protocol = MySIP()
        protocol.connection_made(MagicMock())
        request = make_invite()
        addr = ("192.0.2.1", 5060)
        protocol.request_received(request, addr)
        assert len(received) == 1
        assert received[0] is request
        assert protocol._request_addrs.get(request.headers["Call-ID"]) == addr

    def test_call_received__noop_by_default(self):
        """call_received is a no-op in the base class."""
        protocol = SIP()
        protocol.connection_made(MagicMock())
        protocol.call_received(make_invite())  # must not raise

    def test_answer__sends_200_ok(self):
        """Send a 200 OK response with an SDP body when answering."""

        async def run() -> None:
            protocol = SIP()
            send = MagicMock()
            protocol.send = send
            request = make_invite()
            protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
            await protocol._answer(request, RTP)
            send.assert_called_once()
            response, addr = send.call_args[0]
            assert response.status_code == 200
            assert response.reason == "OK"
            assert addr == ("192.0.2.1", 5060)

        asyncio.run(run())

    def test_answer__sdp_contains_opus_audio_line(self):
        """Include an Opus audio media line in the SDP body of the 200 OK."""

        async def run() -> None:
            protocol = SIP()
            send = MagicMock()
            protocol.send = send
            request = make_invite()
            protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
            await protocol._answer(request, RTP)
            response, _ = send.call_args[0]
            assert b"m=audio" in response.body
            assert b"RTP/AVP 111" in response.body
            assert b"a=rtpmap:111 opus/48000/1" in response.body

        asyncio.run(run())

    def test_answer__copies_dialog_headers(self):
        """Copy Via, To, From, Call-ID, and CSeq headers into the 200 OK."""

        async def run() -> None:
            protocol = SIP()
            send = MagicMock()
            protocol.send = send
            request = make_invite()
            protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
            await protocol._answer(request, RTP)
            response, _ = send.call_args[0]
            assert response.headers["Via"] == "SIP/2.0/UDP pc33.atlanta.com"
            assert response.headers["To"] == "sip:alice@atlanta.com"
            assert response.headers["From"] == "sip:bob@biloxi.com"
            assert response.headers["Call-ID"] == "1234@pc33"
            assert response.headers["CSeq"] == "1 INVITE"

        asyncio.run(run())

    def test_answer__instantiates_call_class_with_caller(self):
        """The call_class is instantiated with the caller from the From header."""
        created = []

        class MyCall(RTP):
            def __init__(self, caller: str = "") -> None:
                super().__init__(caller=caller)
                created.append(self)

        async def run() -> None:
            protocol = SIP()
            protocol.send = MagicMock()
            request = make_invite()
            protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
            await protocol._answer(request, MyCall)

        asyncio.run(run())
        assert len(created) == 1
        assert created[0].caller == "sip:bob@biloxi.com"

    def test_answer__rtp_receives_audio(self):
        """Deliver audio from RTP packets to the call's audio_received via the RTP socket."""
        received_audio = []

        class AudioCapture(RTP):
            def audio_received(self, data: bytes) -> None:
                received_audio.append(data)

        async def run() -> None:
            protocol = SIP()
            send = MagicMock()
            protocol.send = send
            request = make_invite()
            protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
            await protocol._answer(request, AudioCapture)
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

        class AudioCapture(RTP):
            def audio_received(self, data: bytes) -> None:
                received_audio.append(data)

        async def run() -> None:
            protocol = SIP()
            send = MagicMock()
            protocol.send = send
            request = make_invite()
            protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
            await protocol._answer(request, AudioCapture)
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

    def test_answer__content_length_serialized(self):
        """Content-Length is automatically included when the response is serialized."""

        async def run() -> None:
            protocol = SIP()
            send = MagicMock()
            protocol.send = send
            request = make_invite()
            protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
            await protocol._answer(request, RTP)
            response, _ = send.call_args[0]
            serialized = bytes(response)
            parsed = Message.parse(serialized)
            assert "Content-Length" in parsed.headers

        asyncio.run(run())

    def test_answer__logs_info(self, caplog):
        """Log an info message when answering a call."""
        import logging

        async def run():
            protocol = SIP()
            protocol.send = MagicMock()
            request = make_invite()
            protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
            with caplog.at_level(logging.INFO, logger="voip.sip.protocol"):
                await protocol._answer(request, RTP)

        asyncio.run(run())
        assert any("Answering" in r.message for r in caplog.records)

    def test_reject__sends_busy_here_by_default(self):
        """Send a 486 Busy Here response when no status code is given."""
        send = MagicMock()
        protocol = SIP()
        protocol.send = send
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        protocol.reject(request)
        send.assert_called_once()
        response, addr = send.call_args[0]
        assert isinstance(response, Response)
        assert response.status_code == 486
        assert response.reason == "Busy Here"
        assert addr == ("192.0.2.1", 5060)

    def test_reject__custom_status(self):
        """Send the specified status code and reason."""
        send = MagicMock()
        protocol = SIP()
        protocol.send = send
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        protocol.reject(request, status_code=603, reason="Decline")
        response, _ = send.call_args[0]
        assert response.status_code == 603
        assert response.reason == "Decline"

    def test_reject__copies_dialog_headers(self):
        """Copy Via, To, From, Call-ID, and CSeq headers into the response."""
        send = MagicMock()
        protocol = SIP()
        protocol.send = send
        request = make_invite()
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        protocol.reject(request)
        response, _ = send.call_args[0]
        assert response.headers["Via"] == "SIP/2.0/UDP pc33.atlanta.com"
        assert response.headers["To"] == "sip:alice@atlanta.com"
        assert response.headers["From"] == "sip:bob@biloxi.com"
        assert response.headers["Call-ID"] == "1234@pc33"
        assert response.headers["CSeq"] == "1 INVITE"

    @pytest.mark.parametrize("extra_header", ["X-Custom"])
    def test_reject__excludes_extra_headers(self, extra_header):
        """Exclude non-dialog headers from the reject response."""
        send = MagicMock()
        protocol = SIP()
        protocol.send = send
        request = make_invite({extra_header: "value"})
        protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
        protocol.reject(request)
        response, _ = send.call_args[0]
        assert extra_header not in response.headers

    def test_reject__logs_info(self, caplog):
        """Log an info message when rejecting a call."""
        import logging

        with caplog.at_level(logging.INFO, logger="voip.sip.protocol"):
            protocol = SIP()
            protocol.send = MagicMock()
            request = make_invite()
            protocol._request_addrs[request.headers["Call-ID"]] = ("192.0.2.1", 5060)
            protocol.reject(request)
        assert any("Rejecting" in r.message for r in caplog.records)

    def test_datagram_received__keepalive__sends_pong(self):
        """Double-CRLF keepalive (RFC 5626 §4.4.1) is answered with a single-CRLF pong."""
        protocol = SIP()
        transport = MagicMock()
        protocol.connection_made(transport)
        addr = ("192.0.2.1", 5060)
        protocol.datagram_received(b"\r\n\r\n", addr)
        transport.sendto.assert_called_once_with(b"\r\n", addr)

    def test_request_received__unsupported_method__raises(self):
        """Raise NotImplementedError for any non-INVITE SIP request method."""
        protocol = SIP()
        protocol.connection_made(MagicMock())
        request = Request(method="OPTIONS", uri="sip:alice@atlanta.com")
        with pytest.raises(NotImplementedError, match="OPTIONS"):
            protocol.request_received(request, ("192.0.2.1", 5060))

    def test_answer__via_call_received__schedules_answer(self):
        """answer() schedules the async _answer task when called from call_received."""

        async def run():
            answered = []

            class MySIP(SIP):
                def call_received(self, request):
                    self.answer(request=request, call_class=RTP)

                async def _answer(self, request, call_class):
                    answered.append((request, call_class))

            protocol = MySIP()
            protocol.connection_made(MagicMock())
            request = make_invite()
            addr = ("192.0.2.1", 5060)
            protocol._request_addrs[request.headers["Call-ID"]] = addr
            protocol.call_received(request)
            await asyncio.sleep(0.01)
            assert len(answered) == 1
            assert answered[0][1] is RTP

        asyncio.run(run())


class TestSessionInitiationProtocol:
    def test_registrar_uri__strips_user_from_aor(self):
        """Derive registrar URI from AOR by stripping the user part."""
        p = make_register_session(aor="sip:alice@example.com")
        assert p.registrar_uri == "sip:example.com"

    def test_registrar_uri__preserves_port(self):
        """Preserve a non-default port in the derived registrar URI."""
        p = make_register_session(aor="sip:alice@example.com:5080")
        assert p.registrar_uri == "sip:example.com:5080"

    def test_connection_made__sends_register(self):
        """Send a REGISTER request immediately after connection is made."""
        p = make_register_session()
        transport = make_mock_transport()
        p.connection_made(transport)
        transport.sendto.assert_called_once()
        data, addr = transport.sendto.call_args[0]
        assert b"REGISTER sip:example.com SIP/2.0" in data
        assert addr == ("192.0.2.2", 5060)

    def test_connection_made__with_stun__registers_after_discovery(self):
        """Send REGISTER after STUN discovery when a STUN server is configured."""

        async def run():
            p = SessionInitiationProtocol(
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
            p = SessionInitiationProtocol(
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
        p = make_register_session()
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
        p = make_register_session()
        p.connection_made(make_mock_transport())
        assert p.cseq == 1
        p.register()
        assert p.cseq == 2

    def test_register__with_authorization(self):
        """Authorization header is included when credentials are provided."""
        p = make_register_session()
        p.connection_made(make_mock_transport())
        transport = p._transport
        transport.reset_mock()
        p.register(authorization='Digest username="alice"')
        data, _ = transport.sendto.call_args[0]
        assert b'Authorization: Digest username="alice"' in data

    def test_register__with_proxy_authorization(self):
        """Proxy-Authorization header is included for proxy challenges."""
        p = make_register_session()
        p.connection_made(make_mock_transport())
        transport = p._transport
        transport.reset_mock()
        p.register(proxy_authorization='Digest username="alice"')
        data, _ = transport.sendto.call_args[0]
        assert b'Proxy-Authorization: Digest username="alice"' in data

    def test_response_received__200_ok_calls_registered(self):
        """Receiving 200 OK for REGISTER triggers registered()."""
        calls = []

        class ConcreteSession(SessionInitiationProtocol):
            def registered(self):
                calls.append(True)

        p = ConcreteSession(("192.0.2.2", 5060), "sip:alice@example.com", "a", "b")
        p.connection_made(make_mock_transport())
        p.response_received(
            Response(status_code=200, reason="OK", headers={"CSeq": "1 REGISTER"}),
            ("192.0.2.2", 5060),
        )
        assert calls == [True]

    def test_response_received__200_non_register_raises(self):
        """Receiving 200 OK for a non-REGISTER method raises NotImplementedError."""
        p = make_register_session()
        p.connection_made(make_mock_transport())
        with pytest.raises(NotImplementedError):
            p.response_received(
                Response(status_code=200, reason="OK", headers={"CSeq": "1 INVITE"}),
                ("192.0.2.2", 5060),
            )

    def test_response_received__401_retries_with_authorization(self):
        """Receiving 401 triggers a re-REGISTER with an Authorization header."""
        p = make_register_session(username="alice", password="secret")  # noqa: S106
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
        p = make_register_session(username="alice", password="secret")  # noqa: S106
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
        p = make_register_session()
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
        p = make_register_session()
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

        p = make_register_session()
        transport = make_mock_transport("192.0.2.10", 5060)
        p.connection_made(transport)
        data, _ = transport.sendto.call_args[0]
        assert b"Via: SIP/2.0/UDP 192.0.2.10:5060;rport;branch=z9hG4bK" in data
        assert re.search(rb"branch=z9hG4bK[0-9a-f]{32}", data)

    def test_register__via_branch_is_unique_per_request(self):
        """Each REGISTER generates a unique Via branch to prevent transaction conflicts."""
        import re

        p = make_register_session()
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
        p = make_register_session()
        transport = make_mock_transport("10.0.0.5", 5060)
        p.connection_made(transport)
        data, _ = transport.sendto.call_args[0]
        assert b"Contact: <sip:alice@10.0.0.5:5060>" in data

    def test_register__contact_uses_public_addr_when_stun_discovered(self):
        """Contact header uses the STUN-discovered public address."""
        p = make_register_session()
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
            p = make_register_session()
            p.connection_made(make_mock_transport())
            p._stun_transactions = {}
            magic_cookie = 0x2112A442
            transaction_id = b"\x01" * 12
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
            p = make_register_session()
            p.connection_made(make_mock_transport())
            p._stun_transactions = {}
            magic_cookie = 0x2112A442
            transaction_id = b"\x02" * 12
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
        """datagram_received routes STUN messages (first byte 0-3) to handle_stun."""
        p = make_register_session()
        p.connection_made(make_mock_transport())
        stun_data = b"\x01\x01" + b"\x00" * 18
        with patch.object(p, "handle_stun") as mock_handle:
            p.datagram_received(stun_data, ("stun.l.google.com", 19302))
            mock_handle.assert_called_once_with(stun_data, ("stun.l.google.com", 19302))

    def test_datagram_received__sip_response__calls_response_received(self):
        """datagram_received routes SIP messages (first byte >= 4) to response_received."""
        received = []

        class ConcreteSession(SessionInitiationProtocol):
            def response_received(self, response, addr):
                received.append(response)

        p = ConcreteSession(("192.0.2.2", 5060), "sip:alice@example.com", "a", "b")
        p.connection_made(make_mock_transport())
        sip_data = b"SIP/2.0 200 OK\r\nCSeq: 1 REGISTER\r\n\r\n"
        p.datagram_received(sip_data, ("192.0.2.2", 5060))
        assert len(received) == 1
        assert received[0].status_code == 200

    def test_handle_stun__truncated_returns_early(self):
        """handle_stun silently ignores packets shorter than 20 bytes."""
        p = make_register_session()
        transport = make_mock_transport()
        p.connection_made(transport)
        transport.sendto.reset_mock()
        p.handle_stun(b"\x00" * 19, ("stun.l.google.com", 19302))
        transport.sendto.assert_not_called()

    def test_invite_received_after_register(self):
        """INVITE dispatching still works after registration (call_received is called)."""
        received = []

        class ConcreteSession(SessionInitiationProtocol):
            def call_received(self, request):
                received.append(request)

        p = ConcreteSession(("192.0.2.2", 5060), "sip:alice@example.com", "a", "b")
        p.connection_made(make_mock_transport())
        request = Request(
            method="INVITE",
            uri="sip:alice@example.com",
            headers={"From": "sip:bob@example.com", "Call-ID": "test@pc"},
        )
        p.request_received(request, ("192.0.2.1", 5060))
        assert len(received) == 1
        assert received[0] is request

    def test_response_received__200_ok__logs_info(self, caplog):
        """Receiving 200 OK logs an info message."""
        import logging

        p = make_register_session()
        p.connection_made(make_mock_transport())
        with caplog.at_level(logging.INFO, logger="voip.sip.protocol"):
            p.response_received(
                Response(status_code=200, reason="OK", headers={"CSeq": "1 REGISTER"}),
                ("192.0.2.2", 5060),
            )
        assert any("Registration successful" in r.message for r in caplog.records)

    def test_response_received__unexpected_status__logs_warning(self, caplog):
        """An unhandled status code logs a warning and raises NotImplementedError."""
        import logging

        p = make_register_session()
        p.connection_made(make_mock_transport())
        with caplog.at_level(logging.WARNING, logger="voip.sip.protocol"):
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


def make_register_session(
    server_addr=("192.0.2.2", 5060),
    aor="sip:alice@example.com",
    username="alice",
    password="secret",  # noqa: S107
) -> SessionInitiationProtocol:
    """Return a SessionInitiationProtocol session without triggering connection_made."""
    return SessionInitiationProtocol(server_addr, aor, username, password)


def make_mock_transport(host: str = "127.0.0.1", port: int = 5060):
    """Return a MagicMock transport with get_extra_info('sockname') configured."""
    transport = MagicMock()
    transport.get_extra_info.return_value = (host, port)
    return transport
