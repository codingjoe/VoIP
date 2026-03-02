"""Tests for SIP call handling."""

import asyncio
from unittest.mock import MagicMock

import pytest
from sip.calls import IncomingCall, _RTPProtocol
from sip.messages import Message, Request


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


def make_call(transport: MagicMock | None = None) -> IncomingCall:
    """Return an IncomingCall with a mock transport."""
    return IncomingCall(
        make_invite(),
        ("192.0.2.1", 5060),
        transport or MagicMock(),
    )


class TestRTPProtocol:
    def test_datagram_received__forwards_audio(self):
        """Strip RTP header and forward audio payload to the call handler."""
        call = MagicMock()
        protocol = _RTPProtocol(call)
        rtp_packet = b"\x80\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00" + b"audio"
        protocol.datagram_received(rtp_packet, ("192.0.2.1", 5004))
        call.handle.assert_called_once_with(b"audio")

    def test_datagram_received__skips_short_packet(self):
        """Skip packets shorter than the minimum RTP header size."""
        call = MagicMock()
        protocol = _RTPProtocol(call)
        protocol.datagram_received(b"\x80\x00", ("192.0.2.1", 5004))
        call.handle.assert_not_called()

    def test_datagram_received__skips_exact_header_size(self):
        """Skip packets that contain only an RTP header with no audio payload."""
        call = MagicMock()
        protocol = _RTPProtocol(call)
        protocol.datagram_received(b"\x80" * 12, ("192.0.2.1", 5004))
        call.handle.assert_not_called()


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

    def test_handle__returns_not_implemented(self):
        """Return NotImplemented for unhandled audio data."""
        assert make_call().handle(b"audio") is NotImplemented

    def test_reject__sends_busy_here_by_default(self):
        """Send a 486 Busy Here response when no status code is given."""
        transport = MagicMock()
        make_call(transport).reject()
        transport.sendto.assert_called_once()
        sent_bytes, addr = transport.sendto.call_args[0]
        response = Message.parse(sent_bytes)
        assert response.status_code == 486
        assert response.reason == "Busy Here"
        assert addr == ("192.0.2.1", 5060)

    def test_reject__custom_status(self):
        """Send the specified status code and reason."""
        transport = MagicMock()
        make_call(transport).reject(status_code=603, reason="Decline")
        sent_bytes, _ = transport.sendto.call_args[0]
        response = Message.parse(sent_bytes)
        assert response.status_code == 603
        assert response.reason == "Decline"

    def test_reject__copies_dialog_headers(self):
        """Copy Via, To, From, Call-ID, and CSeq headers into the response."""
        transport = MagicMock()
        make_call(transport).reject()
        sent_bytes, _ = transport.sendto.call_args[0]
        response = Message.parse(sent_bytes)
        assert response.headers["Via"] == "SIP/2.0/UDP pc33.atlanta.com"
        assert response.headers["To"] == "sip:alice@atlanta.com"
        assert response.headers["From"] == "sip:bob@biloxi.com"
        assert response.headers["Call-ID"] == "1234@pc33"
        assert response.headers["CSeq"] == "1 INVITE"

    def test_answer__sends_200_ok(self):
        """Send a 200 OK response with an SDP body when answering."""

        async def run() -> None:
            transport = MagicMock()
            await make_call(transport).answer()
            transport.sendto.assert_called_once()
            sent_bytes, addr = transport.sendto.call_args[0]
            response = Message.parse(sent_bytes)
            assert response.status_code == 200
            assert response.reason == "OK"
            assert addr == ("192.0.2.1", 5060)

        asyncio.run(run())

    def test_answer__sdp_contains_audio_line(self):
        """Include an audio media line in the SDP body of the 200 OK."""

        async def run() -> None:
            transport = MagicMock()
            await make_call(transport).answer()
            sent_bytes, _ = transport.sendto.call_args[0]
            response = Message.parse(sent_bytes)
            assert b"m=audio" in response.body
            assert b"RTP/AVP" in response.body

        asyncio.run(run())

    def test_answer__copies_dialog_headers(self):
        """Copy Via, To, From, Call-ID, and CSeq headers into the 200 OK."""

        async def run() -> None:
            transport = MagicMock()
            await make_call(transport).answer()
            sent_bytes, _ = transport.sendto.call_args[0]
            response = Message.parse(sent_bytes)
            assert response.headers["Via"] == "SIP/2.0/UDP pc33.atlanta.com"
            assert response.headers["To"] == "sip:alice@atlanta.com"
            assert response.headers["From"] == "sip:bob@biloxi.com"
            assert response.headers["Call-ID"] == "1234@pc33"
            assert response.headers["CSeq"] == "1 INVITE"

        asyncio.run(run())

    def test_answer__rtp_receives_audio(self):
        """Deliver audio from RTP packets to the call's handle method via the RTP socket."""
        received_audio = []

        class AudioCapture(IncomingCall):
            def handle(self, audio: bytes) -> None:
                received_audio.append(audio)

        async def run() -> None:
            transport = MagicMock()
            call = AudioCapture(make_invite(), ("192.0.2.1", 5060), transport)
            await call.answer()
            sent_bytes, _ = transport.sendto.call_args[0]
            response = Message.parse(sent_bytes)

            # Extract the RTP port from the SDP body and send a test packet
            sdp_line = next(
                line
                for line in response.body.decode().splitlines()
                if line.startswith("m=audio")
            )
            rtp_port = int(sdp_line.split()[1])

            # Send a fake RTP packet directly to the local RTP socket
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
        """Call handle with each RTP payload when multiple packets arrive."""
        received_audio = []

        class AudioCapture(IncomingCall):
            def handle(self, audio: bytes) -> None:
                received_audio.append(audio)

        async def run() -> None:
            transport = MagicMock()
            call = AudioCapture(make_invite(), ("192.0.2.1", 5060), transport)
            await call.answer()
            sent_bytes, _ = transport.sendto.call_args[0]
            response = Message.parse(sent_bytes)
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

    @pytest.mark.parametrize(
        "extra_header",
        ["X-Custom"],
    )
    def test_reject__excludes_extra_headers(self, extra_header):
        """Exclude non-dialog headers from the reject response."""
        transport = MagicMock()
        call = IncomingCall(
            make_invite({extra_header: "value"}),
            ("192.0.2.1", 5060),
            transport,
        )
        call.reject()
        sent_bytes, _ = transport.sendto.call_args[0]
        response = Message.parse(sent_bytes)
        assert extra_header not in response.headers
