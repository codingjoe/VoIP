"""Tests for the RTP protocol implementation (RFC 3550)."""

from __future__ import annotations

import asyncio
import dataclasses
import struct
from unittest.mock import MagicMock

import pytest
from voip.call import Call
from voip.rtp import RTP, RealtimeTransportProtocol, RTPPacket, RTPPayloadType


def make_rtp_packet(
    payload_type: int = 111,
    sequence_number: int = 1,
    timestamp: int = 0,
    ssrc: int = 0,
    payload: bytes = b"audio",
) -> bytes:
    """Build a raw RTP packet with the given fields."""
    header = struct.pack(
        ">BBHII",
        0x80,  # V=2, P=0, X=0, CC=0
        payload_type & 0x7F,
        sequence_number,
        timestamp,
        ssrc,
    )
    return header + payload


class TestRTPPayloadType:
    def test_pcmu__value(self):
        """PCMU payload type is 0 per RFC 3551."""
        assert RTPPayloadType.PCMU == 0

    def test_pcma__value(self):
        """PCMA payload type is 8 per RFC 3551."""
        assert RTPPayloadType.PCMA == 8

    def test_g722__value(self):
        """G722 payload type is 9 per RFC 3551."""
        assert RTPPayloadType.G722 == 9

    def test_opus__value(self):
        """OPUS payload type is 111 per RFC 7587."""
        assert RTPPayloadType.OPUS == 111


class TestRTPPacket:
    def test_parse__extracts_payload_type(self):
        """Parse the payload type from byte 1 of the RTP header."""
        packet = RTPPacket.parse(make_rtp_packet(payload_type=111))
        assert packet.payload_type == 111

    def test_parse__extracts_sequence_number(self):
        """Parse the sequence number from bytes 2–3 of the RTP header."""
        packet = RTPPacket.parse(make_rtp_packet(sequence_number=42))
        assert packet.sequence_number == 42

    def test_parse__extracts_timestamp(self):
        """Parse the timestamp from bytes 4–7 of the RTP header."""
        packet = RTPPacket.parse(make_rtp_packet(timestamp=96000))
        assert packet.timestamp == 96000

    def test_parse__extracts_ssrc(self):
        """Parse the SSRC from bytes 8–11 of the RTP header."""
        packet = RTPPacket.parse(make_rtp_packet(ssrc=0xDEADBEEF))
        assert packet.ssrc == 0xDEADBEEF

    def test_parse__extracts_payload(self):
        """Extract the payload after the 12-byte fixed header."""
        packet = RTPPacket.parse(make_rtp_packet(payload=b"hello"))
        assert packet.payload == b"hello"

    def test_parse__raises_value_error_on_short_packet(self):
        """Raise ValueError when the packet is shorter than 12 bytes."""
        with pytest.raises(ValueError, match="too short"):
            RTPPacket.parse(b"\x80" * 11)

    def test_parse__empty_payload(self):
        """Parse a packet with no payload (header-only)."""
        packet = RTPPacket.parse(make_rtp_packet(payload=b""))
        assert packet.payload == b""

    def test_parse__marker_bit_ignored_in_payload_type(self):
        """The marker bit (bit 7 of byte 1) does not affect the parsed payload type."""
        raw = make_rtp_packet(payload_type=111)
        # Set marker bit: byte 1 becomes 0x80 | 111 = 0xEF
        raw = raw[:1] + bytes([raw[1] | 0x80]) + raw[2:]
        packet = RTPPacket.parse(raw)
        assert packet.payload_type == 111


class TestRealtimeTransportProtocol:
    def test_rtp_header_size__class_attribute(self):
        """rtp_header_size is a class attribute set to the standard 12-byte header."""
        assert RealtimeTransportProtocol.rtp_header_size == 12

    def test_rtp__is_datagram_protocol(self):
        """RealtimeTransportProtocol is an asyncio.DatagramProtocol subclass."""
        assert issubclass(RealtimeTransportProtocol, asyncio.DatagramProtocol)

    def test_rtp__alias(self):
        """RTP is an alias for RealtimeTransportProtocol."""
        assert RTP is RealtimeTransportProtocol

    @pytest.mark.asyncio
    async def test_datagram_received__routes_to_handler(self):
        """Non-STUN datagrams from registered addr are forwarded to the handler."""
        routed: list[bytes] = []

        class RecordCall(Call):
            def datagram_received(self, data: bytes, addr):
                routed.append(data)

        mux = RealtimeTransportProtocol()
        handler = RecordCall(rtp=mux, sip=MagicMock())
        remote_addr = ("127.0.0.1", 5004)
        mux.register_call(remote_addr, handler)
        rtp_packet = make_rtp_packet(payload=b"audio")
        mux.datagram_received(rtp_packet, remote_addr)
        assert routed == [rtp_packet]

    @pytest.mark.asyncio
    async def test_datagram_received__no_handler__drops_packet(self):
        """Packets from an unknown address with no wildcard handler are dropped silently."""
        mux = RealtimeTransportProtocol()
        # No handler registered; must not raise.
        mux.datagram_received(make_rtp_packet(payload=b"ignored"), ("9.9.9.9", 5004))

    async def test_datagram_received__stun_packet__not_forwarded(self):
        """A STUN packet (first byte < 4) must not reach any Call handler."""
        routed: list[bytes] = []

        class RecordCall(Call):
            def datagram_received(self, data: bytes, addr):
                routed.append(data)

        mux = RealtimeTransportProtocol()
        mux.connection_made(MagicMock(spec=asyncio.DatagramTransport))
        handler = RecordCall(rtp=mux, sip=MagicMock())
        mux.register_call(None, handler)
        stun_bytes = b"\x01\x01" + b"\x00" * 18  # first byte = 1 (STUN range [0,3])
        mux.datagram_received(stun_bytes, ("127.0.0.1", 5004))
        assert routed == []

    async def test_stun_discover__uses_actual_socket(self):
        """public_address is resolved via the socket bound to the RTP protocol."""
        from voip.stun import MAGIC_COOKIE, STUNMessageType  # noqa: PLC0415

        received_requests: list[bytes] = []
        server_transport: asyncio.DatagramTransport | None = None

        class _StubSTUNServer(asyncio.DatagramProtocol):
            def connection_made(self, transport):
                nonlocal server_transport
                server_transport = transport

            def datagram_received(self, data, addr):
                received_requests.append(data)
                tid = data[8:20]
                # Build XOR-MAPPED-ADDRESS for 203.0.113.5:54321 (TEST-NET-3, RFC 5737)
                ip_int = (203 << 24) | (0 << 16) | (113 << 8) | 5
                xor_ip = ip_int ^ MAGIC_COOKIE
                xor_port = 54321 ^ (MAGIC_COOKIE >> 16)
                attr = struct.pack(">HH", 0x0020, 8) + struct.pack(
                    ">BBH I", 0, 1, xor_port, xor_ip
                )
                response = (
                    struct.pack(
                        ">HHI12s",
                        STUNMessageType.BINDING_SUCCESS_RESPONSE,
                        len(attr),
                        MAGIC_COOKIE,
                        tid,
                    )
                    + attr
                )
                server_transport.sendto(response, addr)

        loop = asyncio.get_running_loop()
        server_t, _ = await loop.create_datagram_endpoint(
            _StubSTUNServer, local_addr=("127.0.0.1", 0)
        )
        server_addr = server_t.get_extra_info("sockname")

        proto = RealtimeTransportProtocol(stun_server_address=server_addr)
        rtp_t, _ = await loop.create_datagram_endpoint(
            lambda: proto, local_addr=("127.0.0.1", 0)
        )
        try:
            result = await proto.public_address
            assert result == ("203.0.113.5", 54321)
            assert len(received_requests) == 1
        finally:
            rtp_t.close()
            server_t.close()

    @pytest.mark.asyncio
    async def test_register_call__routes_by_addr(self):
        """Packets from a registered remote addr are forwarded to the registered handler."""
        received_wildcard: list[bytes] = []
        received_call: list[bytes] = []

        class WildcardCall(Call):
            def datagram_received(self, data: bytes, addr):
                received_wildcard.append(data)

        class SpecificCall(Call):
            def datagram_received(self, data: bytes, addr):
                received_call.append(data)

        mux = RealtimeTransportProtocol()
        specific_addr = ("1.2.3.4", 5004)
        wildcard_handler = WildcardCall(rtp=mux, sip=MagicMock())
        specific_handler = SpecificCall(rtp=mux, sip=MagicMock())
        mux.register_call(None, wildcard_handler)
        mux.register_call(specific_addr, specific_handler)

        rtp_packet = make_rtp_packet(payload=b"call-audio")
        mux.datagram_received(rtp_packet, specific_addr)
        assert received_call == [rtp_packet]
        assert received_wildcard == []

    @pytest.mark.asyncio
    async def test_register_call__unmatched_addr_uses_wildcard_handler(self):
        """Packets from an unknown addr reach the None-key wildcard handler."""
        received: list[bytes] = []

        class WildcardCall(Call):
            def datagram_received(self, data: bytes, addr):
                received.append(data)

        mux = RealtimeTransportProtocol()
        handler = WildcardCall(rtp=mux, sip=MagicMock())
        mux.register_call(None, handler)

        rtp_packet = make_rtp_packet(payload=b"unmatched")
        mux.datagram_received(rtp_packet, ("9.9.9.9", 9999))
        assert received == [rtp_packet]

    @pytest.mark.asyncio
    async def test_unregister_call__removes_handler(self):
        """After unregister_call, packets from that addr are no longer routed to handler."""
        received: list[bytes] = []

        class RecordCall(Call):
            def datagram_received(self, data: bytes, addr):
                received.append(data)

        mux = RealtimeTransportProtocol()
        handler = RecordCall(rtp=mux, sip=MagicMock())
        remote_addr = ("5.6.7.8", 5004)
        mux.register_call(remote_addr, handler)
        mux.unregister_call(remote_addr)

        mux.datagram_received(make_rtp_packet(payload=b"gone"), remote_addr)
        assert received == []

    @pytest.mark.asyncio
    async def test_register_call__logs_info(self, caplog):
        """register_call emits an info-level log entry."""
        import logging  # noqa: PLC0415

        mux = RealtimeTransportProtocol()
        handler = Call(rtp=mux, sip=MagicMock())
        with caplog.at_level(logging.INFO, logger="voip.rtp"):
            mux.register_call(("1.2.3.4", 5004), handler)
        assert any("rtp_call_registered" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_unregister_call__logs_info(self, caplog):
        """unregister_call emits an info-level log entry."""
        import logging  # noqa: PLC0415

        mux = RealtimeTransportProtocol()
        handler = Call(rtp=mux, sip=MagicMock())
        addr = ("1.2.3.4", 5004)
        mux.register_call(addr, handler)
        with caplog.at_level(logging.INFO, logger="voip.rtp"):
            mux.unregister_call(addr)
        assert any("rtp_call_unregistered" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_packet_received__dispatches_to_handler(self):
        """packet_received forwards the datagram to the registered handler."""
        received: list[tuple[bytes, tuple]] = []

        class CapturingCall(Call):
            def datagram_received(self, data, addr):
                received.append((data, addr))

        mux = RealtimeTransportProtocol()
        handler = CapturingCall(rtp=mux, sip=MagicMock())
        mux.register_call(None, handler)
        packet = make_rtp_packet()
        mux.packet_received(packet, ("1.2.3.4", 5004))
        assert len(received) == 1
        assert received[0][1] == ("1.2.3.4", 5004)

    @pytest.mark.asyncio
    async def test_packet_received__logs_debug_for_dropped_packet(self, caplog):
        """packet_received logs a debug message when no handler is registered."""
        import logging  # noqa: PLC0415

        mux = RealtimeTransportProtocol()
        with caplog.at_level(logging.DEBUG, logger="voip.rtp"):
            mux.packet_received(make_rtp_packet(), ("1.2.3.4", 5004))
        assert any("dropping" in r.message for r in caplog.records)

    def test_inherits_stun_protocol(self):
        """RealtimeTransportProtocol inherits from STUNProtocol."""
        from voip.stun import STUNProtocol  # noqa: PLC0415

        assert issubclass(RealtimeTransportProtocol, STUNProtocol)


class TestSRTPIntegration:
    """Tests for SRTP decryption in the RTP mux (voip.srtp integration)."""

    @pytest.mark.asyncio
    async def test_srtp_packet__decrypted_before_delivery(self):
        """SRTP-encrypted packets are decrypted before being passed to datagram_received."""
        from voip.srtp import SRTPSession  # noqa: PLC0415

        received: list[bytes] = []

        @dataclasses.dataclass
        class SRTPCapture(Call):
            def datagram_received(self, data: bytes, addr) -> None:
                received.append(data)

        mux = RealtimeTransportProtocol()
        session = SRTPSession.generate()
        handler = SRTPCapture(rtp=mux, sip=MagicMock(), srtp=session)
        mux.register_call(None, handler)

        rtp_packet = make_rtp_packet(payload=b"secret")
        srtp_packet = session.encrypt(rtp_packet)
        assert srtp_packet != rtp_packet
        mux.datagram_received(srtp_packet, ("1.2.3.4", 5004))
        assert received == [rtp_packet]

    @pytest.mark.asyncio
    async def test_srtp_invalid_auth_tag__discarded(self, caplog):
        """Discard SRTP packets with invalid authentication tags and log a warning."""
        import logging  # noqa: PLC0415

        from voip.srtp import SRTPSession  # noqa: PLC0415

        received: list[bytes] = []

        @dataclasses.dataclass
        class SRTPCapture(Call):
            def datagram_received(self, data: bytes, addr) -> None:
                received.append(data)

        mux = RealtimeTransportProtocol()
        session = SRTPSession.generate()
        handler = SRTPCapture(rtp=mux, sip=MagicMock(), srtp=session)
        mux.register_call(None, handler)

        rtp_packet = make_rtp_packet(payload=b"tampered")
        srtp_packet = session.encrypt(rtp_packet)
        # Corrupt the authentication tag.
        tampered = srtp_packet[:-1] + bytes([srtp_packet[-1] ^ 0xFF])
        with caplog.at_level(logging.WARNING, logger="voip.rtp"):
            mux.datagram_received(tampered, ("1.2.3.4", 5004))
        assert received == []
        assert any("authentication failed" in r.message for r in caplog.records)
