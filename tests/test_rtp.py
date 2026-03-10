"""Tests for the RTP protocol implementation (RFC 3550)."""

from __future__ import annotations

import asyncio
import struct

import pytest
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

    def test_datagram_received__forwards_audio_payload(self):
        """Strip the RTP header and forward the audio payload to audio_received."""
        received: list[bytes] = []

        class ConcreteRTP(RealtimeTransportProtocol):
            def audio_received(self, data: bytes) -> None:
                received.append(data)

        ConcreteRTP().datagram_received(
            make_rtp_packet(payload=b"audio"), ("127.0.0.1", 5004)
        )
        assert received == [b"audio"]

    def test_datagram_received__skips_packet_shorter_than_header(self):
        """Skip packets shorter than the 12-byte RTP header."""
        received: list[bytes] = []

        class ConcreteRTP(RealtimeTransportProtocol):
            def audio_received(self, data: bytes) -> None:
                received.append(data)

        ConcreteRTP().datagram_received(b"\x80\x00", ("127.0.0.1", 5004))
        assert received == []

    def test_datagram_received__skips_header_only_packet(self):
        """Skip packets that contain only the 12-byte header with no audio payload."""
        received: list[bytes] = []

        class ConcreteRTP(RealtimeTransportProtocol):
            def audio_received(self, data: bytes) -> None:
                received.append(data)

        ConcreteRTP().datagram_received(b"\x80" * 12, ("127.0.0.1", 5004))
        assert received == []
