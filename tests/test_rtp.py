"""Tests for the RTP protocol implementation (RFC 3550)."""

from __future__ import annotations

import asyncio
import struct

import pytest
from voip.rtp import RTP, RealtimeTransportProtocol, RTPPacket, RTPPayloadType
from voip.sdp.types import Attribute, MediaDescription


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

    def test_datagram_received__forwards_audio_payload(self):
        """Parse the RTP packet and forward it to audio_received."""
        received: list[RTPPacket] = []

        class ConcreteRTP(RealtimeTransportProtocol):
            def audio_received(self, packet: RTPPacket) -> None:
                received.append(packet)

        ConcreteRTP().datagram_received(
            make_rtp_packet(payload=b"audio"), ("127.0.0.1", 5004)
        )
        assert len(received) == 1
        assert received[0].payload == b"audio"

    def test_datagram_received__skips_packet_shorter_than_header(self):
        """Skip packets shorter than the 12-byte RTP header."""
        received: list[RTPPacket] = []

        class ConcreteRTP(RealtimeTransportProtocol):
            def audio_received(self, packet: RTPPacket) -> None:
                received.append(packet)

        ConcreteRTP().datagram_received(b"\x80\x00", ("127.0.0.1", 5004))
        assert received == []

    def test_datagram_received__skips_header_only_packet(self):
        """Skip packets that contain only the 12-byte header with no audio payload."""
        received: list[RTPPacket] = []

        class ConcreteRTP(RealtimeTransportProtocol):
            def audio_received(self, packet: RTPPacket) -> None:
                received.append(packet)

        ConcreteRTP().datagram_received(b"\x80" * 12, ("127.0.0.1", 5004))
        assert received == []

    def test_init__stores_media(self):
        """Media parameter is stored on the protocol instance."""
        media = MediaDescription(
            media="audio", port=49170, proto="RTP/AVP", fmt=["8"],
            attributes=[Attribute(name="rtpmap", value="8 PCMA/8000")]
        )
        protocol = RealtimeTransportProtocol(media=media)
        assert protocol.media is media

    def test_init__derives_sample_rate_from_media(self):
        """sample_rate is derived from the rtpmap attribute of the MediaDescription."""
        media = MediaDescription(
            media="audio", port=49170, proto="RTP/AVP", fmt=["9"],
            attributes=[Attribute(name="rtpmap", value="9 G722/8000")]
        )
        protocol = RealtimeTransportProtocol(media=media)
        assert protocol.sample_rate == 8000

    def test_init__default_sample_rate_without_media(self):
        """Default sample_rate is 8000 Hz when no MediaDescription is given."""
        protocol = RealtimeTransportProtocol()
        assert protocol.sample_rate == 8000

    def test_init__derives_payload_type_from_media(self):
        """payload_type is derived from the first fmt entry of the MediaDescription."""
        media = MediaDescription(
            media="audio", port=49170, proto="RTP/AVP", fmt=["8"], attributes=[]
        )
        protocol = RealtimeTransportProtocol(media=media)
        assert protocol.payload_type == 8

    def test_init__default_payload_type_without_media(self):
        """Default payload_type is 0 when no MediaDescription is given."""
        assert RealtimeTransportProtocol().payload_type == 0


class TestNegotiateCodec:
    def _make_media(self, fmts: list[str], rtpmaps: list[str] | None = None) -> MediaDescription:
        """Build a MediaDescription with given format list and optional rtpmap attributes."""
        attributes = []
        for rtpmap in (rtpmaps or []):
            attributes.append(Attribute(name="rtpmap", value=rtpmap))
        return MediaDescription(media="audio", port=49170, proto="RTP/AVP", fmt=fmts, attributes=attributes)

    def test_negotiate_codec__prefers_opus(self):
        """Select Opus when offered alongside lower-priority codecs."""
        media = self._make_media(
            ["0", "8", "111"],
            ["111 opus/48000/2", "8 PCMA/8000"],
        )
        result = RealtimeTransportProtocol.negotiate_codec(media)
        assert result is not None
        fmt, rtpmap, sample_rate = result
        assert fmt == "111"
        assert sample_rate == 48000

    def test_negotiate_codec__falls_back_to_pcma(self):
        """Select PCMA when Opus and G.722 are not offered."""
        media = self._make_media(["0", "8"])
        result = RealtimeTransportProtocol.negotiate_codec(media)
        assert result is not None
        assert result[0] == "8"
        assert result[2] == 8000

    def test_negotiate_codec__falls_back_to_pcmu(self):
        """Select PCMU when only PCMU is offered."""
        media = self._make_media(["0"])
        result = RealtimeTransportProtocol.negotiate_codec(media)
        assert result is not None
        assert result[0] == "0"

    def test_negotiate_codec__empty_fmt__returns_none(self):
        """Return None when the remote side offers no audio formats."""
        media = self._make_media([])
        assert RealtimeTransportProtocol.negotiate_codec(media) is None

    def test_negotiate_codec__unknown_codec__returns_first(self):
        """Fall back to the first offered format for an unrecognised codec."""
        media = self._make_media(
            ["126"], ["126 telephone-event/8000"]
        )
        result = RealtimeTransportProtocol.negotiate_codec(media)
        assert result is not None
        assert result[0] == "126"

    def test_negotiate_codec__subclass_can_override_preferences(self):
        """A subclass with a different PREFERRED_CODECS list uses its own preferences."""
        class PCMAOnlyCall(RealtimeTransportProtocol):
            PREFERRED_CODECS = [("8", "PCMA/8000", 8000)]

        media = self._make_media(["0", "8", "111"])
        result = PCMAOnlyCall.negotiate_codec(media)
        assert result is not None
        assert result[0] == "8"
