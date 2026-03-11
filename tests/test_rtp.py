"""Tests for the RTP protocol implementation (RFC 3550)."""

from __future__ import annotations

import asyncio
import struct

import pytest
from voip.rtp import RTP, RealtimeTransportProtocol, RTPPacket, RTPPayloadType
from voip.sdp.types import MediaDescription, RTPPayloadFormat


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
        """Parse the RTP packet and forward it to audio_received as a list of payloads."""
        received: list[list[bytes]] = []

        class ConcreteRTP(RealtimeTransportProtocol):
            def audio_received(self, packets: list[bytes]) -> None:
                received.append(packets)

        ConcreteRTP().datagram_received(
            make_rtp_packet(payload=b"audio"), ("127.0.0.1", 5004)
        )
        assert len(received) == 1
        assert received[0] == [b"audio"]

    def test_datagram_received__skips_packet_shorter_than_header(self):
        """Skip packets shorter than the 12-byte RTP header."""
        received: list[list[bytes]] = []

        class ConcreteRTP(RealtimeTransportProtocol):
            def audio_received(self, packets: list[bytes]) -> None:
                received.append(packets)

        ConcreteRTP().datagram_received(b"\x80\x00", ("127.0.0.1", 5004))
        assert received == []

    def test_datagram_received__skips_header_only_packet(self):
        """Skip packets that contain only the 12-byte header with no audio payload."""
        received: list[list[bytes]] = []

        class ConcreteRTP(RealtimeTransportProtocol):
            def audio_received(self, packets: list[bytes]) -> None:
                received.append(packets)

        ConcreteRTP().datagram_received(b"\x80" * 12, ("127.0.0.1", 5004))
        assert received == []

    def test_init__stores_media(self):
        """Media parameter is stored on the protocol instance."""
        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ],
        )
        protocol = RealtimeTransportProtocol(media=media)
        assert protocol.media is media

    def test_init__derives_sample_rate_from_media(self):
        """sample_rate is derived from the RTPPayloadFormat sample_rate."""
        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=9, encoding_name="G722", sample_rate=8000)
            ],
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
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=8)],
        )
        protocol = RealtimeTransportProtocol(media=media)
        assert protocol.payload_type == 8

    def test_init__default_payload_type_without_media(self):
        """Default payload_type is 0 when no MediaDescription is given."""
        assert RealtimeTransportProtocol().payload_type == 0

    def test_init__logs_codec_info(self, caplog):
        """Log codec name, sample rate and payload type at INFO level on init."""
        import logging

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ],
        )
        with caplog.at_level(logging.INFO, logger="voip.rtp"):
            RealtimeTransportProtocol(media=media)
        assert any("PCMA" in r.message and "8000" in r.message for r in caplog.records)

    def test_chunk_duration__default_is_zero(self):
        """chunk_duration defaults to 0 (per-packet mode)."""
        assert RealtimeTransportProtocol.chunk_duration == 0

    def test_datagram_received__chunk_duration__buffers_until_threshold(self):
        """Buffer packets and emit audio_received only when the threshold is reached."""
        received: list[list[bytes]] = []
        media = MediaDescription(
            media="audio",
            port=0,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ],
        )

        class ChunkedRTP(RealtimeTransportProtocol):
            chunk_duration = 1  # 1 s @ 8 kHz / 160 samples = 50 packets

            def audio_received(self, packets: list[bytes]) -> None:
                received.append(packets)

        proto = ChunkedRTP(media=media)
        assert proto._packet_threshold == 50
        for _ in range(49):
            proto.datagram_received(make_rtp_packet(), ("127.0.0.1", 5004))
        assert received == []
        proto.datagram_received(make_rtp_packet(payload=b"last"), ("127.0.0.1", 5004))
        assert len(received) == 1
        assert len(received[0]) == 50

    def test_datagram_received__stun_packet__not_forwarded_to_audio_received(self):
        """A STUN packet (first byte < 4) must not reach audio_received."""
        received: list[list[bytes]] = []

        class ConcreteRTP(RealtimeTransportProtocol):
            def audio_received(self, packets: list[bytes]) -> None:
                received.append(packets)

        stun_bytes = b"\x01\x01" + b"\x00" * 18  # first byte = 1 (STUN range [0,3])
        ConcreteRTP().datagram_received(stun_bytes, ("127.0.0.1", 5004))
        assert received == []

    async def test_stun_discover__uses_actual_socket(self):
        """stun_discover() sends a STUN Binding Request through the RTP socket."""
        from voip.stun import MAGIC_COOKIE, STUNMessageType

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
                attr = struct.pack(">HH", 0x0020, 8) + struct.pack(">BBH I", 0, 1, xor_port, xor_ip)
                response = (
                    struct.pack(">HHI12s", STUNMessageType.BINDING_SUCCESS_RESPONSE, len(attr), MAGIC_COOKIE, tid)
                    + attr
                )
                server_transport.sendto(response, addr)

        loop = asyncio.get_running_loop()
        server_t, _ = await loop.create_datagram_endpoint(
            _StubSTUNServer, local_addr=("127.0.0.1", 0)
        )
        server_addr = server_t.get_extra_info("sockname")

        proto = RealtimeTransportProtocol()
        rtp_t, _ = await loop.create_datagram_endpoint(
            lambda: proto, local_addr=("127.0.0.1", 0)
        )
        try:
            result = await proto.stun_discover(server_addr[0], server_addr[1])
            assert result == ("203.0.113.5", 54321)
            assert len(received_requests) == 1
        finally:
            rtp_t.close()
            server_t.close()

    def test_register_call__routes_by_addr(self):
        """Packets from a registered remote addr are forwarded to the registered handler."""
        received_main: list[list[bytes]] = []
        received_call: list[list[bytes]] = []

        class MuxRTP(RealtimeTransportProtocol):
            def audio_received(self, packets):
                received_main.extend(packets)

        class CallRTP(RealtimeTransportProtocol):
            def audio_received(self, packets):
                received_call.extend(packets)

        mux = MuxRTP()
        handler = CallRTP()
        remote_addr = ("1.2.3.4", 5004)
        mux.register_call(remote_addr, handler)

        mux.datagram_received(make_rtp_packet(payload=b"call-audio"), remote_addr)
        assert received_call == [b"call-audio"]
        assert received_main == []

    def test_register_call__unmatched_addr_uses_wildcard_handler(self):
        """Packets from an unknown addr reach the None-key wildcard handler."""
        received: list[bytes] = []

        class WildcardRTP(RealtimeTransportProtocol):
            def audio_received(self, packets):
                received.extend(packets)

        mux = RealtimeTransportProtocol()
        handler = WildcardRTP()
        mux.register_call(None, handler)

        unknown_addr = ("9.9.9.9", 9999)
        mux.datagram_received(make_rtp_packet(payload=b"unmatched"), unknown_addr)
        assert received == [b"unmatched"]

    def test_unregister_call__removes_handler(self):
        """After unregister_call, packets from that addr are no longer routed to handler."""
        received: list[bytes] = []

        class RecordRTP(RealtimeTransportProtocol):
            def audio_received(self, packets):
                received.extend(packets)

        mux = RealtimeTransportProtocol()
        handler = RecordRTP()
        remote_addr = ("5.6.7.8", 5004)
        mux.register_call(remote_addr, handler)
        mux.unregister_call(remote_addr)

        mux.datagram_received(make_rtp_packet(payload=b"gone"), remote_addr)
        assert received == []

    def test_inherits_stun_protocol(self):
        """RealtimeTransportProtocol inherits from STUNProtocol."""
        from voip.stun import STUNProtocol  # noqa: PLC0415

        assert issubclass(RealtimeTransportProtocol, STUNProtocol)


class TestNegotiateCodec:
    def _make_media(
        self, fmts: list[str], rtpmaps: list[str] | None = None
    ) -> MediaDescription:
        """Build a MediaDescription with given format list and optional rtpmap attributes."""
        rtpmap_by_pt: dict[int, RTPPayloadFormat] = {}
        for rtpmap in rtpmaps or []:
            f = RTPPayloadFormat.parse(rtpmap)
            rtpmap_by_pt[f.payload_type] = f
        formats = [
            rtpmap_by_pt.get(int(pt)) or RTPPayloadFormat(payload_type=int(pt))
            for pt in fmts
        ]
        return MediaDescription(media="audio", port=49170, proto="RTP/AVP", fmt=formats)

    def test_negotiate_codec__prefers_opus(self):
        """Select Opus when offered alongside lower-priority codecs."""
        media = self._make_media(
            ["0", "8", "111"],
            ["111 opus/48000/2", "8 PCMA/8000"],
        )
        result = RealtimeTransportProtocol.negotiate_codec(media)
        assert result.fmt[0].payload_type == 111
        assert result.fmt[0].sample_rate == 48000

    def test_negotiate_codec__falls_back_to_pcma(self):
        """Select PCMA when Opus and G.722 are not offered."""
        media = self._make_media(["0", "8"])
        result = RealtimeTransportProtocol.negotiate_codec(media)
        assert result.fmt[0].payload_type == 8
        assert result.fmt[0].sample_rate == 8000

    def test_negotiate_codec__falls_back_to_pcmu(self):
        """Select PCMU when only PCMU is offered."""
        media = self._make_media(["0"])
        result = RealtimeTransportProtocol.negotiate_codec(media)
        assert result.fmt[0].payload_type == 0

    def test_negotiate_codec__empty_fmt__raises(self):
        """Raise NotImplementedError when the remote side offers no audio formats."""
        media = self._make_media([])
        with pytest.raises(NotImplementedError):
            RealtimeTransportProtocol.negotiate_codec(media)

    def test_negotiate_codec__unknown_codec__raises(self):
        """Raise NotImplementedError when no offered codec matches PREFERRED_CODECS."""
        media = self._make_media(["126"], ["126 telephone-event/8000"])
        with pytest.raises(NotImplementedError):
            RealtimeTransportProtocol.negotiate_codec(media)

    def test_negotiate_codec__returns_media_description(self):
        """negotiate_codec returns a MediaDescription object."""
        media = self._make_media(["0", "8", "111"], ["111 opus/48000/2"])
        result = RealtimeTransportProtocol.negotiate_codec(media)
        assert isinstance(result, MediaDescription)
        assert result.media == "audio"
        assert result.proto == "RTP/AVP"

    def test_negotiate_codec__includes_rtpmap_attribute(self):
        """The returned MediaDescription has codec info in its fmt entries."""
        media = self._make_media(["111"], ["111 opus/48000/2"])
        result = RealtimeTransportProtocol.negotiate_codec(media)
        f = result.get_format(111)
        assert f is not None
        assert f.encoding_name.lower() == "opus"
        assert f.sample_rate == 48000

    def test_negotiate_codec__subclass_can_override_preferences(self):
        """A subclass with a different PREFERRED_CODECS list uses its own preferences."""

        class PCMAOnlyCall(RealtimeTransportProtocol):
            PREFERRED_CODECS = [
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ]

        media = self._make_media(["0", "8", "111"])
        result = PCMAOnlyCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 8
