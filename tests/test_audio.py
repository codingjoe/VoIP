"""Tests for audio call handler and codec utilities."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from voip.audio import AudioCall, JitterBuffer, _build_ogg_opus  # noqa: E402
from voip.rtp import RealtimeTransportProtocol, RTPPacket  # noqa: E402
from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: E402


def _make_media(fmt: str, rtpmap: str | None = None) -> MediaDescription:
    """Build a single-codec MediaDescription for use in tests."""
    if rtpmap:
        payload_format = RTPPayloadFormat.parse(rtpmap)
    else:
        payload_format = RTPPayloadFormat(payload_type=int(fmt))
    return MediaDescription(
        media="audio", port=0, proto="RTP/AVP", fmt=[payload_format]
    )


OPUS_MEDIA = _make_media("111", "111 opus/48000/2")
PCMA_MEDIA = _make_media("8", "8 PCMA/8000")
PCMU_MEDIA = _make_media("0")  # static PT, no rtpmap
G722_MEDIA = _make_media("9", "9 G722/8000")


def _make_rtp_packet(sequence_number: int, payload: bytes = b"x") -> RTPPacket:
    """Build an RTPPacket with the given sequence number for use in jitter buffer tests."""
    return RTPPacket(
        payload_type=8,
        sequence_number=sequence_number,
        timestamp=sequence_number * 160,
        ssrc=0,
        payload=payload,
    )


class TestJitterBuffer:
    def test_get__before_put__returns_none(self):
        """Return None when get is called before any packet has been put."""
        assert JitterBuffer().get() is None

    def test_empty__true_initially(self):
        """A freshly created buffer reports itself as empty."""
        assert JitterBuffer().empty is True

    def test_empty__false_after_put(self):
        """Empty is False after at least one packet is stored."""
        buf = JitterBuffer()
        buf.put(_make_rtp_packet(1))
        assert buf.empty is False

    def test_ready__false_below_depth(self):
        """Ready is False when fewer than depth packets have been put."""
        buf = JitterBuffer(depth=2)
        buf.put(_make_rtp_packet(1))
        assert buf.ready is False

    def test_ready__true_at_depth(self):
        """Ready is True once at least depth packets have been buffered."""
        buf = JitterBuffer(depth=2)
        buf.put(_make_rtp_packet(1))
        buf.put(_make_rtp_packet(2))
        assert buf.ready is True

    def test_get__returns_in_order(self):
        """Get returns packets in ascending sequence order regardless of insertion order."""
        buf = JitterBuffer(depth=1)
        buf.put(_make_rtp_packet(2))
        buf.put(_make_rtp_packet(1))
        first = buf.get()
        second = buf.get()
        assert first is not None and first.sequence_number == 1
        assert second is not None and second.sequence_number == 2

    def test_get__returns_none_for_missing_packet(self):
        """Get returns None for a missing sequence slot and still advances the counter."""
        buf = JitterBuffer(depth=1)
        buf.put(_make_rtp_packet(1))
        buf.put(_make_rtp_packet(3))  # seq 2 is absent
        assert buf.get().sequence_number == 1  # type: ignore[union-attr]
        assert buf.get() is None  # seq 2 missing
        assert buf.get().sequence_number == 3  # type: ignore[union-attr]

    def test_get__advances_past_empty_buffer(self):
        """Get advances the sequence counter even when the buffer becomes empty mid-stream."""
        buf = JitterBuffer(depth=1)
        buf.put(_make_rtp_packet(5))
        assert buf.get().sequence_number == 5  # type: ignore[union-attr]
        assert buf.empty
        buf.put(_make_rtp_packet(7))
        # next_seq is now 6, so pkt 7 is not at the head
        assert buf.get() is None  # seq 6 missing
        assert buf.get().sequence_number == 7  # type: ignore[union-attr]

    def test_get__handles_sequence_number_wraparound(self):
        """Get correctly wraps the 16-bit sequence counter from 65535 back to 0."""
        buf = JitterBuffer(depth=1)
        buf.put(_make_rtp_packet(65535))
        buf.put(_make_rtp_packet(0))
        assert buf.get().sequence_number == 65535  # type: ignore[union-attr]
        assert buf.get().sequence_number == 0  # type: ignore[union-attr]

    def test_pause__resets_sequence_anchor(self):
        """Pause clears the sequence anchor so the next burst re-anchors to its minimum."""
        buf = JitterBuffer(depth=1)
        buf.put(_make_rtp_packet(10))
        assert buf.get().sequence_number == 10  # type: ignore[union-attr]
        buf.pause()
        buf.put(_make_rtp_packet(20))
        # Without pause, _next_seq would be 11 and seq 20 would take 9 skips.
        assert buf.get().sequence_number == 20  # type: ignore[union-attr]

    def test_build_ogg_opus__starts_with_ogg_magic(self):
        """The resulting container starts with the Ogg capture pattern 'OggS'."""
        assert _build_ogg_opus(b"packet").startswith(b"OggS")

    def test_build_ogg_opus__contains_opus_head(self):
        """The resulting container includes the OpusHead identification header."""
        assert b"OpusHead" in _build_ogg_opus(b"packet")

    def test_build_ogg_opus__contains_opus_tags(self):
        """The resulting container includes the OpusTags comment header."""
        assert b"OpusTags" in _build_ogg_opus(b"packet")

    def test_build_ogg_opus__non_empty_for_single_packet(self):
        """Produce a non-empty Ogg container for a single Opus packet."""
        assert len(_build_ogg_opus(b"x" * 100)) > 100

    def test_build_ogg_opus__empty_payload(self):
        """Produce a valid Ogg container even when the payload is empty bytes."""
        result = _build_ogg_opus(b"")
        assert b"OggS" in result

    def test_build_ogg_opus__large_packet_uses_255_lacing(self):
        """Produce a valid Ogg container when a packet exceeds 254 bytes (lacing spans 255)."""
        result = _build_ogg_opus(b"x" * 256)
        assert b"OggS" in result
        assert len(result) > 256

    def test_build_ogg_opus__produces_three_pages(self):
        """Produce exactly three Ogg pages: BOS, tags, and data."""
        result = _build_ogg_opus(b"x" * 10)
        assert result.count(b"OggS") == 3


def make_audio_call(**kwargs) -> AudioCall:
    """Create an AudioCall with mock rtp/sip for unit testing."""
    defaults: dict = {
        "rtp": MagicMock(spec=RealtimeTransportProtocol),
        "sip": MagicMock(),
        "media": PCMA_MEDIA,
    }
    defaults.update(kwargs)
    return AudioCall(**defaults)


class TestAudioCall:
    def test_caller__returns_caller_arg(self):
        """Return the caller string passed at construction."""
        call = make_audio_call(caller="sip:bob@biloxi.com")
        assert call.caller == "sip:bob@biloxi.com"

    def test_caller__defaults_to_empty_string(self):
        """Return an empty string when no caller is given."""
        assert make_audio_call().caller == ""

    def test_audio_received__noop_by_default(self):
        """audio_received is a no-op in the base AudioCall class."""
        make_audio_call().audio_received(audio=np.array([]), rms=0.0)  # must not raise

    def test_rtp_and_sip_stored_as_fields(self):
        """Rtp and sip back-references are stored as dataclass fields."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_sip = MagicMock()
        call = AudioCall(rtp=mock_rtp, sip=mock_sip, media=PCMA_MEDIA)
        assert call.rtp is mock_rtp
        assert call.sip is mock_sip

    def test_init__stores_media(self):
        """Media parameter is stored on the AudioCall instance."""
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ],
        )
        call = make_audio_call(media=media)
        assert call.media is media

    def test_init__derives_sample_rate_from_media(self):
        """sample_rate is derived from the RTPPayloadFormat sample_rate."""
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=9, encoding_name="G722", sample_rate=8000)
            ],
        )
        call = make_audio_call(media=media)
        assert call.sample_rate == 8000

    def test_init__default_sample_rate_without_media(self):
        """Default sample_rate is 8000 Hz for G.711 codecs."""
        assert make_audio_call().sample_rate == 8000

    def test_init__derives_payload_type_from_media(self):
        """payload_type is derived from the first fmt entry of the MediaDescription."""
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=8)],
        )
        call = make_audio_call(media=media)
        assert call.payload_type == 8

    def test_init__default_payload_type_without_media(self):
        """Default payload_type is 8 (PCMA) when using the default test media."""
        assert make_audio_call().payload_type == 8

    def test_init__logs_codec_info(self, caplog):
        """Log codec name, sample rate and payload type at INFO level on init."""
        import logging  # noqa: PLC0415

        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

        media = MediaDescription(
            media="audio",
            port=49170,
            proto="RTP/AVP",
            fmt=[
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ],
        )
        with caplog.at_level(logging.INFO, logger="voip.audio"):
            make_audio_call(media=media)
        assert any("PCMA" in r.message and "8000" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_datagram_received__malformed_rtp__noop(self):
        """datagram_received silently ignores packets that fail RTP parsing."""
        call = make_audio_call()
        call.datagram_received(b"\x80" * 5, ("127.0.0.1", 5004))  # too short
        assert call._jitter_buffer.empty

    @pytest.mark.asyncio
    async def test_datagram_received__empty_payload__noop(self):
        """datagram_received ignores valid RTP packets with no payload."""
        import struct  # noqa: PLC0415

        empty_rtp = struct.pack(">BBHII", 0x80, 8 & 0x7F, 1, 0, 0)  # no payload bytes
        call = make_audio_call()
        call.datagram_received(empty_rtp, ("127.0.0.1", 5004))
        assert call._jitter_buffer.empty

    @pytest.mark.asyncio
    async def test_datagram_received__forwards_audio_payload(self):
        """datagram_received buffers packets and calls audio_received after playout delay."""
        import struct  # noqa: PLC0415

        received: list = []

        class ConcreteCall(AudioCall):
            jitter_buffer_depth = 1  # one-packet playout delay for test speed

            def _decode_raw(self, packet: bytes) -> np.ndarray:
                return np.array([1.0], dtype=np.float32)

            def audio_received(self, *, audio: np.ndarray, rms: float) -> None:
                received.append(audio)

        rtp_packet = struct.pack(">BBHII", 0x80, 8 & 0x7F, 1, 0, 0) + b"audio"
        call = ConcreteCall(rtp=MagicMock(), sip=MagicMock(), media=PCMA_MEDIA)
        call.datagram_received(rtp_packet, ("127.0.0.1", 5004))
        await asyncio.sleep(0.15)  # ≥ (depth + 1) × 20 ms playout ticks
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_datagram_received__jitter_buffer__withholds_until_depth(self):
        """audio_received is not called until the jitter buffer reaches its playout depth."""
        import struct  # noqa: PLC0415

        received: list = []

        class ConcreteCall(AudioCall):
            jitter_buffer_depth = 2

            def _decode_raw(self, packet: bytes) -> np.ndarray:
                return np.array([1.0], dtype=np.float32)

            def audio_received(self, *, audio: np.ndarray, rms: float) -> None:
                received.append(audio)

        call = ConcreteCall(rtp=MagicMock(), sip=MagicMock(), media=PCMA_MEDIA)
        # Send only one packet — buffer is below depth, nothing emitted yet.
        call.datagram_received(
            struct.pack(">BBHII", 0x80, 8 & 0x7F, 1, 0, 0) + b"a",
            ("127.0.0.1", 5004),
        )
        await asyncio.sleep(0.01)  # well below one playout tick
        assert received == []
        # Second packet fills the buffer; playout starts and both are emitted.
        call.datagram_received(
            struct.pack(">BBHII", 0x80, 8 & 0x7F, 2, 0, 0) + b"b",
            ("127.0.0.1", 5004),
        )
        await asyncio.sleep(0.15)  # allow playout loop to drain both packets
        assert len(received) == 2


class TestNegotiateCodec:
    def _make_media(self, fmts: list[str], rtpmaps: list[str] | None = None):
        """Build a MediaDescription with given format list and optional rtpmap attributes."""
        from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415

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
        media = self._make_media(["0", "8", "111"], ["111 opus/48000/2", "8 PCMA/8000"])
        result = AudioCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 111
        assert result.fmt[0].sample_rate == 48000

    def test_negotiate_codec__falls_back_to_pcma(self):
        """Select PCMA when Opus and G.722 are not offered."""
        media = self._make_media(["0", "8"])
        result = AudioCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 8
        assert result.fmt[0].sample_rate == 8000

    def test_negotiate_codec__falls_back_to_pcmu(self):
        """Select PCMU when only PCMU is offered."""
        media = self._make_media(["0"])
        result = AudioCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 0

    def test_negotiate_codec__empty_fmt__raises(self):
        """Raise NotImplementedError when the remote side offers no audio formats."""
        media = self._make_media([])
        with pytest.raises(NotImplementedError):
            AudioCall.negotiate_codec(media)

    def test_negotiate_codec__unknown_codec__raises(self):
        """Raise NotImplementedError when no offered codec matches PREFERRED_CODECS."""
        media = self._make_media(["126"], ["126 telephone-event/8000"])
        with pytest.raises(NotImplementedError):
            AudioCall.negotiate_codec(media)

    def test_negotiate_codec__returns_media_description(self):
        """negotiate_codec returns a MediaDescription object."""
        from voip.sdp.types import MediaDescription  # noqa: PLC0415

        media = self._make_media(["0", "8", "111"], ["111 opus/48000/2"])
        result = AudioCall.negotiate_codec(media)
        assert isinstance(result, MediaDescription)
        assert result.media == "audio"
        assert result.proto == "RTP/AVP"

    def test_negotiate_codec__subclass_can_override_preferences(self):
        """A subclass with a different PREFERRED_CODECS list uses its own preferences."""
        from voip.sdp.types import RTPPayloadFormat  # noqa: PLC0415

        class PCMAOnlyCall(AudioCall):
            PREFERRED_CODECS = [
                RTPPayloadFormat(payload_type=8, encoding_name="PCMA", sample_rate=8000)
            ]

        media = self._make_media(["0", "8", "111"])
        result = PCMAOnlyCall.negotiate_codec(media)
        assert result.fmt[0].payload_type == 8

    def test_preferred_codecs__class_attribute(self):
        """PREFERRED_CODECS is a class attribute on AudioCall with Opus first."""
        from voip.sdp.types import RTPPayloadFormat  # noqa: PLC0415

        codecs = AudioCall.PREFERRED_CODECS
        assert isinstance(codecs, list)
        assert all(isinstance(c, RTPPayloadFormat) for c in codecs)
        pts = [c.payload_type for c in codecs]
        assert pts[0] == 111  # Opus is highest priority
        assert 8 in pts  # PCMA present
        assert 0 in pts  # PCMU present
