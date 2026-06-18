"""Tests for the Opus codec (voip.codecs.opus)."""

from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from voip.codecs.opus import Opus, OpusDecoder  # noqa: E402


class TestOggCRC32:
    def test_ogg_crc32__empty_bytes(self):
        """_ogg_crc32 of empty bytes is zero."""
        assert Opus._ogg_crc32(b"") == 0

    def test_ogg_crc32__known_value(self):
        """_ogg_crc32 produces a deterministic 32-bit value."""
        crc = Opus._ogg_crc32(b"OggS")
        assert 0 <= crc <= 0xFFFFFFFF


class TestOggPage:
    def test_ogg_page__starts_with_capture_pattern(self):
        """_ogg_page output starts with the Ogg capture pattern 'OggS'."""
        page = Opus._ogg_page(0x02, 0, 0x12345678, 0, [b"hello"])
        assert page[:4] == b"OggS"

    def test_ogg_page__contains_packet_data(self):
        """_ogg_page embeds the provided packet bytes."""
        page = Opus._ogg_page(0x02, 0, 0, 0, [b"payload"])
        assert b"payload" in page

    def test_ogg_page__large_packet_uses_255_lacing(self):
        """_ogg_page correctly laces a packet exceeding 254 bytes."""
        page = Opus._ogg_page(0x00, 0, 0, 0, [b"x" * 256])
        assert page[:4] == b"OggS"
        assert len(page) > 256


class TestOggContainer:
    def test_ogg_container__starts_with_ogg_magic(self):
        """_ogg_container output starts with the Ogg capture pattern 'OggS'."""
        assert Opus._ogg_container(b"packet").startswith(b"OggS")

    def test_ogg_container__contains_opus_head(self):
        """_ogg_container includes the OpusHead identification header."""
        assert b"OpusHead" in Opus._ogg_container(b"packet")

    def test_ogg_container__contains_opus_tags(self):
        """_ogg_container includes the OpusTags comment header."""
        assert b"OpusTags" in Opus._ogg_container(b"packet")

    def test_ogg_container__non_empty_for_single_packet(self):
        """_ogg_container produces a non-empty Ogg container for a single Opus packet."""
        assert len(Opus._ogg_container(b"x" * 100)) > 100

    def test_ogg_container__empty_payload(self):
        """_ogg_container produces a valid Ogg container even for empty payload."""
        result = Opus._ogg_container(b"")
        assert b"OggS" in result

    def test_ogg_container__produces_three_pages(self):
        """_ogg_container produces exactly three Ogg pages: BOS, tags, and data."""
        result = Opus._ogg_container(b"x" * 10)
        assert result.count(b"OggS") == 3


class TestOpusDecode:
    def test_decode__wraps_in_ogg_format(self):
        """Decode passes the payload through _ogg_container before calling decode_pcm."""
        with patch.object(
            Opus, "decode_pcm", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode_pcm:
            Opus.decode(b"payload", 16000)
        args = mock_decode_pcm.call_args[0]
        assert args[1] == "ogg"

    def test_decode__ignores_input_rate_hz(self):
        """Decode ignores the input_rate_hz argument (Ogg container defines the rate)."""
        with patch.object(
            Opus, "decode_pcm", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode_pcm:
            Opus.decode(b"payload", 16000, input_rate_hz=8000)
        args = mock_decode_pcm.call_args[0]
        assert args[1] == "ogg"

    def test_decode__real_decode_returns_float32(self):
        """Decode produces a float32 array for a real Opus packet."""
        sample = Opus.encode(np.zeros(960, dtype=np.float32))
        result = Opus.decode(sample, 16000)
        assert result.dtype == np.float32

    def test_decode__real_decode_not_empty(self):
        """Decode produces non-empty audio for a non-empty Opus packet.

        Regression test: a too-large OpusHead pre-skip combined with a zero
        granule position previously discarded all decoded samples, yielding
        an empty array and silent calls.
        """
        rng = np.random.default_rng(0)
        sample = Opus.encode(rng.uniform(-0.3, 0.3, 960).astype(np.float32))
        result = Opus.decode(sample, 16000)
        assert result.size > 0


class TestOpusEncode:
    def test_encode__returns_bytes(self):
        """Encode produces non-empty bytes for silent PCM input."""
        result = Opus.encode(np.zeros(960, dtype=np.float32))
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode__produces_single_opus_frame(self):
        """Encode produces exactly one Code-0 Opus frame per 960-sample chunk.

        Regression test: the previous implementation concatenated two raw Opus
        frames (one from `codec.encode(frame)` and one from the flush
        `codec.encode(None)`) into a single RTP payload.  A remote decoder
        receiving such a payload sees Code-0 (single frame) in the TOC byte
        and tries to decode the entire concatenated blob as one frame, which is
        malformed — causing silence on outbound Opus echo calls.
        """
        rng = np.random.default_rng(0)
        result = Opus.encode(rng.uniform(-0.3, 0.3, 960).astype(np.float32))
        # Code-0 single-frame payload: TOC byte only, rest is frame data.
        # Verify size is consistent with a single 20 ms Opus frame (not double).
        assert (result[0] & 0x03) == 0  # TOC code bits: 0 = single frame
        # A correctly encoded single 20 ms Opus frame is well under 1200 bytes.
        # Two concatenated frames from the old code would be ~500+ bytes for noise.
        # Silence is highly compressed; noise at 0.3 amplitude is a better bound.
        assert len(result) < 1200


class TestOpusPacketize:
    def test_packetize__yields_single_frame_packets(self):
        """Packetize yields only Code-0 (single-frame) Opus packets."""
        rng = np.random.default_rng(0)
        audio = rng.uniform(-0.3, 0.3, 48000).astype(np.float32)
        for pkt in Opus.packetize(audio):
            assert (pkt[0] & 0x03) == 0

    def test_packetize__frame_count(self):
        """Packetize yields exactly one packet per 20 ms frame, no flush packet.

        Regression test: the previous implementation appended a flush packet
        (`codec.encode(None)`) after all frames, producing N+1 RTP packets
        for N frames of audio.  `_dispatch_next_packet` sends every yielded
        payload at a fixed 20 ms interval, so the extra packet shifted the
        receiver's playback timeline by one ptime (20 ms), causing audible
        timing glitches.
        """
        # 5 full frames of 960 samples each → exactly 5 packets, no flush
        audio = np.zeros(4800, dtype=np.float32)
        assert len(list(Opus.packetize(audio))) == 5

    def test_packetize__pads_partial_final_frame(self):
        """Packetize zero-pads a partial last frame to a full 960-sample frame."""
        # 5 full frames + 100 extra samples → 6 frames (5 full + 1 padded), no flush
        audio = np.zeros(4900, dtype=np.float32)
        packets = list(Opus.packetize(audio))
        assert len(packets) == 6  # 6 frames (5 full + 1 padded), no flush
        for pkt in packets:
            assert (pkt[0] & 0x03) == 0


class TestOpusConstants:
    def test_channels(self):
        """Opus channel count is 1 (mono), matching the encoder and decoder."""
        assert Opus.channels == 1

    def test_payload_type(self):
        """Opus payload type is 111 per RFC 7587."""
        assert Opus.payload_type == 111

    def test_encoding_name(self):
        """Opus encoding name is 'opus' (lowercase)."""
        assert Opus.encoding_name == "opus"

    def test_sample_rate_hz(self):
        """Opus sample rate is 48 000 Hz."""
        assert Opus.sample_rate_hz == 48000

    def test_rtp_clock_rate_hz(self):
        """Opus RTP clock rate is 48 000 Hz."""
        assert Opus.rtp_clock_rate_hz == 48000

    def test_frame_size(self):
        """Opus frame size is 960 samples at 48 kHz (20 ms)."""
        assert Opus.frame_size == 960

    def test_timestamp_increment(self):
        """Opus timestamp increment is 960 ticks per frame."""
        assert Opus.timestamp_increment == 960


class TestOpusCreateDecoder:
    def test_create_decoder__returns_opus_decoder(self):
        """create_decoder returns an OpusDecoder instance."""
        decoder = Opus.create_decoder(16000)
        assert isinstance(decoder, OpusDecoder)

    def test_create_decoder__ignores_input_rate_hz(self):
        """create_decoder ignores input_rate_hz for API consistency."""
        decoder = Opus.create_decoder(16000, input_rate_hz=8000)
        assert isinstance(decoder, OpusDecoder)
        assert decoder.output_rate_hz == 16000


class TestOpusDecoderDecode:
    def test_decode__returns_float32(self):
        """OpusDecoder.decode produces a float32 array."""
        decoder = Opus.create_decoder(16000)
        payload = Opus.encode(np.zeros(960, dtype=np.float32))
        result = decoder.decode(payload)
        assert result.dtype == np.float32

    def test_decode__non_empty_for_real_packet(self):
        """OpusDecoder.decode produces non-empty audio for a real Opus packet."""
        rng = np.random.default_rng(0)
        decoder = Opus.create_decoder(16000)
        payload = Opus.encode(rng.uniform(-0.3, 0.3, 960).astype(np.float32))
        result = decoder.decode(payload)
        assert result.size > 0

    def test_decode__preserves_state_across_packets(self):
        """OpusDecoder.decode produces consistent per-packet output for sequential packets.

        Regression test: the previous per-packet Ogg-container decode reset the
        `libopus` CELT MDCT overlap window every 20 ms, producing 50 Hz
        window-boundary discontinuities heard as choppiness on echo calls.
        A persistent decoder context preserves overlap state, so packets after
        the first warm-up packet each produce exactly `frame_size / 3` samples
        at the 16 kHz output rate.
        """
        rng = np.random.default_rng(42)
        decoder = Opus.create_decoder(16000)
        counts = []
        for _ in range(10):
            payload = Opus.encode(rng.uniform(-0.3, 0.3, 960).astype(np.float32))
            result = decoder.decode(payload)
            counts.append(len(result))
        # After the first warm-up packet all packets must produce 320 samples.
        assert all(c == 320 for c in counts[1:]), f"Inconsistent counts: {counts}"

    def test_decode__empty_payload_returns_empty(self):
        """OpusDecoder.decode returns an empty array for an empty payload."""
        decoder = Opus.create_decoder(16000)
        result = decoder.decode(b"")
        assert result.size == 0
