"""Tests for the Opus codec (voip.codecs.opus)."""

from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from voip.codecs.opus import Opus  # noqa: E402


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


class TestOpusEncode:
    def test_encode__returns_bytes(self):
        """Encode produces non-empty bytes for silent PCM input."""
        result = Opus.encode(np.zeros(960, dtype=np.float32))
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode__uses_libopus_codec(self):
        """Encode delegates to encode_pcm with libopus codec name."""
        with patch.object(Opus, "encode_pcm", return_value=b"encoded") as mock_enc:
            Opus.encode(np.zeros(960, dtype=np.float32))
        mock_enc.assert_called_once_with(
            pytest.approx(np.zeros(960, dtype=np.float32)),
            "libopus",
            Opus.sample_rate_hz,
        )


class TestOpusConstants:
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
