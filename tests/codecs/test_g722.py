"""Tests for the G.722 codec (voip.codecs.g722)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from voip.codecs.g722 import G722  # noqa: E402


class TestG722Constants:
    def test_payload_type(self):
        """G.722 payload type is 9 per RFC 3551."""
        assert G722.payload_type == 9

    def test_encoding_name(self):
        """G.722 encoding name is g722 (lowercase)."""
        assert G722.encoding_name == "g722"

    def test_sample_rate_hz(self):
        """G.722 actual audio sample rate is 16 000 Hz."""
        assert G722.sample_rate_hz == 16000

    def test_rtp_clock_rate_hz(self):
        """G.722 RTP clock rate is 8 000 Hz per RFC 3551 (despite 16 kHz audio)."""
        assert G722.rtp_clock_rate_hz == 8000

    def test_frame_size(self):
        """G.722 frame size is 320 audio samples per 20 ms."""
        assert G722.frame_size == 320

    def test_timestamp_increment(self):
        """G.722 timestamp increment is 160 ticks at the 8 kHz RTP clock."""
        assert G722.timestamp_increment == 160


class TestG722Decode:
    def test_decode__uses_g722_format(self):
        """Decode calls decode_pcm with the g722 PyAV format string."""
        with patch.object(
            G722, "decode_pcm", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode_pcm:
            G722.decode(b"payload", 16000)
        args = mock_decode_pcm.call_args[0]
        assert args[1] == "g722"

    def test_decode__passes_rtp_clock_rate_as_input(self):
        """Decode passes rtp_clock_rate_hz (8 000) as the PyAV input rate hint."""
        with patch.object(
            G722, "decode_pcm", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode_pcm:
            G722.decode(b"payload", 16000)
        kwargs = mock_decode_pcm.call_args[1]
        assert kwargs.get("input_rate_hz") == G722.rtp_clock_rate_hz

    def test_decode__ignores_input_rate_hz_argument(self):
        """Decode ignores the input_rate_hz argument and always uses rtp_clock_rate_hz."""
        with patch.object(
            G722, "decode_pcm", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode_pcm:
            G722.decode(b"payload", 16000, input_rate_hz=16000)
        kwargs = mock_decode_pcm.call_args[1]
        assert kwargs.get("input_rate_hz") == G722.rtp_clock_rate_hz

    def test_decode__real_decode_returns_float32(self):
        """Decode produces a float32 array for real G.722 encoded input."""
        sample = G722.encode(np.zeros(320, dtype=np.float32))
        result = G722.decode(sample, 16000)
        assert result.dtype == np.float32


class TestG722Encode:
    def test_encode__returns_bytes(self):
        """Encode produces non-empty bytes for silent PCM input."""
        result = G722.encode(np.zeros(320, dtype=np.float32))
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode__uses_g722_codec(self):
        """Encode delegates to encode_pcm with the g722 codec name."""
        with patch.object(G722, "encode_pcm", return_value=b"enc") as mock_enc:
            G722.encode(np.zeros(320, dtype=np.float32))
        mock_enc.assert_called_once_with(
            pytest.approx(np.zeros(320, dtype=np.float32)),
            "g722",
            G722.sample_rate_hz,
        )


class TestG722Packetize:
    def test_packetize__g722_encodes_whole_buffer_at_once(self):
        """Packetize calls encode on the full buffer to preserve ADPCM state."""
        audio = np.zeros(640, dtype=np.float32)
        fake_encoded = b"\xab" * 320
        with patch.object(G722, "encode", return_value=fake_encoded) as mock_enc:
            packets = list(G722.packetize(audio))
        mock_enc.assert_called_once_with(audio)
        assert len(packets) == 2
        assert packets[0] == b"\xab" * 160
        assert packets[1] == b"\xab" * 160

    def test_packetize__yields_160_byte_chunks(self):
        """Packetize yields 160-byte chunks (frame_size // 2 for G.722 2:1 ratio)."""
        audio = np.zeros(320, dtype=np.float32)
        payload_size = G722.frame_size // 2
        with patch.object(G722, "encode", return_value=b"\x00" * payload_size):
            packets = list(G722.packetize(audio))
        assert len(packets) == 1
        assert len(packets[0]) == payload_size
