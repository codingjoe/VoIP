"""Tests for the PCMA and PCMU codecs (voip.codecs.pcma, voip.codecs.pcmu)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from voip.codecs.pcma import PCMA  # noqa: E402
from voip.codecs.pcmu import PCMU  # noqa: E402


class TestPCMAConstants:
    def test_payload_type(self):
        """PCMA payload type is 8 per RFC 3551."""
        assert PCMA.payload_type == 8

    def test_encoding_name(self):
        """PCMA encoding name is pcma."""
        assert PCMA.encoding_name == "pcma"

    def test_sample_rate_hz(self):
        """PCMA sample rate is 8 000 Hz."""
        assert PCMA.sample_rate_hz == 8000

    def test_rtp_clock_rate_hz(self):
        """PCMA RTP clock rate equals sample rate (8 000 Hz)."""
        assert PCMA.rtp_clock_rate_hz == 8000

    def test_frame_size(self):
        """PCMA frame size is 160 samples per 20 ms."""
        assert PCMA.frame_size == 160

    def test_timestamp_increment(self):
        """PCMA timestamp increment is 160 per frame."""
        assert PCMA.timestamp_increment == 160


class TestPCMADecode:
    def test_decode__uses_alaw_format(self):
        """Decode calls decode_pcm with the alaw PyAV format string."""
        with patch.object(
            PCMA, "decode_pcm", return_value=np.zeros(8000, dtype=np.float32)
        ) as mock:
            PCMA.decode(b"payload", 8000)
        assert mock.call_args[0][1] == "alaw"

    def test_decode__passes_sample_rate_as_default(self):
        """Decode passes sample_rate_hz as input_rate_hz when not overridden."""
        with patch.object(
            PCMA, "decode_pcm", return_value=np.zeros(8000, dtype=np.float32)
        ) as mock:
            PCMA.decode(b"payload", 8000)
        assert mock.call_args[1]["input_rate_hz"] == PCMA.sample_rate_hz

    def test_decode__uses_input_rate_hz_override(self):
        """Decode passes the caller-supplied input_rate_hz when provided."""
        with patch.object(
            PCMA, "decode_pcm", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock:
            PCMA.decode(b"payload", 16000, input_rate_hz=16000)
        assert mock.call_args[1]["input_rate_hz"] == 16000

    def test_decode__real_decode_returns_float32(self):
        """Decode produces a float32 array from real A-law encoded input."""
        payload = PCMA.encode(np.zeros(160, dtype=np.float32))
        result = PCMA.decode(payload, 8000)
        assert result.dtype == np.float32


class TestPCMAEncode:
    def test_encode__returns_one_byte_per_sample(self):
        """Encode returns a bytes object with one byte per input sample."""
        assert len(PCMA.encode(np.zeros(160, dtype=np.float32))) == 160

    def test_encode__silence_encodes_consistently(self):
        """Zero-amplitude samples all encode to the same A-law codeword."""
        result = PCMA.encode(np.zeros(10, dtype=np.float32))
        assert all(b == result[0] for b in result)

    def test_encode__positive_and_negative_differ(self):
        """Positive and negative amplitudes produce different A-law codewords."""
        pos = PCMA.encode(np.array([0.5], dtype=np.float32))
        neg = PCMA.encode(np.array([-0.5], dtype=np.float32))
        assert pos != neg


class TestPCMUConstants:
    def test_payload_type(self):
        """PCMU payload type is 0 per RFC 3551."""
        assert PCMU.payload_type == 0

    def test_encoding_name(self):
        """PCMU encoding name is pcmu."""
        assert PCMU.encoding_name == "pcmu"

    def test_sample_rate_hz(self):
        """PCMU sample rate is 8 000 Hz."""
        assert PCMU.sample_rate_hz == 8000

    def test_rtp_clock_rate_hz(self):
        """PCMU RTP clock rate equals sample rate (8 000 Hz)."""
        assert PCMU.rtp_clock_rate_hz == 8000

    def test_frame_size(self):
        """PCMU frame size is 160 samples per 20 ms."""
        assert PCMU.frame_size == 160

    def test_timestamp_increment(self):
        """PCMU timestamp increment is 160 per frame."""
        assert PCMU.timestamp_increment == 160


class TestPCMUDecode:
    def test_decode__uses_mulaw_format(self):
        """Decode calls decode_pcm with the mulaw PyAV format string."""
        with patch.object(
            PCMU, "decode_pcm", return_value=np.zeros(8000, dtype=np.float32)
        ) as mock:
            PCMU.decode(b"payload", 8000)
        assert mock.call_args[0][1] == "mulaw"

    def test_decode__passes_sample_rate_as_default(self):
        """Decode passes sample_rate_hz as input_rate_hz when not overridden."""
        with patch.object(
            PCMU, "decode_pcm", return_value=np.zeros(8000, dtype=np.float32)
        ) as mock:
            PCMU.decode(b"payload", 8000)
        assert mock.call_args[1]["input_rate_hz"] == PCMU.sample_rate_hz

    def test_decode__uses_input_rate_hz_override(self):
        """Decode passes the caller-supplied input_rate_hz when provided."""
        with patch.object(
            PCMU, "decode_pcm", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock:
            PCMU.decode(b"payload", 16000, input_rate_hz=16000)
        assert mock.call_args[1]["input_rate_hz"] == 16000

    def test_decode__real_decode_returns_float32(self):
        """Decode produces a float32 array from real mu-law encoded input."""
        payload = PCMU.encode(np.zeros(160, dtype=np.float32))
        result = PCMU.decode(payload, 8000)
        assert result.dtype == np.float32


class TestPCMUEncode:
    def test_encode__returns_one_byte_per_sample(self):
        """Encode returns a bytes object with one byte per input sample."""
        assert len(PCMU.encode(np.zeros(160, dtype=np.float32))) == 160

    def test_encode__silence_is_0x7f(self):
        """Silence (0.0) must encode to 0x7F (127) per ITU-T G.711."""
        assert PCMU.encode(np.zeros(1, dtype=np.float32))[0] == 0x7F

    def test_encode__max_positive_is_0x00(self):
        """Maximum positive amplitude must encode to 0x00 per ITU-T G.711."""
        assert PCMU.encode(np.array([1.0], dtype=np.float32))[0] == 0x00

    def test_encode__max_negative_is_0x80(self):
        """Maximum negative amplitude must encode to 0x80 per ITU-T G.711."""
        assert PCMU.encode(np.array([-1.0], dtype=np.float32))[0] == 0x80
