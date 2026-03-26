"""Tests for the PCMA and PCMU codecs (voip.codecs.pcma, voip.codecs.pcmu)."""

import pytest

np = pytest.importorskip("numpy")

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
    def test_decode__returns_float32(self):
        """Decode produces float32 samples from A-law encoded input."""
        payload = PCMA.encode(np.zeros(160, dtype=np.float32))
        assert PCMA.decode(payload, 8000).dtype == np.float32

    def test_decode__native_rate_preserves_sample_count(self):
        """Decode at native 8 kHz returns one sample per byte."""
        payload = PCMA.encode(np.zeros(160, dtype=np.float32))
        assert len(PCMA.decode(payload, 8000)) == 160

    def test_decode__resamples_to_output_rate(self):
        """Decode resamples to the requested output rate."""
        payload = PCMA.encode(np.zeros(160, dtype=np.float32))
        assert len(PCMA.decode(payload, 16000)) == 320

    def test_decode__silence_roundtrip(self):
        """Encode then decode silence returns values near zero."""
        payload = PCMA.encode(np.zeros(160, dtype=np.float32))
        result = PCMA.decode(payload, 8000)
        assert np.allclose(result, 0.0, atol=0.02)

    def test_decode__positive_and_negative_differ(self):
        """Positive and negative input decode to values with opposite sign."""
        pos = PCMA.decode(PCMA.encode(np.full(1, 0.5, dtype=np.float32)), 8000)
        neg = PCMA.decode(PCMA.encode(np.full(1, -0.5, dtype=np.float32)), 8000)
        assert pos[0] > 0
        assert neg[0] < 0

    def test_decode__uses_input_rate_hz_as_source_rate(self):
        """input_rate_hz is treated as the source sample rate for resampling."""
        payload = PCMA.encode(np.zeros(160, dtype=np.float32))
        # 160 samples interpreted at 16 kHz, resampled to 8 kHz → 80 output samples.
        result = PCMA.decode(payload, 8000, input_rate_hz=16000)
        assert len(result) == 80

    def test_decode__max_amplitude_codeword(self):
        """A-law codeword 0xAA (max positive) decodes to exactly 0.984375 per ITU-T G.711."""
        decoded = PCMA.decode(bytes([0xAA]), 8000)
        assert abs(decoded[0] - 0.984375) < 1e-6

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

    def test_encode__silence_codeword(self):
        """Silence (0.0) must encode to 0xD5 per ITU-T G.711 A-law."""
        assert PCMA.encode(np.zeros(1, dtype=np.float32))[0] == 0xD5

    def test_encode__max_amplitude_codeword(self):
        """Maximum positive amplitude (1.0) must encode to 0xAA per ITU-T G.711."""
        assert PCMA.encode(np.array([1.0], dtype=np.float32))[0] == 0xAA

    def test_encode_decode__roundtrip_midrange(self):
        """Roundtrip encode→decode stays within one G.711 quantisation step.

        Values in segments 2–5 had ~3× the expected error with the old (wrong)
        mantissa shift; this test guards against regression.
        """
        # Values chosen so all four land in segments 3–5 (magnitudes 1 000–8 191).
        original = np.array([0.05, -0.05, 0.1, -0.1], dtype=np.float32)
        recovered = PCMA.decode(PCMA.encode(original), 8000)
        # Maximum step for segment 5 is 256/32768 ≈ 0.0078; allow ~2 steps of margin.
        assert np.allclose(original, recovered, atol=1 / 64)


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
    def test_decode__returns_float32(self):
        """Decode produces float32 samples from mu-law encoded input."""
        payload = PCMU.encode(np.zeros(160, dtype=np.float32))
        assert PCMU.decode(payload, 8000).dtype == np.float32

    def test_decode__native_rate_preserves_sample_count(self):
        """Decode at native 8 kHz returns one sample per byte."""
        payload = PCMU.encode(np.zeros(160, dtype=np.float32))
        assert len(PCMU.decode(payload, 8000)) == 160

    def test_decode__resamples_to_output_rate(self):
        """Decode resamples to the requested output rate."""
        payload = PCMU.encode(np.zeros(160, dtype=np.float32))
        assert len(PCMU.decode(payload, 16000)) == 320

    def test_decode__max_positive_roundtrip(self):
        """Max positive amplitude (0x00) decodes to a large positive value."""
        decoded = PCMU.decode(bytes([0x00]), 8000)
        assert decoded[0] > 0.9

    def test_decode__max_negative_roundtrip(self):
        """Max negative amplitude (0x80) decodes to a large negative value."""
        decoded = PCMU.decode(bytes([0x80]), 8000)
        assert decoded[0] < -0.9

    def test_decode__uses_input_rate_hz_as_source_rate(self):
        """input_rate_hz is treated as the source sample rate for resampling."""
        payload = PCMU.encode(np.zeros(160, dtype=np.float32))
        # 160 samples interpreted at 16 kHz, resampled to 8 kHz → 80 output samples.
        result = PCMU.decode(payload, 8000, input_rate_hz=16000)
        assert len(result) == 80

    def test_decode__silence_codeword(self):
        """µ-law codeword 0x7F (silence) decodes to exactly 0.0 per ITU-T G.711."""
        decoded = PCMU.decode(bytes([0x7F]), 8000)
        assert decoded[0] == pytest.approx(0.0, abs=1e-7)

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
