"""Tests for the PyAVCodec base class (voip.codecs.av)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from voip.codecs.base import RTPCodec  # noqa: E402
from voip.codecs.pcma import PCMA  # noqa: E402
from voip.sdp.types import RTPPayloadFormat  # noqa: E402


class TestDecodePCM:
    def test_decode_pcm__alaw_returns_float32(self):
        """decode_pcm decodes A-law bytes to a float32 numpy array."""
        payload = PCMA.encode(np.zeros(160, dtype=np.float32))
        result = PCMA.decode_pcm(payload, "alaw", 8000, input_rate_hz=8000)
        assert result.dtype == np.float32

    def test_decode_pcm__resampler_flush_yields_frames(self):
        """Include frames flushed from the resampler after the last input frame."""
        pcm_array = np.zeros(16000, dtype=np.float32)
        flush_frame = MagicMock()
        flush_frame.to_ndarray.return_value = pcm_array
        with patch("voip.codecs.base.av") as mock_av:
            mock_resampler = MagicMock()
            mock_resampler.resample.side_effect = [[], [flush_frame]]
            mock_av.audio.resampler.AudioResampler.return_value = mock_resampler
            mock_container = MagicMock()
            mock_container.__enter__ = lambda s: s
            mock_container.__exit__ = MagicMock(return_value=False)
            mock_container.decode.return_value = [MagicMock()]
            mock_av.open.return_value = mock_container
            result = PCMA.decode_pcm(b"fake", "alaw", 8000, input_rate_hz=8000)
        assert result.dtype == np.float32
        assert len(result) == len(pcm_array)

    def test_decode_pcm__empty_result_when_no_frames(self):
        """decode_pcm returns an empty float32 array when no audio frames are decoded."""
        with patch("voip.codecs.base.av") as mock_av:
            mock_resampler = MagicMock()
            mock_resampler.resample.return_value = []
            mock_av.audio.resampler.AudioResampler.return_value = mock_resampler
            mock_container = MagicMock()
            mock_container.__enter__ = lambda s: s
            mock_container.__exit__ = MagicMock(return_value=False)
            mock_container.decode.return_value = []
            mock_av.open.return_value = mock_container
            result = PCMA.decode_pcm(b"fake", "alaw", 8000)
        assert result.dtype == np.float32
        assert len(result) == 0

    def test_decode_pcm__without_input_rate_passes_no_options(self):
        """decode_pcm passes no sample_rate option when input_rate_hz is None."""
        with patch("voip.codecs.base.av") as mock_av:
            mock_resampler = MagicMock()
            mock_resampler.resample.return_value = []
            mock_av.audio.resampler.AudioResampler.return_value = mock_resampler
            mock_container = MagicMock()
            mock_container.__enter__ = lambda s: s
            mock_container.__exit__ = MagicMock(return_value=False)
            mock_container.decode.return_value = []
            mock_av.open.return_value = mock_container
            PCMA.decode_pcm(b"fake", "alaw", 8000, input_rate_hz=None)
        call_kwargs = mock_av.open.call_args[1]
        assert call_kwargs["options"] == {}


class TestEncodePCM:
    def test_encode_pcm__g722_returns_bytes(self):
        """encode_pcm produces non-empty bytes for G.722."""
        result = RTPCodec.encode_pcm(np.zeros(320, dtype=np.float32), "g722", 16000)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_pcm__opus_returns_bytes(self):
        """encode_pcm produces non-empty bytes for Opus (libopus)."""
        result = RTPCodec.encode_pcm(np.zeros(960, dtype=np.float32), "libopus", 48000)
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestToPayloadFormat:
    def test_to_payload_format__returns_rtp_payload_format(self):
        """to_payload_format returns an RTPPayloadFormat instance."""
        result = PCMA.to_payload_format()
        assert isinstance(result, RTPPayloadFormat)
        assert result.payload_type == 8
        assert result.encoding_name == "pcma"
        assert result.sample_rate == 8000

    def test_to_payload_format__uses_rtp_clock_rate_for_sdp(self):
        """to_payload_format uses rtp_clock_rate_hz as the SDP sample_rate."""
        from voip.codecs.g722 import G722  # noqa: PLC0415

        result = G722.to_payload_format()
        assert result.sample_rate == G722.rtp_clock_rate_hz  # 8000, not 16000


class TestPacketize:
    def test_packetize__default_encodes_per_frame(self):
        """Default packetize yields one encoded chunk per frame_size samples."""
        audio = np.zeros(320, dtype=np.float32)
        with patch.object(PCMA, "encode", return_value=b"\xd5" * 160) as mock_enc:
            packets = list(PCMA.packetize(audio))
        assert mock_enc.call_count == 2
        assert len(packets) == 2


class TestAbstractMethods:
    def test_decode__raises_not_implemented(self):
        """PyAVCodec.decode raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            RTPCodec.decode(b"data", 8000)

    def test_encode__raises_not_implemented(self):
        """PyAVCodec.encode raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            RTPCodec.encode(np.zeros(160, dtype=np.float32))
