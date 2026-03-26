"""Tests for the RTPCodec base class (voip.codecs.base)."""

from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")

from voip.codecs.base import RTPCodec  # noqa: E402
from voip.codecs.pcma import PCMA  # noqa: E402
from voip.sdp.types import RTPPayloadFormat  # noqa: E402


class TestResample:
    def test_resample__passthrough_when_rates_equal(self):
        """Return the original array unchanged when source and destination rates are equal."""
        audio = np.zeros(160, dtype=np.float32)
        assert RTPCodec.resample(audio, 8000, 8000) is audio

    def test_resample__upsample_doubles_length(self):
        """Upsampling from 8 kHz to 16 kHz produces twice as many samples."""
        audio = np.zeros(160, dtype=np.float32)
        result = RTPCodec.resample(audio, 8000, 16000)
        assert len(result) == 320

    def test_resample__downsample_halves_length(self):
        """Downsampling from 16 kHz to 8 kHz halves the sample count."""
        audio = np.zeros(320, dtype=np.float32)
        result = RTPCodec.resample(audio, 16000, 8000)
        assert len(result) == 160

    def test_resample__empty_input_returns_empty(self):
        """Resampling an empty array returns an empty float32 array."""
        result = RTPCodec.resample(np.empty(0, dtype=np.float32), 8000, 16000)
        assert result.dtype == np.float32
        assert len(result) == 0

    def test_resample__single_sample_heavy_downsample_returns_at_least_one(self):
        """Resampling a single sample always yields at least one output sample."""
        audio = np.array([0.5], dtype=np.float32)
        result = RTPCodec.resample(audio, 8000, 100)
        assert len(result) >= 1
        assert result.dtype == np.float32


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
        pytest.importorskip("av")
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
        """RTPCodec.decode raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            RTPCodec.decode(b"data", 8000)

    def test_encode__raises_not_implemented(self):
        """RTPCodec.encode raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            RTPCodec.encode(np.zeros(160, dtype=np.float32))


class TestCreateDecoder:
    def test_create_decoder__returns_per_packet_decoder(self):
        """RTPCodec.create_decoder returns a PerPacketDecoder for stateless codecs."""
        from voip.codecs.base import PerPacketDecoder  # noqa: PLC0415

        decoder = PCMA.create_decoder(16000)
        assert isinstance(decoder, PerPacketDecoder)

    def test_create_decoder__stores_codec_and_rates(self):
        """PerPacketDecoder holds the codec class and both rate parameters."""
        from voip.codecs.base import PerPacketDecoder  # noqa: PLC0415

        decoder = PCMA.create_decoder(16000, input_rate_hz=8000)
        assert isinstance(decoder, PerPacketDecoder)
        assert decoder.codec is PCMA
        assert decoder.output_rate_hz == 16000
        assert decoder.input_rate_hz == 8000

    def test_create_decoder__input_rate_hz_defaults_to_none(self):
        """input_rate_hz defaults to None when not specified."""
        from voip.codecs.base import PerPacketDecoder  # noqa: PLC0415

        decoder = PCMA.create_decoder(16000)
        assert isinstance(decoder, PerPacketDecoder)
        assert decoder.input_rate_hz is None

    def test_per_packet_decoder__delegates_to_codec_decode(self):
        """PerPacketDecoder.decode calls codec.decode with stored rates."""
        with patch.object(
            PCMA, "decode", return_value=np.zeros(160, dtype=np.float32)
        ) as mock_decode:
            decoder = PCMA.create_decoder(16000, input_rate_hz=8000)
            result = decoder.decode(b"payload")
        mock_decode.assert_called_once_with(b"payload", 16000, input_rate_hz=8000)
        assert result.dtype == np.float32
