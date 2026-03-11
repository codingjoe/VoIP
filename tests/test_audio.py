"""Tests for Whisper-based audio transcription."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")
pytest.importorskip("whisper")

import whisper  # noqa: E402
from voip.audio import (  # noqa: E402
    WhisperCall,
    _build_ogg_opus,
    _codec_name_from_media,
)
from voip.rtp import RTP, RTPPacket  # noqa: E402
from voip.sdp.types import Attribute, MediaDescription  # noqa: E402


def packet_threshold(call_class: type[WhisperCall]) -> int:
    """Return the packet count threshold for the given WhisperCall class."""
    return (
        call_class.opus_sample_rate
        * call_class.chunk_duration
        // call_class.opus_frame_size
    )


def _make_media(fmt: str, rtpmap: str | None = None) -> MediaDescription:
    """Build a single-codec MediaDescription for use in tests."""
    attributes = []
    if rtpmap:
        attributes.append(Attribute(name="rtpmap", value=rtpmap))
    return MediaDescription(
        media="audio", port=0, proto="RTP/AVP", fmt=[fmt], attributes=attributes
    )


OPUS_MEDIA = _make_media("111", "111 opus/48000/2")
PCMA_MEDIA = _make_media("8", "8 PCMA/8000")
PCMU_MEDIA = _make_media("0")  # static PT, no rtpmap
G722_MEDIA = _make_media("9", "9 G722/8000")


def make_whisper_call(
    model_mock: MagicMock, call_class=None, media: MediaDescription | None = None
) -> WhisperCall:
    """Return a WhisperCall with a mocked Whisper model."""
    cls = call_class or WhisperCall
    med = media if media is not None else OPUS_MEDIA
    with patch("whisper.load_model", return_value=model_mock):
        return cls(caller="sip:bob@biloxi.com", media=med)


def make_rtp_packet(
    payload: bytes = b"audio",
    payload_type: int = 111,
) -> RTPPacket:
    """Return an RTPPacket with the given payload and payload type."""
    return RTPPacket(
        payload_type=payload_type,
        sequence_number=1,
        timestamp=0,
        ssrc=0,
        payload=payload,
    )


class TestCodecNameFromMedia:
    def test_opus__returns_opus(self):
        """Return 'opus' for an Opus rtpmap."""
        assert _codec_name_from_media(OPUS_MEDIA) == "opus"

    def test_pcma__returns_pcma(self):
        """Return 'pcma' for a PCMA rtpmap."""
        assert _codec_name_from_media(PCMA_MEDIA) == "pcma"

    def test_pcmu__static_no_rtpmap(self):
        """Return 'pcmu' for PCMU payload type 0 (no rtpmap needed, RFC 3551)."""
        assert _codec_name_from_media(PCMU_MEDIA) == "pcmu"

    def test_g722__returns_g722(self):
        """Return 'g722' for G722."""
        assert _codec_name_from_media(G722_MEDIA) == "g722"

    def test_none__defaults_to_opus(self):
        """Default to 'opus' when no MediaDescription is given."""
        assert _codec_name_from_media(None) == "opus"

    def test_empty_fmt__defaults_to_opus(self):
        """Default to 'opus' when the format list is empty."""
        media = MediaDescription(media="audio", port=0, proto="RTP/AVP", fmt=[], attributes=[])
        assert _codec_name_from_media(media) == "opus"

    def test_unknown_dynamic_pt__defaults_to_opus(self):
        """Return 'opus' for an unrecognised dynamic payload type."""
        media = _make_media("126", "126 telephone-event/8000")
        assert _codec_name_from_media(media) == "telephone-event"


class TestWhisperCall:
    def test_whisper_call__is_rtp(self):
        """WhisperCall is a subclass of RTP."""
        assert issubclass(WhisperCall, RTP)

    def test_class_attrs__opus_sample_rate(self):
        """opus_sample_rate is 48000 Hz as required by RFC 7587."""
        assert WhisperCall.opus_sample_rate == 48000

    def test_class_attrs__opus_frame_size(self):
        """opus_frame_size is 960 samples (20 ms at 48 kHz)."""
        assert WhisperCall.opus_frame_size == 960

    def test_class_attrs__chunk_duration(self):
        """chunk_duration controls how many seconds are buffered before transcription."""
        assert WhisperCall.chunk_duration == 30

    def test_init__stores_media(self):
        """Media is stored and accessible as self.media."""
        call = make_whisper_call(MagicMock())
        assert call.media is OPUS_MEDIA

    def test_audio_received__buffers_opus_packet(self):
        """Append each RTP payload to the internal packet buffer."""
        call = make_whisper_call(MagicMock())
        call.audio_received(make_rtp_packet(b"opus_packet"))
        assert call._audio_packets == [b"opus_packet"]

    def test_audio_received__buffers_pcm_below_threshold(self):
        """Don't schedule transcription until the packet threshold is reached."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        call.audio_received(make_rtp_packet(b"opus_packet"))
        model_mock.transcribe.assert_not_called()

    async def test_audio_received__triggers_transcription_when_buffer_full(self):
        """Schedule transcription when enough audio packets have been buffered."""
        transcriptions = []
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": "hello"}

        class SmallChunkCall(WhisperCall):
            chunk_duration = 1  # → 48000 * 1 // 960 = 50 packets

            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        call = make_whisper_call(model_mock, SmallChunkCall)
        call._audio_packets = [b"x"] * (packet_threshold(SmallChunkCall) - 1)
        pcm_samples = np.zeros(whisper.audio.SAMPLE_RATE, dtype=np.float32)
        with patch.object(call, "_decode_audio", return_value=pcm_samples):
            call.audio_received(make_rtp_packet(b"opus_packet"))
            await asyncio.sleep(0.1)
        assert transcriptions == ["hello"]

    async def test_audio_received__clears_transcribed_packets_from_buffer(self):
        """Remove the transcribed packets from the buffer after transcription."""
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": ""}

        class SmallChunkCall(WhisperCall):
            chunk_duration = 1

        call = make_whisper_call(model_mock, SmallChunkCall)
        extra = 5
        call._audio_packets = [b"x"] * (packet_threshold(SmallChunkCall) + extra)
        with patch.object(
            call, "_decode_audio", return_value=np.zeros(16000, dtype=np.float32)
        ):
            await call._transcribe_chunk()
        assert len(call._audio_packets) == extra

    def test_run_transcription__passes_numpy_array_directly(self):
        """Pass a numpy float32 array to the Whisper model without file I/O."""
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": "test"}
        call = make_whisper_call(model_mock)
        audio = np.zeros(16000, dtype=np.float32)
        assert call._run_transcription(audio) == "test"
        model_mock.transcribe.assert_called_once_with(audio)

    def test_run_transcription__no_file_written(self):
        """The transcription path must not write any files to disk."""
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": ""}
        call = make_whisper_call(model_mock)
        with patch(
            "builtins.open", side_effect=AssertionError("open() must not be called")
        ):
            call._run_transcription(np.zeros(16000, dtype=np.float32))

    async def test_transcription_received__strips_whitespace(self):
        """Strip leading and trailing whitespace from the transcription text."""
        transcriptions = []
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": "  hello world  "}

        class Capture(WhisperCall):
            chunk_duration = 1

            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        call = make_whisper_call(model_mock, Capture)
        call._audio_packets = [b"x"] * packet_threshold(Capture)
        with patch.object(
            call, "_decode_audio", return_value=np.zeros(16000, dtype=np.float32)
        ):
            await call._transcribe_chunk()
        assert transcriptions == ["hello world"]

    def test_decode_via_av__opus(self):
        """Pipe Ogg Opus data through PyAV and return a float32 PCM array."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        pcm_array = np.zeros(16000, dtype=np.float32)
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = pcm_array
        mock_resampled = [mock_frame]
        with patch("voip.audio.av") as mock_av:
            mock_resampler = MagicMock()
            mock_resampler.resample.side_effect = [mock_resampled, []]
            mock_av.audio.resampler.AudioResampler.return_value = mock_resampler
            mock_container = MagicMock()
            mock_container.__enter__ = lambda s: s
            mock_container.__exit__ = MagicMock(return_value=False)
            mock_container.decode.return_value = [MagicMock()]
            mock_av.open.return_value = mock_container
            mock_av.AVError = Exception
            result = call._decode_via_av(
                b"fake_ogg_data", input_format="ogg", input_sample_rate=None
            )
        mock_av.open.assert_called_once()
        assert result.dtype == np.float32

    def test_decode_via_av__raises_on_av_error(self):
        """Raise RuntimeError when PyAV raises AVError."""
        call = make_whisper_call(MagicMock())
        with (
            patch("voip.audio.av") as mock_av,
            pytest.raises(RuntimeError, match="Audio decoding failed"),
        ):
            mock_av.AVError = RuntimeError
            mock_av.audio.resampler.AudioResampler.return_value = MagicMock()
            mock_av.open.side_effect = RuntimeError("av error")
            call._decode_via_av(b"bad_data", input_format="ogg", input_sample_rate=None)

    def test_decode_audio__opus__wraps_in_ogg(self):
        """Decode Opus packets by wrapping them in an Ogg container before calling PyAV."""
        call = make_whisper_call(MagicMock(), media=OPUS_MEDIA)
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"])
        mock_decode.assert_called_once()
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "ogg"

    def test_decode_audio__pcma__uses_alaw_format(self):
        """Decode PCMA packets using the alaw PyAV input format."""
        call = make_whisper_call(MagicMock(), media=PCMA_MEDIA)
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "alaw"

    def test_decode_audio__pcmu__uses_mulaw_format(self):
        """Decode PCMU packets using the mulaw PyAV input format."""
        call = make_whisper_call(MagicMock(), media=PCMU_MEDIA)
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "mulaw"

    def test_decode_audio__g722__uses_g722_format(self):
        """Decode G.722 packets using the g722 PyAV input format."""
        call = make_whisper_call(MagicMock(), media=G722_MEDIA)
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "g722"

    def test_decode_audio__unknown__raises(self):
        """Raise NotImplementedError for unsupported codec names."""
        unknown_media = _make_media("126", "126 telephone-event/8000")
        call = make_whisper_call(MagicMock(), media=unknown_media)
        with pytest.raises(NotImplementedError, match="Unsupported"):
            call._decode_audio([b"pkt"])

    def test_decode_audio__uses_sample_rate_from_media(self):
        """Pass the sample rate from the MediaDescription to _decode_via_av."""
        # PCMA/16000 is a wideband variant.
        wideband_pcma = _make_media("8", "8 PCMA/16000")
        call = make_whisper_call(MagicMock(), media=wideband_pcma)
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_sample_rate") == 16000

    def test_audio_received__logs_debug(self, caplog):
        """Log a debug message for each received RTP packet."""
        import logging

        call = make_whisper_call(MagicMock())
        with caplog.at_level(logging.DEBUG, logger="voip.audio"):
            call.audio_received(make_rtp_packet(b"opus_packet"))
        assert any("RTP audio" in r.message for r in caplog.records)


class TestBuildOggOpus:
    def test_build_ogg_opus__starts_with_ogg_magic(self):
        """The resulting container starts with the Ogg capture pattern 'OggS'."""
        assert _build_ogg_opus([b"packet"]).startswith(b"OggS")

    def test_build_ogg_opus__contains_opus_head(self):
        """The resulting container includes the OpusHead identification header."""
        assert b"OpusHead" in _build_ogg_opus([b"packet"])

    def test_build_ogg_opus__contains_opus_tags(self):
        """The resulting container includes the OpusTags comment header."""
        assert b"OpusTags" in _build_ogg_opus([b"packet"])

    def test_build_ogg_opus__non_empty_for_single_packet(self):
        """Produce a non-empty Ogg container for a single Opus packet."""
        assert len(_build_ogg_opus([b"x" * 100])) > 100

    def test_build_ogg_opus__empty_packets_list(self):
        """Produce a valid Ogg container even with an empty packet list."""
        result = _build_ogg_opus([])
        assert b"OggS" in result

    def test_build_ogg_opus__large_packet_uses_255_lacing(self):
        """Produce a valid Ogg container when a packet exceeds 254 bytes (lacing spans 255)."""
        large_packet = b"x" * 256
        result = _build_ogg_opus([large_packet])
        assert b"OggS" in result
        assert len(result) > 256

    def test_build_ogg_opus__multiple_pages(self):
        """Produce multiple Ogg pages when more than 50 packets are provided."""
        packets = [b"x" * 10] * 55
        result = _build_ogg_opus(packets)
        assert result.count(b"OggS") > 3
