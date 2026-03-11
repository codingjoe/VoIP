"""Tests for Whisper-based audio transcription."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")
pytest.importorskip("whisper")

import whisper  # noqa: E402
from voip.audio import WhisperCall, _build_ogg_opus  # noqa: E402
from voip.rtp import RTP, RTPPacket, RTPPayloadType  # noqa: E402


def packet_threshold(call_class: type[WhisperCall]) -> int:
    """Return the packet count threshold for the given WhisperCall class."""
    return (
        call_class.opus_sample_rate
        * call_class.chunk_duration
        // call_class.opus_frame_size
    )


def make_whisper_call(model_mock: MagicMock, call_class=None) -> WhisperCall:
    """Return a WhisperCall with a mocked Whisper model."""
    cls = call_class or WhisperCall
    with patch("whisper.load_model", return_value=model_mock):
        return cls(caller="sip:bob@biloxi.com")


def make_rtp_packet(
    payload: bytes = b"audio",
    payload_type: int = RTPPayloadType.OPUS,
) -> RTPPacket:
    """Return an RTPPacket with the given payload and payload type."""
    return RTPPacket(
        payload_type=payload_type,
        sequence_number=1,
        timestamp=0,
        ssrc=0,
        payload=payload,
    )


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
        call = make_whisper_call(MagicMock())
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"], RTPPayloadType.OPUS)
        mock_decode.assert_called_once()
        _, kwargs = mock_decode.call_args
        assert (
            kwargs.get("input_format") == "ogg" or mock_decode.call_args[0][1] == "ogg"
        )

    def test_decode_audio__pcma__uses_alaw_format(self):
        """Decode PCMA packets using the alaw PyAV input format."""
        call = make_whisper_call(MagicMock())
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"], RTPPayloadType.PCMA)
        args = mock_decode.call_args
        assert args[1].get("input_format") == "alaw" or args[0][1] == "alaw"

    def test_decode_audio__pcmu__uses_mulaw_format(self):
        """Decode PCMU packets using the mulaw PyAV input format."""
        call = make_whisper_call(MagicMock())
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"], RTPPayloadType.PCMU)
        args = mock_decode.call_args
        assert args[1].get("input_format") == "mulaw" or args[0][1] == "mulaw"

    def test_decode_audio__g722__uses_g722_format(self):
        """Decode G.722 packets using the g722 PyAV input format."""
        call = make_whisper_call(MagicMock())
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"], RTPPayloadType.G722)
        args = mock_decode.call_args
        assert args[1].get("input_format") == "g722" or args[0][1] == "g722"

    def test_decode_audio__unknown__raises(self):
        """Raise NotImplementedError for unsupported payload types."""
        call = make_whisper_call(MagicMock())
        with pytest.raises(NotImplementedError, match="Unsupported"):
            call._decode_audio([b"pkt"], 99)

    def test_decode_audio__uses_instance_sample_rate(self):
        """Pass self.sample_rate to _decode_via_av for non-Opus codecs."""
        call = make_whisper_call(MagicMock())
        call.sample_rate = 16000
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"], RTPPayloadType.PCMA)
        _, kwargs = mock_decode.call_args
        assert (
            kwargs.get("input_sample_rate") == 16000
            or mock_decode.call_args[0][2] == 16000
        )

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
