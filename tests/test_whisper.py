"""Tests for Whisper-based Opus audio transcription."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")


from sip.calls import IncomingCall  # noqa: E402
from sip.messages import Request  # noqa: E402
from sip.whisper import WhisperCall  # noqa: E402

import whisper  # noqa: E402


def make_invite() -> Request:
    """Return an INVITE request with default headers."""
    return Request(
        method="INVITE",
        uri="sip:alice@atlanta.com",
        headers={"From": "sip:bob@biloxi.com"},
    )


def make_whisper_call(model_mock: MagicMock, call_class=None) -> WhisperCall:
    """Return a WhisperCall with mocked Whisper model and Opus decoder."""
    cls = call_class or WhisperCall
    with (
        patch.dict("sys.modules", {"opuslib": MagicMock()}),
        patch("sip.whisper.whisper.load_model", return_value=model_mock),
    ):
        return cls(make_invite(), ("192.0.2.1", 5060), MagicMock())


class TestWhisperCall:
    def test_whisper_call__is_incoming_call(self):
        """WhisperCall is a subclass of IncomingCall."""
        assert issubclass(WhisperCall, IncomingCall)

    def test_class_attrs__opus_sample_rate(self):
        """opus_sample_rate is 48000 Hz as required by RFC 7587."""
        assert WhisperCall.opus_sample_rate == 48000

    def test_class_attrs__opus_frame_size(self):
        """opus_frame_size is 960 samples (20 ms at 48 kHz)."""
        assert WhisperCall.opus_frame_size == 960

    def test_class_attrs__chunk_duration(self):
        """chunk_duration controls how many seconds are buffered before transcription."""
        assert WhisperCall.chunk_duration == 30

    def test_audio_received__decodes_opus_packet(self):
        """Decode each Opus packet and add PCM samples to the buffer."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        pcm_bytes = bytes(WhisperCall.opus_frame_size * 2)  # 960 int16 zero samples
        call._decoder.decode.return_value = pcm_bytes
        call.audio_received(b"opus_packet")
        call._decoder.decode.assert_called_once_with(
            b"opus_packet", WhisperCall.opus_frame_size
        )
        # 960 samples at 48 kHz → 320 samples at 16 kHz (every 3rd)
        assert len(call._pcm_buffer) == WhisperCall.opus_frame_size // 3

    def test_audio_received__buffers_pcm_below_threshold(self):
        """Don't transcribe until the chunk_duration threshold is reached."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        pcm_bytes = bytes(WhisperCall.opus_frame_size * 2)
        call._decoder.decode.return_value = pcm_bytes
        call.audio_received(b"opus_packet")
        model_mock.transcribe.assert_not_called()

    def test_audio_received__triggers_transcription_when_buffer_full(self):
        """Schedule transcription when enough PCM has been buffered."""
        transcriptions = []
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": "hello"}

        class SmallChunkCall(WhisperCall):
            chunk_duration = 1  # 1 second = 16 000 samples at 16 kHz

            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        async def run() -> None:
            call = make_whisper_call(model_mock, SmallChunkCall)
            chunk_samples = whisper.audio.SAMPLE_RATE * SmallChunkCall.chunk_duration
            # Pre-fill to one sample below the threshold
            call._pcm_buffer = np.zeros(chunk_samples - 1, dtype=np.float32)
            # One 960-sample Opus packet → 320 samples after decimation
            pcm_bytes = bytes(WhisperCall.opus_frame_size * 2)
            call._decoder.decode.return_value = pcm_bytes
            call.audio_received(b"opus_packet")
            await asyncio.sleep(0.1)  # allow the executor task to complete

        asyncio.run(run())
        assert transcriptions == ["hello"]

    def test_audio_received__clears_transcribed_samples_from_buffer(self):
        """Remove transcribed samples from the buffer after transcription."""
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": ""}

        class SmallChunkCall(WhisperCall):
            chunk_duration = 1

        async def run() -> None:
            call = make_whisper_call(model_mock, SmallChunkCall)
            chunk_samples = whisper.audio.SAMPLE_RATE * SmallChunkCall.chunk_duration
            extra = 100
            call._pcm_buffer = np.zeros(chunk_samples + extra, dtype=np.float32)
            await call._transcribe_chunk()
            assert len(call._pcm_buffer) == extra

        asyncio.run(run())

    def test_run_transcription__passes_numpy_array_directly(self):
        """Pass a numpy float32 array to the Whisper model without file I/O."""
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": "test"}
        call = make_whisper_call(model_mock)
        audio = np.zeros(16000, dtype=np.float32)
        result = call._run_transcription(audio)
        assert result == "test"
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

    def test_transcription_received__returns_not_implemented(self):
        """Return NotImplemented for unhandled transcriptions."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        assert call.transcription_received("hello") is NotImplemented

    def test_transcription_received__strips_whitespace(self):
        """Strip leading and trailing whitespace from the transcription text."""
        transcriptions = []
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": "  hello world  "}

        class Capture(WhisperCall):
            chunk_duration = 1

            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        async def run() -> None:
            call = make_whisper_call(model_mock, Capture)
            chunk_samples = whisper.audio.SAMPLE_RATE * Capture.chunk_duration
            call._pcm_buffer = np.zeros(chunk_samples, dtype=np.float32)
            await call._transcribe_chunk()

        asyncio.run(run())
        assert transcriptions == ["hello world"]

    def test_init__raises_import_error_without_opuslib(self):
        """Raise ImportError when libopus is not available."""
        # opuslib raises a plain Exception (not ImportError) when libopus C lib is missing.
        bad_opuslib = MagicMock()
        bad_opuslib.Decoder.side_effect = Exception(
            "Could not find Opus library. Make sure it is installed."
        )
        with (
            patch.dict("sys.modules", {"opuslib": bad_opuslib}),
            patch("sip.whisper.whisper.load_model"),
            pytest.raises(ImportError, match="libopus"),
        ):
            WhisperCall(make_invite(), ("192.0.2.1", 5060), MagicMock())

    def test_audio_received__logs_debug(self, caplog):
        """Log a debug message for each received RTP packet."""
        import logging

        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        pcm_bytes = bytes(WhisperCall.opus_frame_size * 2)
        call._decoder.decode.return_value = pcm_bytes
        with caplog.at_level(logging.DEBUG, logger="sip.whisper"):
            call.audio_received(b"opus_packet")
        assert any("RTP audio" in r.message for r in caplog.records)
