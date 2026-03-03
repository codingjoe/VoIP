"""Tests for Whisper-based Opus audio transcription."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")


from sip.calls import IncomingCall  # noqa: E402
from sip.messages import Request  # noqa: E402
from sip.whisper import WhisperCall, _build_ogg_opus  # noqa: E402

import whisper  # noqa: E402


def make_invite() -> Request:
    """Return an INVITE request with default headers."""
    return Request(
        method="INVITE",
        uri="sip:alice@atlanta.com",
        headers={"From": "sip:bob@biloxi.com"},
    )


def packet_threshold(call_class: type[WhisperCall]) -> int:
    """Return the packet count threshold for the given WhisperCall class."""
    return call_class.opus_sample_rate * call_class.chunk_duration // call_class.opus_frame_size


def make_whisper_call(model_mock: MagicMock, call_class=None) -> WhisperCall:
    """Return a WhisperCall with a mocked Whisper model."""
    cls = call_class or WhisperCall
    with patch("sip.whisper.whisper.load_model", return_value=model_mock):
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

    def test_audio_received__buffers_opus_packet(self):
        """Append each Opus RTP payload to the internal packet buffer."""
        call = make_whisper_call(MagicMock())
        call.audio_received(b"opus_packet")
        assert call._opus_packets == [b"opus_packet"]

    def test_audio_received__buffers_pcm_below_threshold(self):
        """Don't schedule transcription until the packet threshold is reached."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        call.audio_received(b"opus_packet")
        model_mock.transcribe.assert_not_called()

    def test_audio_received__triggers_transcription_when_buffer_full(self):
        """Schedule transcription when enough Opus packets have been buffered."""
        transcriptions = []
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": "hello"}

        class SmallChunkCall(WhisperCall):
            chunk_duration = 1  # → 48000 * 1 // 960 = 50 packets

            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        async def run() -> None:
            call = make_whisper_call(model_mock, SmallChunkCall)
            call._opus_packets = [b"x"] * (packet_threshold(SmallChunkCall) - 1)
            pcm_samples = np.zeros(whisper.audio.SAMPLE_RATE, dtype=np.float32)
            with patch.object(call, "_decode_opus", return_value=pcm_samples):
                call.audio_received(b"opus_packet")
                await asyncio.sleep(0.1)

        asyncio.run(run())
        assert transcriptions == ["hello"]

    def test_audio_received__clears_transcribed_packets_from_buffer(self):
        """Remove the transcribed packets from the buffer after transcription."""
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": ""}

        class SmallChunkCall(WhisperCall):
            chunk_duration = 1

        async def run() -> None:
            call = make_whisper_call(model_mock, SmallChunkCall)
            extra = 5
            call._opus_packets = [b"x"] * (packet_threshold(SmallChunkCall) + extra)
            with patch.object(call, "_decode_opus", return_value=np.zeros(16000, dtype=np.float32)):
                await call._transcribe_chunk()
            assert len(call._opus_packets) == extra

        asyncio.run(run())

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

    def test_transcription_received__returns_not_implemented(self):
        """Return NotImplemented for unhandled transcriptions."""
        assert make_whisper_call(MagicMock()).transcription_received("hello") is NotImplemented

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
            call._opus_packets = [b"x"] * packet_threshold(Capture)
            with patch.object(call, "_decode_opus", return_value=np.zeros(16000, dtype=np.float32)):
                await call._transcribe_chunk()

        asyncio.run(run())
        assert transcriptions == ["hello world"]

    def test_decode_opus__calls_ffmpeg(self):
        """Pipe Ogg Opus data through ffmpeg and return a float32 PCM array."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        pcm_bytes = np.zeros(16000, dtype=np.float32).tobytes()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = pcm_bytes
        with patch("sip.whisper.subprocess.run", return_value=mock_result) as mock_run:
            result = call._decode_opus(b"fake_ogg_data")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert "-loglevel" in args
        assert "error" in args
        assert "pipe:0" in args
        assert "pipe:1" in args
        assert "f32le" in args
        assert len(result) == 16000

    def test_decode_opus__raises_on_ffmpeg_failure(self):
        """Raise RuntimeError when ffmpeg returns a non-zero exit code."""
        call = make_whisper_call(MagicMock())
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"Invalid data"
        with (
            patch("sip.whisper.subprocess.run", return_value=mock_result),
            pytest.raises(RuntimeError, match="ffmpeg"),
        ):
            call._decode_opus(b"bad_data")

    def test_audio_received__logs_debug(self, caplog):
        """Log a debug message for each received RTP packet."""
        import logging

        call = make_whisper_call(MagicMock())
        with caplog.at_level(logging.DEBUG, logger="sip.whisper"):
            call.audio_received(b"opus_packet")
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

