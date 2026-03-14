"""Tests for AI-powered call handlers (WhisperCall and AgentCall)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faster_whisper")
pytest.importorskip("ollama")
pytest.importorskip("pocket_tts")

from voip.ai import AgentCall, AgentState, TranscribeCall  # noqa: E402
from voip.audio import AudioCall  # noqa: E402
from voip.rtp import RTPPayloadType  # noqa: E402
from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: E402


def _make_media(fmt: str, rtpmap: str | None = None) -> MediaDescription:
    """Build a single-codec MediaDescription for use in tests."""
    if rtpmap:
        payload_format = RTPPayloadFormat.parse(rtpmap)
    else:
        payload_format = RTPPayloadFormat(payload_type=int(fmt))
    return MediaDescription(
        media="audio", port=0, proto="RTP/AVP", fmt=[payload_format]
    )


OPUS_MEDIA = _make_media("111", "111 opus/48000/2")
PCMA_MEDIA = _make_media("8", "8 PCMA/8000")
PCMU_MEDIA = _make_media("0")  # static PT, no rtpmap
G722_MEDIA = _make_media("9", "9 G722/8000")


def make_whisper_call(
    model_mock: MagicMock, call_class=None, media: MediaDescription | None = None
) -> TranscribeCall:
    """Return a WhisperCall with a mocked Whisper model."""
    cls = call_class or TranscribeCall
    med = media if media is not None else OPUS_MEDIA
    with patch("voip.ai.WhisperModel", return_value=model_mock):
        return cls(
            rtp=MagicMock(),
            sip=MagicMock(),
            caller="sip:bob@biloxi.com",
            media=med,
        )


def make_agent_call(
    model_mock: MagicMock,
    tts_mock: MagicMock,
    call_class=None,
    media: MediaDescription | None = None,
) -> AgentCall:
    """Return an AgentCall with mocked Whisper model and Pocket TTS model."""
    cls = call_class or AgentCall
    med = media if media is not None else OPUS_MEDIA
    with (
        patch("voip.ai.WhisperModel", return_value=model_mock),
        patch("voip.ai.TTSModel") as tts_cls,
    ):
        tts_cls.load_model.return_value = tts_mock
        return cls(
            rtp=MagicMock(),
            sip=MagicMock(),
            caller="sip:bob@biloxi.com",
            media=med,
        )


class TestWhisperCall:
    def test_whisper_call__is_audio_call(self):
        """WhisperCall is a subclass of AudioCall."""
        assert issubclass(TranscribeCall, AudioCall)

    def test_init__uses_pre_loaded_model_instance(self):
        """When model is a WhisperModel instance it is stored directly (no re-load)."""
        model_instance = MagicMock()
        with patch("voip.ai.WhisperModel") as wm_cls:
            # Pass the instance directly — the constructor must NOT be called again.
            call = TranscribeCall(
                rtp=MagicMock(),
                sip=MagicMock(),
                media=OPUS_MEDIA,
                model=model_instance,
            )
        wm_cls.assert_not_called()
        assert call._whisper_model is model_instance

    def test_init__stores_media(self):
        """Media is stored and accessible as self.media."""
        call = make_whisper_call(MagicMock())
        assert call.media is OPUS_MEDIA

    def test_init__derives_payload_type_from_opus_media(self):
        """payload_type is 111 (Opus) when given OPUS_MEDIA."""
        call = make_whisper_call(MagicMock())
        assert call.payload_type == RTPPayloadType.OPUS

    def test_init__derives_payload_type_from_pcma_media(self):
        """payload_type is 8 (PCMA) when given PCMA_MEDIA."""
        call = make_whisper_call(MagicMock(), media=PCMA_MEDIA)
        assert call.payload_type == RTPPayloadType.PCMA

    def test_audio_received__initializes_vad_state(self):
        """WhisperCall starts with an empty speech buffer and no timer."""
        call = make_whisper_call(MagicMock())
        assert call._speech_buffer == []
        assert call._transcription_handle is None

    def test_audio_received__silence_audio_accumulates_in_buffer(self):
        """Silence audio (below speech_threshold) is still buffered for transcription."""
        call = make_whisper_call(MagicMock())
        with patch("voip.ai.asyncio.get_event_loop"):
            call.audio_received(audio=np.zeros(320, dtype=np.float32), rms=0.0)
        assert len(call._speech_buffer) == 1

    def test_audio_received__speech_audio_accumulates_in_buffer(self):
        """Audio above speech_threshold is added to _speech_buffer."""
        call = make_whisper_call(MagicMock())
        speech = np.ones(320, dtype=np.float32) * 0.6
        call.audio_received(audio=speech, rms=0.6)
        assert len(call._speech_buffer) == 1

    def test_audio_received__silence_arms_transcription_timer(self):
        """Silence arms the transcription debounce timer."""
        call = make_whisper_call(MagicMock())
        with patch("voip.ai.asyncio.get_event_loop") as mock_loop:
            handle = MagicMock()
            mock_loop.return_value.call_later.return_value = handle
            call.audio_received(audio=np.zeros(320, dtype=np.float32), rms=0.0)
        mock_loop.return_value.call_later.assert_called_once_with(
            call.silence_gap, call._flush_speech_buffer
        )
        assert call._transcription_handle is handle

    def test_audio_received__silence_does_not_rearm_when_timer_running(self):
        """A second silence packet does not create a second timer."""
        call = make_whisper_call(MagicMock())
        call._transcription_handle = MagicMock()
        with patch("voip.ai.asyncio.get_event_loop") as mock_loop:
            call.audio_received(audio=np.zeros(320, dtype=np.float32), rms=0.0)
        mock_loop.return_value.call_later.assert_not_called()

    def test_audio_received__speech_cancels_pending_timer(self):
        """Speech audio cancels any running transcription debounce timer."""
        call = make_whisper_call(MagicMock())
        handle = MagicMock()
        call._transcription_handle = handle
        call.audio_received(audio=np.ones(320, dtype=np.float32) * 0.6, rms=0.6)
        handle.cancel.assert_called_once()
        assert call._transcription_handle is None

    def test_audio_received__empty_array_accumulates_in_buffer(self):
        """Zero-length audio arrays are accepted into the speech buffer."""
        call = make_whisper_call(MagicMock())
        with patch("voip.ai.asyncio.get_event_loop"):
            call.audio_received(audio=np.zeros(0, dtype=np.float32), rms=0.0)
        assert len(call._speech_buffer) == 1

    async def test_flush_speech_buffer__transcribes_accumulated_audio(self):
        """_flush_speech_buffer concatenates speech and schedules transcription."""
        transcriptions = []
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = "hello"
        model_mock.transcribe.return_value = ([seg], MagicMock())

        class Capture(TranscribeCall):
            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        call = make_whisper_call(model_mock, Capture)
        chunk = np.ones(320, dtype=np.float32) * 0.6
        call._speech_buffer = [chunk]
        # Set silence_gap=0 so the minimum-length check passes with one chunk.
        call.silence_gap = 0
        call._flush_speech_buffer()
        await asyncio.sleep(0.1)
        assert transcriptions == ["hello"]
        assert call._speech_buffer == []

    def test_flush_speech_buffer__no_op_when_buffer_empty(self):
        """_flush_speech_buffer does nothing when the speech buffer is empty."""
        call = make_whisper_call(MagicMock())
        with patch("voip.ai.asyncio.create_task") as mock_ct:
            call._flush_speech_buffer()
        mock_ct.assert_not_called()

    def test_flush_speech_buffer__resets_state(self):
        """_flush_speech_buffer clears _transcription_handle and the speech buffer."""
        call = make_whisper_call(MagicMock())
        call._transcription_handle = MagicMock()
        call._speech_buffer = [np.zeros(1, dtype=np.float32)]
        with patch("voip.ai.asyncio.create_task"):
            call._flush_speech_buffer()
        assert call._transcription_handle is None

    async def test_transcribe__strips_whitespace(self):
        """Strip leading and trailing whitespace from the transcription text."""
        transcriptions = []
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = "  hello world  "
        model_mock.transcribe.return_value = ([seg], MagicMock())

        class Capture(TranscribeCall):
            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        call = make_whisper_call(model_mock, Capture)
        await call._transcribe(np.zeros(16000, dtype=np.float32))
        assert transcriptions == ["hello world"]

    def test_run_transcription__passes_numpy_array_directly(self):
        """Pass a numpy float32 array to the Whisper model without file I/O."""
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = "test"
        model_mock.transcribe.return_value = ([seg], MagicMock())
        call = make_whisper_call(model_mock)
        audio = np.zeros(16000, dtype=np.float32)
        assert call._run_transcription(audio) == "test"
        model_mock.transcribe.assert_called_once_with(audio)

    def test_run_transcription__no_file_written(self):
        """The transcription path must not write any files to disk."""
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = ""
        model_mock.transcribe.return_value = ([seg], MagicMock())
        call = make_whisper_call(model_mock)
        with patch(
            "builtins.open", side_effect=AssertionError("open() must not be called")
        ):
            call._run_transcription(np.zeros(16000, dtype=np.float32))

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
        """Propagate the underlying error when PyAV raises during decoding."""
        call = make_whisper_call(MagicMock())
        with (
            patch("voip.audio.av") as mock_av,
            pytest.raises(RuntimeError, match="av error"),
        ):
            mock_av.AVError = RuntimeError
            mock_av.audio.resampler.AudioResampler.return_value = MagicMock()
            mock_av.open.side_effect = RuntimeError("av error")
            call._decode_via_av(b"bad_data", input_format="ogg", input_sample_rate=None)

    def test_decode_raw__opus__wraps_in_ogg(self):
        """Decode Opus packets by wrapping them in an Ogg container before calling PyAV."""
        call = make_whisper_call(MagicMock(), media=OPUS_MEDIA)
        assert call.payload_type == RTPPayloadType.OPUS
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw(b"pkt")
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "ogg"

    def test_decode_raw__pcma__uses_alaw_format(self):
        """Decode PCMA packets using the alaw PyAV input format."""
        call = make_whisper_call(MagicMock(), media=PCMA_MEDIA)
        assert call.payload_type == RTPPayloadType.PCMA
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw(b"pkt")
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "alaw"

    def test_decode_raw__pcmu__uses_mulaw_format(self):
        """Decode PCMU packets using the mulaw PyAV input format."""
        call = make_whisper_call(MagicMock(), media=PCMU_MEDIA)
        assert call.payload_type == RTPPayloadType.PCMU
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw(b"pkt")
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "mulaw"

    def test_decode_raw__g722__uses_g722_format(self):
        """Decode G.722 packets using the g722 PyAV input format."""
        call = make_whisper_call(MagicMock(), media=G722_MEDIA)
        assert call.payload_type == RTPPayloadType.G722
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw(b"pkt")
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "g722"

    def test_decode_raw__uses_sample_rate_from_media(self):
        """Pass the sample rate from the MediaDescription to _decode_via_av."""
        wideband_pcma = _make_media("8", "8 PCMA/16000")
        call = make_whisper_call(MagicMock(), media=wideband_pcma)
        assert call.sample_rate == 16000
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw(b"pkt")
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_sample_rate") == 16000

    async def test_transcribe__raises_on_general_error(self):
        """Exceptions from transcription propagate to the caller."""
        call = make_whisper_call(MagicMock())
        with (
            patch.object(
                call, "_run_transcription", side_effect=RuntimeError("model error")
            ),
            pytest.raises(RuntimeError, match="model error"),
        ):
            await call._transcribe(np.zeros(16000, dtype=np.float32))

    async def test_transcribe__cancelled_error_is_re_raised(self):
        """_transcribe re-raises CancelledError without logging it as an exception."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        with (
            patch.object(
                call, "_run_transcription", side_effect=asyncio.CancelledError
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await call._transcribe(np.zeros(16000, dtype=np.float32))

    async def test_transcribe__empty_transcription_not_delivered(self):
        """Whitespace-only transcription is silently discarded."""
        transcriptions = []
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = "   "
        model_mock.transcribe.return_value = ([seg], MagicMock())

        class Capture(TranscribeCall):
            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        call = make_whisper_call(model_mock, Capture)
        await call._transcribe(np.zeros(16000, dtype=np.float32))
        assert transcriptions == []


class TestAgentCall:
    def test_agent_call__is_whisper_call(self):
        """AgentCall is a subclass of WhisperCall."""
        assert issubclass(AgentCall, TranscribeCall)

    def test_init__loads_tts_model_when_none(self):
        """Load the default Pocket TTS model when tts_model is None."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        with (
            patch("voip.ai.WhisperModel", return_value=MagicMock()),
            patch("voip.ai.TTSModel") as tts_cls,
        ):
            tts_cls.load_model.return_value = tts_mock
            call = AgentCall(rtp=MagicMock(), sip=MagicMock(), media=OPUS_MEDIA)
        tts_cls.load_model.assert_called_once()
        assert call._tts_instance is tts_mock

    def test_init__uses_provided_tts_model(self):
        """Use the provided TTSModel instance instead of loading a new one."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        with (
            patch("voip.ai.WhisperModel", return_value=MagicMock()),
            patch("voip.ai.TTSModel") as tts_cls,
        ):
            call = AgentCall(
                rtp=MagicMock(),
                sip=MagicMock(),
                media=OPUS_MEDIA,
                tts_model=tts_mock,
            )
        tts_cls.load_model.assert_not_called()
        assert call._tts_instance is tts_mock

    def test_init__loads_voice_state(self):
        """Get the voice state from the TTS model on init."""
        tts_mock = MagicMock()
        voice_state = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = voice_state
        with (
            patch("voip.ai.WhisperModel", return_value=MagicMock()),
            patch("voip.ai.TTSModel") as tts_cls,
        ):
            tts_cls.load_model.return_value = tts_mock
            call = AgentCall(
                rtp=MagicMock(), sip=MagicMock(), media=OPUS_MEDIA, voice="alba"
            )
        tts_mock.get_state_for_audio_prompt.assert_called_once_with("alba")
        assert call._voice_state is voice_state

    def test_init__opus_media__sets_rtp_fields(self):
        """Opus media sets 48 kHz sample rate, 960-sample chunks, and 960 ts-increment."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=OPUS_MEDIA)
        assert call._rtp_sample_rate == 48000
        assert call._rtp_chunk_samples == 960
        assert call._rtp_ts_increment == 960

    def test_init__g722_media__sets_rtp_fields(self):
        """G.722 media sets 16 kHz audio but uses 8 kHz RTP clock (160 ts-increment)."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=G722_MEDIA)
        assert call._rtp_sample_rate == 16000
        assert call._rtp_chunk_samples == 320
        assert call._rtp_ts_increment == 160

    def test_init__pcmu_media__sets_rtp_fields(self):
        """PCMU media sets 8 kHz sample rate, 160-sample chunks and ts-increment."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=PCMU_MEDIA)
        assert call._rtp_sample_rate == 8000
        assert call._rtp_chunk_samples == 160
        assert call._rtp_ts_increment == 160

    def test_init__initializes_pending_state(self):
        """AgentCall starts with empty pending text and no response task."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        assert call._pending_text == []
        assert call._response_task is None

    def test_init__initializes_chat_history_with_system_prompt(self):
        """Chat history is seeded with a system prompt mentioning a phone call."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        assert len(call._messages) == 1
        assert call._messages[0]["role"] == "system"
        assert "phone" in call._messages[0]["content"].lower()

    def test_transcription_received__ignores_empty_text(self):
        """transcription_received does not buffer empty text."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call.transcription_received("")
        assert call._pending_text == []

    def test_transcription_received__buffers_non_empty_text(self):
        """transcription_received buffers text and creates a response task."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        with patch(
            "voip.ai.asyncio.create_task",
            side_effect=lambda c: c.close() or MagicMock(),
        ) as mock_ct:
            call.transcription_received("hello")
        assert call._pending_text == ["hello"]
        mock_ct.assert_called_once()

    def test_transcription_received__schedules_response_task(self):
        """transcription_received creates and stores a response task."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        task_mock = MagicMock()
        with patch(
            "voip.ai.asyncio.create_task", side_effect=lambda c: c.close() or task_mock
        ) as mock_ct:
            call.transcription_received("hello world")
        mock_ct.assert_called_once()
        assert call._response_task is task_mock

    def test_transcription_received__cancels_running_task_before_creating_new(self):
        """transcription_received cancels any existing response task."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        old_task = MagicMock()
        old_task.done.return_value = False
        call._response_task = old_task
        with patch("voip.ai.asyncio.create_task", side_effect=lambda c: c.close()):
            call.transcription_received("hello")
        old_task.cancel.assert_called_once()

    async def test_respond__calls_ollama_and_sends_speech(self):
        """_respond fetches an Ollama reply, records it in history, and sends speech."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._pending_text = ["hello"]
        mock_response = MagicMock()
        mock_response.message.content = "I am an AI assistant."
        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            patch.object(
                call, "_send_speech", new_callable=AsyncMock
            ) as mock_send_speech,
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client
            await call._respond()

        mock_send_speech.assert_awaited_once_with("I am an AI assistant.")
        assert {"role": "user", "content": "hello"} in call._messages
        assert {
            "role": "assistant",
            "content": "I am an AI assistant.",
        } in call._messages

    async def test_respond__passes_full_history_to_ollama(self):
        """_respond passes the full message history (including system prompt) to Ollama."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._pending_text = ["hello"]
        mock_response = MagicMock()
        mock_response.message.content = "reply"
        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            patch.object(call, "_send_speech", new_callable=AsyncMock),
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client
            await call._respond()
        _, kwargs = mock_client.chat.call_args
        messages = kwargs.get("messages") or mock_client.chat.call_args[0][0]
        # First message is the system prompt
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "hello"}

    async def test_respond__logs_exception_on_error(self, caplog):
        """Log an exception when Ollama raises an error."""
        import logging

        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._pending_text = ["hello"]
        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            caplog.at_level(logging.ERROR, logger="voip.ai"),
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(side_effect=RuntimeError("ollama error"))
            mock_client_cls.return_value = mock_client
            await call._respond()
        assert any("agent response" in r.message for r in caplog.records)

    async def test_respond__re_raises_cancelled_error(self):
        """Re-raise CancelledError from Ollama and remove the partial user turn."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._pending_text = ["hello"]
        initial_history_len = len(call._messages)
        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            pytest.raises(asyncio.CancelledError),
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(side_effect=asyncio.CancelledError())
            mock_client_cls.return_value = mock_client
            await call._respond()
        # Partial user turn must be rolled back to keep history consistent
        assert len(call._messages) == initial_history_len

    async def test_send_speech__streams_chunks_to_send_rtp_audio(self):
        """_send_speech resamples each TTS chunk and passes it to _send_rtp_audio."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        arr1 = np.zeros(160, dtype=np.float32)
        arr2 = np.ones(160, dtype=np.float32)
        chunk1, chunk2 = MagicMock(), MagicMock()
        chunk1.numpy.return_value = arr1
        chunk2.numpy.return_value = arr2
        tts_mock.generate_audio_stream.return_value = iter([chunk1, chunk2])
        # Match PCMU _rtp_sample_rate so _resample returns the same array (identity check)
        tts_mock.sample_rate = 8000

        call = make_agent_call(MagicMock(), tts_mock, media=PCMU_MEDIA)
        received: list[np.ndarray] = []

        async def _capture(audio: np.ndarray) -> None:
            received.append(audio)

        with patch.object(call, "_send_rtp_audio", side_effect=_capture):
            await call._send_speech("hello")

        assert len(received) == 2
        assert received[0] is arr1
        assert received[1] is arr2

    async def test_send_rtp_audio__sends_to_remote_addr(self):
        """_send_rtp_audio sends RTP packets to the caller's registered address."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=PCMU_MEDIA)

        remote_addr = ("10.0.0.1", 5004)
        call.rtp.calls = {remote_addr: call}

        audio = np.zeros(160, dtype=np.float32)
        with patch.object(call, "send_datagram") as mock_send:
            await call._send_rtp_audio(audio)
            mock_send.assert_called_once()
        data, addr = mock_send.call_args[0]
        assert addr == remote_addr
        assert len(data) == 12 + 160  # 12-byte RTP header + 160 PCMU bytes

    async def test_send_rtp_audio__drops_audio_when_no_remote_addr(self, caplog):
        """Log a warning and drop audio when no RTP address is registered."""
        import logging

        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call.rtp.calls = {}

        with (
            caplog.at_level(logging.WARNING, logger="voip.ai"),
            patch.object(call, "send_datagram") as mock_send,
        ):
            await call._send_rtp_audio(np.zeros(160, dtype=np.float32))
        mock_send.assert_not_called()
        assert any("dropping audio" in r.message for r in caplog.records)

    def test_resample__downsamples_from_24khz_to_8khz(self):
        """_resample reduces 24 000 samples at 24 kHz to 8 000 samples at 8 kHz."""
        audio = np.zeros(24000, dtype=np.float32)
        result = AgentCall._resample(audio, 24000, 8000)
        assert len(result) == 8000

    def test_resample__passthrough_when_rate_matches(self):
        """_resample returns the original array unchanged when rates are equal."""
        audio = np.zeros(8000, dtype=np.float32)
        result = AgentCall._resample(audio, 8000, 8000)
        assert result is audio

    def test_encode_pcmu__returns_one_byte_per_sample(self):
        """_encode_pcmu returns a bytes object with one byte per input sample."""
        samples = np.zeros(160, dtype=np.float32)
        result = AgentCall._encode_pcmu(samples)
        assert isinstance(result, bytes)
        assert len(result) == 160

    def test_encode_pcmu__silence_encodes_to_midpoint(self):
        """Zero-amplitude samples encode to the µ-law midpoint value."""
        samples = np.zeros(10, dtype=np.float32)
        result = AgentCall._encode_pcmu(samples)
        assert result[0] in (127, 128)

    def test_encode_pcma__returns_one_byte_per_sample(self):
        """_encode_pcma returns a bytes object with one byte per input sample."""
        result = AgentCall._encode_pcma(np.zeros(160, dtype=np.float32))
        assert isinstance(result, bytes)
        assert len(result) == 160

    def test_encode_pcma__silence_encodes_consistently(self):
        """Zero-amplitude samples all encode to the same A-law codeword."""
        result = AgentCall._encode_pcma(np.zeros(10, dtype=np.float32))
        assert all(b == result[0] for b in result)

    def test_encode_via_av__g722_returns_bytes(self):
        """_encode_via_av produces non-empty bytes for G.722."""
        result = AgentCall._encode_via_av(
            np.zeros(320, dtype=np.float32), "g722", 16000
        )
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_via_av__opus_returns_bytes(self):
        """_encode_via_av produces non-empty bytes for Opus (libopus)."""
        result = AgentCall._encode_via_av(
            np.zeros(960, dtype=np.float32), "libopus", 48000
        )
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_audio__dispatches_to_pcmu(self):
        """_encode_audio delegates to _encode_pcmu for PCMU media."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=PCMU_MEDIA)
        samples = np.zeros(160, dtype=np.float32)
        with patch.object(
            AgentCall, "_encode_pcmu", return_value=b"x" * 160
        ) as mock_enc:
            call._encode_audio(samples)
        mock_enc.assert_called_once_with(samples)

    def test_encode_audio__dispatches_to_pcma(self):
        """_encode_audio delegates to _encode_pcma for PCMA media."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=PCMA_MEDIA)
        samples = np.zeros(160, dtype=np.float32)
        with patch.object(
            AgentCall, "_encode_pcma", return_value=b"x" * 160
        ) as mock_enc:
            call._encode_audio(samples)
        mock_enc.assert_called_once_with(samples)

    def test_encode_audio__dispatches_to_opus(self):
        """_encode_audio delegates to _encode_via_av with libopus for Opus media."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=OPUS_MEDIA)
        samples = np.zeros(960, dtype=np.float32)
        with patch.object(AgentCall, "_encode_via_av", return_value=b"x") as mock_enc:
            call._encode_audio(samples)
        mock_enc.assert_called_once_with(samples, "libopus", 48000)

    def test_encode_audio__raises_for_unsupported_codec(self):
        """_encode_audio raises NotImplementedError for unrecognised encoding names."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=PCMU_MEDIA)
        call._encoding_name = "h264"
        with pytest.raises(NotImplementedError, match="Unsupported outbound codec"):
            call._encode_audio(np.zeros(160, dtype=np.float32))

    def test_encode_audio__raises_for_g722(self):
        """_encode_audio raises NotImplementedError for G.722 (handled by _packetize)."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=PCMU_MEDIA)
        call._encoding_name = "g722"
        with pytest.raises(NotImplementedError, match="Unsupported outbound codec"):
            call._encode_audio(np.zeros(320, dtype=np.float32))

    def test_packetize__g722_encodes_whole_buffer_at_once(self):
        """_packetize calls _encode_via_av on the full audio buffer for G.722.

        Encoding the whole buffer at once preserves the ADPCM predictor state
        across 20 ms packet boundaries so the decoded audio is continuous.
        """
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=G722_MEDIA)
        # Two 20 ms chunks: 320 samples each → 160 encoded bytes each
        audio = np.zeros(640, dtype=np.float32)
        # _encode_via_av returns 320 bytes for 640-sample input (2:1 ratio)
        fake_encoded = b"\xab" * 320
        with patch.object(
            AgentCall, "_encode_via_av", return_value=fake_encoded
        ) as mock_enc:
            packets = list(call._packetize(audio))
        # Must be called once with the full buffer (not per-chunk)
        mock_enc.assert_called_once_with(audio, "g722", 16000)
        assert len(packets) == 2
        assert packets[0] == b"\xab" * 160
        assert packets[1] == b"\xab" * 160

    def test_packetize__pcmu_encodes_per_chunk(self):
        """_packetize calls _encode_audio once per 20 ms chunk for PCMU."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=PCMU_MEDIA)
        audio = np.zeros(320, dtype=np.float32)
        with patch.object(
            call, "_encode_audio", return_value=b"\x7f" * 160
        ) as mock_enc:
            packets = list(call._packetize(audio))
        assert mock_enc.call_count == 2
        assert len(packets) == 2

    def test_build_rtp_packet__has_twelve_byte_header(self):
        """_build_rtp_packet prepends a 12-byte RTP header to the payload."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        payload = b"\x00" * 160
        packet = call._build_rtp_packet(payload)
        assert len(packet) == 12 + len(payload)
        assert packet[0] == 0x80  # V=2, P=0, X=0, CC=0

    def test_build_rtp_packet__increments_seq_and_ts_each_call(self):
        """Each _build_rtp_packet call increments seq by 1 and ts by chunk size."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=PCMU_MEDIA)
        call._build_rtp_packet(b"\x00" * 160)
        assert call._rtp_seq == 1
        assert call._rtp_ts == 160
        call._build_rtp_packet(b"\x00" * 160)
        assert call._rtp_seq == 2
        assert call._rtp_ts == 320

    def test_build_rtp_packet__uses_negotiated_payload_type(self):
        """_build_rtp_packet uses the negotiated payload type in the RTP header."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=PCMA_MEDIA)
        packet = call._build_rtp_packet(b"\x00" * 160)
        assert packet[1] == RTPPayloadType.PCMA

    def test_preferred_codecs__opus_is_first(self):
        """AgentCall prefers Opus as the highest-priority outbound codec."""
        assert AgentCall.PREFERRED_CODECS[0].payload_type == RTPPayloadType.OPUS

    def test_encode_pcmu__silence_is_0x7f(self):
        """Silence (0.0) must encode to 0x7F (127) per ITU-T G.711."""
        result = AgentCall._encode_pcmu(np.zeros(1, dtype=np.float32))
        assert result[0] == 0x7F

    def test_encode_pcmu__max_positive_is_0x00(self):
        """Maximum positive amplitude must encode to 0x00 per ITU-T G.711."""
        result = AgentCall._encode_pcmu(np.array([1.0], dtype=np.float32))
        assert result[0] == 0x00

    def test_encode_pcmu__max_negative_is_0x80(self):
        """Maximum negative amplitude must encode to 0x80 per ITU-T G.711."""
        result = AgentCall._encode_pcmu(np.array([-1.0], dtype=np.float32))
        assert result[0] == 0x80

    async def test_send_rtp_audio__paces_packets_at_20ms_intervals(self):
        """_send_rtp_audio sleeps _rtp_packet_duration seconds between each packet."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock, media=PCMU_MEDIA)
        remote_addr = ("10.0.0.2", 5006)
        call.rtp.calls = {remote_addr: call}

        # Two 20 ms packets at 8 kHz (320 samples total at chunk_samples=160)
        audio = np.zeros(320, dtype=np.float32)
        sleep_calls: list[float] = []
        original_sleep = asyncio.sleep

        async def _capture_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            await original_sleep(0)  # don't actually wait in tests

        with (
            patch("voip.ai.asyncio.sleep", side_effect=_capture_sleep),
            patch.object(call, "send_datagram"),
        ):
            await call._send_rtp_audio(audio)

        assert len(sleep_calls) == 2
        assert all(s == call._rtp_packet_duration for s in sleep_calls)


class TestAgentState:
    def test_agent_state__has_three_members(self):
        """AgentState defines LISTENING, THINKING, and SPEAKING."""
        assert {s.value for s in AgentState} == {"listening", "thinking", "speaking"}

    def test_agent_state__is_exported(self):
        """AgentState is listed in __all__."""
        import voip.ai

        assert "AgentState" in voip.ai.__all__


class TestAgentCallVAD:
    """Tests for the VAD energy estimation in AgentCall."""

    def test_estimate_payload_rms__silent_bytes_return_zero(self):
        """Constant bytes (silence-like signal) give near-zero std-dev RMS."""
        # All bytes equal → std = 0
        payload = bytes([127] * 160)
        rms = AgentCall._estimate_payload_rms(payload)
        assert rms == pytest.approx(0.0)

    def test_estimate_payload_rms__varying_bytes_return_positive(self):
        """Mixed byte values (speech-like signal) give positive RMS."""
        # Alternating 0 and 255 → high std
        payload = bytes([0, 255] * 80)
        rms = AgentCall._estimate_payload_rms(payload)
        assert rms > 0.5
