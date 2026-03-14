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

    def test_class_attrs__chunk_duration(self):
        """chunk_duration is 0 so every RTP packet is decoded immediately for VAD."""
        assert TranscribeCall.chunk_duration == 0

    def test_init__packet_threshold__opus(self):
        """_packet_threshold is 1 for Opus with chunk_duration=0 (per-packet emit)."""
        call = make_whisper_call(MagicMock(), media=OPUS_MEDIA)
        assert call._packet_threshold == 1

    def test_init__packet_threshold__g722(self):
        """_packet_threshold is 1 for G.722 with chunk_duration=0 (per-packet emit)."""
        call = make_whisper_call(MagicMock(), media=G722_MEDIA)
        assert call._packet_threshold == 1

    def test_init__packet_threshold__pcma(self):
        """_packet_threshold is 1 for PCMA with chunk_duration=0 (per-packet emit)."""
        call = make_whisper_call(MagicMock(), media=PCMA_MEDIA)
        assert call._packet_threshold == 1

    def test_init__packet_threshold__pcmu(self):
        """_packet_threshold is 1 for PCMU with chunk_duration=0 (per-packet emit)."""
        call = make_whisper_call(MagicMock(), media=PCMU_MEDIA)
        assert call._packet_threshold == 1

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
        """WhisperCall starts with an empty speech buffer and not-speaking state."""
        call = make_whisper_call(MagicMock())
        assert call._speech_buffer == []
        assert not call._in_speech
        assert call._transcription_handle is None

    def test_audio_received__silence_audio_does_not_accumulate(self):
        """Silence audio (below speech_threshold) is not added to the speech buffer."""
        call = make_whisper_call(MagicMock())
        call.audio_received(np.zeros(320, dtype=np.float32))  # RMS=0 < 0.01
        assert call._speech_buffer == []
        assert not call._in_speech

    def test_audio_received__speech_audio_accumulates_in_buffer(self):
        """Audio above speech_threshold is added to _speech_buffer."""
        call = make_whisper_call(MagicMock())
        speech = np.ones(320, dtype=np.float32) * 0.5  # RMS=0.5 > 0.01
        call.audio_received(speech)
        assert len(call._speech_buffer) == 1
        assert call._in_speech

    def test_audio_received__silence_after_speech_arms_transcription_timer(self):
        """Silence after speech arms the transcription debounce timer."""
        call = make_whisper_call(MagicMock())
        call.audio_received(np.ones(320, dtype=np.float32) * 0.5)  # speech
        with patch("voip.ai.asyncio.get_event_loop") as mock_loop:
            handle = MagicMock()
            mock_loop.return_value.call_later.return_value = handle
            call.audio_received(np.zeros(320, dtype=np.float32))  # silence
        mock_loop.return_value.call_later.assert_called_once_with(
            call.silence_gap, call._flush_speech_buffer
        )
        assert call._transcription_handle is handle

    def test_audio_received__silence_without_prior_speech_does_not_arm_timer(self):
        """Silence at call start (no prior speech) does not arm the timer."""
        call = make_whisper_call(MagicMock())
        with patch("voip.ai.asyncio.get_event_loop") as mock_loop:
            call.audio_received(np.zeros(320, dtype=np.float32))
        mock_loop.return_value.call_later.assert_not_called()

    def test_audio_received__silence_does_not_rearm_when_timer_running(self):
        """A second silence packet does not create a second timer."""
        call = make_whisper_call(MagicMock())
        call._in_speech = True
        call._transcription_handle = MagicMock()
        with patch("voip.ai.asyncio.get_event_loop") as mock_loop:
            call.audio_received(np.zeros(320, dtype=np.float32))
        mock_loop.return_value.call_later.assert_not_called()

    def test_audio_received__speech_cancels_pending_timer(self):
        """Speech audio cancels any running transcription debounce timer."""
        call = make_whisper_call(MagicMock())
        handle = MagicMock()
        call._transcription_handle = handle
        call.audio_received(np.ones(320, dtype=np.float32) * 0.5)
        handle.cancel.assert_called_once()
        assert call._transcription_handle is None

    def test_audio_received__empty_array_is_ignored(self):
        """Zero-length audio arrays are silently skipped."""
        call = make_whisper_call(MagicMock())
        call.audio_received(np.zeros(0, dtype=np.float32))
        assert call._speech_buffer == []

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
        chunk = np.ones(320, dtype=np.float32) * 0.5
        call._speech_buffer = [chunk]
        call._in_speech = True
        call._flush_speech_buffer()
        await asyncio.sleep(0.1)
        assert transcriptions == ["hello"]
        assert call._speech_buffer == []
        assert not call._in_speech

    def test_flush_speech_buffer__no_op_when_buffer_empty(self):
        """_flush_speech_buffer does nothing when the speech buffer is empty."""
        call = make_whisper_call(MagicMock())
        with patch("voip.ai.asyncio.create_task") as mock_ct:
            call._flush_speech_buffer()
        mock_ct.assert_not_called()

    def test_flush_speech_buffer__resets_state(self):
        """_flush_speech_buffer clears _transcription_handle and _in_speech."""
        call = make_whisper_call(MagicMock())
        call._in_speech = True
        call._transcription_handle = MagicMock()
        call._speech_buffer = [np.zeros(1, dtype=np.float32)]
        with patch("voip.ai.asyncio.create_task"):
            call._flush_speech_buffer()
        assert not call._in_speech
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

    def test_decode_raw__opus__wraps_in_ogg(self):
        """Decode Opus packets by wrapping them in an Ogg container before calling PyAV."""
        call = make_whisper_call(MagicMock(), media=OPUS_MEDIA)
        assert call.payload_type == RTPPayloadType.OPUS
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "ogg"

    def test_decode_raw__pcma__uses_alaw_format(self):
        """Decode PCMA packets using the alaw PyAV input format."""
        call = make_whisper_call(MagicMock(), media=PCMA_MEDIA)
        assert call.payload_type == RTPPayloadType.PCMA
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "alaw"

    def test_decode_raw__pcmu__uses_mulaw_format(self):
        """Decode PCMU packets using the mulaw PyAV input format."""
        call = make_whisper_call(MagicMock(), media=PCMU_MEDIA)
        assert call.payload_type == RTPPayloadType.PCMU
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "mulaw"

    def test_decode_raw__g722__uses_g722_format(self):
        """Decode G.722 packets using the g722 PyAV input format."""
        call = make_whisper_call(MagicMock(), media=G722_MEDIA)
        assert call.payload_type == RTPPayloadType.G722
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "g722"

    def test_decode_raw__unknown__raises(self):
        """Raise NotImplementedError for unsupported encoding names."""
        unknown_media = _make_media("99")
        call = make_whisper_call(MagicMock(), media=unknown_media)
        with pytest.raises(NotImplementedError, match="Unsupported"):
            call._decode_raw([b"pkt"])

    def test_decode_raw__uses_sample_rate_from_media(self):
        """Pass the sample rate from the MediaDescription to _decode_via_av."""
        wideband_pcma = _make_media("8", "8 PCMA/16000")
        call = make_whisper_call(MagicMock(), media=wideband_pcma)
        assert call.sample_rate == 16000
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_raw([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_sample_rate") == 16000

    async def test_transcribe__logs_exception_on_general_error(self):
        """Non-CancelledError exceptions are logged without re-raising."""
        call = make_whisper_call(MagicMock())
        with patch.object(
            call, "_run_transcription", side_effect=RuntimeError("model error")
        ):
            # Must not raise — exception is swallowed and logged
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

    @pytest.mark.asyncio
    async def test_audio_received__logs_debug(self, caplog):
        """Log a debug message for each received audio frame."""
        import logging

        call = make_whisper_call(MagicMock())
        with caplog.at_level(logging.DEBUG, logger="voip.ai"):
            call.audio_received(np.zeros(1, dtype=np.float32))
            await asyncio.sleep(0)  # let the task run so coroutine is not left dangling
        assert any("Audio received" in r.message for r in caplog.records)


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

    def test_transcription_received__ignores_empty_text(self):
        """transcription_received does not buffer empty text."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call.transcription_received("")
        assert call._pending_text == []

    def test_transcription_received__buffers_non_empty_text_when_timer_active(self):
        """transcription_received buffers text when the silence timer is running."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._silence_handle = MagicMock()
        with patch.object(call, "_trigger_response") as mock_trigger:
            call.transcription_received("hello")
        assert call._pending_text == ["hello"]
        mock_trigger.assert_not_called()

    def test_transcription_received__triggers_response_when_timer_already_fired(self):
        """transcription_received immediately triggers a response when the debounce timer has already fired.

        This fixes the race condition where Whisper inference finishes after the
        silence debounce timer has already checked _pending_text and found it empty.
        """
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._silence_handle = None
        assert call._state is AgentState.LISTENING
        with patch.object(call, "_trigger_response") as mock_trigger:
            call.transcription_received("hello world")
        mock_trigger.assert_called_once()

    def test_transcription_received__does_not_trigger_when_thinking(self):
        """transcription_received does not trigger if state is not LISTENING."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._state = AgentState.THINKING
        call._silence_handle = None
        with patch.object(call, "_trigger_response") as mock_trigger:
            call.transcription_received("hello")
        mock_trigger.assert_not_called()
        assert call._pending_text == ["hello"]

    def test_init__state_is_listening(self):
        """Initial state is LISTENING."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        assert call._state is AgentState.LISTENING
        assert call._pending_text == []
        assert call._silence_handle is None
        assert call._response_task is None

    def test_init__initializes_chat_history_with_system_prompt(self):
        """Chat history is seeded with a system prompt mentioning a phone call."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        assert len(call._messages) == 1
        assert call._messages[0]["role"] == "system"
        assert "phone" in call._messages[0]["content"].lower()

    async def test_respond__calls_ollama_and_sends_speech(self):
        """_respond fetches an Ollama reply, records it in history, and sends speech."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
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
            await call._respond("hello")

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
        mock_response = MagicMock()
        mock_response.message.content = "reply"
        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            patch.object(call, "_send_speech", new_callable=AsyncMock),
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client
            await call._respond("hello")
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
        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            caplog.at_level(logging.ERROR, logger="voip.ai"),
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(side_effect=RuntimeError("ollama error"))
            mock_client_cls.return_value = mock_client
            await call._respond("hello")
        assert any("agent response" in r.message for r in caplog.records)

    async def test_respond__re_raises_cancelled_error(self):
        """Re-raise CancelledError from Ollama and remove the partial user turn."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        initial_history_len = len(call._messages)
        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            pytest.raises(asyncio.CancelledError),
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(side_effect=asyncio.CancelledError())
            mock_client_cls.return_value = mock_client
            await call._respond("hello")
        # Partial user turn must be rolled back to keep history consistent
        assert len(call._messages) == initial_history_len
        assert call._state is AgentState.LISTENING

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
        # Same as _PCMU_SAMPLE_RATE so _resample returns the same object (identity check)
        tts_mock.sample_rate = 8000

        call = make_agent_call(MagicMock(), tts_mock)
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
        tts_mock.sample_rate = 8000  # same as PCMU — no resampling needed
        call = make_agent_call(MagicMock(), tts_mock)

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
        tts_mock.sample_rate = 8000
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
        result = AgentCall._resample(audio, 24000)
        assert len(result) == 8000

    def test_resample__passthrough_when_rate_matches(self):
        """_resample returns the original array unchanged when already at 8 kHz."""
        audio = np.zeros(8000, dtype=np.float32)
        result = AgentCall._resample(audio, 8000)
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
        call = make_agent_call(MagicMock(), tts_mock)
        call._build_rtp_packet(b"\x00" * 160)
        assert call._rtp_seq == 1
        assert call._rtp_ts == 160
        call._build_rtp_packet(b"\x00" * 160)
        assert call._rtp_seq == 2
        assert call._rtp_ts == 320

    def test_preferred_codecs__pcmu_is_first(self):
        """AgentCall prefers PCMU so the negotiated codec matches outbound encoding."""
        assert AgentCall.PREFERRED_CODECS[0].payload_type == 0  # PCMU

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

    async def test_send_speech__saves_debug_wav_when_dir_set(self, tmp_path):
        """_send_speech saves a WAV file to debug_audio_dir after streaming."""
        import wave as wave_module

        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        arr = np.zeros(800, dtype=np.float32)  # 100 ms at 8 kHz
        chunk = MagicMock()
        chunk.numpy.return_value = arr
        tts_mock.generate_audio_stream.return_value = iter([chunk])
        tts_mock.sample_rate = 8000

        call = make_agent_call(MagicMock(), tts_mock)
        call.debug_audio_dir = str(tmp_path)
        with patch.object(call, "_send_rtp_audio", new_callable=AsyncMock):
            await call._send_speech("hello")

        wav_files = list(tmp_path.glob("agent_*.wav"))
        assert len(wav_files) == 1
        with wave_module.open(str(wav_files[0]), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 8000

    async def test_send_speech__no_debug_wav_when_dir_not_set(self, tmp_path):
        """_send_speech does not write any files when debug_audio_dir is None."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        arr = np.zeros(160, dtype=np.float32)
        chunk = MagicMock()
        chunk.numpy.return_value = arr
        tts_mock.generate_audio_stream.return_value = iter([chunk])
        tts_mock.sample_rate = 8000

        call = make_agent_call(MagicMock(), tts_mock)
        assert call.debug_audio_dir is None
        with patch.object(call, "_send_rtp_audio", new_callable=AsyncMock):
            await call._send_speech("hello")
        # No files written
        assert list(tmp_path.iterdir()) == []

    async def test_send_rtp_audio__paces_packets_at_20ms_intervals(self):
        """_send_rtp_audio sleeps _RTP_PACKET_DURATION seconds between each packet."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        tts_mock.sample_rate = 8000
        call = make_agent_call(MagicMock(), tts_mock)
        remote_addr = ("10.0.0.2", 5006)
        call.rtp.calls = {remote_addr: call}

        # Two 20 ms packets (320 samples total)
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
        assert all(s == AgentCall._RTP_PACKET_DURATION for s in sleep_calls)


class TestAgentState:
    def test_agent_state__has_three_members(self):
        """AgentState defines LISTENING, THINKING, and SPEAKING."""
        assert {s.value for s in AgentState} == {"listening", "thinking", "speaking"}

    def test_agent_state__is_exported(self):
        """AgentState is listed in __all__."""
        import voip.ai

        assert "AgentState" in voip.ai.__all__


class TestAgentCallVAD:
    """Tests for the VAD state machine in AgentCall."""

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

    def test_on_speech__cancels_silence_handle(self):
        """_on_speech cancels the pending silence debounce timer."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        handle = MagicMock()
        call._silence_handle = handle
        call._on_speech()
        handle.cancel.assert_called_once()
        assert call._silence_handle is None

    def test_on_speech__in_thinking_cancels_response_task(self):
        """_on_speech cancels the response task when THINKING."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._state = AgentState.THINKING
        task = MagicMock()
        task.done.return_value = False
        call._response_task = task
        call._on_speech()
        task.cancel.assert_called_once()
        assert call._state is AgentState.LISTENING
        assert call._response_task is None

    def test_on_speech__in_speaking_cancels_response_task(self):
        """_on_speech cancels the response task when SPEAKING."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._state = AgentState.SPEAKING
        task = MagicMock()
        task.done.return_value = False
        call._response_task = task
        call._on_speech()
        task.cancel.assert_called_once()
        assert call._state is AgentState.LISTENING

    def test_on_speech__in_speaking_clears_pending_text(self):
        """_on_speech clears buffered transcriptions when interrupting."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._state = AgentState.SPEAKING
        call._pending_text = ["hello", "world"]
        task = MagicMock()
        task.done.return_value = False
        call._response_task = task
        call._on_speech()
        assert call._pending_text == []

    def test_on_speech__in_listening_does_nothing_to_state(self):
        """_on_speech in LISTENING state does not change state."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        assert call._state is AgentState.LISTENING
        call._on_speech()
        assert call._state is AgentState.LISTENING

    def test_on_silence__arms_debounce_timer_when_listening(self):
        """_on_silence schedules a call_later when LISTENING with no timer."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        handle = MagicMock()
        with patch("voip.ai.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.call_later.return_value = handle
            call._on_silence()
        mock_loop.return_value.call_later.assert_called_once_with(
            call.silence_duration, call._trigger_response
        )
        assert call._silence_handle is handle

    def test_on_silence__does_not_rearm_when_timer_already_running(self):
        """_on_silence does not create a second timer when one is running."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._silence_handle = MagicMock()  # timer already set
        with patch("voip.ai.asyncio.get_event_loop") as mock_loop:
            call._on_silence()
        mock_loop.return_value.call_later.assert_not_called()

    def test_on_silence__ignored_when_not_listening(self):
        """_on_silence does nothing when agent is THINKING or SPEAKING."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._state = AgentState.THINKING
        with patch("voip.ai.asyncio.get_event_loop") as mock_loop:
            call._on_silence()
        mock_loop.return_value.call_later.assert_not_called()

    def test_trigger_response__does_nothing_when_no_pending_text(self):
        """_trigger_response is a no-op when _pending_text is empty."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        with patch("voip.ai.asyncio.create_task") as mock_ct:
            call._trigger_response()
        mock_ct.assert_not_called()
        assert call._state is AgentState.LISTENING

    def test_trigger_response__combines_pending_text_and_schedules_task(self):
        """_trigger_response joins pending text and creates the response task."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._pending_text = ["hello", "world"]
        with patch("voip.ai.asyncio.create_task") as mock_ct:
            call._trigger_response()
        mock_ct.assert_called_once()
        assert call._state is AgentState.THINKING
        assert call._pending_text == []

    def test_datagram_received__speech_payload_calls_on_speech(self):
        """datagram_received calls _on_speech for high-energy payloads."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        # Build a minimal RTP packet with a high-energy payload
        high_energy_payload = bytes([0, 255] * 80)  # alternating → high std
        import struct

        header = struct.pack(">BBHII", 0x80, 0, 0, 0, 0x12345678)
        rtp_packet = header + high_energy_payload
        with (
            patch.object(call, "_on_speech") as mock_speech,
            patch.object(call, "_on_silence") as mock_silence,
            # Bypass the parent's audio buffering so we can test in isolation
            patch("voip.audio.AudioCall.datagram_received"),
        ):
            call.datagram_received(rtp_packet, ("1.2.3.4", 5004))
        mock_speech.assert_called_once()
        mock_silence.assert_not_called()

    def test_datagram_received__silent_payload_calls_on_silence(self):
        """datagram_received calls _on_silence for low-energy payloads."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        # All-same bytes → zero std → silence
        silence_payload = bytes([127] * 160)
        import struct

        header = struct.pack(">BBHII", 0x80, 0, 0, 0, 0x12345678)
        rtp_packet = header + silence_payload
        with (
            patch.object(call, "_on_speech") as mock_speech,
            patch.object(call, "_on_silence") as mock_silence,
            patch("voip.audio.AudioCall.datagram_received"),
        ):
            call.datagram_received(rtp_packet, ("1.2.3.4", 5004))
        mock_silence.assert_called_once()
        mock_speech.assert_not_called()

    def test_datagram_received__invalid_rtp_is_silently_ignored(self):
        """datagram_received silently ignores datagrams that are not valid RTP."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        with (
            patch.object(call, "_on_speech") as mock_speech,
            patch.object(call, "_on_silence") as mock_silence,
            patch("voip.audio.AudioCall.datagram_received"),
        ):
            call.datagram_received(b"not_rtp", ("1.2.3.4", 5004))
        mock_speech.assert_not_called()
        mock_silence.assert_not_called()

    def test_datagram_received__empty_payload_is_silently_ignored(self):
        """datagram_received ignores RTP packets with no payload."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        import struct

        # Build an RTP header with zero-length payload
        header = struct.pack(">BBHII", 0x80, 0, 0, 0, 0x12345678)
        rtp_packet = header  # no payload
        with (
            patch.object(call, "_on_speech") as mock_speech,
            patch.object(call, "_on_silence") as mock_silence,
            patch("voip.audio.AudioCall.datagram_received"),
        ):
            call.datagram_received(rtp_packet, ("1.2.3.4", 5004))
        mock_speech.assert_not_called()
        mock_silence.assert_not_called()

    async def test_respond__resets_state_to_listening_after_completion(self):
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._state = AgentState.THINKING
        mock_response = MagicMock()
        mock_response.message.content = "done"
        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            patch.object(call, "_send_speech", new_callable=AsyncMock),
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client
            await call._respond("hello")
        assert call._state is AgentState.LISTENING

    async def test_respond__sets_state_to_speaking_before_tts(self):
        """_respond sets state to SPEAKING before calling _send_speech."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        observed_states: list[AgentState] = []
        mock_response = MagicMock()
        mock_response.message.content = "hello"

        async def _capture_state(text: str) -> None:
            observed_states.append(call._state)

        with (
            patch("voip.ai.ollama.AsyncClient") as mock_client_cls,
            patch.object(call, "_send_speech", side_effect=_capture_state),
        ):
            mock_client = MagicMock()
            mock_client.chat = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client
            await call._respond("hi")
        assert AgentState.SPEAKING in observed_states

    def test_on_speech__in_thinking_resets_whisper_transcription_state(self):
        """_on_speech in THINKING cancels transcription timer and clears speech buffer."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._state = AgentState.THINKING
        # Simulate WhisperCall mid-utterance state
        call._in_speech = True
        call._speech_buffer = [np.zeros(320, dtype=np.float32)]
        handle = MagicMock()
        call._transcription_handle = handle
        task = MagicMock()
        task.done.return_value = False
        call._response_task = task
        call._on_speech()
        handle.cancel.assert_called_once()
        assert call._transcription_handle is None
        assert call._speech_buffer == []
        assert not call._in_speech

    def test_on_speech__in_speaking_resets_whisper_transcription_state(self):
        """_on_speech in SPEAKING cancels transcription timer and clears speech buffer."""
        tts_mock = MagicMock()
        tts_mock.get_state_for_audio_prompt.return_value = MagicMock()
        call = make_agent_call(MagicMock(), tts_mock)
        call._state = AgentState.SPEAKING
        call._in_speech = True
        call._speech_buffer = [np.zeros(320, dtype=np.float32)]
        handle = MagicMock()
        call._transcription_handle = handle
        task = MagicMock()
        task.done.return_value = False
        call._response_task = task
        call._on_speech()
        handle.cancel.assert_called_once()
        assert call._speech_buffer == []
        assert not call._in_speech
