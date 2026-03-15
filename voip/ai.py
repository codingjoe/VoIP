"""AI-powered call handlers for RTP streams.

This module provides :class:`TranscribeCall`, which transcribes decoded audio
with faster-whisper, and :class:`AgentCall`, which extends it with an
Ollama-powered response loop and Pocket TTS voice synthesis.

Requires the ``ai`` extra: ``pip install voip[ai]``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any, ClassVar

import numpy as np
import ollama
from faster_whisper import WhisperModel
from pocket_tts import TTSModel

from voip.audio import AudioCall
from voip.rtp import RTPPayloadType
from voip.sdp.types import RTPPayloadFormat

__all__ = ["TranscribeCall", "AgentCall"]

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class TranscribeCall(AudioCall):
    """RTP call handler that transcribes audio with faster-whisper.

    Audio is decoded by `AudioCall` on a per-packet basis and delivered to
    `audio_received`, which applies an energy-based voice activity detector
    (VAD).  Speech packets are accumulated until silence is sustained for
    `silence_gap` seconds, then the entire utterance is sent to Whisper as
    one chunk.  This avoids cutting sentences in the middle and prevents
    background microphone noise from being passed to Whisper as spurious
    audio.

    Override `transcription_received` to handle the resulting text:

    ```python
    class MySession(SessionInitiationProtocol):
        def call_received(self, request: Request) -> None:
            self.answer(request=request, call_class=MyCall)
    ```

    To share one model instance across multiple calls (recommended to avoid
    loading it multiple times) pass a pre-loaded `WhisperModel`:

    ```python
    shared_model = WhisperModel("base")

    class MyCall(TranscribeCall):
        model = shared_model
    ```

    """

    model: str | WhisperModel = dataclasses.field(default="kyutai/stt-1b-en_fr-trfs")
    _whisper_model: WhisperModel = dataclasses.field(init=False, repr=False)

    speech_threshold: float = dataclasses.field(default=0.001)
    silence_gap: float = dataclasses.field(default=0.5)

    _speech_buffer: list[np.ndarray] = dataclasses.field(
        init=False, repr=False, default_factory=list
    )
    _transcription_handle: asyncio.TimerHandle | None = dataclasses.field(
        init=False, repr=False, default=None
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.model, str):
            logger.debug("Loading Whisper model %r", self.model)
            self._whisper_model = WhisperModel(self.model)
        else:
            self._whisper_model = self.model
        self._speech_buffer = []
        self._transcription_handle = None

    def audio_received(self, *, audio: np.ndarray, rms: float) -> None:
        self._speech_buffer.append(audio)
        if rms > self.speech_threshold:
            self._on_audio_speech()
        else:
            self._on_audio_silence()

    def _on_audio_speech(self) -> None:
        """Cancel any pending transcription timer when speech is detected."""
        if self._transcription_handle is not None:
            self._transcription_handle.cancel()
            self._transcription_handle = None

    def _on_audio_silence(self) -> None:
        """Arm the transcription debounce timer on silence if not already running."""
        if self._transcription_handle is None:
            logger.debug("Silence detected")
            loop = asyncio.get_event_loop()
            self._transcription_handle = loop.call_later(
                self.silence_gap,
                self._flush_speech_buffer,
            )

    def _flush_speech_buffer(self) -> None:
        """Concatenate buffered audio and schedule async transcription.

        Resets speech state so the next utterance starts with a clean buffer.
        """
        self._transcription_handle = None
        # Ensure at least one second of audio to avoid cutting words in half.
        if sum(len(c) for c in self._speech_buffer) < self.resampling_rate:
            self._speech_buffer.clear()
            return
        audio = np.concatenate(self._speech_buffer)
        self._speech_buffer.clear()
        asyncio.create_task(self._transcribe(audio))

    async def _transcribe(self, audio: np.ndarray) -> None:
        """Transcribe decoded audio and deliver non-empty text to the handler."""
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, self._run_transcription, audio)
        if text := raw.strip():
            self.transcription_received(text)

    def _run_transcription(self, audio: np.ndarray) -> str:
        """Transcribe a float32 PCM array using the Whisper model.

        Args:
            audio: Float32 mono PCM array at :data:`~voip.audio.SAMPLE_RATE` Hz.

        Returns:
            Concatenated transcription text from all segments.
        """
        segments, _ = self._whisper_model.transcribe(audio)
        result = "".join(segment.text for segment in segments)
        logger.debug("Transcription result: %r", segments)
        return result

    def transcription_received(self, text: str) -> None:
        """Handle a transcription result.  Override in subclasses.

        Args:
            text: Transcribed text for this audio chunk (already stripped).
        """


@dataclasses.dataclass(kw_only=True)
class AgentCall(TranscribeCall):
    """RTP call handler that responds to caller speech using Ollama and Pocket TTS.

    Extends :class:`TranscribeCall` by feeding each transcription to an Ollama
    language model, then synthesising the reply as speech with Pocket TTS
    and streaming it back to the caller via RTP.

    Chat history is maintained across turns so the language model can follow
    the conversation.  A built-in system prompt informs the model that it is
    on a phone call.

    To share the TTS model across multiple calls pass a pre-loaded
    `TTSModel`:

    ```python
    shared_tts = TTSModel.load_model()
    AgentCall(rtp=..., sip=..., tts_model=shared_tts)
    ```
    """

    system_prompt: str = (
        "You are a person on a phone call. "
        "Keep your answers very brief and conversational."
    )
    #: Preferred codecs in priority order (highest first).
    PREFERRED_CODECS: ClassVar[list[RTPPayloadFormat]] = [
        RTPPayloadFormat(
            payload_type=RTPPayloadType.OPUS,
            encoding_name="opus",
            sample_rate=48000,
            channels=2,
        ),
        RTPPayloadFormat(payload_type=RTPPayloadType.G722),
        RTPPayloadFormat(payload_type=RTPPayloadType.PCMU),
        RTPPayloadFormat(payload_type=RTPPayloadType.PCMA),
    ]

    #: Ollama model name for generating replies.
    ollama_model: str = dataclasses.field(default="llama3")
    #: Pocket TTS voice name or path to a conditioning audio file.
    voice: str = dataclasses.field(default="azelma")
    #: Pre-loaded Pocket TTS model.  Pass a shared instance to avoid
    #: loading the model separately for each call.
    tts_model: TTSModel | None = dataclasses.field(default=None)

    _tts_instance: TTSModel = dataclasses.field(init=False, repr=False)
    _voice_state: Any = dataclasses.field(init=False, repr=False)
    _messages: list[dict] = dataclasses.field(init=False, repr=False)
    _pending_text: list[str] = dataclasses.field(init=False, repr=False)
    _response_task: asyncio.Task | None = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._tts_instance = self.tts_model or TTSModel.load_model()
        self._voice_state = self._tts_instance.get_state_for_audio_prompt(self.voice)  # type: ignore[arg-type]
        self._messages = [
            {
                "role": "system",
                "content": self.system_prompt
                + "\n\nYOU MUST NEVER USE NON-VERBAL CHARACTERS IN YOUR RESPONSES!",
            }
        ]
        self._pending_text = []
        self._response_task = None

    def transcription_received(self, text: str) -> None:
        match text:
            case "":
                return
            case _:
                self._pending_text.append(text)
                if self._response_task is not None and not self._response_task.done():
                    self._response_task.cancel()
                self._response_task = asyncio.create_task(self._respond())

    async def _respond(self) -> None:
        """Fetch an Ollama reply for pending text and stream it as speech via RTP.

        On cancellation (human started speaking) the partial user turn is
        removed from the chat history so the history stays consistent.
        """
        self._messages.append(
            {"role": "user", "content": "\n".join(self._pending_text)}
        )
        self._pending_text.clear()
        try:
            response = await ollama.AsyncClient().chat(
                model=self.ollama_model,
                messages=self._messages,
            )
            reply = (response.message.content or "").encode("ascii", "ignore").decode()
            self._messages.append({"role": "assistant", "content": reply})
            logger.info("Agent reply: %r", reply)
            await self._send_speech(reply)
        except asyncio.CancelledError:
            # Remove the partial user turn so history stays consistent.
            if self._messages and self._messages[-1]["role"] == "user":
                self._messages.pop()
            raise
        except Exception:
            logger.exception("Error while generating agent response")

    async def _send_speech(self, text: str) -> None:
        """Stream synthesised speech from Pocket TTS and send via RTP.

        Yields audio chunks from
        :meth:`~pocket_tts.TTSModel.generate_audio_stream` as soon as they
        are decoded, enabling low-latency real-time delivery to the caller.

        Args:
            text: Text to synthesise and transmit.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        def _generate() -> None:
            for chunk in self._tts_instance.generate_audio_stream(
                self._voice_state,
                text,  # type: ignore[too-many-positional-arguments]
            ):
                asyncio.run_coroutine_threadsafe(
                    queue.put(chunk.numpy()), loop
                ).result()
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        future = loop.run_in_executor(None, _generate)
        while (tts_chunk := await queue.get()) is not None:
            resampled = self._resample(
                tts_chunk, self._tts_instance.sample_rate, self._rtp_sample_rate
            )
            await self._send_rtp_audio(resampled)
        await future
