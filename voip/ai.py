"""AI-powered call handlers for RTP streams.

This module provides [`TranscribeCall`][voip.ai.TranscribeCall], which transcribes decoded audio
with faster-whisper, and [`AgentCall`][voip.ai.AgentCall], which extends it with an
Ollama-powered response loop and Pocket TTS voice synthesis.

Requires the ``ai`` extra: ``pip install voip[ai]``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import logging
from typing import Any

import numpy as np
import ollama
from faster_whisper import WhisperModel
from pocket_tts import TTSModel

from voip.audio import VoiceActivityCall

__all__ = ["TranscribeCall", "AgentCall"]

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class TranscribeCall(VoiceActivityCall):
    """RTP call handler that transcribes audio with faster-whisper.

    Audio is decoded by [`AudioCall`][voip.audio.AudioCall] on a per-packet
    basis and delivered to [`audio_received`][voip.audio.AudioCall.audio_received],
    which applies an energy-based voice activity detector (VAD) from
    [`VoiceActivityCall`][voip.audio.VoiceActivityCall].  All audio frames
    (speech and silence) are accumulated until silence is sustained for
    `silence_gap` seconds, then the entire utterance is sent to Whisper as
    one chunk.  This avoids cutting sentences in the middle and prevents
    background microphone noise from being passed to Whisper as spurious audio.

    Override [`transcription_received`][voip.ai.TranscribeCall.transcription_received]
    to handle the resulting text:

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
    whisper_model: WhisperModel = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.model, str):
            logger.debug("Loading Whisper model %r", self.model)
            self.whisper_model = WhisperModel(self.model)
        else:
            self.whisper_model = self.model

    async def speech_buffer_ready(self, audio: np.ndarray) -> None:
        await self.transcribe(audio)

    async def transcribe(self, audio: np.ndarray) -> None:
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, self.run_transcription, audio)
        if text := raw.strip():
            self.transcription_received(text)

    def run_transcription(self, audio: np.ndarray) -> str:
        segments, _ = self.whisper_model.transcribe(audio)
        result = "".join(segment.text for segment in segments)
        logger.debug("Transcription result: %r", result)
        return result

    def transcription_received(self, text: str) -> None:
        """Handle a transcription result.  Override in subclasses.

        Args:
            text: Transcribed text for this audio chunk (already stripped).
        """


@dataclasses.dataclass(kw_only=True)
class AgentCall(TranscribeCall):
    """RTP call handler that responds to caller speech using Ollama and Pocket TTS.

    Extends [`TranscribeCall`][voip.ai.TranscribeCall] by feeding each
    transcription to an Ollama language model, then synthesising the reply as
    speech with Pocket TTS and streaming it back to the caller via RTP.

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

    #: Ollama model name for generating replies.
    ollama_model: str = dataclasses.field(default="llama3")
    #: Pocket TTS voice name or path to a conditioning audio file.
    voice: str = dataclasses.field(default="azelma")
    #: Pre-loaded Pocket TTS model.  Pass a shared instance to avoid
    #: loading the model separately for each call.
    tts_model: TTSModel | None = dataclasses.field(default=None)

    tts_instance: TTSModel = dataclasses.field(init=False, repr=False)
    voice_state: Any = dataclasses.field(init=False, repr=False)
    messages: list[dict] = dataclasses.field(init=False, repr=False)
    pending_text: io.StringIO = dataclasses.field(
        init=False, repr=False, default_factory=io.StringIO
    )
    response_task: asyncio.Task | None = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.tts_instance = self.tts_model or TTSModel.load_model()
        self.voice_state = self.tts_instance.get_state_for_audio_prompt(self.voice)  # type: ignore[arg-type]
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
                + "\n\nYOU MUST NEVER USE NON-VERBAL CHARACTERS IN YOUR RESPONSES!",
            }
        ]
        self.response_task = None

    def transcription_received(self, text: str) -> None:
        self.pending_text.writelines((text, "\n"))
        if self.response_task is not None and not self.response_task.done():
            self.response_task.cancel()
        self.response_task = asyncio.create_task(self.respond())

    async def respond(self) -> None:
        """Fetch an Ollama reply for pending text and stream it as speech via RTP.

        On cancellation (human started speaking) the partial user turn is
        removed from the chat history so the history stays consistent.
        """
        self.messages.append({"role": "user", "content": self.pending_text.getvalue()})
        self.pending_text.seek(0)
        self.pending_text.truncate(0)
        response = await ollama.AsyncClient().chat(
            model=self.ollama_model,
            messages=self.messages,
        )
        reply = (response.message.content or "").encode("ascii", "ignore").decode()
        self.messages.append({"role": "assistant", "content": reply})

        logger.debug("Agent reply: %r", reply)
        await self.send_speech(reply)

    async def send_speech(self, text: str) -> None:
        """Stream synthesised speech from Pocket TTS and send via RTP.

        Yields audio chunks from
        `TTSModel.generate_audio_stream` as soon as they are decoded,
        enabling low-latency real-time delivery to the caller.

        Args:
            text: Text to synthesise and transmit.
        """
        audio = self.tts_instance.generate_audio(
            self.voice_state,
            text,  # type: ignore[too-many-positional-arguments]
        )
        await self.send_rtp_audio(
            self.resample(
                audio.numpy(), self.tts_instance.sample_rate, self.codec.sample_rate_hz
            )
        )
