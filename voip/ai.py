"""AI-powered call handlers for RTP streams.

This module provides [`TranscribeCall`][voip.ai.TranscribeCall], which transcribes decoded audio
with faster-whisper, and [`AgentCall`][voip.ai.AgentCall], which extends it with an
Ollama-powered response loop and Pocket TTS voice synthesis.

Requires the ``ai`` extra: ``pip install voip[ai]``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import logging
import re
import typing

import numpy as np
import ollama
from faster_whisper import WhisperModel
from pocket_tts import TTSModel

from voip.audio import VoiceActivityCall

if typing.TYPE_CHECKING:
    import pathlib

    import torch

__all__ = ["TranscribeCall", "AgentCall"]

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True)
class TranscribeCall(VoiceActivityCall):
    """Transcribe incoming call audio.

    Audio is decoded by [`AudioCall`][voip.audio.AudioCall] on a per-packet
    basis and delivered to [`audio_received`][voip.audio.AudioCall.audio_received],
    which applies an energy-based voice activity detector (VAD) from
    [`VoiceActivityCall`][voip.audio.VoiceActivityCall].  All audio frames
    (speech and silence) are accumulated until silence is sustained for
    `silence_gap` seconds, then the entire utterance is sent to Whisper as
    one chunk.  This avoids cutting sentences in the middle and prevents
    background microphone noise from being passed to Whisper as spurious audio.

    Example:
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

    Args:
        stt_model: Whisper model to use for transcription.  Defaults to "base".

    """

    stt_model: WhisperModel = dataclasses.field(
        default_factory=lambda: WhisperModel("base")
    )

    async def voice_received(self, audio: np.ndarray) -> None:
        await self.transcribe(audio)

    async def transcribe(self, audio: np.ndarray) -> None:
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, self.run_transcription, audio)
        if text := raw.strip():
            self.transcription_received(text)

    def run_transcription(self, audio: np.ndarray) -> str:
        segments, _ = self.stt_model.transcribe(audio)
        result = "".join(segment.text for segment in segments)
        logger.debug("Transcription result: %r", result)
        return result

    def transcription_received(self, text: str) -> None:
        """Handle a transcription result.  Override in subclasses.

        Args:
            text: Transcribed text for this audio chunk (already stripped).
        """


@dataclasses.dataclass(kw_only=True, slots=True)
class AgentCall(TranscribeCall):
    """
    Respond to caller voice inputs with voice responses.

    Uses Ollama to generate responses to transcribed
    text and Pocket TTS to synthesize voice replies.

    Args:
        system_prompt: Prompt to guide the language model.
        llm_model: Ollama model to use for text generation.
        tts_model: Pocket TTS model to use for voice synthesis.
        voice: Voice to use for synthesis.
        audio_interrupt_duration: Time you have to talk over the agent to interrupt the outbound audio.
    """

    system_prompt: str = (
        "You are a person on a phone call."
        " Keep your answers very brief and conversational."
        " YOU MUST NEVER USE NON-VERBAL CHARACTERS IN YOUR RESPONSES!"
    )
    llm_model: str = dataclasses.field(default="ministral-3")
    tts_model: TTSModel = dataclasses.field(
        default_factory=lambda: TTSModel.load_model()
    )
    voice: pathlib.Path | str | torch.Tensor = dataclasses.field(default="azelma")
    audio_interrupt_duration: datetime.timedelta = datetime.timedelta(seconds=0.75)
    _voice_state: dict[str, dict[str, torch.Tensor]] = dataclasses.field(
        init=False, repr=False
    )
    _messages: list[dict] = dataclasses.field(init=False, repr=False)
    _response_task: asyncio.Task | None = dataclasses.field(
        init=False, repr=False, default=None
    )
    _cancel_audio_handle: asyncio.Handle | None = dataclasses.field(
        init=False, repr=False, default=None
    )

    emoji_pattern: typing.ClassVar[typing.Pattern[str]] = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.tts_model = self.tts_model or TTSModel.load_model()
        self._voice_state = self.tts_model.get_state_for_audio_prompt(self.voice)
        self._messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

    def transcription_received(self, text: str) -> None:
        self.cancel_outbound_audio()
        self._messages.append({"role": "user", "content": text})
        if self._response_task is not None and not self._response_task.done():
            self._response_task.cancel()
        self._response_task = asyncio.create_task(self.respond())

    async def respond(self) -> None:
        response = await ollama.AsyncClient().chat(
            model=self.llm_model,
            messages=self._messages,
        )
        # clean non-ascii characters from the response for TTS processing
        reply = self.emoji_pattern.sub("", response.message.content or "")
        self._messages.append({"role": "assistant", "content": reply})
        logger.debug("Agent reply: %r", reply)
        await self.send_speech(reply)

    async def send_speech(self, text: str) -> None:
        audio = self.tts_model.generate_audio(
            self._voice_state,
            text,
        )
        await self.send_audio(
            self.resample(
                audio.numpy(), self.tts_model.sample_rate, self.codec.sample_rate_hz
            )
        )

    def on_audio_speech(self) -> None:
        loop = asyncio.get_event_loop()
        if self._cancel_audio_handle is None:
            self._cancel_audio_handle = loop.call_later(
                self.audio_interrupt_duration.total_seconds(),
                self.cancel_outbound_audio,
            )
        super().on_audio_speech()

    def on_audio_silence(self) -> None:
        super().on_audio_silence()
        try:
            self._cancel_audio_handle.cancel()
        except AttributeError:
            pass
        else:
            self._cancel_audio_handle = None
