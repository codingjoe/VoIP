"""AI-powered call handlers for RTP streams.

This module provides :class:`WhisperCall`, which transcribes decoded audio
with OpenAI Whisper, and :class:`AgentCall`, which extends it with an
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

from voip.audio import SAMPLE_RATE, AudioCall

__all__ = ["AgentCall", "WhisperCall"]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class WhisperCall(AudioCall):
    """RTP call handler that transcribes audio with OpenAI Whisper.

    Audio is decoded by :class:`~voip.audio.AudioCall` and delivered as
    float32 PCM to :meth:`audio_received`, which schedules an async
    transcription job.  Override :meth:`transcription_received` to handle
    the resulting text::

        class MySession(SessionInitiationProtocol):
            def call_received(self, request: Request) -> None:
                self.answer(request=request, call_class=WhisperCall)

    To share one model instance across multiple calls (recommended to avoid
    loading it multiple times) pass a pre-loaded
    :class:`~faster_whisper.WhisperModel` as the *model* argument::

        shared_model = WhisperModel("base")

        class MyCall(WhisperCall):
            model = shared_model
    """

    #: Audio buffered (in seconds) before each transcription is triggered.
    chunk_duration: ClassVar[int] = 5

    #: Whisper model.  Either a model name string (e.g. ``"base"``,
    #: ``"small"``, ``"large-v3"``) that will be loaded on first use, or a
    #: pre-loaded :class:`~faster_whisper.WhisperModel` instance.  Pass a
    #: shared instance to avoid loading the model separately for each call.
    model: str | WhisperModel = dataclasses.field(default="kyutai/stt-1b-en_fr-trfs")
    #: Loaded Whisper model instance (not part of ``__init__``).
    _whisper_model: WhisperModel = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.model, str):
            logger.debug("Loading Whisper model %r", self.model)
            self._whisper_model = WhisperModel(self.model)
        else:
            self._whisper_model = self.model

    def audio_received(self, audio: np.ndarray) -> None:
        """Schedule async transcription for a decoded audio chunk.

        Args:
            audio: Float32 mono PCM array at :data:`~voip.audio.SAMPLE_RATE` Hz.
        """
        logger.debug(
            "Audio received: %d samples (%.1f s)", len(audio), len(audio) / SAMPLE_RATE
        )
        asyncio.create_task(self._transcribe(audio))

    async def _transcribe(self, audio: np.ndarray) -> None:
        """Transcribe decoded audio and deliver the text."""
        loop = asyncio.get_running_loop()
        logger.debug(
            "Transcribing %d samples (%.1f s)",
            len(audio),
            len(audio) / SAMPLE_RATE,
        )
        try:
            text = await loop.run_in_executor(None, self._run_transcription, audio)
            self.transcription_received(text.strip())
        except asyncio.CancelledError:
            logger.debug("Transcription task was cancelled", exc_info=True)
            raise
        except Exception:
            logger.exception("Error while transcribing audio chunk")

    def _run_transcription(self, audio: np.ndarray) -> str:
        """Transcribe a float32 PCM array using the Whisper model.

        Args:
            audio: Float32 mono PCM array at :data:`~voip.audio.SAMPLE_RATE` Hz.

        Returns:
            Concatenated transcription text from all segments.
        """
        segments, _ = self._whisper_model.transcribe(audio)
        result = "".join(segment.text for segment in segments)
        logger.debug("Transcription result: %r", result)
        return result

    def transcription_received(self, text: str) -> None:
        """Handle a transcription result.  Override in subclasses.

        Args:
            text: Transcribed text for this audio chunk.
        """


@dataclasses.dataclass
class AgentCall(WhisperCall):
    """RTP call handler that responds to speech using Ollama and Pocket TTS.

    Extends :class:`WhisperCall` by feeding each transcription to an Ollama
    language model, then synthesising the reply as speech with Pocket TTS.

    Override :meth:`reply_received` to handle the text reply, or
    :meth:`speech_received` to handle the synthesised audio::

        class MyAgent(AgentCall):
            def reply_received(self, text: str) -> None:
                print("Agent says:", text)

            def speech_received(self, audio: np.ndarray) -> None:
                send_rtp(audio)

    To share the TTS model across multiple calls pass a pre-loaded
    :class:`~pocket_tts.TTSModel` as the *tts_model* argument::

        shared_tts = TTSModel.load_model()

        class MyAgent(AgentCall):
            tts_model = shared_tts
    """

    #: Ollama model name for generating replies.
    ollama_model: str = dataclasses.field(default="llama3")
    #: Pocket TTS voice name or path to a conditioning audio file.
    voice: str = dataclasses.field(default="alba")
    #: Pre-loaded Pocket TTS model.  Pass a shared instance to avoid
    #: loading the model separately for each call.
    tts_model: TTSModel | None = dataclasses.field(default=None)

    _tts_instance: TTSModel = dataclasses.field(init=False, repr=False)
    _voice_state: Any = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._tts_instance = self.tts_model or TTSModel.load_model()
        self._voice_state = self._tts_instance.get_state_for_audio_prompt(self.voice)

    def transcription_received(self, text: str) -> None:
        """Schedule an async Ollama→TTS response for *text*."""
        asyncio.create_task(self._respond(text))

    async def _respond(self, text: str) -> None:
        """Fetch an Ollama reply for *text* and synthesise speech."""
        try:
            response = await ollama.AsyncClient().chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": text}],
            )
            reply = response.message.content
            self.reply_received(reply)
            loop = asyncio.get_running_loop()
            audio = await loop.run_in_executor(None, self._synthesize, reply)
            self.speech_received(audio)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error while generating agent response")

    def _synthesize(self, text: str) -> np.ndarray:
        """Synthesise *text* to a float32 PCM array using Pocket TTS.

        Args:
            text: Text to convert to speech.

        Returns:
            Float32 mono PCM array at :attr:`~pocket_tts.TTSModel.sample_rate` Hz.
        """
        return self._tts_instance.generate_audio(self._voice_state, text).numpy()

    def reply_received(self, text: str) -> None:
        """Handle the text reply from Ollama.  Override in subclasses.

        Args:
            text: Text reply generated by the Ollama language model.
        """

    def speech_received(self, audio: np.ndarray) -> None:
        """Handle the synthesised speech audio.  Override in subclasses.

        Args:
            audio: Float32 mono PCM array from Pocket TTS.
        """
