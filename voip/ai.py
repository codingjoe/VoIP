"""AI-powered call handlers for RTP streams.

This module provides :class:`WhisperCall`, which transcribes decoded audio
with faster-whisper, and :class:`AgentCall`, which extends it with an
Ollama-powered response loop and Pocket TTS voice synthesis.

Requires the ``ai`` extra: ``pip install voip[ai]``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import secrets
import struct
from typing import Any, ClassVar

import numpy as np
import ollama
from faster_whisper import WhisperModel
from pocket_tts import TTSModel

from voip.audio import AudioCall, SAMPLE_RATE
from voip.rtp import RTPPayloadType

__all__ = ["AgentCall", "WhisperCall"]

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class WhisperCall(AudioCall):
    """RTP call handler that transcribes audio with faster-whisper.

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
        """Transcribe decoded audio and deliver non-empty text to the handler."""
        loop = asyncio.get_running_loop()
        logger.debug(
            "Transcribing %d samples (%.1f s)",
            len(audio),
            len(audio) / SAMPLE_RATE,
        )
        try:
            raw = await loop.run_in_executor(None, self._run_transcription, audio)
            if text := raw.strip():
                self.transcription_received(text)
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
            text: Transcribed text for this audio chunk (already stripped).
        """


@dataclasses.dataclass(kw_only=True)
class AgentCall(WhisperCall):
    """RTP call handler that responds to caller speech using Ollama and Pocket TTS.

    Extends :class:`WhisperCall` by feeding each transcription to an Ollama
    language model, then synthesising the reply as speech with Pocket TTS
    and streaming it back to the caller via RTP.

    Chat history is maintained across turns so the language model can follow
    the conversation.  A built-in system prompt informs the model that it is
    on a phone call.

    To share the TTS model across multiple calls pass a pre-loaded
    :class:`~pocket_tts.TTSModel`::

        shared_tts = TTSModel.load_model()
        AgentCall(rtp=..., sip=..., tts_model=shared_tts)
    """

    _SYSTEM_PROMPT: ClassVar[str] = (
        "You are a helpful voice assistant on a phone call. "
        "Keep your answers brief and conversational."
    )
    #: PCMU target sample rate (G.711 µ-law, RFC 3551).
    _PCMU_SAMPLE_RATE: ClassVar[int] = 8000
    #: RTP payload samples per packet (20 ms at 8 kHz).
    _RTP_CHUNK_SAMPLES: ClassVar[int] = 160

    #: Ollama model name for generating replies.
    ollama_model: str = dataclasses.field(default="llama3")
    #: Pocket TTS voice name or path to a conditioning audio file.
    voice: str = dataclasses.field(default="alba")
    #: Pre-loaded Pocket TTS model.  Pass a shared instance to avoid
    #: loading the model separately for each call.
    tts_model: TTSModel | None = dataclasses.field(default=None)

    _tts_instance: TTSModel = dataclasses.field(init=False, repr=False)
    _voice_state: Any = dataclasses.field(init=False, repr=False)
    _messages: list[dict] = dataclasses.field(init=False, repr=False)
    _rtp_seq: int = dataclasses.field(init=False, repr=False, default=0)
    _rtp_ts: int = dataclasses.field(init=False, repr=False, default=0)
    _rtp_ssrc: int = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._tts_instance = self.tts_model or TTSModel.load_model()
        self._voice_state = self._tts_instance.get_state_for_audio_prompt(self.voice)
        self._messages = [{"role": "system", "content": self._SYSTEM_PROMPT}]
        self._rtp_ssrc = secrets.randbits(32)

    def transcription_received(self, text: str) -> None:
        """Schedule an async Ollama→TTS→RTP response for *text*.

        Silently ignores empty strings (Whisper occasionally emits them).
        """
        if text:
            asyncio.create_task(self._respond(text))

    async def _respond(self, text: str) -> None:
        """Fetch an Ollama reply for *text* and send as speech via RTP."""
        try:
            self._messages.append({"role": "user", "content": text})
            response = await ollama.AsyncClient().chat(
                model=self.ollama_model,
                messages=self._messages,
            )
            reply = response.message.content
            self._messages.append({"role": "assistant", "content": reply})
            logger.info("Agent reply: %r", reply)
            await self._send_speech(reply)
        except asyncio.CancelledError:
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
                self._voice_state, text
            ):
                asyncio.run_coroutine_threadsafe(queue.put(chunk.numpy()), loop).result()
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        future = loop.run_in_executor(None, _generate)
        while (chunk := await queue.get()) is not None:
            self._send_rtp_audio(chunk)
        await future

    def _send_rtp_audio(self, audio: np.ndarray) -> None:
        """Encode *audio* as PCMU and transmit to the caller's RTP address.

        Looks up the caller's remote RTP address from the shared
        :class:`~voip.rtp.RealtimeTransportProtocol` call registry and
        transmits the encoded audio as 20 ms RTP packets.

        Args:
            audio: Float32 mono PCM chunk from Pocket TTS.
        """
        remote_addr = next(
            (addr for addr, call in self.rtp.calls.items() if call is self),
            None,
        )
        if remote_addr is None:
            logger.warning("No remote RTP address for this call; dropping audio")
            return
        samples = self._resample(audio, self._tts_instance.sample_rate)
        for i in range(0, len(samples), self._RTP_CHUNK_SAMPLES):
            payload = self._encode_pcmu(samples[i : i + self._RTP_CHUNK_SAMPLES])
            self.send_datagram(self._build_rtp_packet(payload), remote_addr)

    @classmethod
    def _resample(cls, audio: np.ndarray, src_rate: int) -> np.ndarray:
        """Resample *audio* from *src_rate* to :attr:`_PCMU_SAMPLE_RATE`.

        Uses linear interpolation via :func:`numpy.interp`.

        Args:
            audio: Float32 mono PCM array.
            src_rate: Sample rate of *audio* in Hz.

        Returns:
            Resampled float32 array at :attr:`_PCMU_SAMPLE_RATE` Hz.
        """
        if src_rate == cls._PCMU_SAMPLE_RATE:
            return audio
        n_out = round(len(audio) * cls._PCMU_SAMPLE_RATE / src_rate)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    @staticmethod
    def _encode_pcmu(samples: np.ndarray) -> bytes:
        """Encode float32 PCM samples to G.711 µ-law (PCMU) bytes.

        Args:
            samples: Float32 mono PCM array in the range ``[-1, 1]``.

        Returns:
            µ-law encoded bytes, one byte per input sample.
        """
        x = np.clip(samples, -1.0, 1.0)
        x_mu = np.sign(x) * np.log1p(255.0 * np.abs(x)) / np.log1p(255.0)
        return np.floor((x_mu + 1.0) / 2.0 * 255.0 + 0.5).astype(np.uint8).tobytes()

    def _build_rtp_packet(self, payload: bytes) -> bytes:
        """Wrap *payload* in a minimal RTP header (RFC 3550 §5.1).

        Increments the sequence number and timestamp after each packet.

        Args:
            payload: Encoded audio payload bytes.

        Returns:
            RTP packet bytes ready for transmission.
        """
        header = struct.pack(
            ">BBHII",
            0x80,  # V=2, P=0, X=0, CC=0
            RTPPayloadType.PCMU,
            self._rtp_seq & 0xFFFF,
            self._rtp_ts & 0xFFFFFFFF,
            self._rtp_ssrc,
        )
        self._rtp_seq += 1
        self._rtp_ts += self._RTP_CHUNK_SAMPLES
        return header + payload
