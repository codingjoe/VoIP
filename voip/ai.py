"""AI-powered call handlers for RTP streams.

This module provides :class:`WhisperCall`, which transcribes decoded audio
with faster-whisper, and :class:`AgentCall`, which extends it with an
Ollama-powered response loop and Pocket TTS voice synthesis.

Requires the ``ai`` extra: ``pip install voip[ai]``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
import secrets
import struct
from typing import Any, ClassVar

import numpy as np
import ollama
from faster_whisper import WhisperModel
from pocket_tts import TTSModel

from voip.audio import AudioCall
from voip.rtp import RTPPayloadType
from voip.sdp.types import RTPPayloadFormat

__all__ = ["AgentCall", "AgentState", "TranscribeCall"]

logger = logging.getLogger(__name__)


class AgentState(enum.Enum):
    """Conversation state for :class:`AgentCall`.

    The state machine drives conversation flow: audio is collected while the
    human speaks, the LLM is queried when silence is detected, and the
    synthesised reply is streamed while the agent speaks.  Inbound speech
    during `THINKING` or `SPEAKING` cancels the current response and returns
    control to the human.
    """

    #: Human speaking; agent collects audio and buffers transcriptions.
    LISTENING = "listening"
    #: Human silent; agent is querying the LLM (Ollama).
    THINKING = "thinking"
    #: Agent transmitting TTS audio via RTP.
    SPEAKING = "speaking"


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

    Override `transcription_received` to handle the resulting text::

        class MySession(SessionInitiationProtocol):
            def call_received(self, request: Request) -> None:
                self.answer(request=request, call_class=WhisperCall)

    To share one model instance across multiple calls (recommended to avoid
    loading it multiple times) pass a pre-loaded `WhisperModel`::

        shared_model = WhisperModel("base")

        class MyCall(WhisperCall):
            model = shared_model
    """

    model: str | WhisperModel = dataclasses.field(default="kyutai/stt-1b-en_fr-trfs")
    _whisper_model: WhisperModel = dataclasses.field(init=False, repr=False)

    speech_threshold: float = dataclasses.field(default=0.5)
    silence_gap: float = dataclasses.field(default=0.5)

    _speech_buffer: list[np.ndarray] = dataclasses.field(
        init=False, repr=False, default_factory=list
    )
    _in_speech: bool = dataclasses.field(init=False, repr=False, default=False)
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
        self._in_speech = False
        self._transcription_handle = None

    def audio_received(self, *, audio: np.ndarray, rms: float) -> None:
        """Classify incoming audio as speech or silence and buffer accordingly.

        Uses RMS energy of *audio* to detect speech.  Speech audio is
        accumulated in the buffer; when silence is sustained for
        `silence_gap` seconds the buffer is flushed and sent to Whisper as a
        single utterance.

        Args:
            audio: Float32 mono PCM array at :data:`~voip.audio.SAMPLE_RATE` Hz.
        """
        self._speech_buffer.append(audio)
        if rms > self.speech_threshold:
            self._on_audio_speech()
        else:
            self._on_audio_silence()

    def _on_audio_speech(self) -> None:
        """Accumulate *audio* into the speech buffer and cancel any pending timer.

        Args:
            audio: Float32 mono PCM array classified as speech.
        """
        if self._transcription_handle is not None:
            self._transcription_handle.cancel()
            self._transcription_handle = None

    def _on_audio_silence(self) -> None:
        """Arm the transcription debounce timer when silence follows speech.

        Does nothing when no speech has been seen yet (e.g. background noise
        at call start) or when the timer is already running.
        """
        if self._transcription_handle is None:
            logger.debug("Silence detected")
            loop = asyncio.get_event_loop()
            self._transcription_handle = loop.call_later(
                self.silence_gap,
                self._flush_speech_buffer,
            )

    def _flush_speech_buffer(self) -> None:
        """Concatenate buffered speech audio and schedule async transcription.

        Resets speech state so the next utterance starts with a clean buffer.
        """
        self._transcription_handle = None
        if len(self._speech_buffer) < self._sample_rate // 50 * self.silence_gap:
            # Not enough speech to transcribe, discard buffer.
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
        "Keep your answers very brief and conversational. YOU MUST NOT USE EMOJIS."
    )
    #: PCMU target sample rate (G.711 µ-law, RFC 3551).
    _PCMU_SAMPLE_RATE: ClassVar[int] = 8000
    #: RTP payload samples per packet (20 ms at 8 kHz).
    _RTP_CHUNK_SAMPLES: ClassVar[int] = 160
    #: Wall-clock duration of one RTP packet in seconds (used for pacing).
    _RTP_PACKET_DURATION: ClassVar[float] = _RTP_CHUNK_SAMPLES / _PCMU_SAMPLE_RATE
    #: Prefer PCMU so outbound audio codec always matches what we send.
    PREFERRED_CODECS: ClassVar[list[RTPPayloadFormat]] = [
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
    _rtp_seq: int = dataclasses.field(init=False, repr=False, default=0)
    _rtp_ts: int = dataclasses.field(init=False, repr=False, default=0)
    _rtp_ssrc: int = dataclasses.field(init=False, repr=False)
    _state: AgentState = dataclasses.field(init=False, repr=False)
    _pending_text: list[str] = dataclasses.field(init=False, repr=False)
    _silence_handle: asyncio.TimerHandle | None = dataclasses.field(
        init=False, repr=False
    )
    _response_task: asyncio.Task | None = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._tts_instance = self.tts_model or TTSModel.load_model()
        self._voice_state = self._tts_instance.get_state_for_audio_prompt(self.voice)
        self._messages = [{"role": "system", "content": self._SYSTEM_PROMPT}]
        self._rtp_ssrc = secrets.randbits(32)
        self._state = AgentState.LISTENING
        self._pending_text = []
        self._silence_handle = None
        self._response_task = None

    def _on_speech(self) -> None:
        """React to a speech-energy packet: cancel any running response.

        Cancels the silence debounce timer (human is still talking) and, if
        the agent is currently `THINKING` or `SPEAKING`, cancels the active
        response task, clears buffered transcriptions and resets the
        `WhisperCall` speech buffer so the next utterance starts fresh.
        """
        if self._silence_handle is not None:
            self._silence_handle.cancel()
            self._silence_handle = None
        match self._state:
            case AgentState.THINKING | AgentState.SPEAKING:
                logger.debug(
                    "Speech detected during %s — cancelling response", self._state.value
                )
                if self._response_task is not None and not self._response_task.done():
                    self._response_task.cancel()
                    self._response_task = None
                self._pending_text.clear()
                # Reset WhisperCall transcription state so next utterance starts fresh
                if self._transcription_handle is not None:
                    self._transcription_handle.cancel()
                    self._transcription_handle = None
                self._speech_buffer.clear()
                self._in_speech = False
                self._state = AgentState.LISTENING

    def transcription_received(self, text: str) -> None:
        """Buffer *text* for the next LLM query, or respond immediately.

        Appends non-empty transcription snippets to the pending buffer.
        In the normal case the buffer is flushed by `_trigger_response` when
        the silence debounce timer fires.

        If the timer has already fired and found an empty buffer (because
        Whisper inference was still running), the transcription is sent to the
        LLM immediately so that late-arriving transcriptions are not silently
        swallowed.

        Args:
            text: Transcribed text (already stripped; empty strings are ignored).
        """
        self._pending_text.append(text)
        if self._response_task is not None and not self._response_task.done():
            self._response_task.cancel()
        self._response_task = asyncio.create_task(self._respond())

    async def _respond(self) -> None:
        """Fetch an Ollama reply for *text* and stream it as speech via RTP.

        Manages state transitions: `THINKING` while waiting for the LLM,
        `SPEAKING` while transmitting TTS audio, `LISTENING` on completion.
        On cancellation (human started speaking) the partial user turn is
        removed from the chat history so the history stays consistent.
        """
        self._messages.append(
            {"role": "user", "content": "\n".join(self._pending_text)}
        )
        self._pending_text.clear()
        try:
            print(self._messages)
            response = await ollama.AsyncClient().chat(
                model=self.ollama_model,
                messages=self._messages,
            )
            reply = response.message.content
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
        finally:
            if self._state is not AgentState.LISTENING:
                self._state = AgentState.LISTENING

    async def _send_speech(self, text: str) -> None:
        """Stream synthesised speech from Pocket TTS and send via RTP.

        Yields audio chunks from
        :meth:`~pocket_tts.TTSModel.generate_audio_stream` as soon as they
        are decoded, enabling low-latency real-time delivery to the caller.
        When :attr:`debug_audio_dir` is set the full 8 kHz PCM is also saved
        to a WAV file for offline inspection.

        Args:
            text: Text to synthesise and transmit.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        def _generate() -> None:
            for chunk in self._tts_instance.generate_audio_stream(
                self._voice_state, text
            ):
                asyncio.run_coroutine_threadsafe(
                    queue.put(chunk.numpy()), loop
                ).result()
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        future = loop.run_in_executor(None, _generate)
        while (tts_chunk := await queue.get()) is not None:
            pcm_8k = self._resample(tts_chunk, self._tts_instance.sample_rate)
            await self._send_rtp_audio(pcm_8k)
        await future

    async def _send_rtp_audio(self, audio: np.ndarray) -> None:
        """Encode *audio* (float32, 8 kHz) as PCMU and transmit to the caller via RTP.

        Looks up the caller's remote RTP address from the shared
        :class:`~voip.rtp.RealtimeTransportProtocol` call registry and
        transmits the encoded audio as 20 ms RTP packets, sleeping
        :attr:`_RTP_PACKET_DURATION` seconds between each packet so that
        packets arrive at the UAS at the correct real-time rate.  Without this
        pacing the UAS jitter buffer receives all packets simultaneously,
        plays them back too fast, and the result sounds cut-up.

        Args:
            audio: Float32 mono PCM at :attr:`_PCMU_SAMPLE_RATE` Hz.
        """
        remote_addr = next(
            (addr for addr, call in self.rtp.calls.items() if call is self),
            None,
        )
        if remote_addr is None:
            logger.warning("No remote RTP address for this call; dropping audio")
            return
        for i in range(0, len(audio), self._RTP_CHUNK_SAMPLES):
            payload = self._encode_pcmu(audio[i : i + self._RTP_CHUNK_SAMPLES])
            self.send_datagram(self._build_rtp_packet(payload), remote_addr)
            await asyncio.sleep(self._RTP_PACKET_DURATION)

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
        """Encode float32 PCM samples to G.711 µ-law (PCMU) bytes per ITU-T G.711.

        The algorithm compresses 16-bit linear PCM using logarithmic µ-law
        companding and inverts all output bits as required by G.711 §A.2.

        Args:
            samples: Float32 mono PCM array in the range ``[-1, 1]``.

        Returns:
            µ-law encoded bytes, one byte per input sample.
        """
        BIAS = 0x84  # 132 — G.711 µ-law bias constant
        CLIP = 32635  # maximum biased magnitude (14-bit saturate)
        # Scale float32 to 16-bit signed linear PCM
        pcm = np.clip(np.round(samples * 32768.0), -32768, 32767).astype(np.int32)
        # Sign bit: 0x80 for positive/zero, 0x00 for negative
        sign = np.where(pcm >= 0, 0x80, 0x00).astype(np.uint8)
        # Biased magnitude, clipped to fit in the encoding table
        biased = np.minimum(np.abs(pcm) + BIAS, CLIP)
        # Segment (chord): floor(log2(biased)) − 7, clamped to [0, 7]
        exp = np.clip(
            np.floor(np.log2(np.maximum(biased, 1))).astype(np.int32) - 7, 0, 7
        )
        # 4-bit quantisation step within the segment
        mantissa = ((biased >> (exp + 3)) & 0x0F).astype(np.uint8)
        # Compose codeword and invert all bits (G.711 §A.2 requirement)
        return (
            (~(sign | (exp.astype(np.uint8) << 4) | mantissa))
            .astype(np.uint8)
            .tobytes()
        )

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
