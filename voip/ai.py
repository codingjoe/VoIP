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
import os
import secrets
import struct
import time
import wave
from typing import Any, ClassVar

import numpy as np
import ollama
from faster_whisper import WhisperModel
from pocket_tts import TTSModel

from voip.audio import SAMPLE_RATE, AudioCall
from voip.rtp import RTPPacket, RTPPayloadType
from voip.sdp.types import RTPPayloadFormat

__all__ = ["AgentCall", "AgentState", "WhisperCall"]

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
    voice: str = dataclasses.field(default="alba")
    #: Pre-loaded Pocket TTS model.  Pass a shared instance to avoid
    #: loading the model separately for each call.
    tts_model: TTSModel | None = dataclasses.field(default=None)
    #: Directory for debug WAV files.  When set, each synthesised response
    #: is saved to ``<debug_audio_dir>/agent_<timestamp>.wav`` at 8 kHz mono
    #: int16 PCM so you can verify TTS output independently of RTP encoding.
    debug_audio_dir: str | None = dataclasses.field(default=None)
    #: Normalised RMS threshold (0–1) above which inbound audio is classified
    #: as speech.  Lower values make the VAD more sensitive.
    vad_threshold: float = dataclasses.field(default=0.02)
    #: Seconds of continuous silence required before the LLM is queried.
    silence_duration: float = dataclasses.field(default=1.0)

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

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Process an inbound RTP datagram and drive VAD state transitions.

        Estimates the energy of the payload to classify the packet as speech
        or silence.  Speech cancels any in-progress response and arms the
        LISTENING state; sustained silence arms the debounce timer that
        eventually triggers the LLM query.

        Args:
            data: Raw (decrypted) RTP datagram bytes.
            addr: Source ``(host, port)`` of the datagram.
        """
        super().datagram_received(data, addr)
        # Parse again (separately from the parent) to inspect individual packets
        # for real-time VAD without waiting for a full 5-second Whisper chunk.
        try:
            packet = RTPPacket.parse(data)
        except ValueError:
            return
        if not packet.payload:
            return
        rms = self._estimate_payload_rms(packet.payload)
        if rms > self.vad_threshold:
            self._on_speech()
        else:
            self._on_silence()

    def _on_speech(self) -> None:
        """React to a speech-energy packet: cancel any running response.

        Cancels the silence debounce timer (human is still talking) and, if
        the agent is currently `THINKING` or `SPEAKING`, cancels the active
        response task and clears buffered transcriptions so the next
        silence-triggered response starts with a clean slate.
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
                self._state = AgentState.LISTENING

    def _on_silence(self) -> None:
        """React to a silence-energy packet: arm the debounce timer.

        Only arms the timer when in `LISTENING` state and no timer is already
        running.  The timer calls `_trigger_response` after
        `silence_duration` seconds of continuous silence.
        """
        if self._state is not AgentState.LISTENING:
            return
        if self._silence_handle is not None:
            return
        loop = asyncio.get_event_loop()
        self._silence_handle = loop.call_later(
            self.silence_duration, self._trigger_response
        )

    def _trigger_response(self) -> None:
        """Combine buffered transcriptions and schedule the LLM response.

        Called by the silence debounce timer.  Transitions to `THINKING` and
        creates the `_respond` task.  Does nothing when no transcriptions have
        been buffered (e.g. background noise caused a false silence transition).
        """
        self._silence_handle = None
        if not self._pending_text:
            return
        text = " ".join(self._pending_text)
        self._pending_text.clear()
        self._state = AgentState.THINKING
        self._response_task = asyncio.create_task(self._respond(text))

    def transcription_received(self, text: str) -> None:
        """Buffer *text* for the next LLM query.

        Appends non-empty transcription snippets to the pending buffer.
        The buffer is flushed and sent to the LLM when the silence debounce
        timer fires (see `_trigger_response`).

        Args:
            text: Transcribed text (already stripped; empty strings are ignored).
        """
        if text:
            self._pending_text.append(text)

    async def _respond(self, text: str) -> None:
        """Fetch an Ollama reply for *text* and stream it as speech via RTP.

        Manages state transitions: `THINKING` while waiting for the LLM,
        `SPEAKING` while transmitting TTS audio, `LISTENING` on completion.
        On cancellation (human started speaking) the partial user turn is
        removed from the chat history so the history stays consistent.
        """
        try:
            self._messages.append({"role": "user", "content": text})
            response = await ollama.AsyncClient().chat(
                model=self.ollama_model,
                messages=self._messages,
            )
            reply = response.message.content
            self._messages.append({"role": "assistant", "content": reply})
            logger.info("Agent reply: %r", reply)
            self._state = AgentState.SPEAKING
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
        debug_chunks: list[np.ndarray] = []

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
            if self.debug_audio_dir is not None:
                debug_chunks.append(pcm_8k)
            await self._send_rtp_audio(pcm_8k)
        await future

        if debug_chunks:
            full_audio = np.concatenate(debug_chunks)
            await loop.run_in_executor(None, self._save_debug_wav, full_audio)

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

    def _save_debug_wav(self, audio: np.ndarray) -> None:
        """Save *audio* as a 16-bit mono WAV file in :attr:`debug_audio_dir`.

        The filename includes a timestamp and process-unique suffix so that
        successive responses in the same call do not overwrite each other.

        Args:
            audio: Float32 mono PCM at :attr:`_PCMU_SAMPLE_RATE` Hz.
        """
        os.makedirs(self.debug_audio_dir, exist_ok=True)  # type: ignore[arg-type]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.debug_audio_dir,  # type: ignore[arg-type]
            f"agent_{timestamp}_{id(audio)}.wav",
        )
        pcm_int16 = np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._PCMU_SAMPLE_RATE)
            wf.writeframes(pcm_int16.tobytes())
        logger.debug("Saved debug audio to %s", filename)

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
    def _estimate_payload_rms(payload: bytes) -> float:
        """Estimate normalised RMS energy from a raw G.711 RTP payload.

        G.711 codecs (PCMU/PCMA) encode silence as a fixed codeword, so speech
        energy manifests as byte variance around that codeword.  Standard
        deviation over the byte values, divided by 128, gives a normalised
        proxy for RMS in the range ``[0, 1]`` that is suitable for thresholding.

        Args:
            payload: Raw RTP payload bytes from a G.711-encoded packet.

        Returns:
            Normalised energy estimate in ``[0, 1]``.
        """
        samples = np.frombuffer(payload, dtype=np.uint8).astype(np.float32)
        return float(np.std(samples) / 128.0)

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
