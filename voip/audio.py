"""Audio call handler for RTP streams.

This module provides [`AudioCall`][voip.audio.AudioCall], which buffers RTP
packets, negotiates codecs, and decodes/encodes audio using the codec
implementations in [`voip.codecs`][voip.codecs].

Requires the ``audio`` extra: ``pip install voip[audio]``.
AI-powered subclasses (Whisper transcription, Ollama agent) live in
[`voip.ai`][voip.ai] and require the ``ai`` extra.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import secrets
from typing import ClassVar

import numpy as np

import voip.codecs as codecs
from voip.codecs import RTPCodec
from voip.codecs.base import PayloadDecoder
from voip.rtp import RTPCall, RTPPacket
from voip.sdp.types import MediaDescription

__all__ = ["AudioCall", "EchoCall", "VoiceActivityCall"]


logger = logging.getLogger(__name__)


def generate_ssrc() -> int:
    """Generate a cryptographically random 32-bit SSRC for an outbound RTP stream.

    Returns:
        A random 32-bit integer suitable for use as an RTP SSRC.
    """
    return secrets.randbits(32)


@dataclasses.dataclass
class AudioCall(RTPCall):
    """RTP call handler with audio buffering, codec negotiation, decoding, and encoding.

    Codec selection is driven by `PREFERRED_CODECS`.
    Override that list in a subclass to change priority.  The selected codec
    class is stored on `codec` after `__post_init__` and used for all
    encode/decode operations.
    """

    #: Preferred codecs in priority order (highest priority first).
    #: Populated from [`voip.codecs.REGISTRY`][voip.codecs.REGISTRY] at import
    #: time; falls back to pure-NumPy codecs when the ``pyav`` extra is absent.
    PREFERRED_CODECS: ClassVar[list[type[RTPCodec]]] = [
        codecs.REGISTRY[name]
        for name in ("opus", "g722", "pcma", "pcmu")
        if name in codecs.REGISTRY
    ]

    #: Target sample rate for decoded audio delivered to `audio_received`.
    RESAMPLING_RATE_HZ: ClassVar[int] = 16000

    #: Wall-clock spacing between outbound RTP packets in seconds.
    RTP_PACKET_DURATION_SECS: ClassVar[float] = 0.02

    #: Resolved codec class for this call, set in `__post_init__`.
    codec: type[RTPCodec] = dataclasses.field(init=False, repr=False)

    #: Per-call payload decoder, set in `__post_init__`.
    #: Stateful for ADPCM codecs (e.g. G.722), stateless for others.
    payload_decoder: PayloadDecoder = dataclasses.field(init=False, repr=False)

    #: Outbound RTP sequence counter.
    rtp_sequence_number: int = dataclasses.field(init=False, repr=False, default=0)
    #: Outbound RTP timestamp counter.
    rtp_timestamp: int = dataclasses.field(init=False, repr=False, default=0)
    #: Outbound RTP synchronisation source identifier.
    rtp_ssrc: int = dataclasses.field(
        init=False, repr=False, default_factory=generate_ssrc
    )

    def __post_init__(self) -> None:
        fmt = self.media.fmt[0]
        if fmt.encoding_name is None:
            raise ValueError(f"No encoding name for payload type {fmt.payload_type}")
        self.codec = codecs.get(fmt.encoding_name)
        self.payload_decoder = self.codec.create_decoder(
            self.RESAMPLING_RATE_HZ, input_rate_hz=self.sample_rate
        )
        logger.info(
            json.dumps(
                {
                    "event": "call_started",
                    "caller": repr(self.caller),
                    "codec": fmt.encoding_name,
                    "sample_rate": fmt.sample_rate or 0,
                    "channels": fmt.channels,
                    "payload_type": fmt.payload_type,
                }
            ),
            extra={
                "caller": repr(self.caller),
                "codec": fmt.encoding_name,
                "payload_type": fmt.payload_type,
            },
        )

    @property
    def payload_type(self) -> int:
        """Negotiated RTP payload type number."""
        return self.codec.payload_type

    @property
    def sample_rate(self) -> int:
        """SDP-negotiated audio sample rate in Hz.

        Reflects the value from the remote `a=rtpmap` line.  For G.722 this
        is 8000 per RFC 3551 even though the codec runs at 16000 Hz
        internally; use `codec.sample_rate_hz` to get the actual audio rate.
        """
        return self.media.fmt[0].sample_rate or 8000

    @classmethod
    def negotiate_codec(cls, remote_media: MediaDescription) -> MediaDescription:
        """Select the best codec from the remote SDP offer.

        Iterates `PREFERRED_CODECS`
        in priority order, matching first by payload type number and then by
        encoding name for dynamic payload types.

        Args:
            remote_media: The `m=audio` section from the remote INVITE SDP.

        Returns:
            A [`MediaDescription`][voip.sdp.types.MediaDescription] with the
            chosen codec.

        Raises:
            NotImplementedError: When no offered codec is in `PREFERRED_CODECS`.
        """
        if not remote_media.fmt:
            raise NotImplementedError("Remote SDP offer contains no audio formats")

        remote_by_pt = {f.payload_type: f for f in remote_media.fmt}
        for codec in cls.PREFERRED_CODECS:
            if codec.payload_type in remote_by_pt:
                remote_fmt = remote_by_pt[codec.payload_type]
                chosen = (
                    remote_fmt
                    if remote_fmt.encoding_name
                    else codec.to_payload_format()
                )
                return MediaDescription(
                    media="audio", port=0, proto=remote_media.proto, fmt=[chosen]
                )
            for remote_fmt in remote_media.fmt:
                if (
                    remote_fmt.encoding_name is not None
                    and remote_fmt.encoding_name.lower() == codec.encoding_name
                ):
                    return MediaDescription(
                        media="audio",
                        port=0,
                        proto=remote_media.proto,
                        fmt=[remote_fmt],
                    )

        raise NotImplementedError(
            f"No supported codec found in remote offer "
            f"{[f.payload_type for f in remote_media.fmt]!r}. "
            f"Supported: {[c.encoding_name for c in cls.PREFERRED_CODECS]!r}"
        )

    def packet_received(self, packet: RTPPacket, addr: tuple[str, int]) -> None:
        """Schedule audio decoding and delivery for *packet*.

        Ignores packets with an empty payload.

        Args:
            packet: Parsed RTP packet.
            addr: Remote ``(host, port)`` the packet arrived from.
        """
        if packet.payload:
            asyncio.create_task(self.emit_audio(packet))

    async def emit_audio(self, packet: RTPPacket) -> None:
        """Decode *packet* and call [`audio_received`][voip.audio.AudioCall.audio_received].

        Args:
            packet: Parsed RTP packet whose payload will be decoded.
        """
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(None, self.decode_payload, packet.payload)
        if audio.size > 0:
            self.audio_received(
                audio=audio, rms=float(np.sqrt(np.mean(np.square(audio))))
            )

    def decode_payload(self, payload: bytes) -> np.ndarray:
        """Decode an RTP payload to float32 PCM at `RESAMPLING_RATE_HZ`.

        Delegates to `payload_decoder`, which is either a
        [`PerPacketDecoder`][voip.codecs.base.PerPacketDecoder] (for stateless
        codecs such as PCMA, PCMU, Opus) or a
        [`G722Decoder`][voip.codecs.g722.G722Decoder] (for G.722, which
        preserves ADPCM predictor state across consecutive packets).

        Args:
            payload: Raw RTP payload bytes.

        Returns:
            Float32 mono PCM array at `RESAMPLING_RATE_HZ` Hz.
        """
        return self.payload_decoder.decode(payload)

    def audio_received(self, *, audio: np.ndarray, rms: float) -> None:
        """Handle decoded audio.  Override in subclasses.

        Args:
            audio: Float32 mono PCM array at `RESAMPLING_RATE_HZ` Hz.
            rms: Root mean square of the decoded PCM, as a proxy for signal
                strength.
        """

    async def send_rtp_audio(self, audio: np.ndarray) -> None:
        """Encode *audio* with the negotiated codec and transmit via RTP.

        Looks up the caller's remote RTP address from the shared
        [`RealtimeTransportProtocol`][voip.rtp.RealtimeTransportProtocol] call
        registry and transmits encoded audio as 20 ms RTP packets, sleeping
        `RTP_PACKET_DURATION_SECS` between each packet.

        Args:
            audio: Float32 mono PCM at `codec.sample_rate_hz` Hz.
        """
        remote_addr = next(
            (addr for addr, call in self.rtp.calls.items() if call is self),
            None,
        )
        if remote_addr is None:
            logger.warning("No remote RTP address for this call; dropping audio")
            return
        for payload in self.codec.packetize(audio):
            self.send_packet(self.next_rtp_packet(payload), remote_addr)
            await asyncio.sleep(self.RTP_PACKET_DURATION_SECS)

    def next_rtp_packet(self, payload: bytes) -> RTPPacket:
        """Create the next outbound RTP packet, incrementing sequence and timestamp.

        Args:
            payload: Encoded audio payload bytes.

        Returns:
            RTP packet ready for transmission.
        """
        packet = RTPPacket(
            payload_type=self.codec.payload_type,
            sequence_number=self.rtp_sequence_number & 0xFFFF,
            timestamp=self.rtp_timestamp & 0xFFFFFFFF,
            ssrc=self.rtp_ssrc,
            payload=payload,
        )
        self.rtp_sequence_number += 1
        self.rtp_timestamp += self.codec.timestamp_increment
        return packet

    @classmethod
    def resample(
        cls, audio: np.ndarray, source_rate_hz: int, destination_rate_hz: int
    ) -> np.ndarray:
        """Resample *audio* from *source_rate_hz* to *destination_rate_hz*.

        Delegates to [`RTPCodec.resample`][voip.codecs.base.RTPCodec.resample].

        Args:
            audio: Float32 mono PCM array.
            source_rate_hz: Sample rate of *audio* in Hz.
            destination_rate_hz: Target sample rate in Hz.

        Returns:
            Resampled float32 array at *destination_rate_hz* Hz.
        """
        return RTPCodec.resample(audio, source_rate_hz, destination_rate_hz)


@dataclasses.dataclass(kw_only=True)
class VoiceActivityCall(AudioCall):
    """AudioCall with energy-based voice activity detection (VAD) and speech buffering.

    Accumulates audio frames into `speech_buffer` based on the result of
    [`collect_audio`][voip.audio.VoiceActivityCall.collect_audio].  A debounce
    timer is armed on silence and fires
    [`flush_speech_buffer`][voip.audio.VoiceActivityCall.flush_speech_buffer]
    after `silence_gap` seconds of sustained quiet.  Subclasses implement
    [`speech_buffer_ready`][voip.audio.VoiceActivityCall.speech_buffer_ready]
    to handle the buffered utterance.

    Override [`collect_audio`][voip.audio.VoiceActivityCall.collect_audio] to
    change which frames are accumulated.  The default implementation buffers
    only speech frames (RMS above `speech_threshold`).  To buffer all frames
    (e.g. for transcription that needs the full utterance including silent
    pauses), override to always return `True`.

    Attributes:
        speech_threshold: RMS level below which audio is treated as silence.
        silence_gap: Seconds of sustained silence required to flush the buffer.
    """

    speech_threshold: float = dataclasses.field(default=0.001)
    silence_gap: float = dataclasses.field(default=0.5)

    speech_buffer: list[np.ndarray] = dataclasses.field(
        init=False, repr=False, default_factory=list
    )
    silence_handle: asyncio.TimerHandle | None = dataclasses.field(
        init=False, repr=False, default=None
    )

    def audio_received(self, *, audio: np.ndarray, rms: float) -> None:
        if self.collect_audio(audio, rms):
            self.speech_buffer.append(audio)
        if rms > self.speech_threshold:
            self.on_audio_speech()
        else:
            self.on_audio_silence()

    def collect_audio(self, audio: np.ndarray, rms: float) -> bool:
        """Return whether to buffer this audio frame.

        The default implementation buffers speech frames only (RMS above
        `speech_threshold`).  Override to change the buffering strategy.

        Args:
            audio: Decoded float32 PCM frame.
            rms: Root mean square of *audio*.

        Returns:
            `True` when the frame should be appended to `speech_buffer`.
        """
        return rms > self.speech_threshold

    def on_audio_speech(self) -> None:
        """Cancel any pending silence timer when speech is detected."""
        if self.silence_handle is not None:
            self.silence_handle.cancel()
            self.silence_handle = None

    def on_audio_silence(self) -> None:
        """Arm the silence debounce timer when speech is buffered."""
        if self.silence_handle is None and self.speech_buffer:
            loop = asyncio.get_running_loop()
            self.silence_handle = loop.call_later(
                self.silence_gap,
                self.flush_speech_buffer,
            )

    def flush_speech_buffer(self) -> None:
        """Concatenate buffered audio and schedule [`speech_buffer_ready`][voip.audio.VoiceActivityCall.speech_buffer_ready].

        Resets speech state so the next utterance starts with a clean buffer.
        """
        self.silence_handle = None
        if not self.speech_buffer:
            return
        audio = np.concatenate(self.speech_buffer)
        self.speech_buffer.clear()
        asyncio.create_task(self.speech_buffer_ready(audio))

    async def speech_buffer_ready(self, audio: np.ndarray) -> None:
        """Handle the flushed speech buffer.  Override in subclasses.

        This base implementation is a no-op.  Subclasses must override this
        method to process the buffered utterance (e.g. echo it back, transcribe
        it, etc.).

        Args:
            audio: Float32 mono PCM array at `RESAMPLING_RATE_HZ` Hz
                containing the full buffered utterance.
        """


@dataclasses.dataclass(kw_only=True)
class EchoCall(VoiceActivityCall):
    """RTP call handler that echoes the caller's speech back after they finish speaking.

    Accumulates speech audio frames (RMS above `speech_threshold`) via the
    [`VoiceActivityCall`][voip.audio.VoiceActivityCall] VAD machinery and
    replays them once a sustained silence lasting `silence_gap` seconds is
    detected.  This gives the caller a natural echo of their own voice,
    useful for network latency testing and call-flow demonstrations.

    Example:
        ```python
        class MySession(SessionInitiationProtocol):
            def call_received(self, request: Request) -> None:
                self.answer(request=request, call_class=EchoCall)
        ```
    """

    async def speech_buffer_ready(self, audio: np.ndarray) -> None:
        """Resample and transmit buffered speech audio back to the caller.

        Args:
            audio: Float32 mono PCM array at `RESAMPLING_RATE_HZ` Hz.
        """
        resampled = self.resample(
            audio, self.RESAMPLING_RATE_HZ, self.codec.sample_rate_hz
        )
        await self.send_rtp_audio(resampled)
