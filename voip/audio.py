"""Audio call handler for RTP streams.

This module provides [`AudioCall`][voip.audio.AudioCall], which buffers RTP
packets, negotiates codecs, and decodes/encodes audio using the codec
implementations in [`voip.codecs`][voip.codecs].

Requires the ``audio`` extra: ``pip install voip[audio]``.
AI-powered subclasses (Whisper transcription, Ollama agent) live in
[`voip.ai`][voip.ai] and require the ``ai`` extra.
"""

import asyncio
import dataclasses
import datetime
import logging
import secrets
from collections.abc import Iterator
from typing import ClassVar

import numpy as np
import pytest

import voip.codecs as codecs
from voip.codecs import RTPCodec
from voip.codecs.base import PayloadDecoder
from voip.rtp import RTPPacket, Session
from voip.sdp.types import MediaDescription

__all__ = ["AudioCall", "EchoCall", "VoiceActivityCall"]


logger = logging.getLogger(__name__)


def generate_ssrc() -> int:
    """Generate a cryptographically random 32-bit SSRC for an outbound RTP stream.

    Returns:
        A random 32-bit integer suitable for use as an RTP SSRC.
    """
    return secrets.randbits(32)


@dataclasses.dataclass(slots=True, kw_only=True)
class AudioCall(Session):
    """
    RTP call handler for audio calls supporting Opus, G.722, PCMA, and PCMU.

    Attributes:
        supported_codecs: Preferred codecs in priority order (highest first).
        rpt_packet_duration: Wall-clock spacing between outbound RTP packets in seconds.

    Args:
        sampling_rate_hz: Target sample rate in Hz for decoded audio
             delivered to `audio_received`.
    """

    supported_codecs: ClassVar[list[type[RTPCodec]]] = [
        codecs.REGISTRY[name]
        for name in ("opus", "g722", "pcma", "pcmu")
        if name in codecs.REGISTRY
    ]
    rpt_packet_duration: ClassVar[datetime.timedelta] = datetime.timedelta(
        milliseconds=20
    )
    sampling_rate_hz: int = 16000

    codec: type[RTPCodec] = dataclasses.field(init=False, repr=False)
    payload_decoder: PayloadDecoder = dataclasses.field(init=False, repr=False)
    rtp_sequence_number: int = dataclasses.field(init=False, repr=False, default=0)
    rtp_timestamp: int = dataclasses.field(init=False, repr=False, default=0)
    rtp_ssrc: int = dataclasses.field(
        init=False, repr=False, default_factory=generate_ssrc
    )
    send_audio_lock: asyncio.Lock = dataclasses.field(
        default_factory=asyncio.Lock,
        init=False,
    )
    outbound_handle: asyncio.TimerHandle | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        fmt = self.media.fmt[0]
        if fmt.encoding_name is None:
            raise ValueError(f"No encoding name for payload type {fmt.payload_type}")
        self.codec = codecs.get(fmt.encoding_name)
        self.payload_decoder = self.codec.create_decoder(
            self.sampling_rate_hz, input_rate_hz=self.sample_rate
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
        if not remote_media.fmt:
            raise NotImplementedError("Remote SDP offer contains no audio formats")

        remote_by_pt = {f.payload_type: f for f in remote_media.fmt}
        for codec in cls.supported_codecs:
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
            f"Supported: {[c.encoding_name for c in cls.supported_codecs]!r}"
        )

    def packet_received(self, packet: RTPPacket, addr: tuple[str, int]) -> None:
        if packet.payload:
            asyncio.create_task(self.emit_audio(packet))

    async def emit_audio(self, packet: RTPPacket) -> None:
        audio = self.decode_payload(packet.payload)
        if audio.size > 0:
            self.audio_received(audio=audio, rms=self.rms(audio))

    def decode_payload(self, payload: bytes) -> np.ndarray:
        return self.payload_decoder.decode(payload)

    def next_rtp_packet(self, payload: bytes) -> RTPPacket:
        packet = RTPPacket(
            payload_type=self.codec.payload_type,
            sequence_number=self.rtp_sequence_number,
            timestamp=self.rtp_timestamp,
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
        return RTPCodec.resample(audio, source_rate_hz, destination_rate_hz)

    @staticmethod
    def rms(audio: np.ndarray) -> float:
        """
        Calculate the Root Mean Square (RMS) of an audio signal.

        Args:
            audio: Float32 mono PCM array.

        Returns:
            RMS value as a proxy for signal strength.
        """
        return float(np.sqrt(np.mean(np.square(audio))))

    def cancel_outbound_audio(self) -> None:
        """Stop the current outbound audio while it is being sent."""
        try:
            self.outbound_handle.cancel()
        except AttributeError:
            pass
        else:
            self.outbound_handle = None

    def _dispatch_next_packet(
        self,
        packets: Iterator[bytes],
        remote_addr: tuple[str, int],
        next_send_at: float,
    ) -> None:
        try:
            payload = next(packets)
        except StopIteration:
            self.outbound_handle = None
        else:
            self.send_packet(self.next_rtp_packet(payload), remote_addr)
            duration_seconds = self.rpt_packet_duration.total_seconds()
            next_deadline = next_send_at + duration_seconds
            loop = asyncio.get_running_loop()
            self.outbound_handle = loop.call_at(
                next_deadline,
                self._dispatch_next_packet,
                packets,
                remote_addr,
                next_deadline,
            )

    async def send_audio(self, audio: np.ndarray) -> None:
        """
        Encode `audio` with the negotiated codec and transmit via RTP.

        Args:
            audio: Float32 mono PCM at `codec.sample_rate_hz` Hz.
        """
        remote_addr = next(
            (addr for addr, call in self.rtp.calls.items() if call is self),
            None,
        )
        match remote_addr:
            case None:
                logger.warning(
                    "No remote RTP address for this call; dropping audio",
                )
                return
            case _:
                pass
        async with self.send_audio_lock:
            self.cancel_outbound_audio()
            loop = asyncio.get_running_loop()
            next_send_at = loop.time()
            self._dispatch_next_packet(
                self.codec.packetize(audio),
                remote_addr,
                next_send_at,
            )

    def audio_received(self, *, audio: np.ndarray, rms: float) -> None:
        """
        Handle decoded audio. Override in subclasses.

        Args:
            audio: Float32 mono PCM array at `RESAMPLING_RATE_HZ` Hz.
            rms: Root Mean Square of the decoded PCM, as a proxy for signal strength.
        """


@pytest.mark.asyncio
async def test_send_audio_with_empty_packet_iterator_does_not_schedule_packets() -> (
    None
):
    empty_audio = np.array([], dtype=np.float32)

    class EmptyPacketCodec:
        def __init__(self) -> None:
            self.payload_type = 0
            self.timestamp_increment = 160
            self.sample_rate_hz = 8000

        def packetize(self, audio: np.ndarray) -> Iterator[bytes]:
            return iter(())

    class SingleCallRtp:
        def __init__(self, call: AudioCall, remote_addr: tuple[str, int]) -> None:
            self.calls = {remote_addr: call}

    call = object.__new__(AudioCall)
    remote_addr = ("127.0.0.1", 4000)
    codec = EmptyPacketCodec()
    send_calls: list[tuple[RTPPacket, tuple[str, int]]] = []

    def send_packet(packet: RTPPacket, addr: tuple[str, int]) -> None:
        send_calls.append((packet, addr))

    call.codec = codec
    call.rtp_sequence_number = 0
    call.rtp_timestamp = 0
    call.rtp_ssrc = 1
    call.rpt_packet_duration = datetime.timedelta(milliseconds=20)
    call.outbound_handle = None
    call.rtp = SingleCallRtp(call, remote_addr)
    call.send_audio_lock = asyncio.Lock()
    call.send_packet = send_packet

    await AudioCall.send_audio(call, empty_audio)

    assert call.outbound_handle is None
    assert send_calls == []


@dataclasses.dataclass(kw_only=True)
class VoiceActivityCall(AudioCall):
    """
    AudioCall with energy-based Voice Activity Detection (VAD) and speech buffering.

    Full utterances are buffered and passed to
    [`voice_received`][voip.audio.VoiceActivityCall.voice_received].
    Silent chunks are dropped from the audio stream.

    Override that method in subclasses to process complete speech segments
    (e.G. transcribe them, echo them back, etc.) instead of raw audio frames.

    An utterance is considered complete when the RMS of the buffered audio
    drops below `voice_rms_threshold` for at least [silence_gap] seconds.

    Full utterances with an RMS sound power below `utterances_rms_threshold`
    are discarded.

    A full utterance must be separated from the previous one by at least the
    `silence_gap` to be considered complete and passed to
    [`voice_received`][voip.audio.VoiceActivityCall.voice_received].

    Example:
        The following example shows how to use `VoiceActivityCall` to echo a caller's
        voice back to them similar to [`EchoCall`][voip.audio.EchoCall].

        ```python
        import dataclasses

        from voip.audio import VoiceActivityCall


        @dataclasses.dataclass(kw_only=True)
        class EchoCall(VoiceActivityCall):

            async def voice_received(self, audio: np.ndarray) -> None:
                resampled = self.resample(
                    audio, self.sampling_rate_hz, self.codec.sample_rate_hz
                )
                await self.send_audio(resampled)
        ```

    Args:
        voice_rms_threshold: Minimum RMS sound power voice detection.
        utterances_rms_threshold: Minimum RMS sound power for an utterance.
        silence_gap: Minimum duration of silence to consider an utterance complete.
    """

    voice_rms_threshold: float = 0.001
    utterances_rms_threshold: float = 0.01
    silence_gap: datetime.timedelta = dataclasses.field(
        default=datetime.timedelta(milliseconds=200)
    )

    _speech_buffer: np.ndarray = dataclasses.field(
        init=False, repr=False, default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    _flush_voice_buffer_handle: asyncio.TimerHandle | None = dataclasses.field(
        init=False, repr=False, default=None
    )

    def audio_received(self, *, audio: np.ndarray, rms: float) -> None:
        self._speech_buffer = np.concatenate((self._speech_buffer, audio))
        if rms > self.voice_rms_threshold:
            self.on_audio_speech()
        else:
            self.on_audio_silence()

    def on_audio_speech(self) -> None:
        if self._flush_voice_buffer_handle is not None:
            self._flush_voice_buffer_handle.cancel()
            self._flush_voice_buffer_handle = None

    def on_audio_silence(self) -> None:
        if self._flush_voice_buffer_handle is None:
            loop = asyncio.get_event_loop()
            self._flush_voice_buffer_handle = loop.call_later(
                self.silence_gap.total_seconds(),
                self.flush_voice_buffer,
            )

    def flush_voice_buffer(self) -> None:
        self._flush_voice_buffer_handle = None
        # Ensure at least one second of audio to avoid cutting words in half.
        if not (
            len(self._speech_buffer)
            < self.sampling_rate_hz * self.silence_gap.total_seconds()
            or self.rms(self._speech_buffer) < self.utterances_rms_threshold
        ):
            asyncio.create_task(self.voice_received(self._speech_buffer.copy()))
        self._speech_buffer = np.empty((0,), dtype=np.float32)

    async def voice_received(self, audio: np.ndarray) -> None:
        """Handle the flushed speech buffer.  Override in subclasses.

        This base implementation is a no-op.  Subclasses must override this
        method to process the buffered utterance (e.g. echo it back, transcribe
        it, etc.).

        Args:
            audio: Float32 mono PCM array at `RESAMPLING_RATE_HZ` Hz
                containing the full buffered utterance.
        """


@dataclasses.dataclass(kw_only=True, slots=True)
class EchoCall(VoiceActivityCall):
    """Echo the caller's speech back after they finish speaking.

    Buffers a full utterance and replays it once a sustained silence lasting
    `silence_gap` seconds is detected. This gives the caller a natural echo
    of their own voice, useful for network latency testing and call-flow
    demonstrations.

    Example:
        ```python
        class MySession(SessionInitiationProtocol):
            def call_received(self, request: Request) -> None:
                self.answer(request=request, call_class=EchoCall)
        ```
    """

    async def voice_received(self, audio: np.ndarray) -> None:
        resampled = self.resample(
            audio, self.sampling_rate_hz, self.codec.sample_rate_hz
        )
        await self.send_audio(resampled)
