"""Audio call handler for RTP streams.

This module provides :class:`AudioCall`, which buffers RTP packets, negotiates
codecs, and decodes raw audio payloads (Opus, G.722, PCMA, PCMU) to float32
PCM via PyAV.

Requires the ``audio`` extra: ``pip install voip[audio]``.
AI-powered subclasses (Whisper transcription, Ollama agent) live in
:mod:`voip.ai` and require the ``ai`` extra.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import logging
import os
import struct
from typing import ClassVar

import av
import numpy as np

from voip.call import Call
from voip.rtp import RTPPacket, RTPPayloadType
from voip.sdp.types import MediaDescription, RTPPayloadFormat

__all__ = ["AudioCall", "JitterBuffer"]

#: Native sample rate expected by Whisper models.
SAMPLE_RATE = 16000

#: Default playout depth for the jitter buffer (packets, ≈ 40 ms at 20 ms/packet).
JITTER_BUFFER_DEPTH = 2

logger = logging.getLogger(__name__)


def _ogg_crc32(data: bytes) -> int:
    """Compute an Ogg CRC32 checksum (polynomial 0x04C11DB7)."""
    crc = 0
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = (crc << 1) ^ (0x04C11DB7 if crc & 0x80000000 else 0)
    return crc & 0xFFFFFFFF


def _ogg_page(
    header_type: int,
    granule_position: int,
    serial_number: int,
    sequence_number: int,
    packets: list[bytes],
) -> bytes:
    """Build a single Ogg page (RFC 3533)."""
    lacing: list[int] = []
    for packet in packets:
        remaining = len(packet)
        while remaining >= 255:
            lacing.append(255)
            remaining -= 255
        lacing.append(remaining)
    header = struct.pack(
        "<4sBBqIIIB",
        b"OggS",
        0,  # stream structure version
        header_type,
        granule_position,
        serial_number,
        sequence_number,
        0,  # CRC placeholder
        len(lacing),
    ) + bytes(lacing)
    page = header + b"".join(packets)
    return page[:22] + struct.pack("<I", _ogg_crc32(page)) + page[26:]


def _build_ogg_opus(packet: bytes) -> bytes:
    """Wrap raw Opus RTP payloads in a minimal Ogg Opus container.

    Opus always uses 48000 Hz internally (RFC 7587 §4), so no sample-rate
    parameter is exposed.
    """
    sample_rate = 48000
    serial_number = int.from_bytes(os.urandom(4), "little")
    opus_head = struct.pack(
        "<8sBBHIhB",
        b"OpusHead",
        1,  # version
        1,  # channel count (mono)
        3840,  # pre-skip: 80 ms at 48 kHz (RFC 7587)
        sample_rate,
        0,  # output gain
        0,  # channel mapping family (mono/stereo)
    )
    vendor = b"voip"
    opus_tags = (
        struct.pack("<8sI", b"OpusTags", len(vendor))
        + vendor
        + struct.pack("<I", 0)  # zero user comments
    )
    pages = [
        _ogg_page(0x02, 0, serial_number, 0, [opus_head]),  # BOS
        _ogg_page(0x00, 0, serial_number, 1, [opus_tags]),
        _ogg_page(0x04, 0, serial_number, 2, [packet]),
    ]
    return b"".join(pages)


@dataclasses.dataclass
class JitterBuffer:
    """Reorder and hold RTP packets to absorb network jitter.

    Incoming packets are stored by their 16-bit sequence number; draining via
    :meth:`get` yields them in ascending order with transparent wraparound
    handling (sequence 0 follows 65535 per RFC 3550 §5.1).

    The sequence anchor is set lazily on the first :meth:`get` call by
    selecting the *earliest* buffered sequence number using modular arithmetic,
    so out-of-order packets that arrive during pre-buffering are emitted in the
    correct order.  Call :meth:`pause` when the playout loop drains the buffer
    so that the next burst of packets re-anchors from scratch.

    Attributes:
        depth: Minimum number of packets to accumulate before :attr:`ready`
            becomes ``True`` (the playout delay expressed in packet units).
    """

    depth: int = JITTER_BUFFER_DEPTH
    _packets: dict[int, RTPPacket] = dataclasses.field(init=False, default_factory=dict)
    _next_seq: int | None = dataclasses.field(init=False, default=None)

    def put(self, packet: RTPPacket) -> None:
        """Store *packet* in the buffer.

        Args:
            packet: Parsed RTP packet to enqueue.
        """
        self._packets[packet.sequence_number] = packet

    def _count_after(self, seq: int) -> int:
        """Count how many buffered packets follow *seq* in 16-bit circular order."""
        return sum(((other - seq) & 0xFFFF) < 0x8000 for other in self._packets)

    def _anchor_sequence(self) -> int:
        """Return the earliest buffered sequence number using wraparound-aware comparison."""
        return max(self._packets, key=self._count_after)

    @property
    def ready(self) -> bool:
        """Return ``True`` when at least :attr:`depth` packets have accumulated."""
        return len(self._packets) >= self.depth

    @property
    def empty(self) -> bool:
        """Return ``True`` when no packets are buffered."""
        return not self._packets

    def get(self) -> RTPPacket | None:
        """Remove and return the next in-order packet, or ``None`` if missing.

        On the first call after construction or :meth:`pause`, the sequence
        anchor is set to the earliest sequence number currently in the buffer.
        Subsequent calls advance the counter by one; a missing slot returns
        ``None`` so the playout loop can emit silence without stalling.

        Returns:
            The next :class:`~voip.rtp.RTPPacket` in sequence order, or
            ``None`` when that slot is missing (packet loss).
        """
        if not self._packets:
            return None
        if self._next_seq is None:
            self._next_seq = self._anchor_sequence()
        packet = self._packets.pop(self._next_seq, None)
        self._next_seq = (self._next_seq + 1) & 0xFFFF
        return packet

    def pause(self) -> None:
        """Reset the sequence anchor so the next burst re-anchors from scratch.

        Call this when the playout loop drains the buffer to avoid emitting
        spurious ``None`` slots if the next burst begins at a different
        sequence number (e.g. after a call pause or a sender restart).
        """
        self._next_seq = None


@dataclasses.dataclass
class AudioCall(Call):
    """RTP call handler with audio buffering, codec negotiation, and decoding."""

    #: Preferred codecs in priority order (highest first).
    PREFERRED_CODECS: ClassVar[list[RTPPayloadFormat]] = [
        RTPPayloadFormat(
            payload_type=RTPPayloadType.OPUS,
            encoding_name="opus",
            sample_rate=48000,
            channels=2,
        ),
        RTPPayloadFormat(payload_type=RTPPayloadType.G722),
        RTPPayloadFormat(payload_type=RTPPayloadType.PCMA),
        RTPPayloadFormat(payload_type=RTPPayloadType.PCMU),
    ]

    #: Playout depth for the jitter buffer in packet units (≈ 40 ms at 20 ms/packet).
    jitter_buffer_depth: ClassVar[int] = JITTER_BUFFER_DEPTH

    _encoding_name: str = dataclasses.field(init=False, repr=False)
    _payload_type: int = dataclasses.field(init=False, default=0, repr=False)
    _sample_rate: int = dataclasses.field(init=False, default=8000, repr=False)
    _frame_duration_s: float = dataclasses.field(init=False, repr=False)
    _jitter_buffer: JitterBuffer = dataclasses.field(init=False, repr=False)
    _playout_task: asyncio.Task[None] | None = dataclasses.field(
        init=False, default=None, repr=False
    )

    def __post_init__(self) -> None:
        fmt = self.media.fmt[0]
        self._encoding_name = fmt.encoding_name.lower()
        self._payload_type = fmt.payload_type
        self._sample_rate = fmt.sample_rate or 8000
        self._frame_duration_s = fmt.frame_size / self._sample_rate
        self._jitter_buffer = JitterBuffer(depth=self.jitter_buffer_depth)
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
        return self._payload_type

    @property
    def sample_rate(self) -> int:
        """Negotiated audio sample rate in Hz."""
        return self._sample_rate

    @classmethod
    def negotiate_codec(cls, remote_media: MediaDescription) -> MediaDescription:
        """Select the best codec from the remote SDP offer.

        Iterates :attr:`PREFERRED_CODECS` in priority order, matching by
        payload type or encoding name.

        Args:
            remote_media: The ``m=audio`` section from the remote INVITE SDP.

        Returns:
            A :class:`~voip.sdp.types.MediaDescription` with the chosen codec.

        Raises:
            NotImplementedError: When no offered codec is in
                :attr:`PREFERRED_CODECS`.
        """
        if not remote_media.fmt:
            raise NotImplementedError("Remote SDP offer contains no audio formats")

        remote_fmts = {f.payload_type for f in remote_media.fmt}
        for preferred in cls.PREFERRED_CODECS:
            if preferred.payload_type in remote_fmts:
                remote_fmt = remote_media.get_format(preferred.payload_type)
                codec = (
                    remote_fmt if remote_fmt and remote_fmt.encoding_name else preferred
                )
                return MediaDescription(
                    media="audio", port=0, proto=remote_media.proto, fmt=[codec]
                )
            for remote_fmt in remote_media.fmt:
                if (
                    remote_fmt.encoding_name is not None
                    and remote_fmt.encoding_name.lower()
                    == preferred.encoding_name.lower()
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

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Add an incoming RTP datagram to the jitter buffer.

        Starts (or restarts) the playout loop when the buffer transitions from
        idle to active.  The playout loop holds packets for at least
        :attr:`jitter_buffer_depth` frame intervals before emitting audio,
        absorbing network jitter and reordering out-of-sequence packets.
        """
        try:
            packet = RTPPacket.parse(data)
        except ValueError:
            return
        if not packet.payload:
            return
        self._jitter_buffer.put(packet)
        if self._playout_task is None or self._playout_task.done():
            self._playout_task = asyncio.create_task(self._playout_loop())

    async def _playout_loop(self) -> None:
        """Pre-buffer then drain the jitter buffer at the codec frame interval.

        Sleeps for :attr:`_frame_duration_s` per iteration, matching the 20 ms
        packetisation interval common to G.722, PCMA, PCMU, and Opus.  Exits
        when the buffer drains so that :meth:`datagram_received` can restart it
        cleanly on the next burst of packets.
        """
        while not self._jitter_buffer.ready:
            await asyncio.sleep(self._frame_duration_s)
        while True:
            await asyncio.sleep(self._frame_duration_s)
            packet = self._jitter_buffer.get()
            if packet is not None:
                await self._emit_audio(packet)
            if self._jitter_buffer.empty:
                self._jitter_buffer.pause()
                return

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

    async def _emit_audio(self, packet: RTPPacket) -> None:
        """Decode *raw_packets* and call :meth:`audio_received` with the result."""
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(None, self._decode_raw, packet.payload)
        if audio.size > 0:
            self.audio_received(
                audio=audio, rms=self._estimate_payload_rms(packet.payload)
            )

    def _decode_raw(self, packet: bytes) -> np.ndarray:
        """Decode raw RTP payloads to a float32 PCM array at :data:`SAMPLE_RATE` Hz.

        The codec is identified from the negotiated :attr:`media` encoding name.

        Args:
            packet: Raw RTP payload bytes for one buffered chunk.

        Returns:
            Float32 mono PCM array resampled to :data:`SAMPLE_RATE` Hz.
        """
        match self._encoding_name:
            case "opus":
                return self._decode_via_av(
                    _build_ogg_opus(packet),
                    input_format="ogg",
                    input_sample_rate=None,
                )
            case "g722":
                return self._decode_via_av(
                    packet,
                    input_format="g722",
                    input_sample_rate=self.sample_rate,
                )
            case "pcma":
                return self._decode_via_av(
                    packet,
                    input_format="alaw",
                    input_sample_rate=self.sample_rate,
                )
            case "pcmu":
                return self._decode_via_av(
                    packet,
                    input_format="mulaw",
                    input_sample_rate=self.sample_rate,
                )

    def _decode_via_av(
        self,
        data: bytes,
        input_format: str,
        input_sample_rate: int | None,
    ) -> np.ndarray:
        """Decode audio bytes via PyAV into float32 PCM at :data:`SAMPLE_RATE` Hz.

        Args:
            data: Raw audio bytes in the codec's wire format.
            input_format: PyAV format string (e.g. ``"ogg"``, ``"alaw"``).
            input_sample_rate: Clock rate to pass to the decoder, or ``None``
                for self-describing formats like Ogg.

        Returns:
            Float32 mono PCM array at :data:`SAMPLE_RATE` Hz.
        """
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=SAMPLE_RATE
        )
        frames: list[np.ndarray] = []
        with av.open(
            io.BytesIO(data),
            format=input_format,
            options=(
                {"sample_rate": str(input_sample_rate)}
                if input_sample_rate is not None
                else {}
            ),
        ) as container:
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    frames.append(resampled.to_ndarray().flatten())
        for resampled in resampler.resample(None):
            frames.append(resampled.to_ndarray().flatten())
        return np.concatenate(frames) if frames else np.array([], dtype=np.float32)

    def audio_received(self, *, audio: np.ndarray, rms: float) -> None:
        """Handle decoded audio.  Override in subclasses.

        Args:
            audio: Float32 mono PCM array at :data:`SAMPLE_RATE` Hz.
            rms: Estimated root mean square of the raw RTP payload bytes, as a
                proxy for signal strength.
        """
