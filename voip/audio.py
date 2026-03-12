"""Audio call handler and Whisper-based transcription for RTP streams.

This module provides :class:`AudioCall`, which buffers RTP packets, negotiates
codecs, and decodes raw audio payloads (Opus, G.722, PCMA, PCMU) to float32
PCM via PyAV.  :class:`WhisperCall` extends it to transcribe decoded audio
with OpenAI Whisper, keeping transcription concerns separate from codec work.

Requires the ``audio`` extra: ``pip install voip[audio]``.
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
from faster_whisper import WhisperModel

from voip.call import Call
from voip.rtp import RTPPacket, RTPPayloadType
from voip.sdp.types import MediaDescription, RTPPayloadFormat

__all__ = ["AudioCall", "WhisperCall"]

#: Native sample rate expected by Whisper models.
SAMPLE_RATE = 16000

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


def _build_ogg_opus(packets: list[bytes]) -> bytes:
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
    ]
    granule = 0
    packets_per_page = 50
    for index, batch_start in enumerate(range(0, len(packets), packets_per_page)):
        batch = packets[batch_start : batch_start + packets_per_page]
        granule += 960 * len(batch)
        is_last = batch_start + packets_per_page >= len(packets)
        header_type = 0x04 if is_last else 0x00  # EOS flag on last page
        pages.append(_ogg_page(header_type, granule, serial_number, index + 2, batch))
    return b"".join(pages)


@dataclasses.dataclass
class AudioCall(Call):
    """RTP call handler with audio buffering, codec negotiation, and decoding.

    Buffers incoming RTP packets and, when :attr:`chunk_duration` seconds of
    audio have been accumulated, decodes them to a float32 PCM array and
    delivers the result to :meth:`audio_received`.

    Subclass and override :meth:`audio_received` to process decoded audio::

        class MyCall(AudioCall):
            def audio_received(self, audio: np.ndarray) -> None:
                save_to_disk(audio)

    Override :attr:`PREFERRED_CODECS` to change the codec priority list, or
    :attr:`chunk_duration` to change the buffering window.

    Attributes:
        chunk_duration: Seconds of audio to buffer (class var; ``0`` = per-packet).
        PREFERRED_CODECS: Codec priority list for :meth:`negotiate_codec`.
    """

    #: Seconds of audio to buffer before emitting :meth:`audio_received`.
    #: ``0`` (default) emits one event per RTP packet.
    chunk_duration: ClassVar[int] = 0

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

    _payload_type: int = dataclasses.field(init=False, default=0, repr=False)
    _sample_rate: int = dataclasses.field(init=False, default=8000, repr=False)
    _audio_buffer: list[bytes] = dataclasses.field(
        init=False, default_factory=list, repr=False
    )
    _packet_threshold: int = dataclasses.field(init=False, default=1, repr=False)

    def __post_init__(self) -> None:
        frame_size = 160  # default for PCMU/PCMA (RFC 3551)
        if self.media is not None and self.media.fmt:
            fmt = self.media.fmt[0]
            self._payload_type = fmt.payload_type
            self._sample_rate = fmt.sample_rate or 8000
            logger.info(
                json.dumps(
                    {
                        "event": "call_started",
                        "caller": repr(self.caller),
                        "codec": fmt.encoding_name or "unknown",
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
            frame_size = fmt.frame_size
        self._packet_threshold = (
            self._sample_rate * self.chunk_duration // frame_size
            if self.chunk_duration
            else 1
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
                    media="audio", port=0, proto="RTP/AVP", fmt=[codec]
                )
            for remote_fmt in remote_media.fmt:
                if (
                    remote_fmt.encoding_name is not None
                    and remote_fmt.encoding_name.lower()
                    == preferred.encoding_name.lower()
                ):
                    return MediaDescription(
                        media="audio", port=0, proto="RTP/AVP", fmt=[remote_fmt]
                    )

        raise NotImplementedError(
            f"No supported codec found in remote offer "
            f"{[f.payload_type for f in remote_media.fmt]!r}. "
            f"Supported: {[c.encoding_name for c in cls.PREFERRED_CODECS]!r}"
        )

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Buffer *data* as an RTP packet; emit when the chunk threshold is reached."""
        try:
            packet = RTPPacket.parse(data)
        except ValueError:
            return
        if not packet.payload:
            return
        self._audio_buffer.append(packet.payload)
        while len(self._audio_buffer) >= self._packet_threshold:
            batch = self._audio_buffer[: self._packet_threshold]
            self._audio_buffer = self._audio_buffer[self._packet_threshold :]
            asyncio.create_task(self._emit_audio(batch))

    async def _emit_audio(self, raw_packets: list[bytes]) -> None:
        """Decode *raw_packets* and call :meth:`audio_received` with the result."""
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(None, self._decode_raw, raw_packets)
        self.audio_received(audio)

    def _decode_raw(self, raw_packets: list[bytes]) -> np.ndarray:
        """Decode raw RTP payloads to a float32 PCM array at :data:`SAMPLE_RATE` Hz.

        The codec is identified from the negotiated :attr:`media` encoding name.

        Args:
            raw_packets: Raw RTP payload bytes for one buffered chunk.

        Returns:
            Float32 mono PCM array resampled to :data:`SAMPLE_RATE` Hz.
        """
        encoding = (
            self.media.fmt[0].encoding_name if self.media and self.media.fmt else ""
        ) or ""
        match encoding.lower():
            case "opus":
                return self._decode_via_av(
                    _build_ogg_opus(raw_packets),
                    input_format="ogg",
                    input_sample_rate=None,
                )
            case "g722":
                return self._decode_via_av(
                    b"".join(raw_packets),
                    input_format="g722",
                    input_sample_rate=self.sample_rate,
                )
            case "pcma":
                return self._decode_via_av(
                    b"".join(raw_packets),
                    input_format="alaw",
                    input_sample_rate=self.sample_rate,
                )
            case "pcmu":
                return self._decode_via_av(
                    b"".join(raw_packets),
                    input_format="mulaw",
                    input_sample_rate=self.sample_rate,
                )
            case _:
                encoding_name = encoding or str(self._payload_type)
                supported = [
                    c.encoding_name for c in self.PREFERRED_CODECS if c.encoding_name
                ]
                raise NotImplementedError(
                    f"Unsupported codec: {encoding_name!r}. "
                    f"Supported: {', '.join(supported)}."
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
        try:
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
        except av.AVError as exc:
            raise RuntimeError(f"Audio decoding failed: {exc}") from exc
        return np.concatenate(frames) if frames else np.array([], dtype=np.float32)

    def audio_received(self, audio: np.ndarray) -> None:
        """Handle decoded audio.  Override in subclasses.

        Args:
            audio: Float32 mono PCM array at :data:`SAMPLE_RATE` Hz
                (16 kHz) covering :attr:`chunk_duration` seconds of audio
                (or one RTP packet when ``chunk_duration == 0``).
        """


@dataclasses.dataclass
class WhisperCall(AudioCall):
    """RTP call handler that transcribes audio with OpenAI Whisper.

    Audio is decoded by :class:`AudioCall` and delivered as float32 PCM to
    :meth:`audio_received`, which schedules an async transcription job.
    Override :meth:`transcription_received` to handle the resulting text::

        class MySession(SessionInitiationProtocol):
            def call_received(self, request: Request) -> None:
                self.answer(request=request, call_class=WhisperCall)
    """

    #: Audio buffered (in seconds) before each transcription is triggered.
    chunk_duration: ClassVar[int] = 5

    #: Whisper model size (e.g. ``"base"``, ``"small"``, ``"large-v3"``).
    model: str = dataclasses.field(default="kyutai/stt-1b-en_fr-trfs")
    #: Loaded Whisper model instance (not part of ``__init__``).
    _whisper_model: WhisperModel = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        logger.debug("Loading Whisper model %r", self.model)
        self._whisper_model = WhisperModel(self.model)

    def audio_received(self, audio: np.ndarray) -> None:
        """Schedule async transcription for a decoded audio chunk.

        Args:
            audio: Float32 mono PCM array at :data:`SAMPLE_RATE` Hz.
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
        text = await loop.run_in_executor(None, self._run_transcription, audio)
        self.transcription_received(text.strip())

    def _run_transcription(self, audio: np.ndarray) -> str:
        """Transcribe a float32 PCM array using the Whisper model.

        Args:
            audio: Float32 mono PCM array at :data:`SAMPLE_RATE` Hz.

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
