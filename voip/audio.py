"""Whisper-based transcription for RTP audio streams (Opus, G.722, PCMA, PCMU)."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import struct
from typing import ClassVar

import av
import numpy as np
from faster_whisper import WhisperModel

from voip.rtp import RealtimeTransportProtocol
from voip.sdp.types import MediaDescription

__all__ = ["WhisperCall"]

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


class WhisperCall(RealtimeTransportProtocol):
    """RTP call handler that decodes audio and transcribes it with OpenAI Whisper.

    Supports Opus, G.722, PCMA (G.711 A-law), and PCMU (G.711 µ-law) based on
    the negotiated RTP payload type. Use it as the *call_class* when answering
    calls in a SIP session::

        class MySession(SessionInitiationProtocol):
            def call_received(self, request: Request) -> None:
                self.answer(request=request, call_class=WhisperCall)
    """

    #: Audio buffered (in seconds) before each transcription is triggered.
    chunk_duration: ClassVar[int] = 5
    #: Maximum seconds to wait for audio decoding to complete.
    decode_timeout_secs: ClassVar[int] = 60

    def __init__(
        self,
        caller: str = "",
        model: str = "base",
        media: MediaDescription | None = None,
    ) -> None:
        super().__init__(caller=caller, media=media)
        logger.debug("Loading Whisper model %r", model)
        self._whisper_model = WhisperModel(model, device="auto")

    def audio_received(self, packets: list[bytes]) -> None:
        """Schedule async transcription for a buffered audio chunk."""
        logger.debug("Audio frame received: %d packets", len(packets))
        asyncio.create_task(self._transcribe_chunk(packets))

    async def _transcribe_chunk(self, packets: list[bytes]) -> None:
        """Decode and transcribe one audio chunk."""
        loop = asyncio.get_running_loop()
        try:
            audio = await loop.run_in_executor(None, self._decode_audio, packets)
            logger.debug(
                "Transcribing %d samples (%.1f s)",
                len(audio),
                len(audio) / SAMPLE_RATE,
            )
            text = await loop.run_in_executor(None, self._run_transcription, audio)
            self.transcription_received(text.strip())
        except asyncio.CancelledError:
            logger.debug("Transcription task was cancelled")
            raise
        except Exception:
            logger.exception("Error while decoding or transcribing audio chunk")

    def _decode_audio(self, packets: list[bytes]) -> np.ndarray:
        """Decode audio packets to a float32 PCM array at Whisper's sample rate.

        The codec is identified from the negotiated :attr:`media` encoding name,
        and the clock rate is taken from :attr:`sample_rate`.
        """
        encoding = (
            self.media.fmt[0].encoding_name if self.media and self.media.fmt else ""
        ) or ""
        match encoding.lower():
            case "opus":
                return self._decode_via_av(
                    _build_ogg_opus(packets),
                    input_format="ogg",
                    input_sample_rate=None,
                )
            case "g722":
                return self._decode_via_av(
                    b"".join(packets),
                    input_format="g722",
                    input_sample_rate=self.sample_rate,
                )
            case "pcma":
                return self._decode_via_av(
                    b"".join(packets),
                    input_format="alaw",
                    input_sample_rate=self.sample_rate,
                )
            case "pcmu":
                return self._decode_via_av(
                    b"".join(packets),
                    input_format="mulaw",
                    input_sample_rate=self.sample_rate,
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported codec: {encoding!r} (PT {self.payload_type}). "
                    f"Supported: opus, g722, pcma, pcmu."
                )

    def _decode_via_av(
        self,
        data: bytes,
        input_format: str,
        input_sample_rate: int | None,
    ) -> np.ndarray:
        """Decode audio data via PyAV into float32 PCM at Whisper's sample rate."""
        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="mono",
            rate=SAMPLE_RATE,
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

    def _run_transcription(self, audio: np.ndarray) -> str:
        """Transcribe a float32 PCM array using the Whisper model."""
        segments, _ = self._whisper_model.transcribe(audio)
        result = "".join(segment.text for segment in segments)
        logger.debug("Transcription result: %r", result)
        return result

    def transcription_received(self, text: str) -> None:
        """Handle a transcription result. Override in subclasses."""
