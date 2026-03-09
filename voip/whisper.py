"""Whisper-based transcription for Opus-encoded incoming SIP calls."""

from __future__ import annotations

import asyncio
import logging
import os
import struct
import subprocess

import ffmpeg
import numpy as np

import whisper

from .call import IncomingCall

__all__ = ["WhisperCall"]

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
    vendor = b"libsip"
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


class WhisperCall(IncomingCall):
    """Inbound SIP call that decodes Opus audio and transcribes it with OpenAI Whisper."""

    #: Opus clock rate (Hz) as specified by RFC 7587 §4.
    opus_sample_rate = 48000
    #: Opus frame size in samples for a standard 20 ms frame at 48 kHz.
    opus_frame_size = 960
    #: Audio buffered (in seconds) before each transcription is triggered.
    chunk_duration = 30

    def __init__(self, *args, model: str = "base", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        logger.debug("Loading Whisper model %r", model)
        self._whisper_model = whisper.load_model(model)
        self._opus_packets: list[bytes] = []
        self._packet_threshold = (
            self.opus_sample_rate * self.chunk_duration // self.opus_frame_size
        )
        self._transcribe_task: asyncio.Task | None = None

    def audio_received(self, data: bytes) -> None:
        """Buffer an Opus RTP payload and transcribe when the chunk threshold is reached."""
        logger.debug("RTP audio packet received: %d bytes", len(data))
        self._opus_packets.append(data)
        # asyncio is single-threaded: this check-and-create is atomic within the event loop.
        if (
            len(self._opus_packets) >= self._packet_threshold
            and self._transcribe_task is None
        ):
            self._transcribe_task = asyncio.create_task(self._transcribe_chunk())

    async def _transcribe_chunk(self) -> None:
        """Decode and transcribe the buffered Opus packets, draining all complete chunks."""
        try:
            while len(self._opus_packets) >= self._packet_threshold:
                packets = self._opus_packets[: self._packet_threshold]
                self._opus_packets = self._opus_packets[self._packet_threshold :]
                ogg_data = _build_ogg_opus(packets)
                loop = asyncio.get_running_loop()
                audio = await loop.run_in_executor(None, self._decode_opus, ogg_data)
                logger.info(
                    "Transcribing %d samples (%.1f s)",
                    len(audio),
                    len(audio) / whisper.audio.SAMPLE_RATE,
                )
                text = await loop.run_in_executor(None, self._run_transcription, audio)
                self.transcription_received(text.strip())
        finally:
            self._transcribe_task = None

    #: Maximum seconds to wait for ffmpeg to decode an audio chunk.
    decode_timeout_secs = 60

    def _decode_opus(self, ogg_data: bytes) -> np.ndarray:
        """Decode Ogg Opus data to a float32 PCM array at 16 kHz via ffmpeg.

        Requires the ``ffmpeg-python`` package and the system ``ffmpeg`` binary.
        """
        try:
            proc = (
                ffmpeg.input(
                    "pipe:0", format="ogg"
                )  # _build_ogg_opus always wraps in Ogg
                .output(
                    "pipe:1",
                    format="f32le",
                    ar=str(whisper.audio.SAMPLE_RATE),
                    ac="1",
                )
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            )
            try:
                out, err = proc.communicate(
                    input=ogg_data, timeout=self.decode_timeout_secs
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise RuntimeError(
                    f"ffmpeg decoding timed out after {self.decode_timeout_secs}s"
                )
            if proc.returncode != 0:
                raise ffmpeg.Error("ffmpeg", b"", err)
        except ffmpeg.Error as exc:
            raise RuntimeError(
                f"ffmpeg decoding failed: {getattr(exc, 'stderr', b'').decode(errors='replace')}"
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg is not installed or not on $PATH. "
                "Install it (e.g. `apt install ffmpeg` or `brew install ffmpeg`)."
            ) from exc
        return np.frombuffer(out, dtype=np.float32)

    def _run_transcription(self, audio: np.ndarray) -> str:
        """Transcribe a float32 PCM array using the Whisper model."""
        result = self._whisper_model.transcribe(audio)["text"]
        logger.debug("Transcription result: %r", result)
        return result

    def transcription_received(self, text: str) -> None:
        """Handle a transcription result. Override in subclasses."""
