"""Audio call handler for RTP streams.

This module provides :class:`AudioCall`, which buffers RTP packets, negotiates
codecs, and decodes raw audio payloads (Opus, G.722, PCMA, PCMU) to float32
PCM via PyAV, and re-encodes float32 PCM for outbound transmission.

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
import secrets
import struct
import typing
from collections.abc import Iterator
from typing import ClassVar, cast

import av
import av.audio.resampler
import numpy as np

from voip.rtp import RTPCall, RTPPacket, RTPPayloadType
from voip.sdp.types import MediaDescription, RTPPayloadFormat

__all__ = ["AudioCall"]


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
class AudioCall(RTPCall):
    """RTP call handler with audio buffering, codec negotiation, decoding, and encoding."""

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

    _encoding_name: str = dataclasses.field(init=False, repr=False)
    _payload_type: int = dataclasses.field(init=False, default=0, repr=False)
    _sample_rate: int = dataclasses.field(init=False, default=8000, repr=False)
    _audio_buffer: list[bytes] = dataclasses.field(
        init=False, default_factory=list, repr=False
    )
    #: Outbound RTP sequence counter.
    _rtp_seq: int = dataclasses.field(init=False, repr=False, default=0)
    #: Outbound RTP timestamp counter.
    _rtp_ts: int = dataclasses.field(init=False, repr=False, default=0)
    #: Outbound RTP synchronisation source identifier.
    _rtp_ssrc: int = dataclasses.field(init=False, repr=False)
    #: Audio sample rate for the negotiated outbound codec in Hz.
    _rtp_sample_rate: int = dataclasses.field(init=False, repr=False)
    #: PCM samples per 20 ms RTP packet at :attr:`_rtp_sample_rate`.
    _rtp_chunk_samples: int = dataclasses.field(init=False, repr=False)
    #: RTP timestamp increment per packet (clock-rate dependent).
    _rtp_ts_increment: int = dataclasses.field(init=False, repr=False)
    #: Wall-clock duration of one RTP packet in seconds (used for pacing).
    _rtp_packet_duration: typing.ClassVar[float] = 0.02
    resampling_rate: typing.ClassVar[int] = 16000

    def __post_init__(self) -> None:
        fmt = self.media.fmt[0]
        if fmt.encoding_name is None:
            raise ValueError(f"No encoding name for payload type {fmt.payload_type}")
        self._encoding_name = fmt.encoding_name.lower()
        self._payload_type = fmt.payload_type
        self._sample_rate = fmt.sample_rate or 8000
        self._rtp_ssrc = secrets.randbits(32)
        match self._encoding_name:
            case "opus":
                self._rtp_sample_rate = 48000
                self._rtp_chunk_samples = 960
                self._rtp_ts_increment = 960
            case "g722":
                # G.722 uses an 8 kHz RTP clock despite 16 kHz audio (RFC 3551 §4.5.2).
                self._rtp_sample_rate = 16000
                self._rtp_chunk_samples = 320
                self._rtp_ts_increment = 160
            case _:  # pcmu, pcma
                self._rtp_sample_rate = 8000
                self._rtp_chunk_samples = 160
                self._rtp_ts_increment = 160
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
                    and preferred.encoding_name is not None
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

    def packet_received(self, packet: RTPPacket, addr: tuple[str, int]) -> None:
        """Schedule audio decoding and delivery for *packet*.

        Ignores packets with an empty payload.

        Args:
            packet: Parsed RTP packet.
            addr: Remote ``(host, port)`` the packet arrived from.
        """
        if packet.payload:
            asyncio.create_task(self._emit_audio(packet))

    async def _emit_audio(self, packet: RTPPacket) -> None:
        """Decode *raw_packets* and call :meth:`audio_received` with the result."""
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(None, self._decode_raw, packet.payload)
        if audio.size > 0:
            self.audio_received(
                audio=audio, rms=float(np.sqrt(np.mean(np.square(audio))))
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
            case _:
                raise NotImplementedError(
                    f"Unsupported inbound codec: {self._encoding_name!r}"
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
            format="fltp", layout="mono", rate=self.resampling_rate
        )
        frames: list[np.ndarray] = []
        with av.open(
            io.BytesIO(data),
            mode="r",
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

    async def _send_rtp_audio(self, audio: np.ndarray) -> None:
        """Encode *audio* with the negotiated codec and transmit to the caller via RTP.

        Looks up the caller's remote RTP address from the shared
        :class:`~voip.rtp.RealtimeTransportProtocol` call registry and
        transmits the encoded audio as 20 ms RTP packets, sleeping
        :attr:`_rtp_packet_duration` seconds between each packet so that
        packets arrive at the UAS at the correct real-time rate.

        Args:
            audio: Float32 mono PCM at :attr:`_rtp_sample_rate` Hz.
        """
        remote_addr = next(
            (addr for addr, call in self.rtp.calls.items() if call is self),
            None,
        )
        if remote_addr is None:
            logger.warning("No remote RTP address for this call; dropping audio")
            return
        for payload in self._packetize(audio):
            self.send_packet(self._next_rtp_packet(payload), remote_addr)
            await asyncio.sleep(self._rtp_packet_duration)

    def _packetize(self, audio: np.ndarray) -> Iterator[bytes]:
        """Encode *audio* and yield one payload bytes object per 20 ms RTP packet.

        G.722 is an ADPCM codec that maintains predictor state across samples.
        Encoding the whole buffer at once preserves that state so the decoded
        audio is continuous.  PCMU, PCMA, and Opus are stateless per-packet and
        are encoded via :meth:`_encode_audio` one chunk at a time.

        Args:
            audio: Float32 mono PCM at :attr:`_rtp_sample_rate` Hz.

        Yields:
            Encoded bytes ready to use as an RTP payload.
        """
        match self._encoding_name:
            case "g722":
                # Encode the whole buffer at once to preserve ADPCM predictor state.
                # G.722 has a fixed 2:1 sample-to-byte ratio, so _rtp_chunk_samples
                # (320) input samples map to 160 output bytes per packet.
                encoded = self._encode_via_av(audio, "g722", self._rtp_sample_rate)
                payload_size = self._rtp_chunk_samples // 2
                for i in range(0, len(encoded), payload_size):
                    yield encoded[i : i + payload_size]
            case _:
                for i in range(0, len(audio), self._rtp_chunk_samples):
                    yield self._encode_audio(audio[i : i + self._rtp_chunk_samples])

    def _next_rtp_packet(self, payload: bytes) -> RTPPacket:
        """Create the next outbound RTP packet with incremented sequence and timestamp.

        Args:
            payload: Encoded audio payload bytes.

        Returns:
            RTP packet ready for transmission.
        """
        packet = RTPPacket(
            payload_type=self._payload_type,
            sequence_number=self._rtp_seq & 0xFFFF,
            timestamp=self._rtp_ts & 0xFFFFFFFF,
            ssrc=self._rtp_ssrc,
            payload=payload,
        )
        self._rtp_seq += 1
        self._rtp_ts += self._rtp_ts_increment
        return packet

    def _encode_audio(self, samples: np.ndarray) -> bytes:
        """Encode float32 PCM to the negotiated outbound codec's bytes.

        Used for stateless per-packet encoding (PCMU, PCMA, Opus).  G.722 is
        handled separately by :meth:`_packetize` which encodes the whole TTS
        buffer at once to preserve the ADPCM predictor state.

        Args:
            samples: Float32 mono PCM array in the range ``[-1, 1]``.

        Returns:
            Encoded bytes for one RTP payload.

        Raises:
            NotImplementedError: When the negotiated codec is not supported
                for outbound encoding.
        """
        match self._encoding_name:
            case "pcmu":
                return self._encode_pcmu(samples)
            case "pcma":
                return self._encode_pcma(samples)
            case "opus":
                return self._encode_via_av(samples, "libopus", self._rtp_sample_rate)
            case _:
                raise NotImplementedError(
                    f"Unsupported outbound codec: {self._encoding_name!r}"
                )

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

    @staticmethod
    def _encode_pcma(samples: np.ndarray) -> bytes:
        """Encode float32 PCM samples to G.711 A-law (PCMA) bytes per ITU-T G.711.

        Args:
            samples: Float32 mono PCM array in the range ``[-1, 1]``.

        Returns:
            A-law encoded bytes, one byte per input sample.
        """
        a_law = 87.6  # G.711 A-law compression parameter
        pcm = np.clip(np.abs(samples), 0, 1.0)
        low = pcm < (1.0 / a_law)
        compressed = np.where(
            low,
            a_law * pcm / (1.0 + np.log(a_law)),
            (1.0 + np.log(np.maximum(a_law * pcm, 1e-10))) / (1.0 + np.log(a_law)),
            # 1e-10 prevents log(0) when pcm is exactly 0.0 in the high range
        )
        # Map to 7-bit integer value
        quantized = np.clip(np.round(compressed * 127), 0, 127).astype(np.uint8)
        sign = np.where(samples >= 0, 0x80, 0x00).astype(np.uint8)
        # XOR even bits per G.711 §A (toggle bits via 0x55)
        return ((sign | quantized) ^ 0x55).astype(np.uint8).tobytes()

    @staticmethod
    def _encode_via_av(samples: np.ndarray, codec_name: str, sample_rate: int) -> bytes:
        """Encode float32 mono PCM to raw codec bytes via PyAV.

        Args:
            samples: Float32 mono PCM array.
            codec_name: PyAV codec name (``"g722"`` or ``"libopus"``).
            sample_rate: Sample rate of *samples* in Hz.

        Returns:
            Encoded audio bytes for one RTP payload.
        """
        codec: av.AudioCodecContext = cast(
            av.AudioCodecContext, av.CodecContext.create(codec_name, "w")
        )
        codec.sample_rate = sample_rate
        codec.format = av.AudioFormat("s16")
        codec.layout = av.AudioLayout("mono")
        codec.open()
        pcm = np.clip(np.round(samples * 32768.0), -32768, 32767).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(
            pcm[np.newaxis, :], format="s16", layout="mono"
        )
        frame.sample_rate = sample_rate
        frame.pts = 0
        return b"".join(
            bytes(packet)
            for segment in (codec.encode(frame), codec.encode(None))
            for packet in segment
        )

    @classmethod
    def _resample(
        cls, audio: np.ndarray, source_rate: int, destination_rate: int
    ) -> np.ndarray:
        """Resample *audio* from *source_rate* to *destination_rate*.

        Uses linear interpolation via :func:`numpy.interp`.

        Args:
            audio: Float32 mono PCM array.
            source_rate: Sample rate of *audio* in Hz.
            destination_rate: Target sample rate in Hz.

        Returns:
            Resampled float32 array at *destination_rate* Hz.
        """
        if source_rate == destination_rate:
            return audio
        n_out = round(len(audio) * destination_rate / source_rate)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
