"""Opus codec implementation for RTP audio streams (RFC 7587).

The [Opus][voip.codecs.opus.Opus] class wraps raw Opus RTP payloads in a
minimal [Ogg][] container before passing them to PyAV for decoding, and
encodes float32 PCM via `libopus`.

Requires the ``pyav`` extra: ``pip install voip[pyav]``.

[Ogg]: https://wiki.xiph.org/Ogg
"""

import dataclasses
import logging
import os
import struct
from collections.abc import Iterator
from typing import ClassVar, cast

import av
import av.audio.resampler
import numpy as np

from voip.codecs.av import PyAVCodec

logger = logging.getLogger(__name__)

__all__ = ["Opus", "OpusDecoder"]


class Opus(PyAVCodec):
    """Opus audio codec ([RFC 7587][]).

    Opus is a highly flexible codec for interactive real-time speech and audio
    transmission. It uses dynamic payload type 111 and always operates at
    48 000 Hz internally.

    Incoming RTP payloads are wrapped in a minimal [Ogg][] container before
    being passed to PyAV. Outbound PCM is encoded via `libopus`.

    [RFC 7587]: https://datatracker.ietf.org/doc/html/rfc7587
    [Ogg]: https://wiki.xiph.org/Ogg
    """

    payload_type: ClassVar[int] = 111
    encoding_name: ClassVar[str] = "opus"
    sample_rate_hz: ClassVar[int] = 48000
    rtp_clock_rate_hz: ClassVar[int] = 48000
    frame_size: ClassVar[int] = 960
    timestamp_increment: ClassVar[int] = 960
    channels: ClassVar[int] = 2

    @staticmethod
    def _ogg_crc32(data: bytes) -> int:
        """Compute an Ogg CRC32 checksum (polynomial 0x04C11DB7).

        Args:
            data: Bytes to checksum.

        Returns:
            32-bit CRC value.
        """
        crc = 0
        for byte in data:
            crc ^= byte << 24
            for _ in range(8):
                crc = (crc << 1) ^ (0x04C11DB7 if crc & 0x80000000 else 0)
        return crc & 0xFFFFFFFF

    @classmethod
    def _ogg_page(
        cls,
        header_type: int,
        granule_position: int,
        serial_number: int,
        sequence_number: int,
        packets: list[bytes],
    ) -> bytes:
        """Build a single Ogg page ([RFC 3533](https://datatracker.ietf.org/doc/html/rfc3533)).

        Args:
            header_type: Page header type flags (e.g. `0x02` for BOS, `0x04` for EOS).
            granule_position: Granule position for this page.
            serial_number: Stream serial number.
            sequence_number: Page sequence number within the stream.
            packets: Packet byte strings to include in this page.

        Returns:
            Complete Ogg page bytes including CRC.
        """
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
        return page[:22] + struct.pack("<I", cls._ogg_crc32(page)) + page[26:]

    @classmethod
    def _ogg_container(cls, packet: bytes) -> bytes:
        """Wrap a raw Opus RTP payload in a minimal Ogg Opus container.

        Produces a three-page Ogg stream: BOS (OpusHead), comment
        (OpusTags), and the single data page.  Opus always uses 48 000 Hz
        internally ([RFC 7587 §4](https://datatracker.ietf.org/doc/html/rfc7587#section-4)).

        Args:
            packet: Raw Opus RTP payload bytes.

        Returns:
            Ogg Opus container bytes suitable for PyAV decoding.
        """
        serial_number = int.from_bytes(os.urandom(4), "little")
        vendor = b"voip"
        opus_head = struct.pack(
            "<8sBBHIhB",
            b"OpusHead",
            1,  # version
            cls.channels,  # channel count
            0,  # pre-skip: each RTP payload is decoded as a standalone stream
            cls.sample_rate_hz,
            0,  # output gain
            0,  # channel mapping family (mono/stereo)
        )
        opus_tags = (
            struct.pack("<8sI", b"OpusTags", len(vendor))
            + vendor
            + struct.pack("<I", 0)  # zero user comments
        )
        return b"".join(
            [
                cls._ogg_page(0x02, 0, serial_number, 0, [opus_head]),  # BOS
                cls._ogg_page(0x00, 0, serial_number, 1, [opus_tags]),
                cls._ogg_page(0x04, cls.frame_size, serial_number, 2, [packet]),
            ]
        )

    @classmethod
    def decode(
        cls,
        payload: bytes,
        output_rate_hz: int,
        *,
        input_rate_hz: int | None = None,
    ) -> np.ndarray:
        return cls.decode_pcm(cls._ogg_container(payload), "ogg", output_rate_hz)

    @classmethod
    def encode(cls, samples: np.ndarray) -> bytes:
        """Encode a single 20 ms PCM chunk to one valid Opus RTP payload.

        Uses [packetize][voip.codecs.opus.Opus.packetize] with a fresh encoder
        and returns the first encoded packet.  The payload is a Code-0
        single-frame Opus packet suitable for direct embedding in an RTP
        packet.

        Args:
            samples: Float32 mono PCM at `sample_rate_hz` Hz,
                nominally `frame_size` (960) samples.

        Returns:
            Raw Opus payload bytes for one RTP packet.
        """
        return next(cls.packetize(samples), b"")

    @classmethod
    def packetize(cls, audio: np.ndarray) -> Iterator[bytes]:
        """Encode *audio* and yield one valid Opus RTP payload per 20 ms frame.

        Creates a single `libopus` encoder context for the entire buffer so
        the encoder's internal state (VBR adaptation, noise shaping) is
        preserved across packet boundaries.  Each call to
        `av.CodecContext.encode` with exactly `frame_size` samples produces
        exactly one Code-0 Opus packet, which is valid as a standalone RTP
        payload.

        Partial last frames are zero-padded to `frame_size` samples so that
        `libopus` always receives the expected number of samples.  The encoder
        is **not** flushed after the last frame: since every chunk is already
        padded to a full frame the flush would only emit encoder look-ahead
        silence, producing an extra RTP packet and shifting the receiver's
        playback timeline by one 20 ms interval.

        Args:
            audio: Float32 mono PCM at `sample_rate_hz` Hz.

        Yields:
            Encoded Opus payload bytes, one per RTP packet.
        """
        codec = cast(av.AudioCodecContext, av.CodecContext.create("libopus", "w"))
        codec.sample_rate = cls.sample_rate_hz
        codec.format = av.AudioFormat("fltp")
        codec.layout = av.AudioLayout("mono")
        codec.open()
        for i in range(0, len(audio), cls.frame_size):
            chunk = audio[i : i + cls.frame_size].astype(np.float32)
            if len(chunk) < cls.frame_size:
                chunk = np.pad(chunk, (0, cls.frame_size - len(chunk)))
            frame = av.AudioFrame.from_ndarray(
                chunk[np.newaxis, :], format="fltp", layout="mono"
            )
            frame.sample_rate = cls.sample_rate_hz
            frame.pts = i
            yield from (bytes(pkt) for pkt in codec.encode(frame))

    @classmethod
    def create_decoder(
        cls, output_rate_hz: int, *, input_rate_hz: int | None = None
    ) -> OpusDecoder:
        """Create a stateful per-call Opus decoder.

        Returns an [OpusDecoder][voip.codecs.opus.OpusDecoder] that preserves
        the `libopus` decoder's internal MDCT overlap state across consecutive
        RTP packets.  Without this, each packet is decoded in an independent
        context, causing CELT window-boundary discontinuities every 20 ms that
        manifest as audible choppiness in echo and playback scenarios.

        The *input_rate_hz* parameter is accepted for API consistency with
        [RTPCodec.create_decoder][voip.codecs.base.RTPCodec.create_decoder]
        but is not used; Opus always decodes at `sample_rate_hz` (48 000 Hz).

        Args:
            output_rate_hz: Target PCM sample rate in Hz for decoded audio.
            input_rate_hz: Ignored.  Opus always decodes at `sample_rate_hz`.

        Returns:
            A new [OpusDecoder][voip.codecs.opus.OpusDecoder] instance.
        """
        return OpusDecoder(output_rate_hz)


@dataclasses.dataclass(slots=True)
class OpusDecoder:
    """Stateful Opus decoder that preserves `libopus` CELT state across packets.

    Creates a single persistent
    [av.CodecContext](https://pyav.basswood-io.com/docs/stable/api/codec.html#av.codec.context.CodecContext)
    for the life of the decoder and feeds each incoming RTP payload directly to
    the same `libopus` context.  This eliminates the per-packet MDCT overlap
    reset that creates 50 Hz window-boundary discontinuities — heard as
    choppiness — when each packet is decoded in a fresh context.

    Use [Opus.create_decoder][voip.codecs.opus.Opus.create_decoder] rather
    than instantiating this class directly.

    Attributes:
        output_rate_hz: Target PCM sample rate in Hz for decoded audio.
        codec_context: Persistent `libopus` decoder context shared across all
            [decode][voip.codecs.opus.OpusDecoder.decode] calls.
        resampler: Persistent resampler targeting `output_rate_hz` Hz.
    """

    output_rate_hz: int
    codec_context: av.AudioCodecContext = dataclasses.field(init=False, repr=False)
    resampler: av.audio.resampler.AudioResampler = dataclasses.field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.codec_context = cast(
            av.AudioCodecContext, av.CodecContext.create("libopus", "r")
        )
        self.codec_context.sample_rate = Opus.sample_rate_hz
        self.codec_context.open()
        self.resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=self.output_rate_hz
        )

    def decode(self, payload: bytes) -> np.ndarray:
        """Decode one Opus RTP payload, preserving CELT state from prior packets.

        Args:
            payload: Raw Opus RTP payload bytes.

        Returns:
            Float32 mono PCM array at `output_rate_hz` Hz.
        """
        return np.concatenate(
            [
                resampled.to_ndarray().flatten()
                for frame in self.codec_context.decode(av.Packet(payload))
                for resampled in self.resampler.resample(frame)
            ]
            or [np.array([], dtype=np.float32)]
        )
