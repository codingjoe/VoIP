"""Opus codec implementation for RTP audio streams (RFC 7587).

The [`Opus`][voip.codecs.opus.Opus] class wraps raw Opus RTP payloads in a
minimal [Ogg][] container before passing them to PyAV for decoding, and
encodes float32 PCM via `libopus`.

Requires the ``pyav`` extra: ``pip install voip[pyav]``.

[Ogg]: https://wiki.xiph.org/Ogg
"""

from __future__ import annotations

import os
import struct
from typing import ClassVar

import numpy as np

from voip.codecs.av import PyAVCodec

__all__ = ["Opus"]


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
            1,  # channel count (mono)
            3840,  # pre-skip: 80 ms at 48 kHz (RFC 7587)
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
                cls._ogg_page(0x04, 0, serial_number, 2, [packet]),
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
        return cls.encode_pcm(samples, "libopus", cls.sample_rate_hz)
