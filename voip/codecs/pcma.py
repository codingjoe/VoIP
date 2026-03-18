"""PCMA (G.711 A-law) codec implementation for RTP audio streams (RFC 3551).

The [`PCMA`][voip.codecs.pcma.PCMA] class decodes and encodes A-law RTP
payloads using a pure-NumPy implementation of the ITU-T G.711 A-law segmented
companding algorithm.  No PyAV dependency is required.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from voip.codecs.base import RTPCodec

__all__ = ["PCMA"]

# G.711 A-law segment upper bounds (16-bit PCM magnitude, inclusive per segment).
# Vectorised segment lookup via np.searchsorted uses side='left' to count thresholds
# strictly exceeded (v > threshold), giving the correct 0–7 segment index.
_ALAW_SEG_UBOUND: np.ndarray = np.array(
    (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF), dtype=np.int32
)


class PCMA(RTPCodec):
    """G.711 A-law codec ([RFC 3551 §4.5.14][]).

    PCMA is the ITU-T G.711 A-law logarithmic companding codec for PSTN
    telephony, standardised in RFC 3551 with static payload type 8.

    Both encode and decode interoperate bit-exactly with real RTP PCMA
    streams.

    [RFC 3551 §4.5.14]: https://datatracker.ietf.org/doc/html/rfc3551#section-4.5.14
    """

    payload_type: ClassVar[int] = 8
    encoding_name: ClassVar[str] = "pcma"
    sample_rate_hz: ClassVar[int] = 8000
    rtp_clock_rate_hz: ClassVar[int] = 8000
    frame_size: ClassVar[int] = 160
    timestamp_increment: ClassVar[int] = 160
    channels: ClassVar[int] = 1

    @classmethod
    def decode(
        cls,
        payload: bytes,
        output_rate_hz: int,
        *,
        input_rate_hz: int | None = None,
    ) -> np.ndarray:
        # XOR with 0x55 to undo the G.711 bit-inversion applied before transmission.
        raw = np.frombuffer(payload, dtype=np.uint8) ^ np.uint8(0x55)
        sign = np.where(raw & 0x80, 1.0, -1.0).astype(np.float32)
        segment = ((raw & 0x70) >> 4).astype(np.int32)
        # t = mantissa bits shifted to the top of the 4-bit slot (×16).
        mantissa_t = ((raw & 0x0F).astype(np.int32)) << 4
        # Segment 0: add step mid-point (8).
        t_seg0 = mantissa_t + 8
        # Segments ≥ 1: add 0x108 bias then left-shift by (segment − 1).
        t_bias = mantissa_t + 0x108
        shift = np.maximum(segment - 1, 0)
        t_segN = np.left_shift(t_bias, shift)
        t = np.where(segment == 0, t_seg0, t_segN).astype(np.float32)
        normalized = (sign * t / 32768.0).astype(np.float32)
        source_rate_hz = (
            input_rate_hz if input_rate_hz is not None else cls.sample_rate_hz
        )
        return cls.resample(normalized, source_rate_hz, output_rate_hz)

    @classmethod
    def encode(cls, samples: np.ndarray) -> bytes:
        # Scale to 16-bit signed PCM range.
        pcm = np.clip(np.round(samples * 32768.0), -32768, 32767).astype(np.int32)
        # Positive samples: mask 0xD5.  Negative samples: mask 0x55, negate−1.
        magnitude = np.where(pcm >= 0, pcm, -pcm - 1).astype(np.int32)
        mask = np.where(pcm >= 0, np.uint8(0xD5), np.uint8(0x55))
        # Find the segment index (0–7) via vectorised binary search on the upper bounds.
        # side='left' counts thresholds strictly less than magnitude, i.e. exceeded.
        seg = np.minimum(
            np.searchsorted(_ALAW_SEG_UBOUND, magnitude, side="left").astype(np.int32),
            7,
        )
        # Extract the 4-bit mantissa in 16-bit space.
        # G.711 A-law quantises 13-bit PCM (16-bit >> 3), so the effective
        # mantissa shift is 4 for segments 0–1 and (seg + 3) for segments ≥ 2.
        shift = np.where(seg < 2, 4, seg + 3).astype(np.int32)
        mantissa = (np.right_shift(magnitude, shift) & 0x0F).astype(np.uint8)
        aval = (seg.astype(np.uint8) << 4) | mantissa
        return (aval ^ mask).astype(np.uint8).tobytes()
