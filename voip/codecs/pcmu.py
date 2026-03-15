"""PCMU (G.711 mu-law) codec implementation for RTP audio streams (RFC 3551).

The [`PCMU`][voip.codecs.pcmu.PCMU] class decodes mu-law RTP payloads via
PyAV and encodes float32 PCM using a pure-NumPy implementation of
ITU-T G.711 mu-law companding.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from voip.codecs.base import RTPCodec

__all__ = ["PCMU"]


class PCMU(RTPCodec):
    """G.711 mu-law codec ([RFC 3551 §4.5.14][]).

    PCMU is the ITU-T G.711 mu-law logarithmic companding codec for PSTN
    telephony, standardised in RFC 3551 with static payload type 0.

    [RFC 3551 §4.5.14]: https://datatracker.ietf.org/doc/html/rfc3551#section-4.5.14
    """

    payload_type: ClassVar[int] = 0
    encoding_name: ClassVar[str] = "pcmu"
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
        return cls.decode_pcm(
            payload,
            "mulaw",
            output_rate_hz,
            input_rate_hz=input_rate_hz
            if input_rate_hz is not None
            else cls.sample_rate_hz,
        )

    @classmethod
    def encode(cls, samples: np.ndarray) -> bytes:
        BIAS = 0x84  # 132: G.711 mu-law bias constant
        CLIP = 32635  # maximum biased magnitude (14-bit saturate)
        pcm = np.clip(np.round(samples * 32768.0), -32768, 32767).astype(np.int32)
        sign = np.where(pcm >= 0, 0x80, 0x00).astype(np.uint8)
        biased = np.minimum(np.abs(pcm) + BIAS, CLIP)
        exp = np.clip(
            np.floor(np.log2(np.maximum(biased, 1))).astype(np.int32) - 7, 0, 7
        )
        mantissa = ((biased >> (exp + 3)) & 0x0F).astype(np.uint8)
        return (
            (~(sign | (exp.astype(np.uint8) << 4) | mantissa))
            .astype(np.uint8)
            .tobytes()
        )
