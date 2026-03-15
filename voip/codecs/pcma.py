"""PCMA (G.711 A-law) codec implementation for RTP audio streams (RFC 3551).

The [`PCMA`][voip.codecs.pcma.PCMA] class decodes A-law RTP payloads via
PyAV and encodes float32 PCM using a pure-NumPy implementation of
ITU-T G.711 A-law companding.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from voip.codecs.base import RTPCodec

__all__ = ["PCMA"]


class PCMA(RTPCodec):
    """G.711 A-law codec ([RFC 3551 §4.5.14][]).

    PCMA is the ITU-T G.711 A-law logarithmic companding codec for PSTN
    telephony, standardised in RFC 3551 with static payload type 8.

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
        return cls.decode_pcm(
            payload,
            "alaw",
            output_rate_hz,
            input_rate_hz=input_rate_hz
            if input_rate_hz is not None
            else cls.sample_rate_hz,
        )

    @classmethod
    def encode(cls, samples: np.ndarray) -> bytes:
        a_law = 87.6  # G.711 A-law compression parameter
        pcm = np.clip(np.abs(samples), 0, 1.0)
        low = pcm < (1.0 / a_law)
        compressed = np.where(
            low,
            a_law * pcm / (1.0 + np.log(a_law)),
            (1.0 + np.log(np.maximum(a_law * pcm, 1e-10))) / (1.0 + np.log(a_law)),
            # 1e-10 prevents log(0) when pcm is exactly 0.0 in the high range
        )
        quantized = np.clip(np.round(compressed * 127), 0, 127).astype(np.uint8)
        sign = np.where(samples >= 0, 0x80, 0x00).astype(np.uint8)
        # XOR even bits per G.711 §A (toggle bits via 0x55)
        return ((sign | quantized) ^ 0x55).astype(np.uint8).tobytes()
