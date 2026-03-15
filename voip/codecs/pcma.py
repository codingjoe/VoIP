"""PCMA (G.711 A-law) codec implementation for RTP audio streams (RFC 3551).

The [`PCMA`][voip.codecs.pcma.PCMA] class decodes and encodes A-law RTP
payloads using a pure-NumPy implementation of ITU-T G.711 A-law companding.
No PyAV dependency is required.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from voip.codecs.base import RTPCodec

__all__ = ["PCMA"]

_A_LAW: float = 87.6
_LN_A: float = float(np.log(_A_LAW))
_COMPRESS_SCALE: float = 1.0 + _LN_A


class PCMA(RTPCodec):
    """G.711 A-law codec ([RFC 3551 §4.5.14][]).

    PCMA is the ITU-T G.711 A-law logarithmic companding codec for PSTN
    telephony, standardised in RFC 3551 with static payload type 8.

    Both encode and decode are pure-NumPy and require no PyAV dependency.

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
        raw = np.frombuffer(payload, dtype=np.uint8) ^ 0x55
        sign = np.where(raw & 0x80, 1.0, -1.0)
        quantized = (raw & 0x7F).astype(np.float32) / 127.0
        threshold = 1.0 / _COMPRESS_SCALE
        linear = np.where(
            quantized < threshold,
            quantized * _COMPRESS_SCALE / _A_LAW,
            np.exp(quantized * _COMPRESS_SCALE - 1.0) / _A_LAW,
        ).astype(np.float32)
        return cls.resample(
            (sign * linear).astype(np.float32), cls.sample_rate_hz, output_rate_hz
        )

    @classmethod
    def encode(cls, samples: np.ndarray) -> bytes:
        pcm = np.clip(np.abs(samples), 0, 1.0)
        low = pcm < (1.0 / _A_LAW)
        compressed = np.where(
            low,
            _A_LAW * pcm / _COMPRESS_SCALE,
            (1.0 + np.log(np.maximum(_A_LAW * pcm, 1e-10))) / _COMPRESS_SCALE,
            # 1e-10 prevents log(0) when pcm is exactly 0.0 in the high range
        )
        quantized = np.clip(np.round(compressed * 127), 0, 127).astype(np.uint8)
        sign = np.where(samples >= 0, 0x80, 0x00).astype(np.uint8)
        # XOR even bits per G.711 §A (toggle bits via 0x55)
        return ((sign | quantized) ^ 0x55).astype(np.uint8).tobytes()
