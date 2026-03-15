"""G.722 wideband codec implementation for RTP audio streams (RFC 3551).

The [`G722`][voip.codecs.g722.G722] class handles the RFC 3551 clock-rate
quirk: SDP advertises 8 000 Hz but the actual audio runs at 16 000 Hz.

Requires the ``pyav`` extra: ``pip install voip[pyav]``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import numpy as np

from voip.codecs.av import PyAVCodec

__all__ = ["G722"]


class G722(PyAVCodec):
    """G.722 wideband audio codec ([RFC 3551 §4.5.2][]).

    G.722 is an ITU-T ADPCM wideband codec.  Despite encoding audio at
    16 000 Hz, the RTP timestamp clock runs at 8 000 Hz per RFC 3551 —
    a well-known quirk of the original specification.

    The entire buffer is encoded at once to preserve the ADPCM predictor
    state across packet boundaries.

    [RFC 3551 §4.5.2]: https://datatracker.ietf.org/doc/html/rfc3551#section-4.5.2
    """

    payload_type: ClassVar[int] = 9
    encoding_name: ClassVar[str] = "g722"
    #: Actual wideband audio sample rate in Hz.
    sample_rate_hz: ClassVar[int] = 16000
    #: RTP timestamp clock rate per RFC 3551 (8 kHz despite 16 kHz audio).
    rtp_clock_rate_hz: ClassVar[int] = 8000
    #: Audio samples per 20 ms frame at 16 kHz.
    frame_size: ClassVar[int] = 320
    #: RTP timestamp ticks per frame at the 8 kHz clock.
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
            "g722",
            output_rate_hz,
            input_rate_hz=cls.rtp_clock_rate_hz,
        )

    @classmethod
    def encode(cls, samples: np.ndarray) -> bytes:
        return cls.encode_pcm(samples, "g722", cls.sample_rate_hz)

    @classmethod
    def packetize(cls, audio: np.ndarray) -> Iterator[bytes]:
        encoded = cls.encode(audio)
        # G.722 2:1 sample-to-byte ratio: frame_size (320) samples → 160 bytes.
        payload_size = cls.frame_size // 2
        for i in range(0, len(encoded), payload_size):
            yield encoded[i : i + payload_size]
