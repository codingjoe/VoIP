"""G.722 wideband codec implementation for RTP audio streams (RFC 3551).

The [`G722`][voip.codecs.g722.G722] class handles the RFC 3551 clock-rate
quirk: SDP advertises 8 000 Hz but the actual audio runs at 16 000 Hz.

Use [`G722Decoder`][voip.codecs.g722.G722Decoder] (via
[`G722.create_decoder`][voip.codecs.g722.G722.create_decoder]) for per-call
stateful decoding that preserves the ADPCM predictor state across consecutive
RTP packets.

Requires the ``hd-audio`` extra: ``pip install voip[hd-audio]``.
"""

import dataclasses
import typing
from collections.abc import Iterator
from typing import ClassVar

import av
import av.audio.resampler
import numpy as np

from voip.codecs.av import PyAVCodec

__all__ = ["G722", "G722Decoder"]


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

    @classmethod
    def create_decoder(
        cls, output_rate_hz: int, *, input_rate_hz: int | None = None
    ) -> G722Decoder:
        """Create a stateful per-call G.722 decoder.

        Returns a [`G722Decoder`][voip.codecs.g722.G722Decoder] that preserves
        the ADPCM predictor state across consecutive RTP packets.  Pass the
        returned decoder to
        [`AudioCall`][voip.audio.AudioCall] (via the `create_decoder`
        factory) to avoid the per-packet state reset that causes robotic
        audio artefacts.

        The *input_rate_hz* parameter is accepted for API consistency with
        [`RTPCodec.create_decoder`][voip.codecs.base.RTPCodec.create_decoder]
        but is not used; G.722 always decodes at 16 000 Hz internally.

        Args:
            output_rate_hz: Target PCM sample rate in Hz for decoded audio.
            input_rate_hz: Ignored.  G.722 always decodes at `sample_rate_hz`.

        Returns:
            A new [`G722Decoder`][voip.codecs.g722.G722Decoder] instance.
        """
        return G722Decoder(output_rate_hz)


@dataclasses.dataclass(slots=True)
class G722Decoder:
    """Stateful G.722 decoder that preserves ADPCM predictor state across packets.

    Creates a single persistent
    [`av.CodecContext`](https://pyav.basswood-io.com/docs/stable/api/codec.html#av.codec.context.CodecContext)
    for the life of the decoder and feeds each incoming RTP packet to the
    same context.  This eliminates the per-packet predictor reset that causes
    robotic artefacts when decoding a G.722 stream with independent codec
    contexts.

    Use [`G722.create_decoder`][voip.codecs.g722.G722.create_decoder] rather
    than instantiating this class directly.

    Attributes:
        output_rate_hz: Target PCM sample rate in Hz for decoded audio.
        codec_context: Persistent G.722 decoder context
            shared across all [decode][voip.codecs.g722.G722Decoder.decode]
            calls on this instance.
        resampler: Sampler targeting `output_rate_hz` Hz.
    """

    output_rate_hz: int
    codec_context: av.AudioCodecContext = dataclasses.field(init=False, repr=False)
    resampler: av.audio.resampler.AudioResampler = dataclasses.field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.codec_context = typing.cast(
            av.AudioCodecContext, av.CodecContext.create("g722", "r")
        )
        self.codec_context.sample_rate = G722.sample_rate_hz
        self.codec_context.open()
        self.resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=self.output_rate_hz
        )

    def decode(self, payload: bytes) -> np.ndarray:
        """Decode one G.722 RTP payload, preserving ADPCM state from prior packets.

        Args:
            payload: Raw G.722 RTP payload bytes (160 bytes per 20 ms frame).

        Returns:
            Float32 mono PCM array at `output_rate_hz` Hz.
        """
        frames = [
            resampled.to_ndarray().flatten()
            for frame in self.codec_context.decode(av.Packet(payload))
            for resampled in self.resampler.resample(frame)
        ]
        return np.concatenate(frames) if frames else np.array([], dtype=np.float32)
