"""Base class for RTP audio codecs.

All concrete codec classes in this package inherit from
[`RTPCodec`][voip.codecs.base.RTPCodec].

Codecs that require [PyAV][] for decode/encode additionally inherit from
[`PyAVCodec`][voip.codecs.av.PyAVCodec], which provides
[`decode_pcm`][voip.codecs.av.PyAVCodec.decode_pcm] and
[`encode_pcm`][voip.codecs.av.PyAVCodec.encode_pcm].

Pure-NumPy codecs ([`PCMA`][voip.codecs.pcma.PCMA], [`PCMU`][voip.codecs.pcmu.PCMU])
inherit directly from `RTPCodec` and require no PyAV dependency.

[PyAV]: https://pyav.basswood-io.com/
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import ClassVar, Protocol

import numpy as np

from voip.sdp.types import RTPPayloadFormat

__all__ = ["PayloadDecoder", "PerPacketDecoder", "RTPCodec"]


class PayloadDecoder(Protocol):
    """Protocol for per-call RTP payload decoders.

    Implementations decode raw RTP payload bytes to float32 mono PCM.
    Stateful implementations (e.g. [`G722Decoder`][voip.codecs.g722.G722Decoder])
    preserve codec predictor state across successive
    [`decode`][voip.codecs.base.PayloadDecoder.decode] calls within a single
    call session.
    """

    def decode(self, payload: bytes) -> np.ndarray:
        """Decode one RTP payload to float32 mono PCM.

        Args:
            payload: Raw RTP payload bytes for a single packet.

        Returns:
            Float32 mono PCM array.
        """
        ...


class RTPCodec:
    """Base class for RTP audio codecs.

    Concrete implementations: [`Opus`][voip.codecs.Opus],
    [`G722`][voip.codecs.G722], [`PCMA`][voip.codecs.pcma.PCMA],
    [`PCMU`][voip.codecs.pcmu.PCMU].

    Codec classes are stateless; every method is a classmethod or staticmethod
    and codecs are referenced as `type[RTPCodec]`, never instantiated.
    Per-call decoder state (required for ADPCM codecs such as G.722) is
    managed by [`PayloadDecoder`][voip.codecs.base.PayloadDecoder] instances
    returned by [`create_decoder`][voip.codecs.base.RTPCodec.create_decoder].

    Concrete subclasses define codec-specific class variables and override
    [`decode`][voip.codecs.base.RTPCodec.decode],
    [`encode`][voip.codecs.base.RTPCodec.encode], and optionally
    [`packetize`][voip.codecs.base.RTPCodec.packetize].

    Subclasses may use the shared PyAV-backed helpers or implement
    [`decode`][voip.codecs.base.RTPCodec.decode] and
    [`encode`][voip.codecs.base.RTPCodec.encode] using alternative backends
    such as NumPy.

    Subclasses that produce variable-length output across frames (e.g. G.722
    ADPCM) should override `packetize` to encode the whole buffer at once and
    preserve predictor state.

    Subclasses that require [PyAV][] additionally inherit from
    [`PyAVCodec`][voip.codecs.av.PyAVCodec].

    [PyAV]: https://pyav.basswood-io.com/
    """

    payload_type: ClassVar[int]
    """RTP payload type number."""

    encoding_name: ClassVar[str]
    """SDP encoding name (lowercase)."""

    sample_rate_hz: ClassVar[int]
    """Actual audio sample rate in Hz."""

    rtp_clock_rate_hz: ClassVar[int]
    """RTP timestamp clock rate in Hz (may differ from `sample_rate_hz`)."""

    frame_size: ClassVar[int]
    """Audio samples per 20 ms RTP frame at `sample_rate_hz`."""

    timestamp_increment: ClassVar[int]
    """RTP timestamp ticks per frame at `rtp_clock_rate_hz`."""

    channels: ClassVar[int]
    """Channel count (1 = mono, 2 = stereo)."""

    @classmethod
    def resample(
        cls, audio: np.ndarray, source_rate_hz: int, destination_rate_hz: int
    ) -> np.ndarray:
        """Resample *audio* from *source_rate_hz* to *destination_rate_hz*.

        Uses linear interpolation via [`numpy.interp`][].

        Args:
            audio: Float32 mono PCM array.
            source_rate_hz: Sample rate of *audio* in Hz.
            destination_rate_hz: Target sample rate in Hz.

        Returns:
            Resampled float32 array at *destination_rate_hz* Hz, or *audio*
            unchanged when both rates are equal.
        """
        if source_rate_hz == destination_rate_hz:
            return audio
        if len(audio) == 0:
            return np.empty(0, dtype=np.float32)
        n_out = max(1, round(len(audio) * destination_rate_hz / source_rate_hz))
        return np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    @classmethod
    def to_payload_format(cls) -> RTPPayloadFormat:
        """Create an [`RTPPayloadFormat`][voip.sdp.types.RTPPayloadFormat] for SDP negotiation.

        Uses `rtp_clock_rate_hz` as the SDP sample rate, which is correct
        per RFC 3551 (e.g. G.722 advertises 8000 Hz in SDP even though the
        actual audio runs at 16000 Hz).

        Returns:
            Payload format descriptor for this codec.
        """
        return RTPPayloadFormat(
            payload_type=cls.payload_type,
            encoding_name=cls.encoding_name,
            sample_rate=cls.rtp_clock_rate_hz,
            channels=cls.channels,
        )

    @classmethod
    def decode(
        cls,
        payload: bytes,
        output_rate_hz: int,
        *,
        input_rate_hz: int | None = None,
    ) -> np.ndarray:
        """Decode an RTP payload to float32 mono PCM.

        Override in subclasses to implement codec-specific decoding.

        Args:
            payload: Raw RTP payload bytes.
            output_rate_hz: Target sample rate in Hz.
            input_rate_hz: Input clock rate override in Hz, or `None` to use
                the subclass default.

        Returns:
            Float32 mono PCM array at *output_rate_hz* Hz.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement decode.")

    @classmethod
    def create_decoder(
        cls, output_rate_hz: int, *, input_rate_hz: int | None = None
    ) -> PayloadDecoder:
        """Create a stateless per-call payload decoder for this codec.

        Override in subclasses that require stateful decoding across RTP
        packets (e.g. G.722 ADPCM — see
        [`G722.create_decoder`][voip.codecs.g722.G722.create_decoder]).

        Args:
            output_rate_hz: Target PCM sample rate in Hz for decoded audio.
            input_rate_hz: Input clock rate override, or `None` to use the
                codec default.

        Returns:
            A [`PayloadDecoder`][voip.codecs.base.PayloadDecoder] that, by
            default, is a [`PerPacketDecoder`][voip.codecs.base.PerPacketDecoder]
            delegating each call to [`decode`][voip.codecs.base.RTPCodec.decode].
        """
        return PerPacketDecoder(cls, output_rate_hz, input_rate_hz)

    @classmethod
    def encode(cls, samples: np.ndarray) -> bytes:
        """Encode float32 mono PCM to an RTP payload.

        Override in subclasses to implement codec-specific encoding.

        Args:
            samples: Float32 mono PCM at `sample_rate_hz` Hz.

        Returns:
            Encoded bytes for one RTP payload.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement encode.")

    @classmethod
    def packetize(cls, audio: np.ndarray) -> Iterator[bytes]:
        """Encode *audio* and yield one encoded payload per 20 ms RTP frame.

        The default implementation encodes one `frame_size` chunk at a time
        using `encode`. Override in subclasses (e.g. G.722) where the entire
        buffer must be encoded at once to preserve codec state.

        Args:
            audio: Float32 mono PCM at `sample_rate_hz` Hz.

        Yields:
            Encoded payload bytes, one per RTP packet.
        """
        for i in range(0, len(audio), cls.frame_size):
            yield cls.encode(audio[i : i + cls.frame_size])


@dataclasses.dataclass(frozen=True)
class PerPacketDecoder:
    """Stateless payload decoder that processes each RTP packet independently.

    Delegate each call to
    [`RTPCodec.decode`][voip.codecs.base.RTPCodec.decode], decoding each
    payload independently without preserving cross-packet state. Suitable for
    stateless codecs such as PCMA, PCMU, and Opus.

    Attributes:
        codec: Codec class to delegate decoding to.
        output_rate_hz: Target PCM sample rate in Hz.
        input_rate_hz: Input clock rate override, or `None` to use the codec default.
    """

    codec: type[RTPCodec]
    output_rate_hz: int
    input_rate_hz: int | None = None

    def decode(self, payload: bytes) -> np.ndarray:
        """Decode one RTP payload to float32 PCM.

        Args:
            payload: Raw RTP payload bytes.

        Returns:
            Float32 mono PCM array at `output_rate_hz` Hz.
        """
        return self.codec.decode(
            payload, self.output_rate_hz, input_rate_hz=self.input_rate_hz
        )
