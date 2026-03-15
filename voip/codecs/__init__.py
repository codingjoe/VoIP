"""Audio codec implementations for RTP streams.

Provides the [`Codec`][voip.codecs.Codec] structural interface and concrete
implementations for all supported RTP audio codecs:

- [`Opus`][voip.codecs.Opus] — Opus (RFC 7587), PT 111
- [`G722`][voip.codecs.G722] — G.722 (RFC 3551), PT 9
- [`PCMA`][voip.codecs.PCMA] — G.711 A-law (RFC 3551), PT 8
- [`PCMU`][voip.codecs.PCMU] — G.711 mu-law (RFC 3551), PT 0

Use [`get`][voip.codecs.get] to look up a codec class by its SDP encoding
name (case-insensitive).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    import numpy as np

    from voip.sdp.types import RTPPayloadFormat

__all__ = ["Codec", "G722", "Opus", "PCMA", "PCMU", "get"]


class Codec(Protocol):
    """Structural interface for RTP audio codec classes.

    Concrete implementations: [`Opus`][voip.codecs.Opus],
    [`G722`][voip.codecs.G722], [`PCMA`][voip.codecs.PCMA],
    [`PCMU`][voip.codecs.PCMU].

    All codec implementations are stateless: every method is a classmethod
    or staticmethod and codecs are referenced as `type[Codec]`, never
    instantiated.
    """

    payload_type: ClassVar[int]
    """RTP payload type number (static or dynamic)."""

    encoding_name: ClassVar[str]
    """SDP encoding name in lowercase (e.g. `"opus"`, `"g722"`)."""

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
    def decode(
        cls,
        payload: bytes,
        output_rate_hz: int,
        *,
        input_rate_hz: int | None = None,
    ) -> np.ndarray:
        """Decode an RTP payload to float32 mono PCM.

        Args:
            payload: Raw RTP payload bytes.
            output_rate_hz: Target sample rate in Hz.
            input_rate_hz: Optional input clock rate override in Hz.

        Returns:
            Float32 mono PCM array at *output_rate_hz* Hz.
        """

    @classmethod
    def encode(cls, samples: np.ndarray) -> bytes:
        """Encode float32 mono PCM to an RTP payload.

        Args:
            samples: Float32 mono PCM at `sample_rate_hz` Hz.

        Returns:
            Encoded bytes for one RTP payload.
        """

    @classmethod
    def packetize(cls, audio: np.ndarray) -> Iterator[bytes]:
        """Encode *audio* and yield one payload per 20 ms RTP frame.

        Args:
            audio: Float32 mono PCM at `sample_rate_hz` Hz.

        Yields:
            Encoded payload bytes, one per RTP packet.
        """

    @classmethod
    def to_payload_format(cls) -> RTPPayloadFormat:
        """Create an [`RTPPayloadFormat`][voip.sdp.types.RTPPayloadFormat] for SDP.

        Returns:
            Payload format descriptor for this codec.
        """


# Concrete implementations — imported after the Protocol to avoid circularity.
from voip.codecs.g722 import G722  # noqa: E402
from voip.codecs.opus import Opus  # noqa: E402
from voip.codecs.pcma import PCMA  # noqa: E402
from voip.codecs.pcmu import PCMU  # noqa: E402

#: Registry mapping lowercase encoding names to codec classes.
REGISTRY: dict[str, type[Codec]] = {
    codec.encoding_name: codec for codec in (Opus, G722, PCMA, PCMU)
}


def get(encoding_name: str) -> type[Codec]:
    """Get a codec class by its SDP encoding name.

    Args:
        encoding_name: SDP encoding name, case-insensitive
            (e.g. `"opus"`, `"G722"`, `"PCMA"`).

    Returns:
        Matching codec class.

    Raises:
        NotImplementedError: When no registered codec matches *encoding_name*.
    """
    try:
        return REGISTRY[encoding_name.lower()]
    except KeyError:
        raise NotImplementedError(
            f"Unsupported codec: {encoding_name!r}. Supported: {list(REGISTRY)!r}"
        )
