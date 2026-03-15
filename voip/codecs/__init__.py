"""Audio codec implementations for RTP streams.

Provides the [`RTPCodec`][voip.codecs.base.RTPCodec] base class and concrete
implementations for all supported RTP audio codecs:

- [`Opus`][voip.codecs.Opus] — Opus (RFC 7587), PT 111
- [`G722`][voip.codecs.G722] — G.722 (RFC 3551), PT 9
- [`PCMA`][voip.codecs.PCMA] — G.711 A-law (RFC 3551), PT 8
- [`PCMU`][voip.codecs.PCMU] — G.711 mu-law (RFC 3551), PT 0

Use [`get`][voip.codecs.get] to look up a codec class by its SDP encoding
name (case-insensitive).
"""

from __future__ import annotations

from voip.codecs.base import RTPCodec
from voip.codecs.g722 import G722
from voip.codecs.opus import Opus
from voip.codecs.pcma import PCMA
from voip.codecs.pcmu import PCMU

__all__ = ["G722", "Opus", "PCMA", "PCMU", "RTPCodec", "get"]

#: Registry mapping lowercase encoding names to codec classes.
REGISTRY: dict[str, type[RTPCodec]] = {
    codec.encoding_name: codec for codec in (Opus, G722, PCMA, PCMU)
}


def get(encoding_name: str) -> type[RTPCodec]:
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
