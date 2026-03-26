"""Audio codec implementations for RTP streams.

Provides the [`RTPCodec`][voip.codecs.base.RTPCodec] base class and concrete
implementations for all supported RTP audio codecs:

- [`PCMA`][voip.codecs.PCMA] — G.711 A-law (RFC 3551), PT 8 *(pure NumPy)*
- [`PCMU`][voip.codecs.PCMU] — G.711 mu-law (RFC 3551), PT 0 *(pure NumPy)*
- [`G722`][voip.codecs.G722] — G.722 (RFC 3551), PT 9 *(requires* ``pyav`` *extra)*
- [`Opus`][voip.codecs.Opus] — Opus (RFC 7587), PT 111 *(requires* ``pyav`` *extra)*

Use [`get`][voip.codecs.get] to look up a codec class by its SDP encoding
name (case-insensitive).

When the ``pyav`` extra is not installed only PCMA and PCMU are registered.
"""

from voip.codecs.base import RTPCodec
from voip.codecs.pcma import PCMA
from voip.codecs.pcmu import PCMU

__all__ = ["PCMA", "PCMU", "RTPCodec", "get"]

#: Registry mapping lowercase encoding names to codec classes.
REGISTRY: dict[str, type[RTPCodec]] = {
    PCMA.encoding_name: PCMA,
    PCMU.encoding_name: PCMU,
}

try:
    from voip.codecs.av import PyAVCodec
    from voip.codecs.g722 import G722
    from voip.codecs.opus import Opus

    REGISTRY |= {G722.encoding_name: G722, Opus.encoding_name: Opus}
    __all__ = [*__all__, "G722", "Opus", "PyAVCodec"]
except ImportError:
    pass


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
