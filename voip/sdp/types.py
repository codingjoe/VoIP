"""SDP field types as defined by RFC 4566."""

from __future__ import annotations

import dataclasses
import enum
from typing import ClassVar, Protocol, runtime_checkable

__all__ = [
    "Field",
    "StrField",
    "IntField",
    "Origin",
    "ConnectionData",
    "Bandwidth",
    "Timing",
    "Attribute",
    "RTPPayloadFormat",
    "StaticPayloadType",
    "MediaDescription",
]


@runtime_checkable
class Field(Protocol):
    """SDP field descriptor protocol."""

    letter: str
    session_attr: str
    is_list: bool
    media_attr: str | None

    @staticmethod
    def parse(value: str) -> object:
        """Parse a raw SDP line value."""


@dataclasses.dataclass(slots=True)
class StrField:
    """Descriptor for SDP fields that parse and serialize as plain strings."""

    letter: str
    session_attr: str
    is_list: bool = False
    media_attr: str | None = None

    @staticmethod
    def parse(value: str) -> str:
        """Return the raw string value unchanged."""
        return value


@dataclasses.dataclass(slots=True)
class IntField:
    """Descriptor for SDP fields that parse and serialize as integers."""

    letter: str
    session_attr: str
    is_list: bool = False
    media_attr: str | None = None

    @staticmethod
    def parse(value: str) -> int:
        """Parse the raw string value as an integer."""
        return int(value)


@dataclasses.dataclass(slots=True)
class Origin:
    """Origin field (o=) as defined by RFC 4566 §5.2."""

    letter: ClassVar[str] = "o"
    session_attr: ClassVar[str] = "origin"
    is_list: ClassVar[bool] = False
    media_attr: ClassVar[str | None] = None

    username: str
    sess_id: str
    sess_version: str
    nettype: str
    addrtype: str
    unicast_address: str

    def __str__(self) -> str:
        """Serialize to SDP o= line value."""
        return (
            f"{self.username} {self.sess_id} {self.sess_version}"
            f" {self.nettype} {self.addrtype} {self.unicast_address}"
        )

    @classmethod
    def parse(cls, value: str) -> Origin:
        """Parse an o= line value into an Origin."""
        username, sess_id, sess_version, nettype, addrtype, unicast_address = (
            value.split(" ", 5)
        )
        return cls(
            username=username,
            sess_id=sess_id,
            sess_version=sess_version,
            nettype=nettype,
            addrtype=addrtype,
            unicast_address=unicast_address,
        )


@dataclasses.dataclass(slots=True)
class ConnectionData:
    """Connection data field (c=) as defined by RFC 4566 §5.7."""

    letter: ClassVar[str] = "c"
    session_attr: ClassVar[str] = "connection"
    is_list: ClassVar[bool] = False
    media_attr: ClassVar[str | None] = "connection"

    nettype: str
    addrtype: str
    connection_address: str

    def __str__(self) -> str:
        """Serialize to SDP c= line value."""
        return f"{self.nettype} {self.addrtype} {self.connection_address}"

    @classmethod
    def parse(cls, value: str) -> ConnectionData:
        """Parse a c= line value into a ConnectionData."""
        nettype, addrtype, connection_address = value.split(" ", 2)
        return cls(
            nettype=nettype,
            addrtype=addrtype,
            connection_address=connection_address,
        )


@dataclasses.dataclass(slots=True)
class Bandwidth:
    """Bandwidth field (b=) as defined by RFC 4566 §5.8."""

    letter: ClassVar[str] = "b"
    session_attr: ClassVar[str] = "bandwidths"
    is_list: ClassVar[bool] = True
    media_attr: ClassVar[str | None] = "bandwidths"

    bwtype: str
    bandwidth: int

    def __str__(self) -> str:
        """Serialize to SDP b= line value."""
        return f"{self.bwtype}:{self.bandwidth}"

    @classmethod
    def parse(cls, value: str) -> Bandwidth:
        """Parse a b= line value into a Bandwidth."""
        bwtype, _, bandwidth = value.partition(":")
        return cls(bwtype=bwtype, bandwidth=int(bandwidth))


@dataclasses.dataclass(slots=True)
class Timing:
    """Timing field (t=) as defined by RFC 4566 §5.9."""

    letter: ClassVar[str] = "t"
    session_attr: ClassVar[str] = "timings"
    is_list: ClassVar[bool] = True
    media_attr: ClassVar[str | None] = None

    start_time: int
    stop_time: int

    def __str__(self) -> str:
        """Serialize to SDP t= line value."""
        return f"{self.start_time} {self.stop_time}"

    @classmethod
    def parse(cls, value: str) -> Timing:
        """Parse a t= line value into a Timing."""
        start_time, stop_time = value.split(" ", 1)
        return cls(start_time=int(start_time), stop_time=int(stop_time))


@dataclasses.dataclass(slots=True)
class Attribute:
    """Attribute field (a=) as defined by RFC 4566 §5.13."""

    letter: ClassVar[str] = "a"
    session_attr: ClassVar[str] = "attributes"
    is_list: ClassVar[bool] = True
    media_attr: ClassVar[str | None] = "attributes"

    name: str
    value: str | None = None

    def __str__(self) -> str:
        """Serialize to SDP a= line value."""
        if self.value is None:
            return self.name
        return f"{self.name}:{self.value}"

    @classmethod
    def parse(cls, value: str) -> Attribute:
        """Parse an a= line value into an Attribute."""
        name, _, attr_value = value.partition(":")
        return cls(name=name, value=attr_value or None)


@dataclasses.dataclass(slots=True)
class RTPPayloadFormat:
    """RTP payload format descriptor (RFC 3551 §6 / RFC 4566 §6).

    Carries the numeric payload type together with the optional codec
    parameters that are conveyed by an ``a=rtpmap`` attribute::

        a=rtpmap:111 opus/48000/2
        a=rtpmap:8 PCMA/8000

    For static payload types the codec parameters are pre-defined by
    RFC 3551 and may be absent from the SDP; for dynamic payload types
    (PT ≥ 96) they are always required.

    Instances serialize to the ``a=rtpmap`` attribute *value* (without the
    leading ``a=rtpmap:`` prefix) when all codec fields are present, so they
    can be embedded directly as :attr:`Attribute.value`.
    """

    payload_type: int
    encoding_name: str | None = None
    clock_rate: int | None = None
    channels: int = 1

    def __str__(self) -> str:
        """Serialize to an ``a=rtpmap`` attribute value (without the ``a=`` prefix).

        Example: ``"111 opus/48000/2"`` or ``"8 PCMA/8000"``.

        Raises:
            ValueError: If *encoding_name* or *clock_rate* are not set.
        """
        if self.encoding_name is None or self.clock_rate is None:
            raise ValueError(
                f"Cannot serialize RtpPayloadFormat for PT {self.payload_type}: "
                "encoding_name and clock_rate are required"
            )
        base = f"{self.payload_type} {self.encoding_name}/{self.clock_rate}"
        return f"{base}/{self.channels}" if self.channels != 1 else base

    @classmethod
    def parse(cls, value: str) -> RTPPayloadFormat:
        """Parse an ``a=rtpmap`` attribute value into an :class:`RtpPayloadFormat`.

        Args:
            value: The attribute value, e.g. ``"111 opus/48000/2"``.

        Raises:
            ValueError: If the value does not conform to the expected format.
        """
        fmt, _, rest = value.partition(" ")
        parts = rest.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid rtpmap value: {value!r}")
        return cls(
            payload_type=int(fmt),
            encoding_name=parts[0],
            clock_rate=int(parts[1]),
            channels=int(parts[2]) if len(parts) > 2 else 1,
        )

    @classmethod
    def from_pt(cls, pt: int) -> RTPPayloadFormat:
        """Create a stub :class:`RtpPayloadFormat` from a payload type number.

        Only the *payload_type* is set.  Codec parameters (*encoding_name*,
        *clock_rate*, *channels*) remain ``None`` / default until an explicit
        ``a=rtpmap`` attribute is parsed and merged in by the SDP parser.

        For static payload types whose codec parameters are defined by RFC 3551
        but not carried in the SDP, use :class:`StaticPayloadType` directly.

        Args:
            pt: RTP payload type number (0–127).
        """
        return cls(payload_type=pt)


class StaticPayloadType(enum.Enum):
    """Static RTP payload types and their clock rates (Hz) as defined by RFC 3551 §6."""

    def __new__(
        cls, pt: int, clock_rate: int, encoding_name: str, channels: int = 1
    ) -> StaticPayloadType:
        obj = object.__new__(cls)
        obj._value_ = pt
        obj.clock_rate = clock_rate
        obj.encoding_name = encoding_name
        obj.channels = channels
        return obj

    PCMU = (0, 8000, "PCMU")  # G.711 µ-law
    GSM = (3, 8000, "GSM")
    G723 = (4, 8000, "G723")
    DVI4_8K = (5, 8000, "DVI4")
    DVI4_16K = (6, 16000, "DVI4")  # 16 kHz variant
    LPC = (7, 8000, "LPC")
    PCMA = (8, 8000, "PCMA")  # G.711 A-law
    G722 = (9, 8000, "G722")  # RTP clock rate is 8000 per RFC 3551 even though wideband
    L16_STEREO = (10, 44100, "L16", 2)
    L16_MONO = (11, 44100, "L16")
    QCELP = (12, 8000, "QCELP")
    CN = (13, 8000, "CN")
    MPA = (14, 90000, "MPA")
    G728 = (15, 8000, "G728")
    G729 = (18, 8000, "G729")


@dataclasses.dataclass(slots=True)
class MediaDescription:
    """Media description section (m=) as defined by RFC 4566 §5.14."""

    letter: ClassVar[str] = "m"
    session_attr: ClassVar[str] = "media"
    is_list: ClassVar[bool] = True
    media_attr: ClassVar[str | None] = None

    media: str
    port: int
    proto: str
    fmt: list[RTPPayloadFormat]
    title: str | None = None
    connection: ConnectionData | None = None
    bandwidths: list[Bandwidth] = dataclasses.field(default_factory=list)
    attributes: list[Attribute] = dataclasses.field(default_factory=list)

    def get_format(self, pt: int | str) -> RTPPayloadFormat | None:
        """Return the :class:`RtpPayloadFormat` for the given payload type, or ``None``.

        Args:
            pt: Payload type as an integer or its string representation
                (e.g. ``111`` or ``"111"``).

        Returns:
            The matching :class:`RtpPayloadFormat` from :attr:`fmt`, or
            ``None`` if not found.
        """
        target = int(pt)
        return next((f for f in self.fmt if f.payload_type == target), None)

    def get_rtpmap(self, pt: int | str) -> RTPPayloadFormat | None:
        """Alias for :meth:`get_format` (backwards compatibility)."""
        return self.get_format(pt)

    @property
    def sample_rate(self) -> int:
        """Clock rate (Hz) of the primary codec.

        Resolved in order:

        1. The *clock_rate* of the first :class:`RtpPayloadFormat` in
           :attr:`fmt` (set when an explicit ``a=rtpmap`` was present).
        2. The RFC 3551 static table via :class:`StaticPayloadType`.

        Raises:
            ValueError: If :attr:`fmt` is empty or the primary format has no
                known clock rate.
        """
        if not self.fmt:
            raise ValueError("No audio format in MediaDescription")
        f = self.fmt[0]
        if f.clock_rate is not None:
            return f.clock_rate
        try:
            return StaticPayloadType(f.payload_type).clock_rate
        except ValueError:
            pass
        raise ValueError(
            f"No clock rate available for payload type {f.payload_type!r}; "
            f"missing a=rtpmap attribute"
        )

    def __str__(self) -> str:
        """Serialize to SDP m= section lines."""
        fmt_str = " ".join(str(f.payload_type) for f in self.fmt)
        lines = [f"m={self.media} {self.port} {self.proto} {fmt_str}"]
        if self.title is not None:
            lines.append(f"i={self.title}")
        if self.connection is not None:
            lines.append(f"c={self.connection}")
        lines.extend(f"b={b}" for b in self.bandwidths)
        for f in self.fmt:
            if f.encoding_name is not None and f.clock_rate is not None:
                lines.append(f"a=rtpmap:{f}")
        lines.extend(f"a={a}" for a in self.attributes)
        return "\r\n".join(lines)

    @classmethod
    def parse(cls, value: str) -> MediaDescription:
        """Parse an m= line value into a MediaDescription."""
        media, port_str, proto, *fmts = value.split(" ")
        fmt = [RTPPayloadFormat.from_pt(int(pt)) for pt in " ".join(fmts).split(" ")]
        return cls(media=media, port=int(port_str), proto=proto, fmt=fmt)
