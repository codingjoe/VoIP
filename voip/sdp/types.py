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
    "RtpMap",
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
class RtpMap:
    """Parsed ``a=rtpmap`` attribute value (RFC 4566 §6 / RFC 3550 §8).

    Associates a payload type number with an encoding name and clock rate::

        a=rtpmap:111 opus/48000/2
        a=rtpmap:8 PCMA/8000

    Instances are compared and serialized without the leading ``a=rtpmap:``
    prefix so they can be stored directly as :attr:`Attribute.value`.
    """

    payload_type: int
    encoding_name: str
    clock_rate: int
    channels: int = 1

    def __str__(self) -> str:
        """Serialize to an ``a=rtpmap`` attribute value (without the ``a=`` prefix).

        Example: ``"111 opus/48000/2"`` or ``"8 PCMA/8000"``.
        """
        base = f"{self.payload_type} {self.encoding_name}/{self.clock_rate}"
        return f"{base}/{self.channels}" if self.channels != 1 else base

    @classmethod
    def parse(cls, value: str) -> RtpMap:
        """Parse an ``a=rtpmap`` attribute value into an :class:`RtpMap`.

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


class StaticPayloadType(enum.Enum):
    """Static RTP payload types and their clock rates (Hz) as defined by RFC 3551 §6."""

    def __new__(cls, pt: int, clock_rate: int) -> StaticPayloadType:
        obj = object.__new__(cls)
        obj._value_ = pt
        obj.clock_rate = clock_rate
        return obj

    PCMU = (0, 8000)  # G.711 µ-law
    GSM = (3, 8000)
    G723 = (4, 8000)
    DVI4_8K = (5, 8000)
    DVI4_16K = (6, 16000)  # 16 kHz variant
    LPC = (7, 8000)
    PCMA = (8, 8000)  # G.711 A-law
    G722 = (9, 8000)  # RTP clock rate is 8000 per RFC 3551 even though wideband
    L16_STEREO = (10, 44100)
    L16_MONO = (11, 44100)
    QCELP = (12, 8000)
    CN = (13, 8000)
    MPA = (14, 90000)
    G728 = (15, 8000)
    G729 = (18, 8000)


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
    fmt: list[str]
    title: str | None = None
    connection: ConnectionData | None = None
    bandwidths: list[Bandwidth] = dataclasses.field(default_factory=list)
    attributes: list[Attribute] = dataclasses.field(default_factory=list)

    def get_rtpmap(self, fmt: str) -> RtpMap | None:
        """Return the :class:`RtpMap` for the given payload type string, or ``None``.

        Searches the ``a=rtpmap`` attributes for an entry whose payload type
        matches *fmt*.

        Args:
            fmt: The payload type as a string (e.g. ``"111"`` or ``"8"``).

        Returns:
            A :class:`RtpMap` if a matching ``a=rtpmap`` attribute is found,
            otherwise ``None``.
        """
        for attr in self.attributes:
            if (
                attr.name == "rtpmap"
                and attr.value
                and attr.value.startswith(f"{fmt} ")
            ):
                return RtpMap.parse(attr.value)
        return None

    @property
    def sample_rate(self) -> int:
        """Clock rate (Hz) of the primary codec as declared in ``a=rtpmap``.

        For static payload types that carry no ``a=rtpmap`` attribute, the rate
        is taken from the RFC 3551 §6 static table.

        Raises:
            ValueError: If the primary format has no ``a=rtpmap`` and is not a
                recognised RFC 3551 static payload type.
        """
        if not self.fmt:
            raise ValueError("No audio format in MediaDescription")
        fmt = self.fmt[0]
        rtpmap = self.get_rtpmap(fmt)
        if rtpmap is not None:
            return rtpmap.clock_rate
        try:
            pt = int(fmt)
        except ValueError as exc:
            raise ValueError(
                f"Cannot determine sample rate for format {fmt!r}"
            ) from exc
        try:
            return StaticPayloadType(pt).clock_rate
        except ValueError:
            pass
        raise ValueError(
            f"No a=rtpmap attribute for dynamic payload type {fmt!r} "
            f"and no RFC 3551 static rate defined"
        )

    def __str__(self) -> str:
        """Serialize to SDP m= section lines."""
        fmt_str = " ".join(self.fmt)
        lines = [f"m={self.media} {self.port} {self.proto} {fmt_str}"]
        if self.title is not None:
            lines.append(f"i={self.title}")
        if self.connection is not None:
            lines.append(f"c={self.connection}")
        lines.extend(f"b={b}" for b in self.bandwidths)
        lines.extend(f"a={a}" for a in self.attributes)
        return "\r\n".join(lines)

    @classmethod
    def parse(cls, value: str) -> MediaDescription:
        """Parse an m= line value into a MediaDescription."""
        media, port_str, proto, *fmts = value.split(" ")
        fmt = " ".join(fmts).split(" ")
        return cls(media=media, port=int(port_str), proto=proto, fmt=fmt)
