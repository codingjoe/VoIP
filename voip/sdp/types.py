"""SDP field types as defined by RFC 4566."""

from __future__ import annotations

import dataclasses
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
        ...


@dataclasses.dataclass
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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


@dataclasses.dataclass
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
