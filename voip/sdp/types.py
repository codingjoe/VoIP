"""SDP field types as defined by RFC 4566."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Generator
from typing import ClassVar, NamedTuple, Protocol, runtime_checkable

from ..types import ByteSerializableObject

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
        return int(value)


@dataclasses.dataclass(slots=True)
class Origin(ByteSerializableObject):
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

    def __bytes__(self) -> bytes:
        return (
            f"{self.username} {self.sess_id} {self.sess_version}"
            f" {self.nettype} {self.addrtype} {self.unicast_address}"
        ).encode()

    @classmethod
    def parse(cls, data: bytes | str) -> Origin:
        value = data.decode() if isinstance(data, bytes) else data
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
class ConnectionData(ByteSerializableObject):
    """Connection data field (c=) as defined by RFC 4566 §5.7."""

    letter: ClassVar[str] = "c"
    session_attr: ClassVar[str] = "connection"
    is_list: ClassVar[bool] = False
    media_attr: ClassVar[str | None] = "connection"

    nettype: str
    addrtype: str
    connection_address: str

    def __bytes__(self) -> bytes:
        return f"{self.nettype} {self.addrtype} {self.connection_address}".encode()

    @classmethod
    def parse(cls, data: bytes | str) -> ConnectionData:
        value = data.decode() if isinstance(data, bytes) else data
        nettype, addrtype, connection_address = value.split(" ", 2)
        return cls(
            nettype=nettype,
            addrtype=addrtype,
            connection_address=connection_address,
        )


@dataclasses.dataclass(slots=True)
class Bandwidth(ByteSerializableObject):
    """Bandwidth field (b=) as defined by RFC 4566 §5.8."""

    letter: ClassVar[str] = "b"
    session_attr: ClassVar[str] = "bandwidths"
    is_list: ClassVar[bool] = True
    media_attr: ClassVar[str | None] = "bandwidths"

    bwtype: str
    bandwidth: int

    def __bytes__(self) -> bytes:
        return f"{self.bwtype}:{self.bandwidth}".encode()

    @classmethod
    def parse(cls, data: bytes | str) -> Bandwidth:
        value = data.decode() if isinstance(data, bytes) else data
        bwtype, _, bandwidth = value.partition(":")
        return cls(bwtype=bwtype, bandwidth=int(bandwidth))


@dataclasses.dataclass(slots=True)
class Timing(ByteSerializableObject):
    """Timing field (t=) as defined by RFC 4566 §5.9."""

    letter: ClassVar[str] = "t"
    session_attr: ClassVar[str] = "timings"
    is_list: ClassVar[bool] = True
    media_attr: ClassVar[str | None] = None

    start_time: int
    stop_time: int

    def __bytes__(self) -> bytes:
        return f"{self.start_time} {self.stop_time}".encode()

    @classmethod
    def parse(cls, data: bytes | str) -> Timing:
        value = data.decode() if isinstance(data, bytes) else data
        start_time, stop_time = value.split(" ", 1)
        return cls(start_time=int(start_time), stop_time=int(stop_time))


@dataclasses.dataclass(slots=True)
class Attribute(ByteSerializableObject):
    """Attribute field (a=) as defined by RFC 4566 §5.13."""

    letter: ClassVar[str] = "a"
    session_attr: ClassVar[str] = "attributes"
    is_list: ClassVar[bool] = True
    media_attr: ClassVar[str | None] = "attributes"

    name: str
    value: str | None = None

    def __bytes__(self) -> bytes:
        match self.value:
            case None:
                return self.name.encode()
            case _:
                return f"{self.name}:{self.value}".encode()

    @classmethod
    def parse(cls, data: bytes | str) -> Attribute:
        value = data.decode() if isinstance(data, bytes) else data
        name, _, attr_value = value.partition(":")
        return cls(name=name, value=attr_value or None)


class PayloadTypeSpec(NamedTuple):
    """Typed specification for a static RTP payload type (RFC 3551 §6)."""

    pt: int
    sample_rate: int
    encoding_name: str
    channels: int = 1
    #: Samples per standard 20 ms RTP frame; 0 = variable or not applicable.
    frame_size: int = 0


class StaticPayloadType(PayloadTypeSpec, enum.Enum):
    """Static RTP payload types as defined by RFC 3551 §6.

    Each member's `value` is a `PayloadTypeSpec` carrying the
    payload type number, sample rate, canonical encoding name, channel count,
    and frame size.  Use `from_pt` to look up a member by its PT number.
    """

    #: G.711 µ-law
    PCMU = PayloadTypeSpec(0, 8000, "PCMU", frame_size=160)
    GSM = PayloadTypeSpec(3, 8000, "GSM", frame_size=160)
    G723 = PayloadTypeSpec(4, 8000, "G723", frame_size=160)
    #: DVI4 at 8 kHz
    DVI4_8K = PayloadTypeSpec(5, 8000, "DVI4", frame_size=160)
    #: DVI4 at 16 kHz
    DVI4_16K = PayloadTypeSpec(6, 16000, "DVI4", frame_size=320)
    LPC = PayloadTypeSpec(7, 8000, "LPC", frame_size=160)
    #: G.711 A-law
    PCMA = PayloadTypeSpec(8, 8000, "PCMA", frame_size=160)
    #: RTP clock rate is 8000 per RFC 3551 even though wideband
    G722 = PayloadTypeSpec(9, 8000, "G722", frame_size=160)
    L16_STEREO = PayloadTypeSpec(10, 44100, "L16", 2, frame_size=882)
    L16_MONO = PayloadTypeSpec(11, 44100, "L16", frame_size=882)
    QCELP = PayloadTypeSpec(12, 8000, "QCELP", frame_size=160)
    CN = PayloadTypeSpec(13, 8000, "CN", frame_size=160)
    MPA = PayloadTypeSpec(14, 90000, "MPA")
    G728 = PayloadTypeSpec(15, 8000, "G728", frame_size=160)
    #: DVI4 at 11.025 kHz
    DVI4_11K = PayloadTypeSpec(16, 11025, "DVI4", frame_size=220)
    #: DVI4 at 22.05 kHz
    DVI4_22K = PayloadTypeSpec(17, 22050, "DVI4", frame_size=441)
    G729 = PayloadTypeSpec(18, 8000, "G729", frame_size=160)
    CELB = PayloadTypeSpec(25, 90000, "CelB")
    JPEG = PayloadTypeSpec(26, 90000, "JPEG")
    NV = PayloadTypeSpec(28, 90000, "nv")
    H261 = PayloadTypeSpec(31, 90000, "H261")
    #: MPEG-1 and MPEG-2 video
    MPV = PayloadTypeSpec(32, 90000, "MPV")
    #: MPEG-2 transport stream
    MP2T = PayloadTypeSpec(33, 90000, "MP2T")
    H263 = PayloadTypeSpec(34, 90000, "H263")

    @classmethod
    def from_pt(cls, pt: int) -> StaticPayloadType:
        """Look up a static payload type by its PT number."""
        for member in cls:
            if member.value.pt == pt:
                return member
        raise ValueError(f"No static payload type with PT {pt}")


@dataclasses.dataclass(slots=True)
class RTPPayloadFormat(ByteSerializableObject):
    """RTP payload format descriptor (RFC 3551 §6 / RFC 4566 §6).

    Codec parameters from ``a=rtpmap`` are merged in by the SDP parser.
    Static payload types fall back to the `StaticPayloadType` table.
    Dynamic payload types (PT ≥ 96) require an explicit ``a=rtpmap``.

    Serialises to the ``a=rtpmap`` value when codec fields are present.
    """

    payload_type: int
    fmtp: str | None = None
    encoding_name: str | None = None
    channels: int = 1
    sample_rate: int | None = None

    def __post_init__(self):
        try:
            default = StaticPayloadType.from_pt(self.payload_type)
        except ValueError:
            pass
        else:
            self.sample_rate = self.sample_rate or default.sample_rate
            self.encoding_name = self.encoding_name or default.encoding_name
            self.channels = self.channels or default.channels

    def __bytes__(self) -> bytes:
        base = f"{self.payload_type} {self.encoding_name}/{self.sample_rate}"
        match self.channels:
            case 1:
                return base.encode()
            case _:
                return f"{base}/{self.channels}".encode()

    @classmethod
    def parse(cls, data: bytes | str) -> RTPPayloadFormat:
        value = data.decode() if isinstance(data, bytes) else data
        fmt, _, rest = value.partition(" ")
        parts = rest.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid rtpmap value: {value!r}")
        return cls(
            payload_type=int(fmt),
            encoding_name=parts[0],
            sample_rate=int(parts[1]),
            channels=int(parts[2]) if len(parts) > 2 else 1,
        )

    @classmethod
    def from_pt(cls, pt: int) -> RTPPayloadFormat:
        """Create an `RTPPayloadFormat` from a payload type number."""
        return cls(payload_type=pt)

    @property
    def frame_size(self) -> int:
        """Samples per standard 20 ms RTP frame.

        For static payload types the value comes from `StaticPayloadType`.
        For dynamic payload types (e.g. Opus, PT ≥ 96) it is derived from
        `sample_rate` assuming a 20 ms packetisation interval.
        """
        try:
            spec = StaticPayloadType.from_pt(self.payload_type)
            if spec.frame_size:
                return spec.frame_size
        except ValueError:
            pass
        return (self.sample_rate or 8000) * 20 // 1000


@dataclasses.dataclass(slots=True)
class MediaDescription(ByteSerializableObject):
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
        """Return the `RTPPayloadFormat` for payload type *pt*, or ``None``."""
        target = int(pt)
        return next((f for f in self.fmt if f.payload_type == target), None)

    def apply_attribute(self, attr: Attribute) -> bool:
        """Apply a media-level ``a=`` attribute, returning ``True`` if consumed.

        Handles ``a=rtpmap`` and ``a=fmtp`` by updating the matching
        `RTPPayloadFormat` entry.  Other attributes go to `attributes`.
        """
        if attr.name == "rtpmap" and attr.value is not None:
            rtpfmt = RTPPayloadFormat.parse(attr.value)
            for i, f in enumerate(self.fmt):
                if f.payload_type == rtpfmt.payload_type:
                    # Preserve fmtp if it was already applied out-of-order.
                    if f.fmtp is not None and rtpfmt.fmtp is None:
                        rtpfmt.fmtp = f.fmtp
                    self.fmt[i] = rtpfmt
                    break
            return True
        if attr.name == "fmtp" and attr.value is not None:
            pt_str, _, params = attr.value.partition(" ")
            try:
                pt = int(pt_str)
            except ValueError:
                return False
            for f in self.fmt:
                if f.payload_type == pt:
                    f.fmtp = params
                    return True
        return False

    def _lines(self) -> Generator[str]:
        """Yield each SDP line in canonical field order."""
        yield f"m={self.media} {self.port} {self.proto} {' '.join(str(f.payload_type) for f in self.fmt)}"
        match self.title:
            case str() as title:
                yield f"i={title}"
        match self.connection:
            case ConnectionData() as connection:
                yield f"c={connection}"
        yield from (f"b={b}" for b in self.bandwidths)
        for fmt in self.fmt:
            match fmt.encoding_name, fmt.sample_rate:
                case (str(), int()):
                    yield f"a=rtpmap:{fmt}"
            match fmt.fmtp:
                case str() as fmtp:
                    yield f"a=fmtp:{fmt.payload_type} {fmtp}"
        yield from (f"a={a}" for a in self.attributes)

    def __bytes__(self) -> bytes:
        return "\r\n".join(self._lines()).encode()

    @classmethod
    def parse(cls, data: bytes | str) -> MediaDescription:
        value = data.decode() if isinstance(data, bytes) else data
        lines = value.splitlines()
        first = lines[0].rstrip("\r").removeprefix("m=")
        media_type, port_str, proto, *fmts = first.split()
        fmt = [RTPPayloadFormat.from_pt(int(pt)) for pt in fmts]
        obj = cls(media=media_type, port=int(port_str), proto=proto, fmt=fmt)
        for line in lines[1:]:
            line = line.rstrip("\r")
            if not line or "=" not in line:
                continue
            letter, _, attr_value = line.partition("=")
            match letter:
                case "i":
                    obj.title = attr_value
                case "c":
                    obj.connection = ConnectionData.parse(attr_value)
                case "b":
                    obj.bandwidths.append(Bandwidth.parse(attr_value))
                case "a":
                    attr = Attribute.parse(attr_value)
                    if not obj.apply_attribute(attr):
                        obj.attributes.append(attr)
        return obj
