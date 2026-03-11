"""SDP field types as defined by RFC 4566."""

from __future__ import annotations

import dataclasses
import enum
from typing import ClassVar, NamedTuple, Protocol, runtime_checkable

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


class PayloadTypeSpec(NamedTuple):
    """Typed specification for a static RTP payload type (RFC 3551 §6)."""

    pt: int
    sample_rate: int
    encoding_name: str
    channels: int = 1


class StaticPayloadType(PayloadTypeSpec, enum.Enum):
    """Static RTP payload types as defined by RFC 3551 §6.

    Each member's :attr:`value` is a :class:`PayloadTypeSpec` carrying the
    payload type number, sample rate, canonical encoding name, and channel
    count.  Use :meth:`from_pt` to look up a member by its PT number.
    """

    #: G.711 µ-law
    PCMU = PayloadTypeSpec(0, 8000, "PCMU")
    GSM = PayloadTypeSpec(3, 8000, "GSM")
    G723 = PayloadTypeSpec(4, 8000, "G723")
    #: DVI4 at 8 kHz
    DVI4_8K = PayloadTypeSpec(5, 8000, "DVI4")
    #: DVI4 at 16 kHz
    DVI4_16K = PayloadTypeSpec(6, 16000, "DVI4")
    LPC = PayloadTypeSpec(7, 8000, "LPC")
    #: G.711 A-law
    PCMA = PayloadTypeSpec(8, 8000, "PCMA")
    #: RTP clock rate is 8000 per RFC 3551 even though wideband
    G722 = PayloadTypeSpec(9, 8000, "G722")
    L16_STEREO = PayloadTypeSpec(10, 44100, "L16", 2)
    L16_MONO = PayloadTypeSpec(11, 44100, "L16")
    QCELP = PayloadTypeSpec(12, 8000, "QCELP")
    CN = PayloadTypeSpec(13, 8000, "CN")
    MPA = PayloadTypeSpec(14, 90000, "MPA")
    G728 = PayloadTypeSpec(15, 8000, "G728")
    #: DVI4 at 11.025 kHz
    DVI4_11K = PayloadTypeSpec(16, 11025, "DVI4")
    #: DVI4 at 22.05 kHz
    DVI4_22K = PayloadTypeSpec(17, 22050, "DVI4")
    G729 = PayloadTypeSpec(18, 8000, "G729")
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


@dataclasses.dataclass(init=False, slots=True, repr=False)
class RTPPayloadFormat:
    """RTP payload format descriptor (RFC 3551 §6 / RFC 4566 §6).

    Carries the numeric payload type together with the optional codec
    parameters that are conveyed by an ``a=rtpmap`` attribute::

        a=rtpmap:111 opus/48000/2
        a=rtpmap:8 PCMA/8000

    For static payload types the codec parameters are pre-defined by
    RFC 3551 and may be absent from the SDP; :attr:`sample_rate` falls
    back to the :class:`StaticPayloadType` table in that case.  For
    dynamic payload types (PT ≥ 96) an explicit ``a=rtpmap`` is always
    required.

    Instances serialise to the ``a=rtpmap`` attribute *value* (without the
    leading ``a=rtpmap:`` prefix) when all codec fields are present.
    """

    payload_type: int
    encoding_name: str | None
    _sample_rate: int | None
    channels: int
    fmtp: str | None

    def __init__(
        self,
        payload_type: int,
        encoding_name: str | None = None,
        sample_rate: int | None = None,
        channels: int = 1,
        fmtp: str | None = None,
    ) -> None:
        self.payload_type = payload_type
        self.encoding_name = encoding_name
        self._sample_rate = sample_rate
        self.channels = channels
        self.fmtp = fmtp
        if self._sample_rate is None:
            try:
                self._sample_rate = StaticPayloadType.from_pt(
                    self.payload_type
                ).sample_rate
            except ValueError:
                pass  # Dynamic PT; sample_rate will be supplied via a=rtpmap.

    @property
    def sample_rate(self) -> int:
        """RTP clock rate in Hz.

        Raises:
            ValueError: If the sample rate is not known — i.e. a dynamic
                payload type with no ``a=rtpmap`` attribute parsed yet.
        """
        if self._sample_rate is None:
            raise ValueError(
                f"No sample rate for payload type {self.payload_type}; "
                "supply an explicit a=rtpmap attribute"
            )
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int | None) -> None:
        self._sample_rate = value

    def __repr__(self) -> str:
        return (
            f"RTPPayloadFormat("
            f"payload_type={self.payload_type!r}, "
            f"encoding_name={self.encoding_name!r}, "
            f"sample_rate={self._sample_rate!r}, "
            f"channels={self.channels!r}, "
            f"fmtp={self.fmtp!r})"
        )

    def __str__(self) -> str:
        """Serialize to an ``a=rtpmap`` attribute value (without the ``a=`` prefix).

        Example: ``"111 opus/48000/2"`` or ``"8 PCMA/8000"``.

        Raises:
            ValueError: If *encoding_name* or *sample_rate* are not explicitly set.
        """
        base = f"{self.payload_type} {self.encoding_name}/{self.sample_rate}"
        return f"{base}/{self.channels}" if self.channels != 1 else base

    @classmethod
    def parse(cls, value: str) -> RTPPayloadFormat:
        """Parse an ``a=rtpmap`` attribute value into an :class:`RTPPayloadFormat`.

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
            sample_rate=int(parts[1]),
            channels=int(parts[2]) if len(parts) > 2 else 1,
        )

    @classmethod
    def from_pt(cls, pt: int) -> RTPPayloadFormat:
        """Create a stub :class:`RTPPayloadFormat` from a payload type number.

        Only the *payload_type* is set.  Codec parameters (*encoding_name*,
        *sample_rate*, *channels*) remain ``None`` / default until an explicit
        ``a=rtpmap`` attribute is parsed and merged in by the SDP parser.
        :attr:`sample_rate` will fall back to :class:`StaticPayloadType` for
        known static payload types.

        Args:
            pt: RTP payload type number (0–127).
        """
        return cls(payload_type=pt)


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
        """Return the :class:`RTPPayloadFormat` for the given payload type, or ``None``.

        Args:
            pt: Payload type as an integer or its string representation
                (e.g. ``111`` or ``"111"``).

        Returns:
            The matching :class:`RTPPayloadFormat` from :attr:`fmt`, or
            ``None`` if not found.
        """
        target = int(pt)
        return next((f for f in self.fmt if f.payload_type == target), None)

    def apply_attribute(self, attr: Attribute) -> bool:
        """Apply a media-level ``a=`` attribute to this description.

        Handles ``a=rtpmap`` and ``a=fmtp`` by updating the matching
        :class:`RTPPayloadFormat` entry in :attr:`fmt`.  All other
        attributes are appended to :attr:`attributes`.

        Args:
            attr: The parsed :class:`Attribute` to apply.

        Returns:
            ``True`` when the attribute was consumed as a format-specific
            attribute (``rtpmap`` or ``fmtp``), ``False`` otherwise.
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
            if f.encoding_name is not None and f._sample_rate is not None:
                lines.append(f"a=rtpmap:{f}")
            if f.fmtp is not None:
                lines.append(f"a=fmtp:{f.payload_type} {f.fmtp}")
        lines.extend(f"a={a}" for a in self.attributes)
        return "\r\n".join(lines)

    @classmethod
    def parse(cls, value: str) -> MediaDescription:
        """Parse an ``m=`` line value into a :class:`MediaDescription`.

        *value* may be either just the ``m=`` line's value (e.g.
        ``"audio 49170 RTP/AVP 0 111"``) or a multi-line block as produced
        by :meth:`__str__` — i.e. including a leading ``m=`` and subsequent
        ``i=``, ``c=``, ``b=``, ``a=rtpmap``, ``a=fmtp`` and generic ``a=``
        lines.  This allows :meth:`parse` and :meth:`__str__` to round-trip
        without going through :class:`~voip.sdp.messages.SessionDescription`.

        Args:
            value: The ``m=`` line value or a full media-section block.
        """
        lines = value.splitlines()
        first = lines[0].rstrip("\r")
        if first.startswith("m="):
            first = first[2:]
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
