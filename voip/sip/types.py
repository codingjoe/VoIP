from __future__ import annotations

import dataclasses
import enum
import ipaddress
import re

__all__ = ["CallerID", "DigestAlgorithm", "DigestQoP", "SipUri", "Status"]

import typing
import urllib.parse
from collections.abc import Generator
from typing import Any


@dataclasses.dataclass(slots=True, eq=True, order=True, unsafe_hash=True)
class SipUri:
    """A parsed SIP or SIPS URI per [RFC 3261 §19.1].

    Format: ``sip:user:password@host:port;uri-parameters?headers``

    The `parse` classmethod decodes a raw SIP URI string into structured
    fields.  IPv6 addresses in the host part must be enclosed in square
    brackets per [RFC 2732] (e.g. ``sip:alice@[::1]:5060``); the stored
    `host` is the bare address without brackets.

    [RFC 3261 §19.1]: https://datatracker.ietf.org/doc/html/rfc3261#section-19.1
    [RFC 2732]: https://datatracker.ietf.org/doc/html/rfc2732

    Examples:
        >>> SipUri.parse("sip:alice@example.com")
        SipURI(scheme='sip', user='alice', host='example.com', ...)
        >>> SipUri.parse("sips:+15551234567@carrier.com:5061")
        SipURI(scheme='sips', user='+15551234567', host='carrier.com', port=5061, ...)
        >>> SipUri.parse("sip:alice@[::1]:5060")
        SipURI(scheme='sip', user='alice', host='::1', port=5060, ...)
    """

    scheme: str
    """URI scheme — ``"sip"`` or ``"sips"``."""
    host: str | ipaddress.IPv6Address | ipaddress.IPv4Address
    """Host as a bare string — no brackets for IPv6 addresses."""
    user: str | None = None
    """SIP user part (phone number or username)."""
    password: str | None = None
    """Optional password from the user-info component (``user:password@host``)."""
    port: int | None = None
    """Optional port number; ``None`` when not present in the URI."""
    uri_parameters: dict[str, str | None] = dataclasses.field(default_factory=dict)
    """URI parameters as a mapping of name → value (``None`` for flag parameters)."""
    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    """SIP URI headers (``?Header=value``) as a mapping of name → value."""

    def __post_init__(self):
        self.port = (
            self.port
            if self.port is not None
            else 5061
            if self.scheme == "sips"
            else 5060
        )
        try:
            self.host = ipaddress.ip_address(self.host)
        except ValueError:
            pass

    SIP_URL_PATTERN: typing.ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<scheme>sips?):"
        r"((?P<user>[^@;:]+)(?P<password>:[^@;]*)?@)?"
        r"(?P<host>([^;?:@]+|\[[0-9a-fA-F:]+\]))"
        r"(?P<port>:[0-9]+)?"
        r"(?P<uri_parameters>;[^?]+)?"
        r"(?P<headers>\?[^?]+)?$",
        re.IGNORECASE,
    )

    @classmethod
    def parse(cls, value: str) -> SipUri:
        """Parse a SIP or SIPS URI string into a `SipURI` instance.

        Implements the full ``sip:user:password@host:port;uri-parameters?headers``
        grammar from [RFC 3261 §19.1].  IPv6 host literals must be bracketed
        per [RFC 2732], e.g. ``sip:alice@[::1]:5060``.

        [RFC 3261 §19.1]: https://datatracker.ietf.org/doc/html/rfc3261#section-19.1
        [RFC 2732]: https://datatracker.ietf.org/doc/html/rfc2732

        Args:
            value: Raw SIP URI string.

        Returns:
            Parsed `SipURI` instance.

        Raises:
            ValueError: When the URI is malformed (missing scheme, missing
                ``user@host``, unclosed IPv6 bracket, empty host, or invalid port).
        """
        if match := cls.SIP_URL_PATTERN.fullmatch(value):
            host = match.group("host")
            if host.startswith("[") and host.endswith("]"):
                host = host[1:-1]
            host = urllib.parse.unquote(host)
            try:
                ipaddress.ip_address(host)
            except ValueError:
                pass  # Not an IP address, treat as a regular hostname

            return cls(
                scheme=match.group("scheme").lower(),
                user=urllib.parse.unquote(match.group("user"))
                if match.group("user")
                else None,
                host=host,
                password=urllib.parse.unquote(match.group("password")[1:])
                if match.group("password")
                else None,
                port=int(match.group("port")[1:]) if match.group("port") else None,
                uri_parameters=dict(
                    cls._parse_uri_parameters(match.group("uri_parameters"))
                )
                if match.group("uri_parameters")
                else {},
                headers=dict(cls._parse_headers(match.group("headers")[1:]))
                if match.group("headers")
                else {},
            )
        raise ValueError(f"Invalid SIP URI: {value!r}")

    @classmethod
    def _parse_uri_parameters(
        cls, params: str
    ) -> Generator[tuple[str, str | None], Any]:
        """Parse SIP URI parameters from a query string format."""
        for part in params[1:].split(";"):
            if "=" in part:
                name, val = part.split("=", 1)
                yield urllib.parse.unquote(name), urllib.parse.unquote(val)
            elif part:
                yield urllib.parse.unquote(part), None

    @classmethod
    def _parse_headers(cls, headers: str) -> Generator[tuple[str, str], Any]:
        """Parse SIP URI headers from a query string format."""
        for part in headers.split("&"):
            if "=" in part:
                name, val = part.split("=", 1)
                yield urllib.parse.unquote(name), urllib.parse.unquote(val)
            elif part:
                yield urllib.parse.unquote(part), ""

    def __str__(self) -> str:
        """Return the SIP URI as a string."""
        parts = [f"{self.scheme}:"]
        if self.user:
            parts.append(urllib.parse.quote(self.user))
            if self.password:
                parts.append(f":{urllib.parse.quote(self.password)}")
            parts.append("@")
        print(type(self.host), self.host)
        parts.append(
            f"[{str(self.host)}]"
            if isinstance(self.host, ipaddress.IPv6Address)
            else str(self.host)
        )
        parts.append(f":{self.port}")
        for name, val in self.uri_parameters.items():
            if val is not None:
                parts.append(f";{urllib.parse.quote(name)}={urllib.parse.quote(val)}")
            else:
                parts.append(f";{urllib.parse.quote(name)}")
        if self.headers:
            parts.append("?")
            parts.append(
                "&".join(
                    f"{urllib.parse.quote(name)}={urllib.parse.quote(val)}"
                    for name, val in self.headers.items()
                )
            )
        return "".join(parts)


class CallerID(str):
    """SIP From/To header value with structured access and privacy-safe repr.

    Behaves as a plain ``str`` so it is wire-format compatible and can be
    stored in header dicts unchanged.  ``repr()`` returns a short anonymized
    form that shows only the last four characters of the user part and the
    carrier domain — useful for log messages.

    Examples:
        >>> str(CallerID('"015114455910" <sip:015114455910@telefonica.de>;tag=abc'))
        '"015114455910" <sip:015114455910@telefonica.de>;tag=abc'
        >>> repr(CallerID('"015114455910" <sip:015114455910@telefonica.de>;tag=abc'))
        '****5910@telefonica.de'
        >>> repr(CallerID('sip:alice@example.com'))
        '*lice@example.com'
    """

    @property
    def display_name(self) -> str | None:
        """Display name from the From/To header, if present."""
        m = re.match(r'^"([^"]+)"\s*<|^([^<"]+?)\s*<', self)
        if m:
            return (m.group(1) or m.group(2) or "").strip() or None
        return None

    @property
    def user(self) -> str | None:
        """SIP user part (phone number or username)."""
        m = re.search(r"sips?:([^@>;\s]+)@", self)
        return m.group(1) if m else None

    @property
    def host(self) -> str | None:
        """Carrier domain extracted from the SIP URI."""
        m = re.search(r"sips?:[^@>;\s]+@([^>;)\s,]+)", self)
        return m.group(1) if m else None

    @property
    def tag(self) -> str | None:
        """Dialog tag parameter value, if present."""
        m = re.search(r";tag=([^\s;]+)", self)
        return m.group(1) if m else None

    def __repr__(self) -> str:
        """Anonymized label: last 4 chars of user + carrier domain."""
        user = self.display_name or self.user or ""
        host = self.host or ""
        masked = ("*" * max(0, len(user) - 4)) + user[-4:] if user else "****"
        return f"{masked}@{host}" if host else masked


Status = enum.IntEnum(
    value="Status",
    names={
        # 1xx Provisional
        "Trying": 100,
        "Ringing": 180,
        "Call Is Being Forwarded": 181,
        "Queued": 182,
        "Session Progress": 183,
        # 2xx Success
        "OK": 200,
        # 3xx Redirection
        "Multiple Choices": 300,
        "Moved Permanently": 301,
        "Moved Temporarily": 302,
        "Use Proxy": 305,
        "Alternative Service": 380,
        # 4xx Client Failure
        "Bad Request": 400,
        "Unauthorized": 401,
        "Payment Required": 402,
        "Forbidden": 403,
        "Not Found": 404,
        "Method Not Allowed": 405,
        "Not Acceptable": 406,
        "Proxy Authentication Required": 407,
        "Request Timeout": 408,
        "Gone": 410,
        "Request Entity Too Large": 413,
        "Request-URI Too Long": 414,
        "Unsupported Media Type": 415,
        "Unsupported URI Scheme": 416,
        "Bad Extension": 420,
        "Extension Required": 421,
        "Interval Too Brief": 423,
        "Temporarily Unavailable": 480,
        "Call/Transaction Does Not Exist": 481,
        "Loop Detected": 482,
        "Too Many Hops": 483,
        "Address Incomplete": 484,
        "Ambiguous": 485,
        "Busy Here": 486,
        "Request Terminated": 487,
        "Not Acceptable Here": 488,
        "Request Pending": 491,
        "Undecipherable": 493,
        # 5xx Server Failure
        "Server Internal Error": 500,
        "Not Implemented": 501,
        "Bad Gateway": 502,
        "Service Unavailable": 503,
        "Server Time-out": 504,
        "Version Not Supported": 505,
        "Message Too Large": 513,
        # 6xx Global Failure
        "Busy Everywhere": 600,
        "Decline": 603,
        "Does Not Exist Anywhere": 604,
    },
)


class DigestAlgorithm(enum.StrEnum):
    """Hash algorithms for SIP Digest Authentication (RFC 3261, RFC 8760).

    RFC 8760 deprecates MD5 in favour of SHA-256 and SHA-512-256.
    """

    MD5 = "MD5"
    MD5_SESS = "MD5-sess"
    SHA_256 = "SHA-256"
    SHA_256_SESS = "SHA-256-sess"
    SHA_512_256 = "SHA-512-256"
    SHA_512_256_SESS = "SHA-512-256-sess"


class DigestQoP(enum.StrEnum):
    """Quality of protection values for HTTP Digest Authentication (RFC 2617)."""

    AUTH = "auth"
    AUTH_INT = "auth-int"
