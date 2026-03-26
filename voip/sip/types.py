import dataclasses
import enum
import ipaddress
import re

__all__ = [
    "CallerID",
    "DigestAlgorithm",
    "DigestQoP",
    "SipUri",
    "SIPStatus",
    "SIPMethod",
]

import typing
import urllib.parse
from collections.abc import Iterator

from voip.types import NetworkAddress

if typing.TYPE_CHECKING:
    pass


@dataclasses.dataclass(slots=True, eq=True)
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
        SipUri(scheme='sip', user='alice', host='example.com', ...)
        >>> SipUri.parse("sips:+15551234567@carrier.com:5061")
        SipUri(scheme='sips', user='+15551234567', host='carrier.com', port=5061, ...)
        >>> SipUri.parse("sip:alice@[::1]:5060")
        SipUri(scheme='sip', user='alice', host=IPv6Address('::1'), port=5060, ...)

    Args:
        scheme: URI scheme — `sip` or `sips`.
        host: Host as a bare string — no brackets for IPv6 addresses.
        user: SIP user part (phone number or username).
        port: Port number. 5061 for `sips:` and 5060 for `sip:`.
        parameters: URI parameters as a mapping of name → value (`None` for flag parameters).
        headers: SIP headers as a mapping of name → value.

    """

    scheme: str
    host: str | ipaddress.IPv6Address | ipaddress.IPv4Address
    user: str | None = None
    password: str | None = dataclasses.field(default=None, repr=False)
    port: int | None = None
    parameters: dict[str, str | None] = dataclasses.field(default_factory=dict)
    headers: dict[str, str] = dataclasses.field(default_factory=dict)

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
        r"(?P<host>(\[[0-9a-fA-F:]+]|[^;?:@\[\]]+))"
        r"(?P<port>:[0-9]+)?"
        r"(?P<parameters>;[^?]+)?"
        r"(?P<headers>\?[^?]+)?$",
        re.IGNORECASE,
    )

    @classmethod
    def parse(cls, value: str) -> SipUri:
        """
        Parse a SIP or SIPS URI string into a `SipUri` instance.

        Returns:
            Parsed `SipUri` instance.

        Raises:
            ValueError: When the URI is malformed (missing scheme, invalid
                characters, unclosed IPv6 bracket, empty host, or invalid port).
        """
        if match := cls.SIP_URL_PATTERN.fullmatch(value):
            host = match.group("host")
            if host.startswith("[") and host.endswith("]"):
                host = host[1:-1]
            host = urllib.parse.unquote(host)

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
                parameters=dict(cls._parse_parameters(match.group("parameters")))
                if match.group("parameters")
                else {},
                headers=dict(cls._parse_headers(match.group("headers")[1:]))
                if match.group("headers")
                else {},
            )
        raise ValueError(f"Invalid SIP URI: {value!r}")

    @classmethod
    def _parse_parameters(cls, params: str) -> Iterator[tuple[str, str | None]]:
        for part in params[1:].split(";"):
            if "=" in part:
                name, val = part.split("=", 1)
                yield urllib.parse.unquote(name), urllib.parse.unquote(val)
            elif part:
                yield urllib.parse.unquote(part), None

    @classmethod
    def _parse_headers(cls, headers: str) -> Iterator[tuple[str, str]]:
        for part in headers.split("&"):
            if "=" in part:
                name, val = part.split("=", 1)
                yield urllib.parse.unquote(name), urllib.parse.unquote(val)
            elif part:
                yield urllib.parse.unquote(part), ""

    def __str__(self) -> str:
        parts = [f"{self.scheme}:"]
        if self.user:
            parts.append(urllib.parse.quote(self.user))
            if self.password:
                parts.append(f":{urllib.parse.quote(self.password)}")
            parts.append("@")
        parts.append(
            f"[{str(self.host)}]"
            if isinstance(self.host, ipaddress.IPv6Address)
            else str(self.host)
        )
        parts.append(f":{self.port}")
        for name, val in self.parameters.items():
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

    @property
    def maddr(self) -> NetworkAddress:
        try:
            return NetworkAddress.parse(self.parameters["maddr"])
        except KeyError:
            return NetworkAddress(self.host, self.port)

    @property
    def ttl(self) -> int | None:
        try:
            return int(self.parameters["ttl"])
        except KeyError:
            return None

    @property
    def transport(self):
        return (
            self.parameters.get("transport", "TLS").upper()
            if self.scheme == "sip"
            else "TLS"
        )


class CallerID(str):
    """SIP From/To header value with structured access and privacy-safe repr.

    Behaves as a plain ``str`` so it is wire-format compatible and can be
    stored in header dicts unchanged.  ``repr()`` returns a short anonymized
    form that shows only the last four characters of the user part and the
    carrier domain — useful for log messages.

    Examples:
        >>> str(CallerID('"08001234567" <sip:08001234567@telefonica.de>;tag=abc'))
        '"08001234567" <sip:08001234567@telefonica.de>;tag=abc'
        >>> repr(CallerID('"08001234567" <sip:08001234567@telefonica.de>;tag=abc'))
        '***4567@telefonica.de'
        >>> repr(CallerID('sip:alice@example.com'))
        '*lice@example.com'
    """

    @property
    def display_name(self) -> str | None:
        """Display name from the From/To header, if present."""
        if m := re.match(r'^"([^"]+)"\s*<|^([^<"]+?)\s*<', self):
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
        user = self.display_name or self.user or ""
        host = self.host or ""
        masked = ("*" * max(0, len(user) - 4)) + user[-4:] if user else "****"
        return f"{masked}@{host}" if host else masked


class SIPStatus(enum.IntEnum):
    """
    SIP Status Codes based on [RFC 3261].

    [RFC 3261]: https://datatracker.ietf.org/doc/html/rfc3261#section-21
    """

    def __new__(cls, value, phrase, description=""):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.phrase = phrase
        obj.description = description
        return obj

    TRYING = (
        100,
        "Trying",
        "The request is being processed. No final response is available yet.",
    )
    RINGING = 180, "Ringing", "The called party is being alerted of the call."
    CALL_IS_BEING_FORWARDED = (
        181,
        "Call Is Being Forwarded",
        "The called party is being alerted of the call, but the call is not yet established.",
    )
    QUEUED = (
        182,
        "Queued",
        "The called party is being alerted of the call, but the call is not yet established.",
    )
    SESSION_PROGRESS = (
        183,
        "Session Progress",
        "The called party is being alerted of the call, but the call is not yet established.",
    )

    OK = 200, "OK", "The request has succeeded."

    MULTIPLE_CHOICES = (
        300,
        "Multiple Choices",
        "The requested resource has multiple representations, each with its own specific location.",
    )
    MOVED_PERMANENTLY = (
        301,
        "Moved Permanently",
        "The requested resource has been assigned a new permanent URI and any future references to this resource ought to use one of the returned URIs.",
    )
    MOVED_TEMPORARILY = (
        302,
        "Moved Temporarily",
        "The requested resource is temporarily unavailable and the server is asking the client to try again later.",
    )
    USE_PROXY = (
        305,
        "Use Proxy",
        "The requested resource is available only through a proxy, the address for which is provided in the response.",
    )
    ALTERNATIVE_SERVICE = (
        380,
        "Alternative Service",
        "The server has fulfilled a request for the service indicated by the URI.",
    )

    BAD_REQUEST = (
        400,
        "Bad Request",
        "The request has bad syntax or cannot be fulfilled due to bad syntax.",
    )
    UNAUTHORIZED = 401, "Unauthorized", "The request requires user authentication."
    PAYMENT_REQUIRED = 402, "Payment Required", "Further action is required."
    FORBIDDEN = (
        403,
        "Forbidden",
        "The server understood the request but refuses to fulfill it.",
    )
    NOT_FOUND = 404, "Not Found", "The requested resource could not be found."
    METHOD_NOT_ALLOWED = (
        405,
        "Method Not Allowed",
        "The method specified in the Request-URI is not allowed for the resource identified by the request URI.",
    )
    NOT_ACCEPTABLE = (
        406,
        "Not Acceptable",
        "The server cannot produce a response matching the Accept headers.",
    )
    PROXY_AUTHENTICATION_REQUIRED = (
        407,
        "Proxy Authentication Required",
        "The client must authenticate itself with the proxy.",
    )
    REQUEST_TIMEOUT = (
        408,
        "Request Timeout",
        "The server timed out waiting for the request.",
    )
    GONE = (
        410,
        "Gone",
        "The requested resource is no longer available at the server and no longer exists.",
    )
    REQUEST_ENTITY_TOO_LARGE = (
        413,
        "Request Entity Too Large",
        "The server will not accept the request, because the entity of the request is too large.",
    )
    REQUEST_URI_TOO_LONG = (
        414,
        "Request-URI Too Long",
        "The server will not accept the request, because the Request-URI is too long.",
    )
    UNSUPPORTED_MEDIA_TYPE = (
        415,
        "Unsupported Media Type",
        "The server will not accept the request, because the media type of the request is unsupported.",
    )
    UNSUPPORTED_URI_SCHEME = (
        416,
        "Unsupported URI Scheme",
        "The server will not accept the request, because the URI scheme of the request is unsupported.",
    )
    BAD_EXTENSION = (
        420,
        "Bad Extension",
        "This status code indicates that the server does not recognize the value of any of the parameters that it needs to understand in the request.",
    )
    EXTENSION_REQUIRED = (
        421,
        "Extension Required",
        "This status code indicates that the server requires the client to identify itself (usually, using the Contact header field) before it will proceed with the request.",
    )
    INTERVAL_TOO_BRIEF = (
        423,
        "Interval Too Brief",
        "This status code indicates that the server is unwilling to process the request because either an individual header field, or all the header fields collectively, are too large.",
    )
    TEMPORARILY_UNAVAILABLE = (
        480,
        "Temporarily Unavailable",
        "This status code indicates that the server is currently unable to handle the request due to a temporary overloading or maintenance of the server.",
    )
    CALL_TRANSACTION_DOES_NOT_EXIST = (
        481,
        "Call/Transaction Does Not Exist",
        "This status code indicates that the server has received a final response for the transaction which it is still attempting to complete.",
    )
    LOOP_DETECTED = (
        482,
        "Loop Detected",
        "This status code indicates that the server has detected an infinite loop while processing the request.",
    )
    TOO_MANY_HOPS = (
        483,
        "Too Many Hops",
        "This status code indicates that the server has exceeded the maximum number of hops allowed in the request URI.",
    )
    ADDRESS_INCOMPLETE = (
        484,
        "Address Incomplete",
        "This status code indicates that the server has received a final response for the transaction which it is still attempting to complete, but has an invalid value for one or more of the header fields included in the request message.",
    )
    AMBIGUOUS = (
        485,
        "Ambiguous",
        "This status code indicates that the server cannot decide on a response to the request because multiple responses are possible.",
    )
    BUSY_HERE = (
        486,
        "Busy Here",
        "This status code indicates that the server is busy here.",
    )
    REQUEST_TERMINATED = (
        487,
        "Request Terminated",
        "This status code indicates that the server has received a final response for the transaction which it is still attempting to complete, but has received a termination request for that transaction from the client.",
    )
    NOT_ACCEPTABLE_HERE = (
        488,
        "Not Acceptable Here",
        "This status code indicates that the server is not able to produce a response which is acceptable to the client, according to the proactive negotiation header fields received in the request, and the server is unwilling to supply a default reason phrase.",
    )
    REQUEST_PENDING = (
        491,
        "Request Pending",
        "This status code indicates that the server has received a final response for the transaction which it is still attempting to complete, but has not yet delivered that response to the client.",
    )
    UNDECIPHERABLE = (
        493,
        "Undecipherable",
        "This status code indicates that the server was unable to decrypt a message after performing the necessary decryption(s).",
    )

    SERVER_INTERNAL_ERROR = (
        500,
        "Server Internal Error",
        "The server encountered an unexpected condition which prevented it from fulfilling the request.",
    )
    NOT_IMPLEMENTED = (
        501,
        "Not Implemented",
        "The server does not support the functionality required to fulfill the request.",
    )
    BAD_GATEWAY = (
        502,
        "Bad Gateway",
        "The server, while acting as a gateway or proxy, received an invalid response from the upstream server it accessed in attempting to fulfill the request.",
    )
    SERVICE_UNAVAILABLE = (
        503,
        "Service Unavailable",
        "The server is currently unable to handle the request due to a temporary overloading or maintenance of the server.",
    )
    SERVER_TIME_OUT = (
        504,
        "Server Time-out",
        "The server, while acting as a gateway or proxy, did not receive a timely response from the upstream server specified by the URI (e.g., HTTP, FTP, LDAP) or some other auxiliary server (e.g., DNS) it needed to access in attempting to complete the request.",
    )
    VERSION_NOT_SUPPORTED = (
        505,
        "Version Not Supported",
        "The server does not support, or refuses to support, the protocol version that was used in the request message.",
    )
    MESSAGE_TOO_LARGE = (
        513,
        "Message Too Large",
        "The server is unwilling to process the request because its header fields are too large.",
    )

    BUSY_EVERYWHERE = (
        600,
        "Busy Everywhere",
        "The server is not able to process the request because it is busy.  For example, this error might be given if a server is overloaded with requests and is unable to process one of the requests.",
    )
    DECLINE = 603, "Decline", "The call has been declined."
    DOES_NOT_EXIST_ANYWHERE = (
        604,
        "Does Not Exist Anywhere",
        "The server has received a final response for the transaction which it is still attempting to complete, but has received a termination request for that transaction from a server which it does not control.",
    )
    NOT_ACCEPTABLE_ANYWHERE = (
        606,
        "Not Acceptable",
        "The server is not able to produce a response which is acceptable to the client, according to the proactive negotiation header fields received in the request, and the server is unwilling to supply a default reason phrase.",
    )


class SIPMethod(enum.StrEnum):
    """SIP methods and descriptions as defined in [RFC 3261].

    Extended in [RFC 3262], [RFC 3265], [RFC 3515], [RFC 3428], and [RFC 3311].

    [RFC 3261]: https://tools.ietf.org/html/rfc3261
    [RFC 3262]: https://tools.ietf.org/html/rfc3262
    [RFC 3265]: https://tools.ietf.org/html/rfc3265
    [RFC 3515]: https://tools.ietf.org/html/rfc3515
    [RFC 3428]: https://tools.ietf.org/html/rfc3428
    [RFC 3311]: https://tools.ietf.org/html/rfc3311
    """

    def __new__(cls, value: str, description: str = "") -> SIPMethod:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj

    INVITE = "INVITE", "The client is requesting to initiate a call."
    ACK = "ACK", "The client is acknowledging the receipt of a previous request."
    BYE = "BYE", "The client is requesting to end the call."
    CANCEL = "CANCEL", "The client is requesting to cancel a previous request."
    REGISTER = (
        "REGISTER",
        "The client requests that the server register itself with the server's registration agent.",
    )
    OPTIONS = (
        "OPTIONS",
        "The client requests information about the server's capabilities or configuration.",
    )
    NOTIFY = "NOTIFY", "The client is requesting to send a notification to the server."
    SUBSCRIBE = "SUBSCRIBE", "The client is requesting to subscribe to a resource."
    PUBLISH = "PUBLISH", "The client is requesting to publish a resource."
    REFER = (
        "REFER",
        "The client is requesting that the server refer the client to another resource.",
    )
    PRACK = (
        "PRACK",
        "The client is requesting to confirm the receipt of a previous request.",
    )
    INFO = "INFO", "The client is requesting information about a session."
    MESSAGE = "MESSAGE", "The client is requesting to send a message."
    UPDATE = "UPDATE", "The client is requesting to update a resource."


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


def _mask_caller(header: str) -> str:
    """Return a privacy-safe label from a SIP From/To header value.

    Strips the `tag=` parameter, extracts the display name or SIP user part,
    and replaces all but the last four characters with `*`.

    Examples:
    ```
    >>> _mask_caller('"08001234567" <sip:08001234567@example.com>;tag=abc')
    '*******4567'
    >>> _mask_caller('sip:alice@example.com')
    '*lice'
    ```
    """
    # Drop the tag and any subsequent parameters
    value = header.split(";")[0].strip()
    # Extract display name: "Name" <sip:…> or Name <sip:…>
    m = re.match(r'^"?([^"<]+?)"?\s*<', value)
    name = m.group(1).strip() if m else None
    if not name:
        # Bare or angle-bracket URI: sip:user@host or <sip:user@host>
        m = re.search(r"sips?:([^@>;\s]+)", value)
        name = m.group(1) if m else value
    if len(name) > 4:
        return "*" * (len(name) - 4) + name[-4:]
    return name
