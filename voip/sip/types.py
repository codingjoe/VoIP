from __future__ import annotations

import enum
import re

__all__ = ["CallerID", "DigestAlgorithm", "DigestQoP", "Status"]


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
