"""SIP message types as defined by RFC 3261."""

import abc
import dataclasses
import logging
import platform
import typing

from urllib3 import HTTPHeaderDict

import voip
from voip.sdp.messages import SessionDescription

from ..types import ByteSerializableObject
from .types import CallerID, SIPMethod, SIPStatus, SipURI

if typing.TYPE_CHECKING:
    from voip.sip.dialog import Dialog

__all__ = ["Request", "Response", "Message"]

logger = logging.getLogger("voip.sip")

#: Headers whose values are parsed as `CallerID` objects.
CALLER_IDS_HEADERS = frozenset({"From", "To", "Route", "Record-Route", "Contact"})

#: User-Agent header value to use in generated messages.
USER_AGENT = (
    f"VoIP/{voip.__version__}"
    f" {platform.python_implementation()}/{platform.python_version()}"
    f" {platform.system()}/{platform.platform()}"
)


class SIPHeaderDict(ByteSerializableObject, HTTPHeaderDict):
    """Header map for SIP messages, mapping header names to their values."""

    def __bytes__(self) -> bytes:
        return b"".join(f"{name}: {value}\r\n".encode() for name, value in self.items())

    @classmethod
    def parse(cls, data: bytes) -> SIPHeaderDict:
        self = SIPHeaderDict()
        for line in data.decode().split("\r\n"):
            name, sep, value = line.partition(":")
            if not sep:
                raise ValueError(f"Invalid header: {line!r}")
            name = name.strip()
            value = value.strip()
            self.add(name, CallerID(value) if name in CALLER_IDS_HEADERS else value)
        return self


@dataclasses.dataclass(slots=True, kw_only=True)
class Message(ByteSerializableObject, abc.ABC):
    """
    A SIP message [RFC 3261 §7].

    [RFC 3261 §7]: https://datatracker.ietf.org/doc/html/rfc3261#section-7
    """

    headers: SIPHeaderDict | dict[str, str | CallerID] = dataclasses.field(
        default_factory=SIPHeaderDict, repr=False
    )
    body: SessionDescription | None = dataclasses.field(default=None, repr=False)
    version: str = "SIP/2.0"

    def __post_init__(self):
        if not isinstance(self.headers, SIPHeaderDict):
            self.headers: SIPHeaderDict = SIPHeaderDict(dict(self.headers))

    @classmethod
    def parse(cls, data: bytes) -> Request | Response:
        header_section, _, body = data.partition(b"\r\n\r\n")
        first_line, _, header_section = header_section.partition(b"\r\n")
        headers = SIPHeaderDict.parse(header_section)
        parts = first_line.split(b" ", 2)
        if first_line.startswith(b"SIP/"):
            version, status_code_str, reason = parts
            return Response(
                status_code=int(status_code_str),
                phrase=reason.decode("ascii"),
                headers=headers,
                body=cls._parse_body(headers, body),
                version=version.decode("ascii"),
            )
        try:
            method, uri, version = parts
        except ValueError:
            raise ValueError(f"Invalid SIP message first line: {data!r}")
        return Request(
            method=method.decode("ascii"),
            uri=uri.decode("ascii"),
            headers=headers,
            body=cls._parse_body(headers, body),
            version=version.decode("ascii"),
        )

    @staticmethod
    def _parse_body(headers: dict[str, str], body: bytes) -> SessionDescription | None:
        """Parse the body according to the Content-Type header."""
        if headers.get("Content-Type") == "application/sdp" and body:
            return SessionDescription.parse(body)
        return None

    def __bytes__(self) -> bytes:
        if raw_body := bytes(self.body) if self.body is not None else b"":
            self.headers["Content-Length"] = str(len(raw_body))
        return b"\r\n".join(
            (self._first_line().encode(), bytes(self.headers), raw_body)
        )

    @property
    def branch(self) -> str | None:
        """Branch parameter from the top Via header (RFC 3261 §20.42)."""
        _, uri = self.headers["Via"].split()
        return SipURI.parse(f"sip:{uri}").parameters["branch"]

    @property
    def remote_tag(self) -> str | None:
        """To-tag used with From-tag to identify the SIP dialog (RFC 3261 §12.2.2)."""
        return CallerID(self.headers["To"]).tag

    @property
    def local_tag(self) -> str:
        """From-tag used with To-tag to identify the SIP dialog (RFC 3261 §12.2.2)."""
        return CallerID(self.headers["From"]).tag

    @property
    def sequence(self) -> int:
        """Sequence number a transaction within a dialog."""
        return int(self.headers["CSeq"].split()[0])

    @abc.abstractmethod
    def _first_line(self) -> str: ...


@dataclasses.dataclass(kw_only=True)
class Request(Message):
    """
    A SIP request message [RFC 3261 §7.1].

    [RFC 3261 §7.1]: https://datatracker.ietf.org/doc/html/rfc3261#section-7.1
    """

    method: SIPMethod | str
    uri: SipURI | str

    def _first_line(self) -> str:
        return f"{self.method} {self.uri} {self.version}"

    @classmethod
    def from_dialog(cls, *, dialog: Dialog, headers, **kwargs) -> Request:
        """Create a request from a dialog, copying relevant headers."""
        return cls(
            headers=headers | dialog.headers,
            **kwargs,
        )


@dataclasses.dataclass(kw_only=True)
class Response(Message):
    """
    A SIP response message [RFC 3261 §7.2].

    [RFC 3261 §7.2]: https://datatracker.ietf.org/doc/html/rfc3261#section-7.2
    """

    status_code: SIPStatus | int
    phrase: str

    def _first_line(self) -> str:
        return f"{self.version} {self.status_code} {self.phrase}"

    @classmethod
    def from_request(
        cls, request: Request, *, headers=None, dialog: Dialog = None, **kwargs
    ) -> Response:
        """Create a response from a request, copying relevant headers."""
        headers = {
            "Via": request.headers["Via"],
            "From": request.headers["From"],
            "To": f"{request.headers['To']};tag={dialog.remote_tag}"
            if dialog and dialog.remote_tag
            else request.headers["To"],
            "Call-ID": request.headers["Call-ID"],
            "CSeq": request.headers["CSeq"],
        } | (headers or {})
        return cls(headers=headers, **kwargs)
