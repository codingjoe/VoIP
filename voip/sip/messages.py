"""SIP message types as defined by RFC 3261."""

import abc
import dataclasses
import datetime
import socket
import uuid

from voip.sdp.messages import SessionDescription

from ..types import ByteSerializableObject
from .types import CallerID, SIPMethod, SIPStatus, SipUri

__all__ = ["Request", "Response", "Message", "Dialog"]

#: Headers whose values are parsed as `CallerID` objects.
_CALLER_HEADERS = frozenset({"From", "To"})


@dataclasses.dataclass(slots=True, kw_only=True)
class Message(ByteSerializableObject, abc.ABC):
    """
    A SIP message [RFC 3261 §7].

    [RFC 3261 §7]: https://datatracker.ietf.org/doc/html/rfc3261#section-7
    """

    headers: dict[str, str | CallerID] = dataclasses.field(
        default_factory=dict, repr=False
    )
    body: SessionDescription | None = dataclasses.field(default=None, repr=False)
    version: str = "SIP/2.0"

    @classmethod
    def parse(cls, data: bytes) -> Request | Response:
        header_section, _, body = data.partition(b"\r\n\r\n")
        lines = header_section.decode().split("\r\n")
        first_line, *header_lines = lines
        headers = {}
        for line in header_lines:
            name, sep, value = line.partition(":")
            if not sep:
                continue
            name = name.strip()
            value = value.strip()
            headers[name] = CallerID(value) if name in _CALLER_HEADERS else value
        parts = first_line.split(" ", 2)
        if first_line.startswith("SIP/"):
            version, status_code_str, reason = parts
            return Response(
                status_code=int(status_code_str),
                phrase=reason,
                headers=headers,
                body=cls._parse_body(headers, body),
                version=version,
            )
        try:
            method, uri, version = parts
        except ValueError:
            raise ValueError(f"Invalid SIP message first line: {data!r}")
        return Request(
            method=method,
            uri=uri,
            headers=headers,
            body=cls._parse_body(headers, body),
            version=version,
        )

    @staticmethod
    def _parse_body(headers: dict[str, str], body: bytes) -> SessionDescription | None:
        """Parse the body according to the Content-Type header."""
        if headers.get("Content-Type") == "application/sdp" and body:
            return SessionDescription.parse(body)
        return None

    def __bytes__(self) -> bytes:
        headers = dict(self.headers)
        raw_body = bytes(self.body) if self.body is not None else b""
        if raw_body:
            headers.setdefault("Content-Length", str(len(raw_body)))
        header_lines = "".join(
            f"{name}: {value}\r\n" for name, value in headers.items()
        )
        return f"{self._first_line()}\r\n{header_lines}\r\n".encode() + raw_body

    @property
    def branch(self) -> str | None:
        """Branch parameter from the top Via header (RFC 3261 §20.42)."""
        _, uri = self.headers["Via"].split()
        return SipUri.parse(f"sip:{uri}").parameters["branch"]

    @property
    def remote_tag(self) -> str | None:
        """To-tag used with From-tag to identify the SIP dialog (RFC 3261 §12.2.2)."""
        return self.headers["To"].tag

    @property
    def local_tag(self) -> str:
        """From-tag used with To-tag to identify the SIP dialog (RFC 3261 §12.2.2)."""
        return self.headers["From"].tag

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
    uri: SipUri | str

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


@dataclasses.dataclass(kw_only=True, slots=True)
class Dialog:
    """
    Peer-to-peer SIP relationship between two user agents.

    A dialog is identified by the tuple of (Call-ID, From tag, To tag) and
    established by a non-final response to the INVITE, see also: [RFC 3261 §12]

    [RFC 3261 §12]: https://datatracker.ietf.org/doc/html/rfc3261#section-12

    Args:
        uac: The user agent that initiated the dialog.
        call_id: The Call-ID header value for this dialog.
        local_tag: The From-header tag parameter value for this dialog.
        remote_tag: The To-header tag parameter value for this dialog.

    """

    uac: SipUri | None = None
    call_id: str = dataclasses.field(
        default_factory=lambda: f"{uuid.uuid4()}@{socket.gethostname()}",
        compare=False,
    )
    local_tag: str = dataclasses.field(
        default_factory=lambda: str(uuid.uuid4()), compare=True
    )
    remote_tag: str | None = dataclasses.field(default=None, compare=True)
    remote_contact: SipUri | None = dataclasses.field(default=None, compare=True)
    route_set: list[SipUri] = dataclasses.field(default_factory=list)

    created: datetime.datetime = dataclasses.field(
        init=False, default_factory=datetime.datetime.now
    )

    @property
    def from_header(self) -> str:
        """The logical sender of a request."""
        return f"{self.uac.scheme}:{self.uac.user}@{socket.gethostname()};tag={self.local_tag}"

    @property
    def to_header(self) -> str:
        """The logical recipient of a request."""
        part = f"{self.uac.scheme}:{self.uac.user}@{self.uac.host}:{self.uac.port};transport={self.uac.parameters.get('transport', 'TLS')}"
        if self.remote_tag:
            part += f";tag={self.remote_tag}"
        return part

    @property
    def headers(self) -> dict[str, str]:
        """Return a dict of headers for this dialog."""
        return {
            "From": self.from_header,
            "To": self.to_header,
            "Call-ID": self.call_id,
        }

    @classmethod
    def from_request(cls, request: Request) -> Dialog:
        """Create a dialog from a request, extracting relevant headers."""
        return cls(
            call_id=request.headers["Call-ID"],
            local_tag=request.local_tag,
            remote_tag=request.remote_tag or str(uuid.uuid4()),
            remote_contact=request.headers.get("Contact"),
        )
