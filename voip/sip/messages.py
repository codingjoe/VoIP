"""SIP message types as defined by RFC 3261."""

from __future__ import annotations

import abc
import dataclasses
import typing

from voip.sdp.messages import SessionDescription

from ..types import ByteSerializableObject
from .types import CallerID, SIPMethod, SIPStatus

if typing.TYPE_CHECKING:
    from . import SipUri

__all__ = ["Request", "Response", "Message"]

#: Headers whose values are parsed as `CallerID` objects.
_CALLER_HEADERS = frozenset({"From", "To"})


@dataclasses.dataclass(kw_only=True)
class Message(ByteSerializableObject, abc.ABC):
    """
    A SIP message [RFC 3261 §7].

    [RFC 3261 §7]: https://datatracker.ietf.org/doc/html/rfc3261#section-7
    """

    headers: dict[str, str] = dataclasses.field(default_factory=dict)
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
