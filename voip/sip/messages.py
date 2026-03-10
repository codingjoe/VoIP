"""SIP message types as defined by RFC 3261."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voip.sdp.messages import SessionDescription

__all__ = ["Request", "Response", "Message"]


@dataclasses.dataclass(kw_only=True)
class Message:
    """A SIP message (RFC 3261 §7)."""

    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    body: bytes | SessionDescription = dataclasses.field(default=b"", repr=False)
    version: str = "SIP/2.0"

    @classmethod
    def parse(cls, data: bytes) -> Request | Response:
        """Parse a SIP message from raw bytes."""
        header_section, _, body = data.partition(b"\r\n\r\n")
        lines = header_section.decode().split("\r\n")
        first_line, *header_lines = lines
        headers = {}
        for line in header_lines:
            if not line:
                continue
            name, sep, value = line.partition(":")
            if not sep:
                continue
            headers[name.strip()] = value.strip()
        parts = first_line.split(" ", 2)
        if first_line.startswith("SIP/"):
            version, status_code_str, reason = parts
            return Response(
                status_code=int(status_code_str),
                reason=reason,
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
    def _parse_body(headers: dict[str, str], body: bytes) -> bytes | SessionDescription:
        """Parse the body according to the Content-Type header."""
        if headers.get("Content-Type") == "application/sdp" and body:
            from voip.sdp.messages import SessionDescription

            return SessionDescription.parse(body)
        return body

    def __bytes__(self) -> bytes:
        """Serialize to bytes."""
        headers = dict(self.headers)
        raw_body = bytes(self.body) if self.body else b""
        if raw_body:
            headers.setdefault("Content-Length", str(len(raw_body)))
        header_lines = "".join(
            f"{name}: {value}\r\n" for name, value in headers.items()
        )
        return f"{self._first_line()}\r\n{header_lines}\r\n".encode() + raw_body

    def __str__(self) -> str:
        return self.__bytes__().decode()

    def _first_line(self) -> str: ...


@dataclasses.dataclass(kw_only=True)
class Request(Message):
    """A SIP request message (RFC 3261 §7.1)."""

    method: str
    uri: str

    def _first_line(self) -> str:
        return f"{self.method} {self.uri} {self.version}"


@dataclasses.dataclass(kw_only=True)
class Response(Message):
    """A SIP response message (RFC 3261 §7.2)."""

    status_code: int
    reason: str

    def _first_line(self) -> str:
        return f"{self.version} {self.status_code} {self.reason}"
