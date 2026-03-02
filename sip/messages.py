"""SIP message types as defined by RFC 3261."""

from __future__ import annotations

import dataclasses

__all__ = ["Request", "Response"]


@dataclasses.dataclass(kw_only=True)
class Message:
    """A SIP message (RFC 3261 §7)."""

    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    body: bytes = dataclasses.field(default=b"", repr=False)
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
                body=body,
                version=version,
            )
        method, uri, version = parts
        return Request(
            method=method, uri=uri, headers=headers, body=body, version=version
        )

    def __bytes__(self) -> bytes:
        """Serialize to bytes."""
        header_lines = "".join(
            f"{name}: {value}\r\n" for name, value in self.headers.items()
        )
        return f"{self._first_line()}\r\n{header_lines}\r\n".encode() + self.body

    def _first_line(self) -> str:
        return NotImplemented


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
