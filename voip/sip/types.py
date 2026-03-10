from __future__ import annotations

import enum


class SIPStatus(enum.StrEnum):
    """Common SIP response status codes and reason phrases (RFC 3261)."""

    OK = "200 OK"
    BUSY_HERE = "486 Busy Here"

    @property
    def status_code(self) -> int:
        """Return the numeric status code."""
        return int(self.value.split(" ", 1)[0])

    @property
    def reason(self) -> str:
        """Return the reason phrase."""
        return self.value.split(" ", 1)[1]


class SIPStatusCode(enum.IntEnum):
    """Common SIP response status codes (RFC 3261)."""

    OK = 200
    UNAUTHORIZED = 401
    PROXY_AUTHENTICATION_REQUIRED = 407
