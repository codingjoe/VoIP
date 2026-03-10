from __future__ import annotations

import enum


class DigestQoP(enum.StrEnum):
    """Quality of protection values for HTTP Digest Authentication (RFC 2617)."""

    AUTH = "auth"
    AUTH_INT = "auth-int"
