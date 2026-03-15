"""
Session Description Protocol (SDP) implementation of [RFC 4566].

[RFC 4566]: https://datatracker.ietf.org/doc/html/rfc4566
"""

from .messages import SessionDescription

__all__ = ["SessionDescription"]
