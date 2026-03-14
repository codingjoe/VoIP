from __future__ import annotations

import enum
import typing


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


class ByteSerializableObject:
    """Parse and serialize objects to and from raw bytes."""

    __slots__ = ()

    @classmethod
    def parse(cls, data: bytes) -> typing.Self:
        """Parse an object from raw bytes."""

    def __bytes__(self) -> bytes:
        """Serialize the object to raw bytes."""

    def __str__(self) -> str:
        return self.__bytes__().decode()
