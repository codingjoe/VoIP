from __future__ import annotations

import typing


class ByteSerializableObject(typing.Protocol):
    """Parse and serialize objects to and from raw bytes."""

    __slots__ = ()

    @classmethod
    def parse(cls, data: bytes) -> typing.Self:
        """Parse an object from raw bytes."""

    def __bytes__(self) -> bytes:
        """Serialize the object to raw bytes."""

    def __str__(self) -> str:
        return self.__bytes__().decode()
