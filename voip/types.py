from __future__ import annotations

import abc
import typing


class ByteSerializableObject(abc.ABC):
    """Parse and serialize objects to and from raw bytes."""

    __slots__ = ()

    @classmethod
    @abc.abstractmethod
    def parse(cls, data: bytes) -> typing.Self:
        """Parse an object from raw bytes."""

    @abc.abstractmethod
    def __bytes__(self) -> bytes:
        """Serialize the object to raw bytes."""

    def __str__(self) -> str:
        return self.__bytes__().decode()
