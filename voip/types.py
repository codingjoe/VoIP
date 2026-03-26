import abc
import ipaddress
import re
import typing
from ipaddress import IPv4Address, IPv6Address


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


# Match host and optional port. Host can be a domain name, ipv4 or ipv6 address.
NETLOC_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<host>
        # IPv6 addresses enclosed in square brackets (with optional port)
        \[[^]]+]
        |
        # Bare IPv6 addresses (at least two colons, no port)
        (?:[0-9a-fA-F]*:){2,}[0-9a-fA-F]*
        |
        # Hostname or IPv4 (no colon except the port separator)
        [^:]+
    )
    (?::(?P<port>\d+))?
    \s*$
    """,
    re.VERBOSE,
)


class NetworkAddress(typing.NamedTuple):
    """Parse and serialize an address."""

    host: str | IPv4Address | IPv6Address
    port: int | None = None

    def __str__(self):
        if self.port and isinstance(self.host, IPv6Address):
            return f"[{self.host}]:{self.port}"
        elif self.port is None:
            return str(self.host)
        return f"{self.host}:{self.port}"

    @classmethod
    def parse(cls, data: str) -> NetworkAddress:
        if match := NETLOC_PATTERN.match(data):
            host, port = match.group("host").strip("[]"), match.group("port")
            try:
                host = ipaddress.ip_address(host)
            except ValueError:
                pass
            return cls(host=host, port=int(port) if port else None)
        raise ValueError(f"Invalid network address: {data!r}")
