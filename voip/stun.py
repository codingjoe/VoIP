"""Session Traversal Utilities for NAT (STUN) implementation of RFC 5389."""

import asyncio
import dataclasses
import enum
import ipaddress
import logging
import struct
import uuid

__all__ = ["STUNAttributeType", "STUNMessageType", "STUNProtocol"]

from voip.types import NetworkAddress

logger = logging.getLogger(__name__)

MAGIC_COOKIE = 0x2112A442


class STUNMessageType(enum.IntEnum):
    """STUN message types (RFC 5389 §6)."""

    BINDING_REQUEST = 0x0001
    BINDING_SUCCESS_RESPONSE = 0x0101


class STUNAttributeType(enum.IntEnum):
    """STUN attribute types (RFC 5389 §15)."""

    MAPPED_ADDRESS = 0x0001
    XOR_MAPPED_ADDRESS = 0x0020


def _parse_address(
    value: bytes, xor_key: bytes
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int] | None:
    """Decode a STUN MAPPED-ADDRESS or XOR-MAPPED-ADDRESS attribute value.

    When *xor_key* is non-empty the port and address bytes are XORed with
    the key per RFC 5389 §15.2; pass an empty byte string for plain
    MAPPED-ADDRESS attributes.

    Args:
        value: Raw attribute value bytes (everything after the type/length TLV header).
        xor_key: XOR key bytes — must be exactly 16 bytes
            (``MAGIC_COOKIE (4 bytes) || transaction_id (12 bytes)``)
            for XOR-MAPPED-ADDRESS, or empty bytes for plain MAPPED-ADDRESS.

    Returns:
        ``(ip_address, port)`` on success, ``None`` when *value* is
        too short or the address family is unrecognised.
    """
    assert not xor_key or len(xor_key) == 16, "xor_key must be 16 bytes or empty"  # noqa: S101
    if len(value) < 4:
        return None
    family = value[1]
    raw_port = struct.unpack(">H", value[2:4])[0]
    port = raw_port ^ (MAGIC_COOKIE >> 16) if xor_key else raw_port
    match family:
        case 0x01 if len(value) >= 8:  # IPv4
            raw_ip = value[4:8]
            ip_bytes = (
                bytes(a ^ b for a, b in zip(raw_ip, xor_key[:4], strict=False))
                if xor_key
                else raw_ip
            )
            return ipaddress.IPv4Address(ip_bytes), port
        case 0x02 if len(value) >= 20:  # IPv6
            raw_ip = value[4:20]
            ip_bytes = (
                bytes(a ^ b for a, b in zip(raw_ip, xor_key, strict=False))
                if xor_key
                else raw_ip
            )
            return ipaddress.IPv6Address(ip_bytes), port
        case _:
            return None


@dataclasses.dataclass(kw_only=True, slots=True)
class STUNProtocol(asyncio.DatagramProtocol):
    """
    Protocol for demultiplexing STUN (RFC 5389/7983) from other traffic.

    Use this as the base class for any protocol that shares a UDP socket with
    STUN. Incoming datagrams whose first byte is in `[0, 3]` (RFC 7983) are
    treated as STUN messages and routed to the STUN handler. All other
    datagrams are forwarded to `packet_received`.

    When the socket is ready and the reachable address is known,
    `stun_connection_made` is called.  If `stun_server_address` is
    `None` this happens synchronously from `connection_made` with the
    local socket address.  If STUN is configured it is called from
    `datagram_received` when the Binding Response arrives, with the
    discovered public address.  Subclasses only need to override
    `stun_connection_made` — no `connection_made` override is
    required:

    ```python
    class MyProtocol(STUNProtocol):
        def stun_connection_made(
            self,
            transport: asyncio.DatagramTransport,
            addr: tuple[str, int],
        ) -> None:
            # socket is ready; addr is the reachable (public or local) address
            ...

        def packet_received(self, data: bytes, addr: tuple[str, int]) -> None:
            process(data)
    ```
    """

    stun_server_address: NetworkAddress | None = NetworkAddress(
        "stun.cloudflare.com", 3478
    )
    _stun_transaction_id: bytes = dataclasses.field(init=False, default=b"")
    transport: asyncio.DatagramTransport | None = dataclasses.field(
        init=False, default=None
    )

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self.transport = transport
        if self.stun_server_address is None:
            host, port = transport.get_extra_info("sockname")[:2]
            self.stun_connection_made(
                transport, NetworkAddress(host=ipaddress.ip_address(host), port=port)
            )
        else:
            self._stun_transaction_id = uuid.uuid4().bytes[:12]
            self._send_stun_request()

    def stun_connection_made(
        self,
        transport: asyncio.DatagramTransport,
        addr: NetworkAddress,
    ) -> None:
        """Called when the socket is ready and the reachable address is known.

        When STUN is configured, *addr* is the **public** ``(ip, port)``
        discovered from the STUN Binding Response and this method is called
        by `datagram_received`.  When ``stun_server_address=None``,
        *addr* is the local socket address and this method is called
        synchronously from `connection_made`.

        Subclasses override this method to trigger protocol-specific
        initialisation once the socket is ready.

        Args:
            transport: The UDP transport bound to this protocol.
            addr: Reachable ``(host, port)`` — public when STUN is used,
                local otherwise.
        """  # noqa: D401

    def send(self, data: bytes, addr: NetworkAddress) -> None:
        """Send a raw datagram through the shared UDP socket.

        Args:
            data: Raw bytes to transmit.
            addr: Destination ``(host, port)``.
        """
        if self.transport is not None:
            self.transport.sendto(data, addr)

    def close(self) -> None:
        """Close the underlying UDP transport."""
        if self.transport is not None:
            self.transport.close()

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        if data and data[0] < 4:
            # RFC 7983: first byte in [0, 3] indicates a STUN packet.
            if (
                len(data) >= 20
                and self._stun_transaction_id
                and data[8:20] == self._stun_transaction_id
            ):
                self._parse_stun_response(data)
            return
        self.packet_received(data, NetworkAddress(*addr))

    def connection_lost(self, exc: Exception | None) -> None:
        """Clear the internal transport reference on disconnect."""
        self.transport = None

    def error_received(self, exc: Exception) -> None:
        logger.warning("UDP transport error (ignored): %s", exc)

    def packet_received(self, data: bytes, addr: NetworkAddress) -> None:
        """Override in subclasses to handle non-STUN datagrams.

        Args:
            data: Raw datagram payload (first byte ≥ 4, not a STUN packet).
            addr: Source ``(host, port)`` of the datagram.
        """

    def _send_stun_request(self) -> None:
        """Send a STUN Binding Request through the protocol's own transport.

        Sends the request through the transport bound to this protocol so the
        server observes the same NAT mapping as real traffic.  Responses are
        demultiplexed from normal datagrams via the RFC 7983 first-byte rule.
        """
        if self.transport is None or self.stun_server_address is None:
            return
        request = struct.pack(
            ">HHI12s",
            STUNMessageType.BINDING_REQUEST,
            0,
            MAGIC_COOKIE,
            self._stun_transaction_id,
        )
        logger.debug("Sending STUN Binding Request to %s:%s", *self.stun_server_address)
        self.transport.sendto(request, self.stun_server_address)

    def _parse_stun_response(self, data: bytes) -> None:
        """Parse a STUN Binding Success Response and invoke :meth:`stun_connection_made`."""
        if len(data) < 20:
            return
        message_type, _message_len, magic_cookie = struct.unpack(">HHI", data[:8])
        response_tid = data[8:20]
        if (
            magic_cookie != MAGIC_COOKIE
            or message_type != STUNMessageType.BINDING_SUCCESS_RESPONSE
            or response_tid != self._stun_transaction_id
        ):
            return
        # Clear transaction ID so duplicate responses are ignored.
        self._stun_transaction_id = b""
        offset = 20
        xor_mapped: tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int] | None = (
            None
        )
        mapped: tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int] | None = None
        xor_key = struct.pack(">I", MAGIC_COOKIE) + response_tid
        while offset + 4 <= len(data):
            attribute_type, attribute_len = struct.unpack(
                ">HH", data[offset : offset + 4]
            )
            attribute_value = data[offset + 4 : offset + 4 + attribute_len]
            match attribute_type:
                case STUNAttributeType.XOR_MAPPED_ADDRESS:
                    xor_mapped = _parse_address(attribute_value, xor_key)
                case STUNAttributeType.MAPPED_ADDRESS:
                    mapped = _parse_address(attribute_value, b"")
            offset += 4 + ((attribute_len + 3) & ~3)  # 4-byte aligned
        try:
            host, port = xor_mapped or mapped
            host = ipaddress.ip_address(host)
        except ValueError, TypeError:
            logger.exception("No address attribute in STUN response")
        else:
            self.stun_connection_made(self.transport, NetworkAddress(host, port))
