"""Session Traversal Utilities for NAT (STUN) implementation of RFC 5389."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import enum
import logging
import socket
import struct
import uuid

__all__ = ["STUNAttributeType", "STUNMessageType", "STUNProtocol", "stun_discover"]

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


def _parse_stun_response(
    data: bytes,
    transaction_id: bytes,
    future: asyncio.Future[tuple[str, int]],
) -> None:
    """Parse a STUN Binding Success Response and resolve the given future."""
    if len(data) < 20:
        return
    message_type, _message_len, magic_cookie = struct.unpack(">HHI", data[:8])
    response_tid = data[8:20]
    if (
        magic_cookie != MAGIC_COOKIE
        or message_type != STUNMessageType.BINDING_SUCCESS_RESPONSE
        or response_tid != transaction_id
    ):
        return
    if future.done():
        return
    offset = 20
    xor_mapped: tuple[str, int] | None = None
    mapped: tuple[str, int] | None = None
    while offset + 4 <= len(data):
        attribute_type, attribute_len = struct.unpack(">HH", data[offset : offset + 4])
        attribute_value = data[offset + 4 : offset + 4 + attribute_len]
        if (
            attribute_type == STUNAttributeType.XOR_MAPPED_ADDRESS
            and len(attribute_value) >= 8
            and attribute_value[1] == 0x01  # IPv4
        ):
            port = struct.unpack(">H", attribute_value[2:4])[0] ^ (MAGIC_COOKIE >> 16)
            ip_int = struct.unpack(">I", attribute_value[4:8])[0] ^ MAGIC_COOKIE
            xor_mapped = (socket.inet_ntoa(struct.pack(">I", ip_int)), port)
        elif (
            attribute_type == STUNAttributeType.MAPPED_ADDRESS
            and len(attribute_value) >= 8
            and attribute_value[1] == 0x01  # IPv4
        ):
            port = struct.unpack(">H", attribute_value[2:4])[0]
            mapped = (socket.inet_ntoa(attribute_value[4:8]), port)
        offset += 4 + ((attribute_len + 3) & ~3)  # 4-byte aligned
    result = xor_mapped or mapped
    if result:
        future.set_result(result)
    else:
        future.set_exception(RuntimeError("No address attribute in STUN response"))


async def stun_discover(
    host: str, port: int = 3478, timeout_secs: float = 3.0
) -> tuple[str, int]:
    """Discover the public IP:port using STUN on a dedicated ephemeral socket.

    Creates a temporary UDP socket exclusively for the STUN exchange so that
    STUN traffic does not interfere with SIP or RTP transports.  Any protocol
    handler that needs its public address can simply ``await`` this coroutine.

    Args:
        host: STUN server hostname or IP address.
        port: STUN server UDP port (typically 3478).
        timeout_secs: Seconds to wait for a response before raising
            :exc:`asyncio.TimeoutError`.

    Returns:
        A ``(ip, port)`` tuple with the publicly visible address of the
        ephemeral socket as reported by the STUN server.

    Raises:
        asyncio.TimeoutError: If the server does not respond in time.
        RuntimeError: If the STUN response contains no address attribute.
    """
    loop = asyncio.get_running_loop()
    transaction_id = uuid.uuid4().bytes[:12]
    future: asyncio.Future[tuple[str, int]] = loop.create_future()

    class _STUNClientProtocol(asyncio.DatagramProtocol):
        def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
            _parse_stun_response(data, transaction_id, future)

        def error_received(self, exc: OSError) -> None:
            if not future.done():
                future.set_exception(exc)

    request = struct.pack(
        ">HHI12s",
        STUNMessageType.BINDING_REQUEST,
        0,
        MAGIC_COOKIE,
        transaction_id,
    )
    logger.debug("Sending STUN Binding Request to %s:%s", host, port)
    transport, _ = await loop.create_datagram_endpoint(
        _STUNClientProtocol,
        remote_addr=(host, port),
    )
    transport.sendto(request)
    try:
        return await asyncio.wait_for(future, timeout=timeout_secs)
    finally:
        transport.close()


@dataclasses.dataclass(slots=True)
class STUNProtocol(asyncio.DatagramProtocol):
    """
    Demultiplexes STUN (RFC 5389/7983) from other traffic.

    Use this as the base class for any protocol that shares a UDP socket with
    STUN. Incoming datagrams whose first byte is in ``[0, 3]`` (RFC 7983) are
    treated as STUN messages and routed to the pending discovery future. All
    other datagrams are forwarded to :meth:`packet_received`.

    :attr:`public_address` is an :class:`asyncio.Future` resolved automatically
    in :meth:`connection_made` via STUN so that subclasses can simply
    ``await`` it to obtain the correct routable address::

        class MyProtocol(STUNProtocol):
            async def on_ready(self) -> None:
                ip, port = await self.public_address

            def packet_received(self, data: bytes, addr: tuple[str, int]) -> None:
                process(data)
    """

    stun_server_address: tuple[str, int] = "stun.cloudflare.com", 3478
    stun_server_timeout: datetime.timedelta = datetime.timedelta(seconds=3)
    public_address: asyncio.Future[tuple[str, int]] = dataclasses.field(init=False)
    _stun_pending: dict[bytes, asyncio.Future[tuple[str, int]]] = dataclasses.field(
        init=False, default_factory=dict
    )
    transport: asyncio.DatagramTransport = dataclasses.field(init=False)

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore[override]
        """Store the transport, initialise :attr:`public_address`, and start STUN discovery."""
        self.transport = transport
        loop = asyncio.get_running_loop()
        self.public_address = loop.create_future()
        loop.create_task(self.stun_discover())

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Demultiplex STUN responses (RFC 7983) from other traffic.

        Datagrams with first byte in ``[0, 3]`` are dispatched to any pending
        discovery future; all others are forwarded to :meth:`packet_received`.
        """
        if data and data[0] < 4:
            # RFC 7983: first byte in [0, 3] indicates a STUN packet.
            if len(data) >= 20:
                tid = data[8:20]
                fut = self._stun_pending.get(tid)
                if fut is not None:
                    _parse_stun_response(data, tid, fut)
            return
        self.packet_received(data, addr)

    def packet_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Override in subclasses to handle non-STUN datagrams.

        Args:
            data: Raw datagram payload (first byte ≥ 4, not a STUN packet).
            addr: Source ``(host, port)`` of the datagram.
        """

    async def stun_discover(self) -> None:
        """Send a STUN Binding Request and resolve :attr:`public_address`.

        Sends the request through the transport bound to this protocol so the
        server observes the same NAT mapping as real traffic.  Responses are
        demultiplexed from normal datagrams via the RFC 7983 first-byte rule.

        Raises:
            asyncio.TimeoutError: If the server does not respond in time.
            RuntimeError: If the STUN response contains no address attribute.
        """
        transaction_id = uuid.uuid4().bytes[:12]
        self._stun_pending[transaction_id] = self.public_address
        request = struct.pack(
            ">HHI12s",
            STUNMessageType.BINDING_REQUEST,
            0,
            MAGIC_COOKIE,
            transaction_id,
        )
        logger.debug("Sending STUN Binding Request to %s:%s", *self.stun_server_address)
        self.transport.sendto(request, self.stun_server_address)
        try:
            await asyncio.wait_for(
                asyncio.shield(self.public_address),
                self.stun_server_timeout.total_seconds(),
            )
            logger.info(
                "STUN: public address for %s is %s:%d",
                type(self).__name__,
                *self.public_address.result(),
            )
        except Exception as exc:
            if not self.public_address.done():
                self.public_address.set_exception(exc)
        finally:
            self._stun_pending.pop(transaction_id, None)
