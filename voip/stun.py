"""Session Traversal Utilities for NAT (STUN) implementation of RFC 5389."""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import socket
import struct

__all__ = ["STUNAttributeType", "STUNMessageType", "stun_discover"]

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
    host: str, port: int, timeout_secs: float = 3.0
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
    transaction_id = os.urandom(12)
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
