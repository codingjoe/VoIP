"""STUN protocol implementation (RFC 5389)."""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import socket
import struct

__all__ = ["STUNAttributeType", "STUNMessageType", "STUNProtocol"]

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


class STUNProtocol:
    """Mixin providing STUN Binding Request/Response handling (RFC 5389).

    Mix into an ``asyncio.DatagramProtocol`` subclass that sets
    ``self._transport`` in ``connection_made``. Call ``_handle_stun`` for
    datagrams whose first byte is 0–3 (RFC 7983 multiplexing).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stun_transactions: dict[bytes, asyncio.Future] = {}

    async def _stun_discover(
        self, host: str, port: int, timeout_secs: float = 3.0
    ) -> tuple[str, int]:
        """Send a STUN Binding Request and return the discovered public address."""
        transaction_id = os.urandom(12)
        request = struct.pack(
            ">HHI12s",
            STUNMessageType.BINDING_REQUEST,
            0,
            MAGIC_COOKIE,
            transaction_id,
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, int]] = loop.create_future()
        self._stun_transactions[transaction_id] = future
        logger.debug("Sending STUN Binding Request to %s:%s", host, port)
        self._transport.sendto(request, (host, port))
        try:
            return await asyncio.wait_for(future, timeout=timeout_secs)
        finally:
            self._stun_transactions.pop(transaction_id, None)

    def _handle_stun(self, data: bytes, addr: tuple[str, int]) -> None:
        """Parse a STUN Binding Success Response and resolve the pending future."""
        if len(data) < 20:
            return
        message_type, _message_len, magic_cookie = struct.unpack(">HHI", data[:8])
        transaction_id = data[8:20]
        if (
            magic_cookie != MAGIC_COOKIE
            or message_type != STUNMessageType.BINDING_SUCCESS_RESPONSE
        ):
            return
        future = self._stun_transactions.get(transaction_id)
        if future is None or future.done():
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
