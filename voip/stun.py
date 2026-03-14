"""Session Traversal Utilities for NAT (STUN) implementation of RFC 5389."""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
import socket
import struct
import uuid

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


@dataclasses.dataclass(kw_only=True, slots=True)
class STUNProtocol(asyncio.DatagramProtocol):
    """
    Demultiplexes STUN (RFC 5389/7983) from other traffic.

    Use this as the base class for any protocol that shares a UDP socket with
    STUN. Incoming datagrams whose first byte is in ``[0, 3]`` (RFC 7983) are
    treated as STUN messages and routed to the STUN handler. All other
    datagrams are forwarded to `packet_received`.

    When the socket is ready and the reachable address is known,
    `stun_connection_made` is called.  If ``stun_server_address`` is
    ``None`` this happens synchronously from `connection_made` with the
    local socket address.  If STUN is configured it is called from
    `datagram_received` when the Binding Response arrives, with the
    discovered public address.  Subclasses only need to override
    `stun_connection_made` — no `connection_made` override is
    required::

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
    """

    stun_server_address: tuple[str, int] | None = ("stun.cloudflare.com", 3478)
    _stun_transaction_id: bytes = dataclasses.field(init=False, default=b"")
    transport: asyncio.DatagramTransport | None = dataclasses.field(
        init=False, default=None
    )

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:  # type: ignore[override]
        """Store the transport and start STUN discovery.

        When ``stun_server_address`` is ``None`` the socket is considered
        ready immediately: :meth:`stun_connection_made` is called right away
        with the local socket address.  Otherwise a STUN Binding Request is
        sent and :meth:`stun_connection_made` will be called by
        :meth:`datagram_received` once the server's response arrives.
        """
        self.transport = transport
        if self.stun_server_address is None:
            self.stun_connection_made(transport, transport.get_extra_info("sockname"))
        else:
            self._stun_transaction_id = uuid.uuid4().bytes[:12]
            self._send_stun_request()

    def stun_connection_made(
        self,
        transport: asyncio.DatagramTransport,
        addr: tuple[str, int],
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

    def send(self, data: bytes, addr: tuple[str, int]) -> None:
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
        """Demultiplex STUN responses (RFC 7983) from other traffic.

        Datagrams with first byte in ``[0, 3]`` are dispatched to the STUN
        handler; all others are forwarded to :meth:`packet_received`.
        """
        if data and data[0] < 4:
            # RFC 7983: first byte in [0, 3] indicates a STUN packet.
            if (
                len(data) >= 20
                and self._stun_transaction_id
                and data[8:20] == self._stun_transaction_id
            ):
                self._parse_stun_response(data)
            return
        self.packet_received(data, addr)

    def connection_lost(self, exc: Exception | None) -> None:
        """Clear the internal transport reference on disconnect."""
        self.transport = None

    def error_received(self, exc: Exception) -> None:
        """Handle transport-level errors without closing the socket.

        On Windows, sending to an unreachable UDP port triggers an ICMP
        "Port Unreachable" response, which surfaces as ``ConnectionResetError``
        (``WSAECONNRESET``) on the next receive.  Logging and ignoring it keeps
        the socket alive so subsequent datagrams (e.g. RTP) are still delivered.
        """
        logger.warning("UDP transport error (ignored): %s", exc)

    def packet_received(self, data: bytes, addr: tuple[str, int]) -> None:
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
        if self.transport is None:
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
        xor_mapped: tuple[str, int] | None = None
        mapped: tuple[str, int] | None = None
        while offset + 4 <= len(data):
            attribute_type, attribute_len = struct.unpack(
                ">HH", data[offset : offset + 4]
            )
            attribute_value = data[offset + 4 : offset + 4 + attribute_len]
            if (
                attribute_type == STUNAttributeType.XOR_MAPPED_ADDRESS
                and len(attribute_value) >= 8
                and attribute_value[1] == 0x01  # IPv4
            ):
                port = struct.unpack(">H", attribute_value[2:4])[0] ^ (
                    MAGIC_COOKIE >> 16
                )
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
            logger.debug("STUN response: %s:%s", *result)
            self.stun_connection_made(self.transport, result)
        else:
            logger.error("No address attribute in STUN response")
