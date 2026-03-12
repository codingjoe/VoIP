"""Tests for the STUN utility functions (RFC 5389)."""

from __future__ import annotations

import asyncio
import socket
import struct
import unittest.mock

from voip.stun import (
    MAGIC_COOKIE,
    STUNAttributeType,
    STUNMessageType,
    STUNProtocol,
)


def make_xor_mapped_address_attribute(ip: str, port: int) -> bytes:
    """Build a XOR-MAPPED-ADDRESS attribute for the given IPv4 address and port."""
    ip_int = struct.unpack(">I", socket.inet_aton(ip))[0] ^ MAGIC_COOKIE
    xor_port = port ^ (MAGIC_COOKIE >> 16)
    value = struct.pack(">BBH I", 0x00, 0x01, xor_port, ip_int)
    return struct.pack(">HH", STUNAttributeType.XOR_MAPPED_ADDRESS, len(value)) + value


def make_mapped_address_attribute(ip: str, port: int) -> bytes:
    """Build a MAPPED-ADDRESS attribute for the given IPv4 address and port."""
    value = struct.pack(">BBH4s", 0x00, 0x01, port, socket.inet_aton(ip))
    return struct.pack(">HH", STUNAttributeType.MAPPED_ADDRESS, len(value)) + value


def make_success_response(transaction_id: bytes, *attributes: bytes) -> bytes:
    """Build a STUN Binding Success Response with the given attributes."""
    body = b"".join(attributes)
    return (
        struct.pack(
            ">HHI12s",
            STUNMessageType.BINDING_SUCCESS_RESPONSE,
            len(body),
            MAGIC_COOKIE,
            transaction_id,
        )
        + body
    )


class TestSTUNMessageType:
    def test_binding_request__value(self):
        """BINDING_REQUEST has the correct numeric value per RFC 5389."""
        assert STUNMessageType.BINDING_REQUEST == 0x0001

    def test_binding_success_response__value(self):
        """BINDING_SUCCESS_RESPONSE has the correct numeric value per RFC 5389."""
        assert STUNMessageType.BINDING_SUCCESS_RESPONSE == 0x0101


class TestSTUNAttributeType:
    def test_mapped_address__value(self):
        """MAPPED_ADDRESS has the correct numeric value per RFC 5389."""
        assert STUNAttributeType.MAPPED_ADDRESS == 0x0001

    def test_xor_mapped_address__value(self):
        """XOR_MAPPED_ADDRESS has the correct numeric value per RFC 5389."""
        assert STUNAttributeType.XOR_MAPPED_ADDRESS == 0x0020


class TestSTUNProtocol:
    def test_is_datagram_protocol(self):
        """STUNProtocol is an asyncio.DatagramProtocol subclass."""
        assert issubclass(STUNProtocol, asyncio.DatagramProtocol)

    async def test_connection_made__stores_transport(self):
        """connection_made() stores the transport and initialises public_address."""
        proto = STUNProtocol()
        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        proto.connection_made(transport)
        assert proto.transport is transport
        assert isinstance(proto.public_address, asyncio.Future)

    async def test_datagram_received__stun_packet__not_forwarded(self):
        """A STUN packet (first byte < 4) does not reach packet_received."""
        received: list[bytes] = []

        class ConcreteProto(STUNProtocol):
            def packet_received(self, data, addr):
                received.append(data)

        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        proto = ConcreteProto()
        proto.connection_made(transport)
        stun_bytes = b"\x01\x01" + b"\x00" * 18
        proto.datagram_received(stun_bytes, ("127.0.0.1", 5004))
        assert received == []

    async def test_datagram_received__non_stun__forwarded_to_packet_received(self):
        """Non-STUN datagrams (first byte >= 4) are forwarded to packet_received."""
        received: list[bytes] = []

        class ConcreteProto(STUNProtocol):
            def packet_received(self, data, addr):
                received.append(data)

        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        proto = ConcreteProto()
        proto.connection_made(transport)
        proto.datagram_received(b"\x80hello", ("127.0.0.1", 5004))
        assert received == [b"\x80hello"]

    async def test_stun_discover__uses_actual_socket(self):
        """public_address is resolved via the socket bound to the protocol."""
        received_requests: list[bytes] = []
        server_transport: asyncio.DatagramTransport | None = None

        class _StubSTUNServer(asyncio.DatagramProtocol):
            def connection_made(self, transport):
                nonlocal server_transport
                server_transport = transport

            def datagram_received(self, data, addr):
                received_requests.append(data)
                tid = data[8:20]
                ip_int = (203 << 24) | (0 << 16) | (113 << 8) | 5
                xor_ip = ip_int ^ MAGIC_COOKIE
                xor_port = 12345 ^ (MAGIC_COOKIE >> 16)
                attr = struct.pack(">HH", 0x0020, 8) + struct.pack(
                    ">BBH I", 0, 1, xor_port, xor_ip
                )
                response = (
                    struct.pack(
                        ">HHI12s",
                        STUNMessageType.BINDING_SUCCESS_RESPONSE,
                        len(attr),
                        MAGIC_COOKIE,
                        tid,
                    )
                    + attr
                )
                server_transport.sendto(response, addr)

        loop = asyncio.get_running_loop()
        server_t, _ = await loop.create_datagram_endpoint(
            _StubSTUNServer, local_addr=("127.0.0.1", 0)
        )
        server_addr = server_t.get_extra_info("sockname")

        proto = STUNProtocol(stun_server_address=server_addr)
        client_t, _ = await loop.create_datagram_endpoint(
            lambda: proto, local_addr=("127.0.0.1", 0)
        )
        try:
            result = await proto.public_address
            assert result == ("203.0.113.5", 12345)
            assert len(received_requests) == 1
        finally:
            client_t.close()
            server_t.close()
