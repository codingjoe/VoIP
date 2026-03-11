"""Tests for the STUN utility functions (RFC 5389)."""

from __future__ import annotations

import asyncio
import socket
import struct

import pytest
from voip.stun import (
    MAGIC_COOKIE,
    STUNAttributeType,
    STUNMessageType,
    _parse_stun_response,
    stun_discover,
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


class TestParseStunResponse:
    async def test_xor_mapped_address(self):
        """Resolve the future with the XOR-MAPPED-ADDRESS when present."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, int]] = loop.create_future()
        transaction_id = b"\x01" * 12
        _parse_stun_response(
            make_success_response(
                transaction_id,
                make_xor_mapped_address_attribute("203.0.113.5", 54321),
            ),
            transaction_id,
            future,
        )
        assert future.result() == ("203.0.113.5", 54321)

    async def test_mapped_address_fallback(self):
        """Fall back to MAPPED-ADDRESS when XOR-MAPPED-ADDRESS is absent."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, int]] = loop.create_future()
        transaction_id = b"\x02" * 12
        _parse_stun_response(
            make_success_response(
                transaction_id,
                make_mapped_address_attribute("198.51.100.7", 60001),
            ),
            transaction_id,
            future,
        )
        assert future.result() == ("198.51.100.7", 60001)

    async def test_xor_mapped_address_preferred_over_mapped(self):
        """Prefer XOR-MAPPED-ADDRESS over MAPPED-ADDRESS when both are present."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, int]] = loop.create_future()
        transaction_id = b"\x03" * 12
        _parse_stun_response(
            make_success_response(
                transaction_id,
                make_xor_mapped_address_attribute("203.0.113.1", 12345),
                make_mapped_address_attribute("198.51.100.1", 9999),
            ),
            transaction_id,
            future,
        )
        assert future.result() == ("203.0.113.1", 12345)

    def test_truncated_packet__returns_early(self):
        """Silently ignore packets shorter than the minimum 20-byte STUN header."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        loop.close()
        _parse_stun_response(b"\x00" * 19, b"\x00" * 12, future)
        assert not future.done()

    async def test_wrong_magic_cookie__returns_early(self):
        """Ignore responses with an incorrect magic cookie."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, int]] = loop.create_future()
        transaction_id = b"\x04" * 12
        data = struct.pack(
            ">HHI12s",
            STUNMessageType.BINDING_SUCCESS_RESPONSE,
            0,
            0xDEADBEEF,  # wrong magic cookie
            transaction_id,
        )
        _parse_stun_response(data, transaction_id, future)
        assert not future.done()

    async def test_wrong_message_type__returns_early(self):
        """Ignore non-success-response STUN messages."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, int]] = loop.create_future()
        transaction_id = b"\x05" * 12
        data = struct.pack(
            ">HHI12s",
            STUNMessageType.BINDING_REQUEST,  # not a success response
            0,
            MAGIC_COOKIE,
            transaction_id,
        )
        _parse_stun_response(data, transaction_id, future)
        assert not future.done()

    async def test_mismatched_transaction_id__returns_early(self):
        """Ignore responses whose transaction ID does not match."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, int]] = loop.create_future()
        transaction_id = b"\x06" * 12
        other_transaction_id = b"\xff" * 12
        _parse_stun_response(
            make_success_response(
                other_transaction_id,
                make_xor_mapped_address_attribute("1.2.3.4", 1234),
            ),
            transaction_id,
            future,
        )
        assert not future.done()

    async def test_no_address_attribute__sets_exception(self):
        """Set a RuntimeError on the future when no address attribute is present."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[str, int]] = loop.create_future()
        transaction_id = b"\x07" * 12
        _parse_stun_response(
            make_success_response(transaction_id),
            transaction_id,
            future,
        )
        assert future.done()
        with pytest.raises(RuntimeError, match="No address attribute"):
            future.result()


class TestStunDiscover:
    async def test_stun_discover__sends_binding_request_and_returns_address(self):
        """stun_discover creates a dedicated UDP socket and returns the server-reported address.

        Sends a STUN Binding Request and resolves with the XOR-MAPPED-ADDRESS
        from the server's success response.
        """
        received_requests: list[bytes] = []

        # Start a minimal local STUN server on a free port.
        server_transport: asyncio.DatagramTransport | None = None
        server_future: asyncio.Future[tuple[str, int]] = (
            asyncio.get_running_loop().create_future()
        )

        class _StubSTUNServer(asyncio.DatagramProtocol):
            def connection_made(self, transport):
                nonlocal server_transport
                server_transport = transport

            def datagram_received(self, data, addr):
                received_requests.append(data)
                tid = data[8:20]
                response = (
                    struct.pack(
                        ">HHI12s",
                        STUNMessageType.BINDING_SUCCESS_RESPONSE,
                        0,
                        MAGIC_COOKIE,
                        tid,
                    )
                    + make_xor_mapped_address_attribute("1.2.3.4", 54321)
                )
                server_transport.sendto(response, addr)
                if not server_future.done():
                    server_future.set_result(addr)

        loop = asyncio.get_running_loop()
        server_t, _ = await loop.create_datagram_endpoint(
            _StubSTUNServer, local_addr=("127.0.0.1", 0)
        )
        server_addr = server_t.get_extra_info("sockname")
        try:
            result = await stun_discover(server_addr[0], server_addr[1])
        finally:
            server_t.close()

        assert result == ("1.2.3.4", 54321)
        assert len(received_requests) == 1

    async def test_stun_discover__raises_on_timeout(self):
        """stun_discover raises asyncio.TimeoutError when no response arrives."""
        # Use a blackhole address that accepts but never responds.
        loop = asyncio.get_running_loop()
        blackhole_t, _ = await loop.create_datagram_endpoint(
            asyncio.DatagramProtocol, local_addr=("127.0.0.1", 0)
        )
        blackhole_addr = blackhole_t.get_extra_info("sockname")
        try:
            with pytest.raises((TimeoutError, asyncio.TimeoutError)):
                await stun_discover(
                    blackhole_addr[0], blackhole_addr[1], timeout_secs=0.05
                )
        finally:
            blackhole_t.close()
