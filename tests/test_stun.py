"""Tests for the STUN protocol implementation (RFC 5389)."""

from __future__ import annotations

import asyncio
import socket
import struct
from unittest.mock import MagicMock

import pytest
from voip.stun import MAGIC_COOKIE, STUNAttributeType, STUNMessageType, STUNProtocol


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


class ConcreteSTUN(STUNProtocol):
    """Concrete STUNProtocol subclass for testing."""

    def __init__(self) -> None:
        super().__init__()
        self._transport = MagicMock()


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
    async def test_handle_stun__xor_mapped_address(self):
        """Resolve the future with the XOR-MAPPED-ADDRESS when present."""
        protocol = ConcreteSTUN()
        transaction_id = b"\x01" * 12
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        protocol._stun_transactions[transaction_id] = future
        protocol.handle_stun(
            make_success_response(
                transaction_id,
                make_xor_mapped_address_attribute("203.0.113.5", 54321),
            ),
            ("stun.example.com", 3478),
        )
        assert future.result() == ("203.0.113.5", 54321)

    async def test_handle_stun__mapped_address_fallback(self):
        """Fall back to MAPPED-ADDRESS when XOR-MAPPED-ADDRESS is absent."""
        protocol = ConcreteSTUN()
        transaction_id = b"\x02" * 12
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        protocol._stun_transactions[transaction_id] = future
        protocol.handle_stun(
            make_success_response(
                transaction_id,
                make_mapped_address_attribute("198.51.100.7", 60001),
            ),
            ("stun.example.com", 3478),
        )
        assert future.result() == ("198.51.100.7", 60001)

    async def test_handle_stun__xor_mapped_address_preferred_over_mapped(self):
        """Prefer XOR-MAPPED-ADDRESS over MAPPED-ADDRESS when both are present."""
        protocol = ConcreteSTUN()
        transaction_id = b"\x03" * 12
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        protocol._stun_transactions[transaction_id] = future
        protocol.handle_stun(
            make_success_response(
                transaction_id,
                make_xor_mapped_address_attribute("203.0.113.1", 12345),
                make_mapped_address_attribute("198.51.100.1", 9999),
            ),
            ("stun.example.com", 3478),
        )
        assert future.result() == ("203.0.113.1", 12345)

    def test_handle_stun__truncated_packet__returns_early(self):
        """Silently ignore packets shorter than the minimum 20-byte STUN header."""
        protocol = ConcreteSTUN()
        protocol.handle_stun(b"\x00" * 19, ("stun.example.com", 3478))
        # No exception raised, no future resolved

    async def test_handle_stun__wrong_magic_cookie__returns_early(self):
        """Ignore responses with an incorrect magic cookie."""
        protocol = ConcreteSTUN()
        transaction_id = b"\x04" * 12
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        protocol._stun_transactions[transaction_id] = future
        data = struct.pack(
            ">HHI12s",
            STUNMessageType.BINDING_SUCCESS_RESPONSE,
            0,
            0xDEADBEEF,  # wrong magic cookie
            transaction_id,
        )
        protocol.handle_stun(data, ("stun.example.com", 3478))
        assert not future.done()

    async def test_handle_stun__wrong_message_type__returns_early(self):
        """Ignore non-success-response STUN messages."""
        protocol = ConcreteSTUN()
        transaction_id = b"\x05" * 12
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        protocol._stun_transactions[transaction_id] = future
        data = struct.pack(
            ">HHI12s",
            STUNMessageType.BINDING_REQUEST,  # not a success response
            0,
            MAGIC_COOKIE,
            transaction_id,
        )
        protocol.handle_stun(data, ("stun.example.com", 3478))
        assert not future.done()

    def test_handle_stun__unknown_transaction__returns_early(self):
        """Ignore responses whose transaction ID has no pending future."""
        protocol = ConcreteSTUN()
        protocol.handle_stun(
            make_success_response(
                b"\xff" * 12,
                make_xor_mapped_address_attribute("1.2.3.4", 1234),
            ),
            ("stun.example.com", 3478),
        )
        # No exception raised

    async def test_handle_stun__no_address_attribute__sets_exception(self):
        """Set a RuntimeError on the future when no address attribute is present."""
        protocol = ConcreteSTUN()
        transaction_id = b"\x06" * 12
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        protocol._stun_transactions[transaction_id] = future
        protocol.handle_stun(
            make_success_response(transaction_id),
            ("stun.example.com", 3478),
        )
        assert future.done()
        with pytest.raises(RuntimeError, match="No address attribute"):
            future.result()

    async def test_stun_discover__sends_binding_request(self):
        """Send a STUN Binding Request datagram to the specified server."""
        protocol = ConcreteSTUN()
        transaction_id_holder: list[bytes] = []

        def capture_sendto(data, address):
            transaction_id_holder.append(data[8:20])
            future = protocol._stun_transactions.get(data[8:20])
            if future:
                future.set_result(("1.2.3.4", 4321))

        protocol._transport.sendto.side_effect = capture_sendto
        result = await protocol.stun_discover("stun.example.com", 3478)
        protocol._transport.sendto.assert_called_once()
        _, called_address = protocol._transport.sendto.call_args[0]
        assert called_address == ("stun.example.com", 3478)
        assert result == ("1.2.3.4", 4321)

    async def test_stun_discover__cleans_up_transaction_on_timeout(self):
        """Remove the pending transaction from the dict when the request times out."""
        protocol = ConcreteSTUN()
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await protocol.stun_discover("stun.example.com", 3478, timeout_secs=0.01)
        assert len(protocol._stun_transactions) == 0
