"""Tests for the STUN utility functions (RFC 5389)."""

import asyncio
import ipaddress
import socket
import struct
import unittest.mock

from voip.stun import (
    MAGIC_COOKIE,
    STUNAttributeType,
    STUNMessageType,
    STUNProtocol,
    _parse_address,
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


def make_xor_mapped_address_attribute_ipv6(
    ip: str, port: int, transaction_id: bytes
) -> bytes:
    """Build a XOR-MAPPED-ADDRESS attribute for the given IPv6 address and port.

    Per RFC 5389 §15.2 the port is XORed with the top 16 bits of the magic
    cookie and the 128-bit address is XORed with the concatenation of the
    magic cookie and the transaction ID.
    """
    xor_port = port ^ (MAGIC_COOKIE >> 16)
    xor_key = struct.pack(">I", MAGIC_COOKIE) + transaction_id
    raw_addr = socket.inet_pton(socket.AF_INET6, ip)
    xor_addr = bytes(a ^ b for a, b in zip(raw_addr, xor_key, strict=False))
    value = struct.pack(">BBH", 0x00, 0x02, xor_port) + xor_addr
    return struct.pack(">HH", STUNAttributeType.XOR_MAPPED_ADDRESS, len(value)) + value


def make_mapped_address_attribute_ipv6(ip: str, port: int) -> bytes:
    """Build a MAPPED-ADDRESS attribute for the given IPv6 address and port."""
    raw_addr = socket.inet_pton(socket.AF_INET6, ip)
    value = struct.pack(">BBH", 0x00, 0x02, port) + raw_addr
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


class TestParseAddress:
    def test_too_short__returns_none(self):
        """Return None when the attribute value is shorter than 4 bytes."""
        assert _parse_address(b"\x00\x01", b"") is None

    def test_unknown_family__returns_none(self):
        """Return None for an unrecognised address family byte."""
        value = struct.pack(">BBH4s", 0x00, 0x03, 1234, b"\x00" * 4)
        assert _parse_address(value, b"") is None


class TestSTUNProtocol:
    def test_is_datagram_protocol(self):
        """STUNProtocol is an asyncio.DatagramProtocol subclass."""
        assert issubclass(STUNProtocol, asyncio.DatagramProtocol)

    async def test_connection_made__stun_disabled__calls_stun_connection_made(self):
        """When stun_server_address is None, stun_connection_made is called with the local addr."""
        received: list[tuple] = []

        class ConcreteProto(STUNProtocol):
            def stun_connection_made(self, transport, addr):
                received.append((transport, addr))

        proto = ConcreteProto(stun_server_address=None)
        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        transport.get_extra_info.return_value = ("127.0.0.1", 5060)
        proto.connection_made(transport)
        # No STUN request was sent — the callback is immediate.
        transport.sendto.assert_not_called()
        # stun_connection_made is called once with the local socket address.
        assert len(received) == 1
        assert received[0][0] is transport
        assert received[0][1] == (ipaddress.IPv4Address("127.0.0.1"), 5060)

    async def test_connection_made__stun_enabled__sends_binding_request(self):
        """When stun_server_address is set, a STUN Binding Request is sent."""
        proto = STUNProtocol(stun_server_address=("127.0.0.1", 3478))
        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        proto.connection_made(transport)
        # A STUN request should have been sent to the STUN server.
        assert transport.sendto.called
        call_args = transport.sendto.call_args
        assert call_args[0][1] == ("127.0.0.1", 3478)

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

    async def test_stun_discover__calls_stun_connection_made_with_public_addr(self):
        """stun_connection_made is invoked with the public address discovered via STUN."""
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

        done: asyncio.Future[
            tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int]
        ] = loop.create_future()

        class TrackingProto(STUNProtocol):
            def stun_connection_made(self, transport, addr):
                if not done.done():
                    done.set_result(addr)

        proto = TrackingProto(stun_server_address=server_addr)
        client_t, _ = await loop.create_datagram_endpoint(
            lambda: proto, local_addr=("127.0.0.1", 0)
        )
        try:
            result = await asyncio.wait_for(done, 2.0)
            assert result == (ipaddress.IPv4Address("203.0.113.5"), 12345)
            assert len(received_requests) == 1
        finally:
            client_t.close()
            server_t.close()

    async def test_parse_stun_response__xor_mapped_address_ipv6(self):
        """XOR-MAPPED-ADDRESS with IPv6 family resolves to the correct address."""
        transaction_id = b"\x01" * 12
        attr = make_xor_mapped_address_attribute_ipv6(
            "2001:db8::1", 54321, transaction_id
        )
        response = make_success_response(transaction_id, attr)

        received: list[tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int]] = []

        class TrackingProto(STUNProtocol):
            def stun_connection_made(self, transport, addr):
                received.append(addr)

        proto = TrackingProto(stun_server_address=("::1", 3478))
        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        proto.connection_made(transport)
        proto._stun_transaction_id = transaction_id
        proto.datagram_received(response, ("::1", 3478))
        assert len(received) == 1
        assert received[0] == (ipaddress.IPv6Address("2001:db8::1"), 54321)

    async def test_parse_stun_response__mapped_address_ipv6(self):
        """MAPPED-ADDRESS with IPv6 family is used when XOR-MAPPED-ADDRESS is absent."""
        transaction_id = b"\x02" * 12
        attr = make_mapped_address_attribute_ipv6("::1", 12345)
        response = make_success_response(transaction_id, attr)

        received: list[tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int]] = []

        class TrackingProto(STUNProtocol):
            def stun_connection_made(self, transport, addr):
                received.append(addr)

        proto = TrackingProto(stun_server_address=("::1", 3478))
        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        proto.connection_made(transport)
        proto._stun_transaction_id = transaction_id
        proto.datagram_received(response, ("::1", 3478))
        assert len(received) == 1
        assert received[0] == (ipaddress.IPv6Address("::1"), 12345)

    async def test_parse_stun_response__mapped_address_ipv4_fallback(self):
        """MAPPED-ADDRESS with IPv4 family is used when XOR-MAPPED-ADDRESS is absent."""
        transaction_id = b"\x03" * 12
        attr = make_mapped_address_attribute("203.0.113.1", 9999)
        response = make_success_response(transaction_id, attr)

        received: list[tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, int]] = []

        class TrackingProto(STUNProtocol):
            def stun_connection_made(self, transport, addr):
                received.append(addr)

        proto = TrackingProto(stun_server_address=("127.0.0.1", 3478))
        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        proto.connection_made(transport)
        proto._stun_transaction_id = transaction_id
        proto.datagram_received(response, ("127.0.0.1", 3478))
        assert len(received) == 1
        assert received[0] == (ipaddress.IPv4Address("203.0.113.1"), 9999)

    async def test_parse_stun_response__no_address_attribute_logs_error(self, caplog):
        """Log an error when the STUN response contains no address attribute."""
        import logging  # noqa: PLC0415

        transaction_id = b"\x04" * 12
        response = make_success_response(transaction_id)

        class TrackingProto(STUNProtocol):
            def stun_connection_made(self, transport, addr):
                pass  # pragma: no cover

        proto = TrackingProto(stun_server_address=("127.0.0.1", 3478))
        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        proto.connection_made(transport)
        proto._stun_transaction_id = transaction_id
        with caplog.at_level(logging.ERROR):
            proto.datagram_received(response, ("127.0.0.1", 3478))
        assert any("No address attribute" in r.message for r in caplog.records)

    async def test_close__closes_transport(self):
        """close() calls close() on the underlying transport."""
        proto = STUNProtocol(stun_server_address=None)
        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        transport.get_extra_info.return_value = ("127.0.0.1", 0)
        proto.connection_made(transport)
        proto.close()
        transport.close.assert_called_once()

    def test_send__delivers_datagram(self):
        """send() passes data and address to the underlying transport."""
        proto = STUNProtocol(stun_server_address=None)
        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        transport.get_extra_info.return_value = ("127.0.0.1", 0)
        proto.connection_made(transport)
        proto.send(b"hello", ("127.0.0.1", 5004))
        transport.sendto.assert_any_call(b"hello", ("127.0.0.1", 5004))

    def test_error_received__logs_warning(self, caplog):
        """error_received() logs a warning and does not raise."""
        import logging  # noqa: PLC0415

        proto = STUNProtocol(stun_server_address=None)
        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        transport.get_extra_info.return_value = ("127.0.0.1", 0)
        proto.connection_made(transport)
        with caplog.at_level(logging.WARNING):
            proto.error_received(OSError("network error"))
        assert any("network error" in r.message for r in caplog.records)

    def test_send_stun_request__no_op_when_transport_is_none(self):
        """_send_stun_request() is a no-op when the transport is not set."""
        proto = STUNProtocol(stun_server_address=("127.0.0.1", 3478))
        # transport is None (never connected)
        proto._send_stun_request()  # must not raise

    def test_parse_stun_response__too_short__ignored(self):
        """_parse_stun_response() silently ignores responses shorter than 20 bytes."""
        proto = STUNProtocol(stun_server_address=("127.0.0.1", 3478))
        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        transport.get_extra_info.return_value = ("127.0.0.1", 0)
        proto.connection_made(transport)
        proto._stun_transaction_id = b"\x05" * 12
        proto._parse_stun_response(b"\x01\x01" + b"\x00" * 10)  # only 12 bytes

    def test_parse_stun_response__wrong_transaction_id__ignored(self):
        """_parse_stun_response() ignores responses with a mismatched transaction ID."""
        transaction_id = b"\x06" * 12
        received: list = []

        class TrackingProto(STUNProtocol):
            def stun_connection_made(self, transport, addr):
                received.append(addr)  # pragma: no cover

        proto = TrackingProto(stun_server_address=("127.0.0.1", 3478))
        transport = unittest.mock.MagicMock(spec=asyncio.DatagramTransport)
        transport.get_extra_info.return_value = ("127.0.0.1", 0)
        proto.connection_made(transport)
        proto._stun_transaction_id = transaction_id
        wrong_tid = b"\xff" * 12
        response = make_success_response(
            wrong_tid, make_xor_mapped_address_attribute("203.0.113.5", 1234)
        )
        proto._parse_stun_response(response)
        assert received == []
