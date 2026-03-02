"""Tests for the SIP asyncio protocol handler."""

import errno
from unittest.mock import MagicMock

import pytest
from sip.aio import SessionInitiationProtocol
from sip.calls import IncomingCall
from sip.messages import Request, Response


class ConcreteProtocol(SessionInitiationProtocol):
    """Concrete subclass for testing that records received messages."""

    def __init__(self):
        self.requests = []
        self.responses = []
        self.calls = []

    def request_received(self, request, addr):
        self.requests.append((request, addr))

    def response_received(self, response, addr):
        self.responses.append((response, addr))

    def invite_received(self, call, addr):
        self.calls.append((call, addr))


class InviteProtocol(SessionInitiationProtocol):
    """Concrete subclass for testing that records incoming calls."""

    def __init__(self):
        self.calls = []

    def invite_received(self, call, addr):
        self.calls.append((call, addr))


class TestSessionInitiationProtocol:
    def test_connection_made__stores_transport(self):
        """Store the transport for later use when a connection is established."""
        protocol = SessionInitiationProtocol()
        transport = MagicMock()
        protocol.connection_made(transport)
        assert protocol._transport is transport

    def test_datagram_received__request(self):
        """Dispatch a received SIP request datagram to request_received."""
        protocol = ConcreteProtocol()
        data = (
            b"INVITE sip:bob@biloxi.com SIP/2.0\r\n"
            b"Via: SIP/2.0/UDP pc33.atlanta.com\r\n"
            b"\r\n"
        )
        addr = ("192.0.2.1", 5060)
        protocol.datagram_received(data, addr)
        assert len(protocol.requests) == 1
        request, called_addr = protocol.requests[0]
        assert isinstance(request, Request)
        assert request.method == "INVITE"
        assert called_addr == addr

    def test_datagram_received__response(self):
        """Dispatch a received SIP response datagram to response_received."""
        protocol = ConcreteProtocol()
        data = b"SIP/2.0 200 OK\r\nVia: SIP/2.0/UDP pc33.atlanta.com\r\n\r\n"
        addr = ("192.0.2.1", 5060)
        protocol.datagram_received(data, addr)
        assert len(protocol.responses) == 1
        response, called_addr = protocol.responses[0]
        assert isinstance(response, Response)
        assert response.status_code == 200
        assert called_addr == addr

    def test_request_received__invite__calls_invite_received(self):
        """Dispatch an INVITE request to invite_received with an IncomingCall."""
        protocol = InviteProtocol()
        protocol._transport = MagicMock()
        request = Request(
            method="INVITE",
            uri="sip:alice@atlanta.com",
            headers={"From": "sip:bob@biloxi.com"},
        )
        addr = ("192.0.2.1", 5060)
        protocol.request_received(request, addr)
        assert len(protocol.calls) == 1
        call, called_addr = protocol.calls[0]
        assert isinstance(call, IncomingCall)
        assert call.caller == "sip:bob@biloxi.com"
        assert called_addr == addr

    def test_request_received__returns_not_implemented(self):
        """Return NotImplemented for unhandled SIP requests."""
        protocol = SessionInitiationProtocol()
        request = Request(method="OPTIONS", uri="sip:bob@biloxi.com")
        result = protocol.request_received(request, ("192.0.2.1", 5060))
        assert result is NotImplemented

    def test_invite_received__returns_not_implemented(self):
        """Return NotImplemented for unhandled incoming calls."""
        protocol = SessionInitiationProtocol()
        call = MagicMock(spec=IncomingCall)
        result = protocol.invite_received(call, ("192.0.2.1", 5060))
        assert result is NotImplemented

    def test_response_received__returns_not_implemented(self):
        """Return NotImplemented for unhandled SIP responses."""
        protocol = SessionInitiationProtocol()
        response = Response(status_code=200, reason="OK")
        result = protocol.response_received(response, ("192.0.2.1", 5060))
        assert result is NotImplemented

    def test_error_received__blocking_io(self):
        """Log blocking IO errors without re-raising."""
        protocol = SessionInitiationProtocol()
        exc = OSError(errno.EAGAIN, "Resource temporarily unavailable")
        protocol.error_received(exc)  # should not raise

    def test_error_received__reraises(self):
        """Re-raise unexpected transport errors."""
        protocol = SessionInitiationProtocol()
        exc = OSError("Unexpected error")
        with pytest.raises(OSError):
            protocol.error_received(exc)

    def test_connection_lost__no_exception(self):
        """Handle a clean connection close without raising."""
        protocol = SessionInitiationProtocol()
        protocol.connection_lost(None)  # should not raise

    def test_connection_lost__with_exception(self):
        """Log an exception on connection lost without re-raising."""
        protocol = SessionInitiationProtocol()
        protocol.connection_lost(Exception("Connection reset"))  # should not raise
