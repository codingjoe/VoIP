"""Tests for the SIP asyncio protocol handler."""

import errno

import pytest
from voip.sip.messages import Request, Response
from voip.sip.protocol import SessionInitiationProtocol


class ConcreteProtocol(SessionInitiationProtocol):
    """Concrete subclass for testing that records received messages."""

    def __init__(self):
        super().__init__()
        self.requests = []
        self.responses = []

    def request_received(self, request, addr):
        self.requests.append((request, addr))

    def response_received(self, response, addr):
        self.responses.append((response, addr))


class TestSessionInitiationProtocol:
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
