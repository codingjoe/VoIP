"""SIP asyncio protocol handler."""

from __future__ import annotations

import asyncio
import errno
import logging

from .messages import Message, Request, Response

logger = logging.getLogger(__name__)

__all__ = ["SIP", "SessionInitiationProtocol"]


class SessionInitiationProtocol(asyncio.DatagramProtocol):
    """An asyncio protocol handler for the Session Initiation Protocol (RFC 3261)."""

    __slots__ = ()

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Dispatch a received datagram to the appropriate handler."""
        match Message.parse(data):
            case Request() as request:
                self.request_received(request, addr)
            case Response() as response:
                self.response_received(response, addr)

    def request_received(self, request: Request, addr: tuple[str, int]) -> None:
        """Handle a received SIP request. Override in subclasses to process requests."""
        return NotImplemented

    def response_received(self, response: Response, addr: tuple[str, int]) -> None:
        """Handle a received SIP response. Override in subclasses to process responses."""
        return NotImplemented

    def error_received(self, exc: OSError) -> None:
        """Handle a transport-level error."""
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            logger.exception("Blocking IO error", exc_info=exc)
        else:
            raise exc

    def connection_lost(self, exc: Exception | None) -> None:
        """Handle a lost connection."""
        if exc is not None:
            logger.exception("Connection lost", exc_info=exc)


SIP = SessionInitiationProtocol
