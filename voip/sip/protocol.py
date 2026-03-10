"""SIP asyncio protocol handler."""

from __future__ import annotations

import asyncio
import errno
import logging

from voip.rtp import RTP, RealtimeTransportProtocol

from .messages import Message, Request, Response
from .types import SIPStatus

logger = logging.getLogger(__name__)

__all__ = ["SIP", "SessionInitiationProtocol"]


class _RTPProtocol(RealtimeTransportProtocol):
    """Internal asyncio protocol that strips RTP headers and forwards audio to an RTP handler."""

    def __init__(self, handler: RTP) -> None:
        self._handler = handler

    def audio_received(self, data: bytes) -> None:
        """Forward the RTP audio payload to the associated call handler."""
        self._handler.audio_received(data)


class SIP(asyncio.DatagramProtocol):
    """SIP session handler (RFC 3261).

    Subclass and override :meth:`call_received` to handle incoming calls::

        class MySession(SIP):
            def call_received(self, request: Request) -> None:
                self.answer(request=request, call_class=MyCall)

    Use :class:`voip.sip.RegisterSIP` if you need to register with a SIP carrier first.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: Pending INVITE addresses keyed by Call-ID.
        self._request_addrs: dict[str, tuple[str, int]] = {}

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        """Store the transport for sending SIP messages."""
        logger.debug("SIP transport connected")
        self._transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Handle RFC 5626 keepalive pings, then dispatch SIP messages."""
        if data == b"\r\n\r\n":  # RFC 5626 §4.4.1 double-CRLF keepalive ping
            logger.debug("RFC 5626 keepalive from %s, sending pong", addr)
            self._transport.sendto(b"\r\n", addr)
            return
        match Message.parse(data):
            case Request() as request:
                self.request_received(request, addr)
            case Response() as response:
                self.response_received(response, addr)

    def send(self, message: Response | Request, addr: tuple[str, int]) -> None:
        """Serialize and send a SIP message to the given address."""
        logger.debug("Sending %r to %r", message, addr)
        self._transport.sendto(bytes(message), addr)

    def request_received(self, request: Request, addr: tuple[str, int]) -> None:
        """Dispatch a received SIP request to the appropriate handler."""
        match request.method:
            case "INVITE":
                logger.info("INVITE received from %s", addr[0])
                call_id = request.headers.get("Call-ID", "")
                self._request_addrs[call_id] = addr
                self.call_received(request)
            case _:
                raise NotImplementedError(
                    f"Unsupported SIP request method: {request.method}"
                )

    def response_received(self, response: Response, addr: tuple[str, int]) -> None:
        """Handle a received SIP response. Override in subclasses to process responses."""

    def call_received(self, request: Request) -> None:
        """Handle an incoming call.

        Override in subclasses to accept or reject the call::

            def call_received(self, request: Request) -> None:
                self.answer(request=request, call_class=MyCall)

        Args:
            request: The SIP INVITE request.
        """

    def answer(self, request: Request, *, call_class: type[RTP] = RTP) -> None:
        """Answer an incoming call by setting up RTP and sending 200 OK with SDP.

        Schedules the asynchronous RTP setup and SIP response without blocking.

        Args:
            request: The SIP INVITE request (from :meth:`call_received`).
            call_class: The :class:`~voip.rtp.RTP` subclass to instantiate for the call.
        """
        asyncio.get_running_loop().create_task(self._answer(request, call_class))

    async def _answer(self, request: Request, call_class: type[RTP]) -> None:
        """Perform the asynchronous part of answering: set up RTP, send 200 OK."""
        call_id = request.headers.get("Call-ID", "")
        addr = self._request_addrs.pop(call_id, None)
        if addr is None:
            logger.error("No address found for INVITE with Call-ID %r", call_id)
            return
        call = call_class(caller=request.headers.get("From", ""))
        logger.info("Answering call from %s", call.caller)
        loop = asyncio.get_running_loop()
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: _RTPProtocol(call),
            local_addr=("0.0.0.0", 0),  # noqa: S104
        )
        local_addr = rtp_transport.get_extra_info("sockname")
        sdp_ip = self._contact_ip or local_addr[0]
        logger.debug("RTP listening on %s:%s", local_addr[0], local_addr[1])
        sdp = (
            f"v=0\r\n"
            f"c=IN IP4 {sdp_ip}\r\n"
            f"m=audio {local_addr[1]} RTP/AVP 111\r\n"
            f"a=rtpmap:111 opus/48000/1\r\n"
        ).encode()
        self.send(
            Response(
                status_code=SIPStatus.OK.status_code,
                reason=SIPStatus.OK.reason,
                headers={
                    **{
                        key: value
                        for key, value in request.headers.items()
                        if key in ("Via", "To", "From", "Call-ID", "CSeq")
                    },
                    "Content-Type": "application/sdp",
                },
                body=sdp,
            ),
            addr,
        )

    def reject(
        self,
        request: Request,
        status_code: int = SIPStatus.BUSY_HERE.status_code,
        reason: str = SIPStatus.BUSY_HERE.reason,
    ) -> None:
        """Reject an incoming call.

        Args:
            request: The SIP INVITE request (from :meth:`call_received`).
            status_code: SIP response status code (default: 486 Busy Here).
            reason: SIP response reason phrase.
        """
        call_id = request.headers.get("Call-ID", "")
        addr = self._request_addrs.pop(call_id, None)
        if addr is None:
            logger.error("No address found for INVITE with Call-ID %r", call_id)
            return
        logger.info(
            "Rejecting call from %s with %s %s",
            request.headers.get("From", "unknown"),
            status_code,
            reason,
        )
        self.send(
            Response(
                status_code=status_code,
                reason=reason,
                headers={
                    key: value
                    for key, value in request.headers.items()
                    if key in ("Via", "To", "From", "Call-ID", "CSeq")
                },
            ),
            addr,
        )

    @property
    def _contact_ip(self) -> str | None:
        """Return the IP address to advertise in SDP (None if not available)."""
        return None

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


#: Alias for backward compatibility.
SessionInitiationProtocol = SIP
