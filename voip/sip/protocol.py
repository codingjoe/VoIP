"""
SIP asyncio protocol handler.

See also: https://datatracker.ietf.org/doc/html/rfc3261
"""

import asyncio
import dataclasses
import datetime
import ipaddress
import logging
import typing

from voip.rtp import RealtimeTransportProtocol

from ..types import NetworkAddress
from . import types
from .messages import Dialog, Message, Request, Response
from .transactions import InviteTransaction, RegistrationTransaction, Transaction
from .types import (
    SIPMethod,
    SIPStatus,
)

logger = logging.getLogger("voip.sip")

#: RFC 5626 §4.4 keepalive PING sequence.
PING: typing.Final[bytes] = b"\r\n\r\n"

#: RFC 5626 §4.4 keepalive PONG reply.
PONG: typing.Final[bytes] = b"\r\n"

__all__ = [
    "SIP",
    "SessionInitiationProtocol",
    "InviteTransaction",
    "RegistrationTransaction",
]


@dataclasses.dataclass(kw_only=True, slots=True)
class SessionInitiationProtocol(asyncio.Protocol):
    """
    SIP User Agent Client (UAC) over TLS/TCP [RFC 3261].

    Handles incoming calls and, optionally, carrier registration with digest
    authentication [RFC 3261 §22].  All signaling is sent over a single
    persistent TLS/TCP connection.

    ```python
    class MyTransaction(Transaction):
        def call_received(self, request: Request) -> None:
            asyncio.create_task(self.answer(call_class=MyCall))

    class MySession(SessionInitiationProtocol):
        transaction_class = MyTransaction
    ```

    To register with a carrier on startup, pass the registration parameters:

    ```python
    session = SessionInitiationProtocol(
        aor="sips:alice@example.com",
        username="alice",
        password="secret",
    )
    ```

    [RFC 3261]: https://datatracker.ietf.org/doc/html/rfc3261
    [RFC 3261 §22]: https://datatracker.ietf.org/doc/html/rfc3261#section-22

    Attributes:
        VIA_BRANCH_PREFIX:
            RFC 3261 §8.1.1.7 Via branch magic cookie (indicates RFC 3261 compliance).

    Args:
        aor: SIP Address of Record (AOR) to register with the carrier, e.g.
        rtp: Shared RTP mux for call media.  When provided, call handlers can register
            their RTP addresses with the mux to receive media packets.
        transaction_class: Transaction subclass to handle SIP transactions.
        registration_class: Transaction subclass to handle registration transactions.
        keepalive_interval: Keep-alive ping interval. Should be between 30 and 90 seconds.

    """

    VIA_BRANCH_PREFIX: typing.ClassVar[str] = "z9hG4bK"

    aor: types.SipUri
    rtp: RealtimeTransportProtocol
    transaction_class: type[InviteTransaction]
    registration_class: type[RegistrationTransaction] = RegistrationTransaction
    keepalive_interval: datetime.timedelta = datetime.timedelta(seconds=30)

    keepalive_task: asyncio.Task | None = dataclasses.field(init=False, default=None)
    local_address: NetworkAddress = dataclasses.field(init=False)
    dialogs: dict[tuple[str, str], Dialog] = dataclasses.field(
        init=False, default_factory=dict
    )
    transactions: dict[str, Transaction] = dataclasses.field(
        init=False, default_factory=dict
    )
    disconnected_event: asyncio.Event = dataclasses.field(
        init=False, default_factory=asyncio.Event
    )
    transport: asyncio.Transport | None = dataclasses.field(init=False, default=None)
    is_secure: bool = dataclasses.field(init=False, default=False)
    recv_buffer: bytearray = dataclasses.field(init=False, default_factory=bytearray)

    def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
        """Store the TLS/TCP transport and start RTP mux + carrier registration."""
        self.transport = transport
        # IPv6 sockets return a 4-tuple (host, port, flowinfo, scope_id);
        # we only need the first two elements.
        host, port = transport.get_extra_info("sockname")[:2]
        self.local_address = NetworkAddress(ipaddress.ip_address(host), port)
        self.is_secure = transport.get_extra_info("ssl_object") is not None
        try:
            loop = asyncio.get_running_loop()
            tx = RegistrationTransaction(sip=self, method=SIPMethod.REGISTER)
            self.transactions[tx.branch] = tx
            self.keepalive_task = loop.create_task(self.send_keepalive())
        except RuntimeError:
            pass  # no running loop in synchronous test setups

    async def send_keepalive(self) -> None:
        while True:
            await asyncio.sleep(self.keepalive_interval.total_seconds())
            if self.transport is None:
                return
            logger.info("PING", extra={"addr": self.local_address})
            self.transport.write(PING)

    def data_received(self, data: bytes) -> None:
        self.recv_buffer.extend(data)
        for frame in self._extract_frames():
            self._dispatch_frame(frame)

    def _extract_frames(self) -> typing.Generator[memoryview | bytes]:  # noqa: C901
        while self.recv_buffer:
            if self.recv_buffer[0:1] != b"\r":
                # SIP message: wait for the header-body separator.
                header_end = self.recv_buffer.find(b"\r\n\r\n")
                if header_end == -1:
                    break  # incomplete headers – wait for more data
                content_length = 0
                for line in self.recv_buffer[:header_end].split(b"\r\n")[1:]:
                    name, sep, value = line.partition(b":")
                    if sep and name.strip().lower() == b"content-length":
                        try:
                            content_length = int(value.strip())
                        except ValueError:
                            pass
                        break
                message_end = header_end + 4 + content_length
                if len(self.recv_buffer) < message_end:
                    break  # incomplete body – wait for more data
                frame = memoryview(self.recv_buffer)[:message_end]
                yield frame
                frame.release()
                del self.recv_buffer[:message_end]
            elif len(self.recv_buffer) >= 4 and self.recv_buffer[:4] == PING:
                yield PING
                del self.recv_buffer[:4]
            elif len(self.recv_buffer) >= 3 and self.recv_buffer[2:3] == b"\r":
                # Third byte is CR – could be the start of PING; wait for 4th byte.
                break
            elif self.recv_buffer[:2] == PONG:
                yield PONG
                del self.recv_buffer[:2]
            else:
                # Single CR or other incomplete sequence – wait for more data.
                break

    def _dispatch_frame(self, frame: memoryview | bytes) -> None:
        peer = NetworkAddress(*self.transport.get_extra_info("peername"))
        if frame == PONG:
            logger.info("PONG", extra={"addr": peer})
        elif frame == PING:
            logger.info("PING", extra={"addr": peer})
            if self.transport:
                logger.info("PONG", extra={"addr": self.local_address})
                self.transport.write(PONG)
        else:
            match Message.parse(bytes(frame)):
                case Request() as request:
                    logger.info(
                        "Request received: %r",
                        request,
                        extra={"addr": peer},
                    )
                    self.request_received(request)
                case Response() as response:
                    logger.info(
                        "Response received %r",
                        response,
                        extra={"addr": peer},
                    )
                    self.response_received(response)

    def send(self, message: Response | Request) -> None:
        """Serialize and send a SIP message over the TLS/TCP connection."""
        logger.debug("Sending %r", message)
        if self.transport is not None:
            self.transport.write(bytes(message))

    def close(self) -> None:
        """Close the TLS/TCP transport and the RTP mux."""
        if self.transport is not None:
            self.transport.close()

    @property
    def allowed_methods(self) -> frozenset[SIPMethod]:
        """SIP methods supported by this UA.

        A method is included when the class defines a ``<method_lower>_received``
        handler (e.g. ``register_received`` enables REGISTER).

        Returns:
            Frozenset of [`SIPMethod`][voip.sip.types.SIPMethod] values.
        """
        return frozenset(
            (
                *(
                    m
                    for m in SIPMethod
                    if hasattr(self.transaction_class, f"{m.lower()}_received")
                ),
                SIPMethod.OPTIONS,
            )
        )

    @property
    def allow_header(self) -> str:
        """Comma-separated Allow header value in SIPMethod enum order."""
        return ",".join(m for m in SIPMethod if m in self.allowed_methods)

    def method_not_allowed(self, request: Request) -> None:
        """Respond with 405 Method Not Allowed.

        Override to customise the error response or add logging.

        Args:
            request: The unhandled SIP request.
        """
        logger.warning("SIP method %r is not supported", request.method)
        dialog_headers = {
            key: value
            for key, value in request.headers.items()
            if key in ("Via", "To", "From", "Call-ID", "CSeq")
        }
        self.send(
            Response(
                status_code=SIPStatus.METHOD_NOT_ALLOWED,
                phrase=SIPStatus.METHOD_NOT_ALLOWED.phrase,
                headers={**dialog_headers, "Allow": self.allow_header},
            ),
        )

    def request_received(self, request: Request) -> None:
        """Dispatch request to transaction methods."""
        match request.method:
            case SIPMethod.CANCEL:
                try:
                    tx = self.transactions[request.branch]
                except KeyError:
                    self.send(
                        Response.from_request(
                            request,
                            status_code=SIPStatus.GONE,
                            phrase=SIPStatus.GONE.phrase,
                        )
                    )
                    return
            case SIPMethod.OPTIONS:
                self.send(
                    Response.from_request(
                        request,
                        status_code=SIPStatus.OK,
                        phrase=SIPStatus.OK.phrase,
                        headers={"Allow": self.allow_header},
                    )
                )
                return
            case _:
                tx = self.transaction_class.from_request(request=request, sip=self)
                self.transactions[request.branch] = tx
        try:
            handler: typing.Callable[[Request], Response | None] = getattr(
                tx, f"{request.method.lower()}_received"
            )
        except AttributeError:
            handler = self.method_not_allowed
        handler(request)

    def response_received(self, response: Response) -> None:
        """Delegate REGISTER responses to the registration transaction.

        Args:
            response: The parsed SIP response.
        """
        try:
            tx = self.transactions[response.branch]
        except KeyError:
            logger.warning(
                "Received response with unknown branch %r: %r",
                response.branch,
                response,
            )
        else:
            tx.response_received(response)

    @property
    def contact(self) -> str:
        """Return a ``Contact:`` header value for this UA.

        The URI scheme mirrors `aor`: a ``sips:`` AOR produces a
        ``sips:`` Contact (the strongest TLS guarantee); a ``sip:`` AOR over
        TLS produces ``sip:`` with ``transport=tls``; plain TCP produces plain
        ``sip:``.

        When *ob* is ``True`` the ``ob`` URI parameter ([RFC 5626 §5]) is
        appended inside the angle brackets to advertise outbound keep-alive
        support to the registrar.

        [RFC 5626 §5]: https://datatracker.ietf.org/doc/html/rfc5626#section-5
        """
        address = (
            f"{self.aor.user}@{self.local_address}"
            if self.aor.user
            else str(self.local_address)
        )
        ob_uri_param = ";ob"
        if self.aor.scheme == "sips":
            return f"<sips:{address}{ob_uri_param}>"
        tls_param = ";transport=tls" if self.is_secure else ""
        return f"<sip:{address}{tls_param}{ob_uri_param}>"

    def connection_lost(self, exc: Exception | None) -> None:
        """Handle a lost TLS/TCP connection."""
        if exc is not None:
            logger.exception("Connection lost", exc_info=exc)
        if self.keepalive_task is not None:
            self.keepalive_task.cancel()
            self.keepalive_task = None
        self.transport = None
        self.disconnected_event.set()


#: Short alias for `SessionInitiationProtocol`.
SIP = SessionInitiationProtocol
