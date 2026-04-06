"""
SIP asyncio protocol handler.

See also: https://datatracker.ietf.org/doc/html/rfc3261
"""

import asyncio
import dataclasses
import datetime
import logging
import typing

from voip.rtp import RealtimeTransportProtocol

from ..types import NetworkAddress
from . import types
from .dialog import Dialog
from .messages import USER_AGENT, Message, Request, Response
from .transactions import (
    ByeTransaction,
    InviteTransaction,
    RegistrationTransaction,
    Transaction,
)
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

    Handles SIP message parsing, carrier registration, and transaction management.

    Example:
        You can use the handler like any [asyncio.Protocol][asyncio.Protocol] in Python.

        ```python
        import asyncio

        from voip.sip import SessionInitiationProtocol

        async def main():
            loop = asyncio.get_running_loop()

            transport, protocol = await loop.create_connection(
                SessionInitiationProtocol,
                '0.0.0.0', 5060)

            try:
                await asyncio.Future()
            finally:
                transport.close()


        asyncio.run(main())
        ```

        However, this example is incomplete, since the protocol will require some
        arguments, like a reference to the RTP protocol and an AOR.

    > [!Note]
    > The support is limited to UAC (client mode).
    > This library currently does not implement server (UAS) functionality.

    [RFC 3261]: https://datatracker.ietf.org/doc/html/rfc3261

    Args:
        aor: SIP Address of Record (AOR) to register with the carrier.
        rtp: Shared RTP mux for call media.
        dialog_class: [Dialog][voip.sip.Dialog] subclass used to
            create dialogs for incoming calls.  Defaults to the base
            [Dialog][voip.sip.Dialog] which rejects all calls with
            ``486 Busy Here``.
        keepalive_interval: Keep-alive ping interval. Should be between 30 and 90 seconds.

    """

    aor: types.SipURI
    rtp: RealtimeTransportProtocol
    dialog_class: type[Dialog] = dataclasses.field(default=Dialog)
    keepalive_interval: datetime.timedelta = datetime.timedelta(seconds=30)

    keepalive_task: asyncio.Task | None = dataclasses.field(init=False, default=None)
    public_address: NetworkAddress = None
    _dialogs: dict[tuple[str, str], Dialog] = dataclasses.field(
        init=False, default_factory=dict
    )
    _transactions: dict[str, Transaction] = dataclasses.field(
        init=False, default_factory=dict
    )
    disconnected_event: asyncio.Event = dataclasses.field(
        init=False, default_factory=asyncio.Event
    )
    transport: asyncio.Transport | None = dataclasses.field(init=False, default=None)
    is_secure: bool = dataclasses.field(init=False, default=False)
    recv_buffer: bytearray = dataclasses.field(init=False, default_factory=bytearray)

    def __post_init__(self):
        self.public_address = self.public_address or self.rtp.public_address

    def add_dialog(self, dialog: Dialog) -> None:
        """Register *dialog* keyed by ``(dialog.local_tag, dialog.remote_tag)``."""
        self._dialogs[dialog.local_tag, dialog.remote_tag] = dialog

    def del_dialog(self, dialog: Dialog) -> None:
        """Remove *dialog* from the registry."""
        self._dialogs.pop((dialog.local_tag, dialog.remote_tag), None)

    def add_transaction(self, tx: Transaction) -> None:
        """Register *tx* by its branch parameter."""
        self._transactions[tx.branch] = tx

    def del_transaction(self, tx: Transaction) -> None:
        """Remove *tx* from the registry."""
        self._transactions.pop(tx.branch, None)

    def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
        """Store the TLS/TCP transport and start RTP mux + carrier registration."""
        self.transport = transport
        self.is_secure = transport.get_extra_info("ssl_object") is not None
        try:
            loop = asyncio.get_running_loop()
            tx = RegistrationTransaction(sip=self, method=SIPMethod.REGISTER)
            self._transactions[tx.branch] = tx
            loop.create_task(self.handle_registration(tx))
            self.keepalive_task = loop.create_task(self.send_keepalive())
        except RuntimeError:
            pass  # no running loop in synchronous test setups

    async def send_keepalive(self) -> None:
        while True:
            await asyncio.sleep(self.keepalive_interval.total_seconds())
            if self.transport is None:
                return
            logger.info("PING", extra={"addr": self.public_address})
            self.transport.write(PING)

    async def handle_registration(self, tx: RegistrationTransaction) -> None:
        await tx
        self.on_registered()

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
                logger.info("PONG", extra={"addr": self.public_address})
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
        message.headers.setdefault("User-Agent", USER_AGENT)
        if self.transport is not None:
            self.transport.write(bytes(message))

    def close(self) -> None:
        """Close the TLS/TCP transport and the RTP mux."""
        if self.transport is not None:
            self.transport.close()

    @property
    def allowed_methods(self) -> frozenset[SIPMethod]:
        """SIP methods supported by this UA."""
        return frozenset(
            {
                SIPMethod.INVITE,
                SIPMethod.ACK,
                SIPMethod.BYE,
                SIPMethod.CANCEL,
                SIPMethod.OPTIONS,
            }
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
        """Dispatch an incoming SIP request to the appropriate transaction."""
        match request.method:
            case SIPMethod.INVITE:
                asyncio.create_task(
                    InviteTransaction.receive(request=request, sip=self)
                )
            case SIPMethod.ACK:
                # For non-2xx ACKs the INVITE tx is still present; route by branch.
                tx = self._transactions.get(request.branch)
                if isinstance(tx, InviteTransaction):
                    tx.ack_received(request)
                    return
                # For 2xx ACKs the tx is gone; route by established dialog.
                try:
                    dialog = self._dialogs[request.remote_tag, request.local_tag]
                    dialog.invite_transaction.ack_received(request)
                except KeyError, AttributeError:
                    logger.warning("ACK for unknown dialog: %r", request)
            case SIPMethod.BYE:
                asyncio.create_task(ByeTransaction.receive(request=request, sip=self))
            case SIPMethod.CANCEL:
                try:
                    tx = self._transactions[request.branch]
                except KeyError:
                    self.send(
                        Response.from_request(
                            request,
                            status_code=SIPStatus.GONE,
                            phrase=SIPStatus.GONE.phrase,
                        )
                    )
                    return
                tx.cancel_received(request)
            case SIPMethod.OPTIONS:
                self.send(
                    Response.from_request(
                        request,
                        status_code=SIPStatus.OK,
                        phrase=SIPStatus.OK.phrase,
                        headers={"Allow": self.allow_header},
                    )
                )
            case _:
                self.method_not_allowed(request)

    def response_received(self, response: Response) -> None:
        """Delegate REGISTER responses to the registration transaction.

        Args:
            response: The parsed SIP response.
        """
        try:
            tx = self._transactions[response.branch]
        except KeyError:
            logger.warning(
                "Received response with unknown branch %r: %r",
                response.branch,
                response,
            )
        else:
            tx.response_received(response)

    def on_registered(self) -> None:
        """Handle successful carrier registration.

        Override in subclasses to initiate outbound calls or start other
        post-registration activity. The base implementation is a no-op.
        """

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
            f"{self.aor.user}@{self.public_address}"
            if self.aor.user
            else str(self.public_address)
        )
        ob_uri_param = ";ob"
        if self.aor.scheme == "sips":
            return f"<sips:{address}{ob_uri_param}>"
        tls_param = ";transport=tls" if self.is_secure else ";transport=tcp"
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
