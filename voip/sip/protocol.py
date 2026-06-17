"""
SIP asyncio protocol handler.

See also: https://datatracker.ietf.org/doc/html/rfc3261
"""

import asyncio
import dataclasses
import datetime
import ipaddress
import logging
import ssl
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
class SessionInitiationProtocol(asyncio.Protocol, asyncio.DatagramProtocol):
    """
    SIP User Agent Client (UAC) over TLS/TCP or UDP [RFC 3261].

    Handles SIP message parsing, carrier registration, and transaction management.
    The transport is selected automatically from the AOR's ``transport`` parameter:

    | `aor.transport` | Underlying transport |
    |-----------------|----------------------|
    | ``TLS`` (default) | TCP with TLS |
    | ``TCP`` | plain TCP |
    | ``UDP`` | UDP datagram socket |

    Use [`run`][voip.sip.protocol.SessionInitiationProtocol.run] for a single
    outbound connection and [`serve`][voip.sip.protocol.SessionInitiationProtocol.serve]
    for a persistent inbound server with automatic reconnection.

    Example:
        ```python
        import asyncio
        from voip.sip import SessionInitiationProtocol, Dialog

        async def main():
            protocol = await SessionInitiationProtocol.run(
                aor=SipURI.parse("sip:alice@carrier.example;transport=UDP"),
                dialog_class=Dialog,
            )
            # place outbound calls via protocol …
        ```

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
        keepalive_interval: Keep-alive ping interval for TCP transports.
            Should be between 30 and 90 seconds (RFC 5626).

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
    registered_event: asyncio.Event = dataclasses.field(
        init=False, default_factory=asyncio.Event
    )
    transport: asyncio.BaseTransport | None = dataclasses.field(
        init=False, default=None
    )
    is_secure: bool = dataclasses.field(init=False, default=False)
    recv_buffer: bytearray = dataclasses.field(init=False, default_factory=bytearray)

    def __post_init__(self):
        if self.public_address is None and self.rtp.public_address is not None:
            self.public_address = self.rtp.public_address.result()

    @classmethod
    async def run(
        cls,
        aor: types.SipURI,
        dialog_class: type[Dialog],
        *,
        rtp: RealtimeTransportProtocol | None = None,
        no_verify_tls: bool = False,
        stun_server: NetworkAddress | None = None,
        **kwargs: typing.Any,
    ) -> SessionInitiationProtocol:
        """Connect to the SIP proxy and return once registered.

        Establishes RTP (if not provided) and SIP/TLS connections derived from
        *aor*, then **suspends until SIP registration is confirmed** before
        returning the ready protocol.  After this call returns the caller may
        safely place outbound calls or start an MCP server.

        The transport protocol (TLS vs plain TCP) and proxy address are read
        from *aor* directly — no extra arguments are needed.

        Args:
            aor: SIP Address of Record, e.g. ``sip:alice@carrier.example``.
                The host, port, and ``transport`` parameter are used to connect
                to the SIP proxy.
            dialog_class: [`Dialog`][voip.sip.Dialog] subclass used for
                inbound calls.  Defaults to the base
                [`Dialog`][voip.sip.Dialog], which rejects all calls.
            rtp: Existing RTP endpoint to reuse.  When ``None`` (default) a
                new datagram endpoint is created from *aor* and *stun_server*.
                Pass an existing instance to share one endpoint across
                reconnections (see [`serve`][voip.sip.protocol.SessionInitiationProtocol.serve]).
            no_verify_tls: Disable TLS certificate verification. Insecure; for
                testing only. Defaults to ``False``.
            stun_server: STUN server for RTP NAT traversal. Ignored when *rtp*
                is provided.
            **kwargs: Extra keyword arguments forwarded to the protocol
                constructor, e.g. ``verbose=2`` for
                [`ConsoleMessageProtocol`][voip.__main__.ConsoleMessageProtocol].

        Returns:
            The registered [`SessionInitiationProtocol`][voip.sip.protocol.SessionInitiationProtocol]
            instance, ready to place calls.
        """
        loop = asyncio.get_running_loop()
        if rtp is None:
            rtp_bind_address = (
                "::" if isinstance(aor.maddr[0], ipaddress.IPv6Address) else "0.0.0.0"  # noqa: S104
            )
            rtp = await RealtimeTransportProtocol.create(rtp_bind_address, stun_server)
        if aor.transport == "UDP":
            _, protocol = await loop.create_datagram_endpoint(
                lambda: cls(aor=aor, rtp=rtp, dialog_class=dialog_class, **kwargs),
                remote_addr=(str(aor.maddr[0]), aor.maddr[1]),
            )
        else:
            ssl_context: ssl.SSLContext | None = None
            if aor.transport == "TLS":
                ssl_context = ssl.create_default_context()
                if no_verify_tls:
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
            _, protocol = await loop.create_connection(
                lambda: cls(aor=aor, rtp=rtp, dialog_class=dialog_class, **kwargs),
                host=str(aor.maddr[0]),
                port=aor.maddr[1],
                ssl=ssl_context,
            )
        await protocol.registered_event.wait()
        return protocol

    @classmethod
    async def serve(
        cls,
        aor: types.SipURI,
        dialog_class: type[Dialog],
        *,
        no_verify_tls: bool = False,
        stun_server: NetworkAddress | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Register with a carrier and handle inbound calls, reconnecting on disconnect.

        Creates one RTP endpoint for the lifetime of the process, then enters a
        persistent loop: connect to the SIP proxy, wait for the connection to drop,
        and reconnect with exponential back-off.  Use this for long-running
        inbound-call servers.

        The transport protocol (TLS vs plain TCP) and proxy address are read from
        *aor* directly.

        Args:
            aor: SIP Address of Record, e.g. ``sip:alice@carrier.example``.
            dialog_class: [`Dialog`][voip.sip.Dialog] subclass used for
                inbound calls.
            no_verify_tls: Disable TLS certificate verification. Insecure; for
                testing only. Defaults to ``False``.
            stun_server: STUN server for RTP NAT traversal.
            **kwargs: Extra keyword arguments forwarded to the protocol
                constructor, e.g. ``verbose=2`` for
                [`ConsoleMessageProtocol`][voip.__main__.ConsoleMessageProtocol].
        """
        rtp_bind_address = (
            "::" if isinstance(aor.maddr[0], ipaddress.IPv6Address) else "0.0.0.0"  # noqa: S104
        )
        rtp = await RealtimeTransportProtocol.create(rtp_bind_address, stun_server)
        backoff_secs = 1
        while True:
            try:
                protocol = await cls.run(
                    aor, dialog_class, rtp=rtp, no_verify_tls=no_verify_tls, **kwargs
                )
                backoff_secs = 1
                await protocol.disconnected_event.wait()
                logger.info("SIP connection closed; reconnecting in %s s", backoff_secs)
            except (OSError, ssl.SSLError) as exc:
                logger.warning(
                    "SIP connection failed (%s); retrying in %s s", exc, backoff_secs
                )
            await asyncio.sleep(backoff_secs)
            backoff_secs = min(backoff_secs * 2, 60)

    def register_dialog(self, dialog: Dialog) -> None:
        """Register *dialog* keyed by ``(dialog.local_tag, dialog.remote_tag)``."""
        if dialog.remote_tag is None:
            logger.warning("Dialog without remote tag cannot be registered: %r", dialog)
        else:
            self._dialogs[dialog.local_tag, dialog.remote_tag] = dialog

    def drop_dialog(self, dialog: Dialog) -> None:
        """Remove *dialog* from the registry."""
        if dialog.remote_tag is None:
            logger.warning("Dialog without remote tag cannot be removed: %r", dialog)
        else:
            try:
                del self._dialogs[dialog.local_tag, dialog.remote_tag]
            except KeyError:
                logger.warning("Dialog not found for removal: %r", dialog)

    def register_transaction(self, tx: Transaction) -> None:
        """Register *tx* by its branch parameter."""
        self._transactions[tx.branch] = tx

    def drop_transaction(self, tx: Transaction) -> None:
        """Remove *tx* from the registry."""
        try:
            del self._transactions[tx.branch]
        except KeyError:
            logger.warning("Transaction not found for removal: %r", tx)

    def connection_made(self, transport: asyncio.BaseTransport) -> None:  # type: ignore[override]
        """Store the transport and start carrier registration.

        Keepalive pings (RFC 5626) are only started for TCP/TLS transports.
        """
        self.transport = transport
        self.is_secure = (
            not isinstance(transport, asyncio.DatagramTransport)
            and transport.get_extra_info("ssl_object") is not None
        )
        try:
            loop = asyncio.get_running_loop()
            tx = RegistrationTransaction(sip=self, method=SIPMethod.REGISTER)
            self.register_transaction(tx)
            loop.create_task(self.handle_registration(tx))
            if not isinstance(transport, asyncio.DatagramTransport):
                self.keepalive_task = loop.create_task(self.send_keepalive())
        except RuntimeError:
            pass  # no running loop in synchronous test setups

    async def send_keepalive(self) -> None:
        while True:
            await asyncio.sleep(self.keepalive_interval.total_seconds())
            if self.transport is None or isinstance(
                self.transport, asyncio.DatagramTransport
            ):
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

    def datagram_received(self, data: bytes, addr: tuple) -> None:  # type: ignore[override]
        """Dispatch a complete UDP SIP datagram."""
        self._dispatch_frame(data)

    def error_received(self, exc: Exception) -> None:  # type: ignore[override]
        """Log a UDP transport error."""
        logger.warning("UDP error received", exc_info=exc)

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
        peer = NetworkAddress(*self.transport.get_extra_info("peername")[:2])
        if frame == PONG:
            logger.info("PONG", extra={"addr": peer})
        elif frame == PING:
            logger.info("PING", extra={"addr": peer})
            if self.transport and not isinstance(
                self.transport, asyncio.DatagramTransport
            ):
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
        """Serialize and send a SIP message over the active transport."""
        logger.debug("Sending %r", message)
        message.headers.setdefault("User-Agent", USER_AGENT)
        if self.transport is None:
            return
        if isinstance(self.transport, asyncio.DatagramTransport):
            self.transport.sendto(bytes(message))
        else:
            self.transport.write(bytes(message))

    def close(self) -> None:
        """Close the transport."""
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
                try:
                    tx = self._dialogs[
                        request.remote_tag, request.local_tag
                    ].invite_transaction
                except KeyError:
                    self.send(
                        Response.from_request(
                            request,
                            status_code=SIPStatus.GONE,
                            phrase=SIPStatus.GONE.phrase,
                        )
                    )
                else:
                    tx.ack_received(request)
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
        self.registered_event.set()

    @property
    def contact(self) -> str:
        """Return a ``Contact:`` header value for this UA.

        The URI scheme and transport parameter mirror the active transport:

        | Transport | Contact URI |
        |-----------|-------------|
        | SIPS AOR  | ``sips:…;ob`` |
        | TLS       | ``sip:…;transport=tls;ob`` |
        | TCP       | ``sip:…;transport=tcp;ob`` |
        | UDP       | ``sip:…;transport=udp`` |

        The ``ob`` parameter ([RFC 5626 §5]) advertises outbound keep-alive
        support to the registrar for TCP/TLS transports.

        [RFC 5626 §5]: https://datatracker.ietf.org/doc/html/rfc5626#section-5
        """
        address = (
            f"{self.aor.user}@{self.public_address}"
            if self.aor.user
            else str(self.public_address)
        )
        if self.aor.scheme == "sips":
            return f"<sips:{address};ob>"
        if isinstance(self.transport, asyncio.DatagramTransport):
            return f"<sip:{address};transport=udp>"
        transport_param = "tls" if self.is_secure else "tcp"
        return f"<sip:{address};transport={transport_param};ob>"

    def connection_lost(self, exc: Exception | None) -> None:
        """Handle a lost or closed transport connection."""
        if exc is not None:
            logger.exception("Connection lost", exc_info=exc)
        if self.keepalive_task is not None:
            self.keepalive_task.cancel()
            self.keepalive_task = None
        self.transport = None
        self.disconnected_event.set()


#: Short alias for `SessionInitiationProtocol`.
SIP = SessionInitiationProtocol
