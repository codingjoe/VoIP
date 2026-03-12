"""
SIP asyncio protocol handler.

See also: https://datatracker.ietf.org/doc/html/rfc3261
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import hashlib
import json
import logging
import re
import secrets
import socket
import typing
import uuid

from voip.call import Call
from voip.rtp import RealtimeTransportProtocol
from voip.sdp.messages import SessionDescription
from voip.sdp.types import (
    Attribute,
    ConnectionData,
    MediaDescription,
    Origin,
    RTPPayloadFormat,
    Timing,
)
from voip.stun import STUNProtocol
from voip.types import DigestQoP

from .messages import Message, Request, Response
from .types import CallerID, Status

logger = logging.getLogger("voip.sip")

__all__ = ["SIP", "SessionInitiationProtocol"]


def _mask_caller(header: str) -> str:
    """Return a privacy-safe label from a SIP From/To header value.

    Strips the ``tag=`` parameter, extracts the display name or SIP user part,
    and replaces all but the last four characters with ``*``.

    Examples::

        >>> _mask_caller('"015114455910" <sip:015114455910@example.com>;tag=abc')
        '********5910'
        >>> _mask_caller('sip:alice@example.com')
        '*lice'
    """
    # Drop the tag and any subsequent parameters
    value = header.split(";")[0].strip()
    # Extract display name: "Name" <sip:…> or Name <sip:…>
    m = re.match(r'^"?([^"<]+?)"?\s*<', value)
    name = m.group(1).strip() if m else None
    if not name:
        # Bare or angle-bracket URI: sip:user@host or <sip:user@host>
        m = re.search(r"sips?:([^@>;\s]+)", value)
        name = m.group(1) if m else value
    if len(name) > 4:
        return "*" * (len(name) - 4) + name[-4:]
    return name


@dataclasses.dataclass(kw_only=True, slots=True)
class SessionInitiationProtocol(STUNProtocol):
    """SIP session handler (RFC 3261).

    Handles incoming calls and, optionally, carrier registration with digest
    auth (RFC 3261 §22).

    Subclass and override :meth:`call_received` to handle incoming calls::

        class MySession(SessionInitiationProtocol):
            def call_received(self, request: Request) -> None:
                self.answer(request=request, call_class=MyCall)

    To register with a carrier on startup, pass the registration parameters::

        session = SessionInitiationProtocol(
            server_address=("sip.example.com", 5060),
            aor="sip:alice@example.com",
            username="alice",
            password="secret",
        )
    """

    #: RFC 3261 §8.1.1.7 Via branch magic cookie (indicates RFC 3261 compliance).
    VIA_BRANCH_PREFIX: typing.ClassVar[str] = "z9hG4bK"

    #: RFC 3261 §11 – methods supported by this UA (used in Allow header).
    ALLOW: typing.ClassVar[str] = "INVITE, ACK, BYE, CANCEL, OPTIONS"

    _request_addrs: dict[str, tuple[str, int]] = dataclasses.field(
        init=False, default_factory=dict
    )
    _answered_calls: collections.OrderedDict[str, None] = dataclasses.field(
        init=False, default_factory=collections.OrderedDict
    )
    answered_call_backlog: int = 1000
    _to_tags: dict[str, str] = dataclasses.field(init=False, default_factory=dict)
    _rtp_protocol: RealtimeTransportProtocol | None = dataclasses.field(
        init=False, default=None
    )
    _rtp_transport: asyncio.DatagramTransport | None = dataclasses.field(
        init=False, default=None
    )
    _call_rtp_addrs: dict[str, tuple[str, int] | None] = dataclasses.field(
        init=False, default_factory=dict
    )
    transport: asyncio.DatagramTransport = dataclasses.field(init=False)
    public_address: asyncio.Future[tuple[str, int]] = dataclasses.field(init=False)
    server_address: tuple[str, int]
    aor: str
    username: str | None = None
    password: str | None = None
    call_id: str = dataclasses.field(init=False)
    cseq: int = dataclasses.field(init=False, default=0)

    def __post_init__(self):
        self.call_id = f"{uuid.uuid4()}@{socket.gethostname()}"

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        """Store the transport, create :attr:`public_address`, and start STUN discovery."""
        logger.debug("SIP transport connected")
        self.transport = transport
        self.public_address = asyncio.get_running_loop().create_future()
        STUNProtocol.connection_made(self, transport)
        if self.stun_server_address is None:
            # No STUN: use the local socket address and start up immediately.
            self.public_address.set_result(transport.get_extra_info("sockname"))
            try:
                asyncio.get_running_loop().create_task(self._initialize())
            except RuntimeError:
                pass  # no running loop in synchronous test setups

    def stun_connection_made(
        self,
        transport: asyncio.DatagramTransport,
        addr: tuple[str, int],
    ) -> None:
        """Resolve :attr:`public_address` and begin initialisation on STUN completion."""
        if not self.public_address.done():
            self.public_address.set_result(addr)
        # Schedule RTP mux creation and (optionally) registration in a single task so
        # that both the SIP and RTP public addresses are known before we send REGISTER.
        try:
            asyncio.get_running_loop().create_task(self._initialize())
        except RuntimeError:
            pass  # no running loop in synchronous test setups

    async def _initialize(self) -> None:
        """Set up the RTP mux and register with the carrier (in that order).

        Scheduled by :meth:`stun_connection_made` after STUN completes, so
        :attr:`public_address` is already resolved when :meth:`register` runs.
        """
        await self._start_rtp_mux()
        await self.register()

    def packet_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Handle RFC 5626 keepalive pings, then dispatch SIP messages."""
        if data == b"\r\n\r\n":  # RFC 5626 §4.4.1 double-CRLF keepalive ping
            logger.debug("RFC 5626 keepalive from %s, sending pong", addr)
            self.transport.sendto(b"\r\n", addr)
            return
        match Message.parse(data):
            case Request() as request:
                self.request_received(request, addr)
            case Response() as response:
                self.response_received(response, addr)

    def send(self, message: Response | Request, addr: tuple[str, int]) -> None:
        """Serialize and send a SIP message to the given address."""
        logger.debug("Sending %r to %r", message, addr)
        self.transport.sendto(bytes(message), addr)

    def close(self) -> None:
        """Close the underlying UDP transport."""
        if self.transport is not None:
            self.transport.close()

    def _cleanup_rtp_call(self, call_id: str) -> None:
        """Remove the call handler registered with the shared RTP mux, if any."""
        if call_id in self._call_rtp_addrs and self._rtp_protocol is not None:
            self._rtp_protocol.unregister_call(self._call_rtp_addrs.pop(call_id))

    async def _start_rtp_mux(self) -> None:
        """Create and connect the shared RTP multiplexer socket (idempotent).

        Passes :attr:`stun_server_address` to the mux so that it discovers its
        own public address automatically on ``connection_made``.  Waits for the
        RTP STUN task to settle before returning so that callers can rely on
        ``self._rtp_protocol.public_address`` being populated.
        """
        if self._rtp_protocol is not None:
            return
        loop = asyncio.get_running_loop()
        mux = RealtimeTransportProtocol(stun_server_address=self.stun_server_address)
        self._rtp_transport, self._rtp_protocol = await loop.create_datagram_endpoint(
            lambda: mux,
            local_addr=("0.0.0.0", 0),  # noqa: S104
        )
        rtp_addr = self._rtp_transport.get_extra_info("sockname")
        logger.debug("RTP mux listening on %s:%d", *rtp_addr)
        # Wait for RTP STUN to complete before returning so that the mux's
        # public_address is available when _answer() builds the SDP.
        await mux.public_address

    def _mark_call_answered(self, call_id: str) -> None:
        """Record *call_id* as answered, evicting the oldest entry if the LRU is full."""
        if call_id in self._answered_calls:
            self._answered_calls.move_to_end(call_id)
        else:
            if len(self._answered_calls) >= self.answered_call_backlog:
                self._answered_calls.popitem(last=False)
            self._answered_calls[call_id] = None

    def request_received(self, request: Request, addr: tuple[str, int]) -> None:
        """Dispatch a received SIP request to the appropriate handler."""
        call_id = request.headers.get("Call-ID", "")
        match request.method:
            case "INVITE":
                caller = CallerID(request.headers.get("From", ""))
                logger.info(
                    json.dumps(
                        {
                            "event": "incoming_call",
                            "caller": repr(caller),
                            "ip": addr[0],
                            "call_id": call_id,
                        }
                    ),
                    extra={"caller": repr(caller), "ip": addr[0], "call_id": call_id},
                )
                if call_id in self._answered_calls:
                    logger.debug(
                        "Ignoring INVITE retransmission for Call-ID %r", call_id
                    )
                    return
                # Mark immediately (before async answering) so retransmissions
                # that arrive while RTP setup is in progress are suppressed.
                self._mark_call_answered(call_id)
                self._request_addrs[call_id] = addr
                self._to_tags[call_id] = secrets.token_hex(8)
                self.call_received(request)
            case "ACK":
                self.ack_received(request)
            case "BYE":
                self._answered_calls.pop(call_id, None)
                caller = CallerID(request.headers.get("From", ""))
                logger.info(
                    json.dumps(
                        {
                            "event": "call_ended",
                            "caller": repr(caller),
                            "ip": addr[0],
                            "call_id": call_id,
                        }
                    ),
                    extra={"caller": repr(caller), "ip": addr[0], "call_id": call_id},
                )
                self.send(
                    Response(
                        status_code=Status["OK"],
                        reason=Status["OK"].name,
                        headers=self._with_to_tag(
                            {
                                key: value
                                for key, value in request.headers.items()
                                if key in ("Via", "To", "From", "Call-ID", "CSeq")
                            },
                            call_id,
                        ),
                    ),
                    addr,
                )
                self._to_tags.pop(call_id, None)
                self._cleanup_rtp_call(call_id)
                self.bye_received(request)
            case "CANCEL":
                caller = CallerID(request.headers.get("From", ""))
                logger.info(
                    json.dumps(
                        {
                            "event": "call_cancelled",
                            "caller": repr(caller),
                            "ip": addr[0],
                            "call_id": call_id,
                        }
                    ),
                    extra={"caller": repr(caller), "ip": addr[0], "call_id": call_id},
                )
                self.send(
                    Response(
                        status_code=Status["OK"],
                        reason=Status["OK"].name,
                        headers={
                            key: value
                            for key, value in request.headers.items()
                            if key in ("Via", "To", "From", "Call-ID", "CSeq")
                        },
                    ),
                    addr,
                )
                invite_addr = self._request_addrs.pop(call_id, None)
                if invite_addr is not None:
                    self.send(
                        Response(
                            status_code=Status["Request Terminated"],
                            reason=Status["Request Terminated"].name,
                            headers=self._with_to_tag(
                                {
                                    key: value
                                    for key, value in request.headers.items()
                                    if key in ("Via", "To", "From", "Call-ID", "CSeq")
                                },
                                call_id,
                            ),
                        ),
                        invite_addr,
                    )
                self._answered_calls.pop(call_id, None)
                self._to_tags.pop(call_id, None)
                self._cleanup_rtp_call(call_id)
                self.cancel_received(request)
            case _:
                raise NotImplementedError(
                    f"Unsupported SIP request method: {request.method}"
                )

    def response_received(self, response: Response, addr: tuple[str, int]) -> None:
        """Handle REGISTER responses including digest auth challenges (RFC 3261 §22).

        Only processes responses when registration parameters are configured.
        """
        if response.status_code == Status["OK"] and "REGISTER" in response.headers.get(
            "CSeq", ""
        ):
            logger.info("Registration successful")
            self.registered()
            return
        if response.status_code in (
            Status["Unauthorized"],
            Status["Proxy Authentication Required"],
        ):
            if not self.username or not self.password:
                logger.error(
                    "Auth challenge received but username/password are not configured"
                )
                return
            logger.debug(
                "Auth challenge received (%s), retrying with credentials",
                response.status_code,
            )
            is_proxy = response.status_code == Status["Proxy Authentication Required"]
            challenge_key = "Proxy-Authenticate" if is_proxy else "WWW-Authenticate"
            params = self.parse_auth_challenge(response.headers.get(challenge_key, ""))
            realm = params.get("realm", "")
            nonce = params.get("nonce", "")
            opaque = params.get("opaque")
            qop_options = params.get("qop", "")
            qop = (
                DigestQoP.AUTH.value
                if DigestQoP.AUTH.value in qop_options.split(",")
                else None
            )
            nc = "00000001"
            cnonce = secrets.token_hex(8) if qop else None
            digest = self.digest_response(
                username=self.username,
                password=self.password,
                realm=realm,
                nonce=nonce,
                method="REGISTER",
                uri=self.registrar_uri,
                qop=qop,
                nc=nc,
                cnonce=cnonce,
            )
            auth_value = (
                f'Digest username="{self.username}", realm="{realm}", '
                f'nonce="{nonce}", uri="{self.registrar_uri}", '
                f'response="{digest}", algorithm="MD5"'
            )
            if qop:
                auth_value += f', qop={qop}, nc={nc}, cnonce="{cnonce}"'
            if opaque:
                auth_value += f', opaque="{opaque}"'
            if is_proxy:
                asyncio.create_task(self.register(proxy_authorization=auth_value))
            else:
                asyncio.create_task(self.register(authorization=auth_value))
            return
        logger.warning(
            "Unexpected REGISTER response: %s %s", response.status_code, response.reason
        )
        raise NotImplementedError("Unexpected REGISTER response")

    def call_received(self, request: Request) -> None:
        """Handle an incoming call.

        Override in subclasses to accept or reject the call::

            def call_received(self, request: Request) -> None:
                self.answer(request=request, call_class=MyCall)

        Args:
            request: The SIP INVITE request.
        """

    def ack_received(self, request: Request) -> None:
        """Handle an ACK confirming dialog establishment.

        Override in subclasses to react to the ACK.

        Args:
            request: The SIP ACK request.
        """

    def bye_received(self, request: Request) -> None:
        """Handle a BYE terminating a dialog.

        Override in subclasses to tear down the call.

        Args:
            request: The SIP BYE request.
        """

    def cancel_received(self, request: Request) -> None:
        """Handle a CANCEL request for a pending INVITE.

        Override in subclasses to react to caller cancellation before the call
        is answered.

        Args:
            request: The SIP CANCEL request.
        """

    async def answer(self, request: Request, *, call_class: type[Call]) -> None:
        """Answer an incoming call by setting up RTP and sending 200 OK with SDP.

        This coroutine can be awaited directly or wrapped in a task::

            # inside a sync call_received:
            asyncio.create_task(self.answer(request=request, call_class=MyCall))

            # inside an async call_received:
            await self.answer(request=request, call_class=MyCall)

        Args:
            request: The SIP INVITE request (from :meth:`call_received`).
            call_class: A :class:`~voip.call.Call` subclass whose
                :meth:`~voip.call.Call.negotiate_codec` selects the codec.
                The class is constructed with ``rtp``, ``sip``, ``caller``,
                and ``media`` keyword arguments.

        Raises:
            NotImplementedError: When :meth:`~voip.call.Call.negotiate_codec`
                raises (no supported codec in the remote SDP offer).
        """
        await self._answer(request, call_class)

    async def _answer(self, request: Request, call_class: type[Call]) -> None:
        """Perform the asynchronous part of answering: set up RTP, send 200 OK."""
        call_id = request.headers.get("Call-ID", "")
        addr = self._request_addrs.pop(call_id, None)
        if addr is None:
            logger.error("No address found for INVITE with Call-ID %r", call_id)
            return
        caller = CallerID(request.headers.get("From", ""))
        logger.info(
            json.dumps(
                {
                    "event": "call_answered",
                    "caller": repr(caller),
                    "ip": addr[0],
                    "call_id": call_id,
                }
            ),
            extra={"caller": repr(caller), "ip": addr[0], "call_id": call_id},
        )
        remote_audio = next(
            (
                m
                for m in (request.body.media if request.body else [])
                if m.media == "audio"
            ),
            None,
        )
        # Codec negotiation is delegated to the call class.  If the remote SDP
        # offers no supported codec, negotiate_codec raises NotImplementedError
        # and the exception propagates — the call is not answered.
        if remote_audio is not None:
            negotiated_media = call_class.negotiate_codec(remote_audio)
        else:
            negotiated_media = MediaDescription(
                media="audio",
                port=0,
                proto="RTP/AVP",
                fmt=[RTPPayloadFormat.from_pt(0)],
            )

        # Instantiate the per-call handler and register it with the shared mux.
        call_handler = call_class(
            rtp=self._rtp_protocol, sip=self, caller=caller, media=negotiated_media
        )
        # Determine the remote RTP address for routing.  If the INVITE SDP
        # specifies a connection address, use that; otherwise fall back to the
        # source IP of the SIP message.  When no remote audio port is known
        # (no SDP), register under the ``None`` wildcard key so that the mux
        # delivers all unmatched traffic to this handler.
        if remote_audio is not None:
            conn = request.body.connection if request.body else None
            remote_ip = conn.connection_address if conn else addr[0]
            remote_rtp_addr: tuple[str, int] | None = (remote_ip, remote_audio.port)
        else:
            remote_rtp_addr = None
        self._rtp_protocol.register_call(remote_rtp_addr, call_handler)
        self._call_rtp_addrs[call_id] = remote_rtp_addr

        local_rtp_addr = self._rtp_transport.get_extra_info("sockname")
        rtp_public = await self._rtp_protocol.public_address
        sdp_ip = rtp_public[0] if rtp_public else local_rtp_addr[0]
        sdp_port = rtp_public[1] if rtp_public else local_rtp_addr[1]
        contact_addr = await self.public_address
        logger.debug("RTP mux at %s:%s; contact %s:%s", sdp_ip, sdp_port, *contact_addr)
        record_route = request.headers.get("Record-Route")
        sess_id = str(secrets.randbelow(2**32) + 1)
        sdp_media_attributes = [
            Attribute(name="sendrecv"),
        ]
        self.send(
            Response(
                status_code=Status["OK"],
                reason=Status["OK"].name,
                headers={
                    **self._with_to_tag(
                        {
                            key: value
                            for key, value in request.headers.items()
                            if key in ("Via", "To", "From", "Call-ID", "CSeq")
                        },
                        call_id,
                    ),
                    **({"Record-Route": record_route} if record_route else {}),
                    "Contact": f"<sip:{contact_addr[0]}:{contact_addr[1]}>",
                    "Allow": self.ALLOW,
                    "Supported": "replaces",
                    "Content-Type": "application/sdp",
                },
                body=SessionDescription(
                    origin=Origin(
                        username="-",
                        sess_id=sess_id,
                        sess_version=sess_id,
                        nettype="IN",
                        addrtype="IP4",
                        unicast_address=sdp_ip,
                    ),
                    timings=[Timing(start_time=0, stop_time=0)],
                    connection=ConnectionData(
                        nettype="IN", addrtype="IP4", connection_address=sdp_ip
                    ),
                    media=[
                        MediaDescription(
                            media="audio",
                            port=sdp_port,
                            proto="RTP/AVP",
                            fmt=negotiated_media.fmt,
                            attributes=sdp_media_attributes,
                        )
                    ],
                ),
            ),
            addr,
        )
        self._mark_call_answered(call_id)
        self._to_tags.pop(call_id, None)

    def _with_to_tag(self, headers: dict[str, str], call_id: str) -> dict[str, str]:
        """Return headers with the To tag appended (RFC 3261 §8.2.6.2)."""
        tag = self._to_tags.get(call_id, "")
        return {
            **headers,
            "To": headers.get("To", "") + (f";tag={tag}" if tag else ""),
        }

    def ringing(self, request: Request) -> None:
        """Send a 180 Ringing provisional response to the caller.

        Call this from :meth:`call_received` before answering to indicate
        that the call is being processed (e.g. while a user is alerted).

        Args:
            request: The SIP INVITE request (from :meth:`call_received`).
        """
        call_id = request.headers.get("Call-ID", "")
        address = self._request_addrs.get(call_id)
        if address is None:
            logger.error("No address found for INVITE with Call-ID %r", call_id)
            return
        caller = CallerID(request.headers.get("From", ""))
        logger.info(
            json.dumps(
                {"event": "call_ringing", "caller": repr(caller), "call_id": call_id}
            ),
            extra={"caller": repr(caller), "call_id": call_id},
        )
        self.send(
            Response(
                status_code=Status["Ringing"],
                reason=Status["Ringing"].name,
                headers=self._with_to_tag(
                    {
                        key: value
                        for key, value in request.headers.items()
                        if key in ("Via", "To", "From", "Call-ID", "CSeq")
                    },
                    call_id,
                ),
            ),
            address,
        )

    def reject(
        self,
        request: Request,
        status_code: int = Status["Busy Here"],
        reason: str = Status["Busy Here"].name,
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
        caller = CallerID(request.headers.get("From", ""))
        logger.info(
            json.dumps(
                {
                    "event": "call_rejected",
                    "caller": repr(caller),
                    "ip": addr[0],
                    "call_id": call_id,
                    "status": status_code,
                    "reason": reason,
                }
            ),
            extra={
                "caller": repr(caller),
                "ip": addr[0],
                "call_id": call_id,
                "status": status_code,
            },
        )
        self.send(
            Response(
                status_code=status_code,
                reason=reason,
                headers=self._with_to_tag(
                    {
                        key: value
                        for key, value in request.headers.items()
                        if key in ("Via", "To", "From", "Call-ID", "CSeq")
                    },
                    call_id,
                ),
            ),
            addr,
        )
        self._to_tags.pop(call_id, None)

    @property
    def registrar_uri(self) -> str:
        """Registrar Request-URI derived from the AOR (e.g. sip:example.com)."""
        if not self.aor:
            raise ValueError("AOR is not configured; cannot derive registrar URI")
        scheme, _, rest = self.aor.partition(":")
        _, _, hostport = rest.partition("@")
        return f"{scheme}:{hostport}"

    async def register(
        self,
        authorization: str | None = None,
        proxy_authorization: str | None = None,
    ) -> None:
        """Send a REGISTER request to the carrier, optionally with credentials."""
        self.cseq += 1
        logger.debug(
            "Sending REGISTER to %s:%s (CSeq %s)",
            self.server_address[0],
            self.server_address[1],
            self.cseq,
        )
        # Prefer the publicly routable SIP address (from STUN) for Via and Contact.
        # Falls back to the local address when STUN is not configured.
        public_address = await self.public_address
        branch = f"{self.VIA_BRANCH_PREFIX}{secrets.token_hex(16)}"
        logger.debug("REGISTER Via branch: %s", branch)
        # Extract SIP user part from AOR (e.g. "sip:alice@example.com" -> "alice")
        aor_rest = self.aor.partition(":")[2] if self.aor else ""
        user = aor_rest.partition("@")[0] if "@" in aor_rest else aor_rest
        headers = {
            "Via": f"SIP/2.0/UDP {public_address[0]}:{public_address[1]};rport;branch={branch}",
            "From": self.aor,
            "To": self.aor,
            "Call-ID": self.call_id,
            "CSeq": f"{self.cseq} REGISTER",
            "Contact": f"<sip:{user}@{public_address[0]}:{public_address[1]}>",
            "Expires": "3600",  # 1 hour
            "Max-Forwards": "70",
        }
        if authorization is not None:
            headers["Authorization"] = authorization
        if proxy_authorization is not None:
            headers["Proxy-Authorization"] = proxy_authorization
        self.send(
            Request(method="REGISTER", uri=self.registrar_uri, headers=headers),
            self.server_address,
        )

    def registered(self) -> None:
        """Handle a confirmed carrier registration. Override to react."""

    @staticmethod
    def parse_auth_challenge(header: str) -> dict[str, str]:
        """Parse Digest challenge parameters from a WWW-Authenticate/Proxy-Authenticate header."""
        _, _, params_str = header.partition(" ")
        params = {}
        for part in re.split(r",\s*(?=[a-zA-Z])", params_str):
            key, _, value = part.partition("=")
            if key.strip():
                params[key.strip()] = value.strip().strip('"')
        return params

    @staticmethod
    def digest_response(
        *,
        username: str,
        password: str,
        realm: str,
        nonce: str,
        method: str,
        uri: str,
        qop: str | None = None,
        nc: str = "00000001",
        cnonce: str | None = None,
    ) -> str:
        """Compute an RFC 2617 / RFC 3261 §22 MD5 digest response."""
        ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode()).hexdigest()  # noqa: S324
        ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()  # noqa: S324
        if qop in (DigestQoP.AUTH, DigestQoP.AUTH_INT):
            return hashlib.md5(  # noqa: S324
                f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}".encode()
            ).hexdigest()
        return hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()  # noqa: S324

    def error_received(self, exc: OSError) -> None:
        """Handle a transport-level error, delegating to the STUN base class."""
        STUNProtocol.error_received(self, exc)

    def connection_lost(self, exc: Exception | None) -> None:
        """Handle a lost connection."""
        if exc is not None:
            logger.exception("Connection lost", exc_info=exc)
        self.transport = None
        STUNProtocol.connection_lost(self, exc)


#: Short alias for :class:`SessionInitiationProtocol`.
SIP = SessionInitiationProtocol
