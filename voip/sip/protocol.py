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
from voip.srtp import SRTPSession
from voip.types import DigestAlgorithm, DigestQoP

from .messages import Message, Request, Response
from .types import CallerID, Status

logger = logging.getLogger("voip.sip")

__all__ = ["RegistrationError", "SIP", "SessionInitiationProtocol"]


class RegistrationError(Exception):
    """Raised when a SIP REGISTER request fails with an unexpected response.

    The exception message includes the response status code and reason phrase
    from the server, e.g. ``"403 Forbidden"`` or ``"500 Server Error"``.
    """


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
class SessionInitiationProtocol(asyncio.Protocol):
    """
    SIP User Agent Client (UAC) over TLS/TCP [RFC 3261].

    Handles incoming calls and, optionally, carrier registration with digest
    authentication [RFC 3261 §22].  All signalling is sent over a single
    persistent TLS/TCP connection.

    RFC 3261 topology overview
    --------------------------
    *Outbound proxy* (§8.1.2): the SIP server this UA sends all requests to.
    It may be a carrier edge proxy whose address differs from the registrar.

    *Registrar* (§10): the server that maintains location bindings for a
    domain.  Its URI is derived automatically from the `aor` by
    stripping the user part (e.g. ``sips:alice@example.com`` →
    ``sips:example.com``).  When no `outbound_proxy` is configured,
    the UA is expected to connect directly to the registrar server.

    When an `outbound_proxy` is configured it acts as the first SIP
    hop and may differ from the registrar domain — for example when a carrier
    provides a dedicated proxy at ``proxy.carrier.com`` while the AOR domain
    (and thus the registrar Request-URI) is ``carrier.com``.

    Subclass and override `call_received` to handle incoming calls::

        class MySession(SessionInitiationProtocol):
            def call_received(self, request: Request) -> None:
                self.answer(request=request, call_class=MyCall)

    To register with a carrier on startup, pass the registration parameters::

        session = SessionInitiationProtocol(
            aor="sips:alice@example.com",
            username="alice",
            password="secret",
            # Optional: connect via a separate outbound proxy
            # outbound_proxy=("proxy.carrier.com", 5061),
        )

    [RFC 3261]: https://datatracker.ietf.org/doc/html/rfc3261
    [RFC 3261 §22]: https://datatracker.ietf.org/doc/html/rfc3261#section-22
    """

    #: RFC 3261 §8.1.1.7 Via branch magic cookie (indicates RFC 3261 compliance).
    VIA_BRANCH_PREFIX: typing.ClassVar[str] = "z9hG4bK"

    #: RFC 3261 §11 – methods supported by this UA (used in Allow header).
    ALLOW: typing.ClassVar[str] = "INVITE, ACK, BYE, CANCEL, OPTIONS"

    _pending_invites: set[str] = dataclasses.field(init=False, default_factory=set)
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
    _initialize_task: asyncio.Task | None = dataclasses.field(init=False, default=None)
    _call_rtp_addrs: dict[str, tuple[str, int] | None] = dataclasses.field(
        init=False, default_factory=dict
    )
    _buffer: bytearray = dataclasses.field(init=False, default_factory=bytearray)
    #: RFC 3261 §8.1.2 — outbound SIP proxy address ``(host, port)``.
    #: When ``None`` the caller connects directly to the registrar server.
    #: The address may differ from the registrar domain derived from
    #: `aor` (e.g. ``proxy.carrier.com`` vs ``carrier.com``).
    outbound_proxy: tuple[str, int] | None = None
    aor: str
    username: str | None = None
    password: str | None = None
    #: STUN server used for RTP NAT traversal (SIP uses TLS/TCP; no STUN needed).
    rtp_stun_server_address: tuple[str, int] | None = ("stun.cloudflare.com", 3478)
    call_id: str = dataclasses.field(init=False)
    cseq: int = dataclasses.field(init=False, default=0)
    #: Local TCP socket address (host, port) — set when connection is established.
    local_address: tuple[str, int] = dataclasses.field(init=False)
    transport: asyncio.Transport | None = dataclasses.field(init=False, default=None)
    #: True when the underlying transport is TLS-wrapped; False for plain TCP.
    _is_tls: bool = dataclasses.field(init=False, default=False)

    def __post_init__(self):
        self.call_id = f"{uuid.uuid4()}@{socket.gethostname()}"

    def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
        """Store the TLS/TCP transport and start RTP mux + carrier registration."""
        self.transport = transport
        self.local_address = transport.get_extra_info("sockname")
        self._is_tls = transport.get_extra_info("ssl_object") is not None
        try:
            self._initialize_task = asyncio.get_running_loop().create_task(
                self._initialize()
            )
        except RuntimeError:
            pass  # no running loop in synchronous test setups

    async def _initialize(self) -> None:
        """Set up the RTP mux and register with the carrier (in that order).

        Creates a dedicated UDP socket for RTP (with optional STUN discovery
        for NAT traversal) before sending REGISTER so the SDP answer can
        advertise the correct public RTP address.
        """
        loop = asyncio.get_running_loop()
        self._rtp_transport, self._rtp_protocol = await loop.create_datagram_endpoint(
            lambda: RealtimeTransportProtocol(
                stun_server_address=self.rtp_stun_server_address
            ),
            local_addr=("0.0.0.0", 0),  # noqa: S104
        )
        await self.register()

    def data_received(self, data: bytes) -> None:
        """Buffer incoming bytes and dispatch complete SIP messages.

        SIP over TCP uses the ``Content-Length`` header to frame messages
        (RFC 3261 §18.3).  Partial datagrams are accumulated until a full
        message is available.
        """
        self._buffer.extend(data)
        while True:
            end_of_headers = self._buffer.find(b"\r\n\r\n")
            if end_of_headers == -1:
                break
            header_bytes = bytes(self._buffer[:end_of_headers])
            # Determine body length from Content-Length header.
            content_length = 0
            for line in header_bytes.decode(errors="replace").split("\r\n")[1:]:
                low = line.lower()
                if low.startswith("content-length:"):
                    try:
                        content_length = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                    break
            message_end = end_of_headers + 4 + content_length
            if len(self._buffer) < message_end:
                break
            message_data = bytes(self._buffer[:message_end])
            del self._buffer[:message_end]
            addr = self.transport.get_extra_info("peername") if self.transport else None
            self.packet_received(message_data, addr)

    def packet_received(self, data: bytes, addr: tuple[str, int] | None) -> None:
        """Handle RFC 5626 keepalive pings, then dispatch SIP messages."""
        if data == b"\r\n\r\n":  # RFC 5626 §4.4.1 double-CRLF keepalive ping
            logger.debug("RFC 5626 keepalive from %s, sending pong", addr)
            if self.transport:
                self.transport.write(b"\r\n")
            return
        match Message.parse(data):
            case Request() as request:
                self.request_received(request, addr)
            case Response() as response:
                self.response_received(response, addr)

    def send(self, message: Response | Request) -> None:
        """Serialize and send a SIP message over the TLS/TCP connection."""
        logger.debug("Sending %r", message)
        if self.transport is not None:
            self.transport.write(bytes(message))

    def close(self) -> None:
        """Close the TLS/TCP transport and the RTP mux."""
        if self.transport is not None:
            self.transport.close()
        if self._rtp_transport is not None:
            self._rtp_transport.close()

    def _cleanup_rtp_call(self, call_id: str) -> None:
        """Remove the call handler registered with the shared RTP mux, if any."""
        if call_id in self._call_rtp_addrs and self._rtp_protocol is not None:
            self._rtp_protocol.unregister_call(self._call_rtp_addrs.pop(call_id))

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
        peer_ip = addr[0] if addr else None
        match request.method:
            case "INVITE":
                caller = CallerID(request.headers.get("From", ""))
                logger.info(
                    json.dumps(
                        {
                            "event": "incoming_call",
                            "caller": repr(caller),
                            "ip": peer_ip,
                            "call_id": call_id,
                        }
                    ),
                    extra={"caller": repr(caller), "ip": peer_ip, "call_id": call_id},
                )
                if call_id in self._answered_calls:
                    logger.debug(
                        "Ignoring INVITE retransmission for Call-ID %r", call_id
                    )
                    return
                # Mark immediately (before async answering) so retransmissions
                # that arrive while RTP setup is in progress are suppressed.
                self._mark_call_answered(call_id)
                self._pending_invites.add(call_id)
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
                            "ip": peer_ip,
                            "call_id": call_id,
                        }
                    ),
                    extra={"caller": repr(caller), "ip": peer_ip, "call_id": call_id},
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
                            "ip": peer_ip,
                            "call_id": call_id,
                        }
                    ),
                    extra={"caller": repr(caller), "ip": peer_ip, "call_id": call_id},
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
                )
                if call_id in self._pending_invites:
                    self._pending_invites.discard(call_id)
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
                    )
                self._answered_calls.pop(call_id, None)
                self._to_tags.pop(call_id, None)
                self._cleanup_rtp_call(call_id)
                self.cancel_received(request)
            case _:
                raise NotImplementedError(
                    f"Unsupported SIP request method: {request.method}"
                )

    def response_received(
        self, response: Response, addr: tuple[str, int] | None
    ) -> None:
        """Handle REGISTER responses including digest auth challenges (RFC 3261 §22).

        Only processes responses when registration parameters are configured.
        """
        if response.status_code == Status["OK"] and response.headers.get(
            "CSeq", ""
        ).split()[-1:] == ["REGISTER"]:
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
            algorithm = params.get("algorithm", DigestAlgorithm.SHA_256)
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
                algorithm=algorithm,
                qop=qop,
                nc=nc,
                cnonce=cnonce,
            )
            auth_value = (
                f'Digest username="{self.username}", realm="{realm}", '
                f'nonce="{nonce}", uri="{self.registrar_uri}", '
                f'response="{digest}", algorithm="{algorithm}"'
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
        raise RegistrationError(f"{response.status_code} {response.reason}")

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
            request: The SIP INVITE request (from `call_received`).
            call_class: A `Call` subclass whose `negotiate_codec` selects the codec.
                The class is constructed with ``rtp``, ``sip``, ``caller``,
                and ``media`` keyword arguments.

        Raises:
            NotImplementedError: When `negotiate_codec` raises (no supported codec in the remote SDP offer).
        """
        await self._answer(request, call_class)

    async def _answer(self, request: Request, call_class: type[Call]) -> None:
        """Perform the asynchronous part of answering: set up RTP, send 200 OK."""
        call_id = request.headers.get("Call-ID", "")
        if call_id not in self._pending_invites:
            logger.error("No pending INVITE found for Call-ID %r", call_id)
            return
        self._pending_invites.discard(call_id)
        # Ensure the RTP mux has been created before answering.  Under normal
        # operation _initialize() completes before any INVITE arrives, but an
        # early INVITE must wait for the mux.  Skip if already available.
        if self._rtp_protocol is None:
            if self._initialize_task is not None:
                await self._initialize_task
            if self._rtp_protocol is None:
                logger.error("RTP mux not ready; cannot answer call")
                return
        peer = self.transport.get_extra_info("peername") if self.transport else None
        caller = CallerID(request.headers.get("From", ""))
        logger.info(
            json.dumps(
                {
                    "event": "call_answered",
                    "caller": repr(caller),
                    "ip": peer[0] if peer else None,
                    "call_id": call_id,
                }
            ),
            extra={
                "caller": repr(caller),
                "ip": peer[0] if peer else None,
                "call_id": call_id,
            },
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
                proto="RTP/SAVP",
                fmt=[RTPPayloadFormat.from_pt(0)],
            )

        # Generate a fresh SRTP session only when the negotiated transport is SRTP.
        use_srtp = negotiated_media.proto == "RTP/SAVP"
        srtp_session = SRTPSession.generate() if use_srtp else None

        # Instantiate the per-call handler and register it with the shared mux.
        call_handler = call_class(
            rtp=self._rtp_protocol,
            sip=self,
            caller=caller,
            media=negotiated_media,
            srtp=srtp_session,
        )
        # Determine the remote RTP address for routing.
        # Per RFC 4566 §5.7 the effective connection address is taken from the
        # media-level c= line first, then the session-level c= line, then the
        # SIP peer IP as last resort.
        #
        # When the media port is 0, the stream is inactive (RFC 4566 §5.14);
        # registering an address and hole-punching are skipped, and we fall
        # through to ``remote_rtp_addr = None`` so the mux wildcard is used
        # (if any traffic arrives at all).
        #
        # When no SDP was present in the INVITE we also use the wildcard so
        # the mux delivers all unmatched traffic to this handler.
        if remote_audio is not None and remote_audio.port != 0:
            media_conn = remote_audio.connection
            session_conn = request.body.connection if request.body else None
            conn = media_conn or session_conn
            if conn is not None:
                remote_ip = conn.connection_address
            else:
                remote_ip = peer[0] if peer else "0.0.0.0"  # noqa: S104
            remote_rtp_addr: tuple[str, int] | None = (remote_ip, remote_audio.port)
        else:
            remote_rtp_addr = None
        self._rtp_protocol.register_call(remote_rtp_addr, call_handler)
        self._call_rtp_addrs[call_id] = remote_rtp_addr

        # NAT hole-punch: send a dummy datagram to the carrier's RTP address so
        # that our router creates a return-path mapping allowing the carrier's
        # media packets to reach our UDP socket (RFC 4787 / address-restricted NAT).
        if remote_rtp_addr is not None:
            self._rtp_protocol.send(b"\x00", remote_rtp_addr)

        record_route = request.headers.get("Record-Route")
        sess_id = str(secrets.randbelow(2**32) + 1)
        rtp_public = await self._rtp_protocol.public_address
        sdp_media_attributes = [Attribute(name="sendrecv")]
        if srtp_session is not None:
            sdp_media_attributes.append(
                Attribute(name="crypto", value=srtp_session.sdes_attribute)
            )
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
                    "Contact": self._build_contact(),
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
                        unicast_address=rtp_public[0],
                    ),
                    timings=[Timing(start_time=0, stop_time=0)],
                    connection=ConnectionData(
                        nettype="IN",
                        addrtype="IP4",
                        connection_address=rtp_public[0],
                    ),
                    media=[
                        MediaDescription(
                            media="audio",
                            port=rtp_public[1],
                            proto=negotiated_media.proto,
                            fmt=negotiated_media.fmt,
                            attributes=sdp_media_attributes,
                        )
                    ],
                ),
            ),
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

    def _build_contact(self, user: str | None = None) -> str:
        """Return a ``Contact:`` header value for this UA.

        The URI scheme mirrors `aor`: a ``sips:`` AOR produces a
        ``sips:`` Contact (the strongest TLS guarantee); a ``sip:`` AOR over
        TLS produces ``sip:`` with ``transport=tls``; plain TCP produces plain
        ``sip:``.

        Args:
            user: SIP user part (e.g. ``"alice"``).  When provided the Contact
                is of the form ``<scheme:user@host:port>``; otherwise just
                ``<scheme:host:port>``.
        """
        aor_scheme = self.aor.partition(":")[0]  # "sip" or "sips"
        host_port = f"{self.local_address[0]}:{self.local_address[1]}"
        addr = f"{user}@{host_port}" if user else host_port
        if aor_scheme == "sips":
            return f"<sips:{addr}>"
        tls_param = ";transport=tls" if self._is_tls else ""
        return f"<sip:{addr}{tls_param}>"

    def ringing(self, request: Request) -> None:
        """Send a 180 Ringing provisional response to the caller.

        Call this from `call_received` before answering to indicate
        that the call is being processed (e.g. while a user is alerted).

        Args:
            request: The SIP INVITE request (from `call_received`).
        """
        call_id = request.headers.get("Call-ID", "")
        if call_id not in self._pending_invites:
            logger.error("No pending INVITE found for Call-ID %r", call_id)
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
        )

    def reject(
        self,
        request: Request,
        status_code: int = Status["Busy Here"],
        reason: str = Status["Busy Here"].name,
    ) -> None:
        """Reject an incoming call.

        Args:
            request: The SIP INVITE request (from `call_received`).
            status_code: SIP response status code (default: 486 Busy Here).
            reason: SIP response reason phrase.
        """
        call_id = request.headers.get("Call-ID", "")
        if call_id not in self._pending_invites:
            logger.error("No pending INVITE found for Call-ID %r", call_id)
            return
        self._pending_invites.discard(call_id)
        peer = self.transport.get_extra_info("peername") if self.transport else None
        caller = CallerID(request.headers.get("From", ""))
        logger.info(
            json.dumps(
                {
                    "event": "call_rejected",
                    "caller": repr(caller),
                    "ip": peer[0] if peer else None,
                    "call_id": call_id,
                    "status": status_code,
                    "reason": reason,
                }
            ),
            extra={
                "caller": repr(caller),
                "ip": peer[0] if peer else None,
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
        )
        self._to_tags.pop(call_id, None)

    @property
    def registrar_uri(self) -> str:
        """Registrar Request-URI derived from the AOR, preserving its scheme.

        The scheme (``sip:`` or ``sips:``) is taken directly from `aor`
        so the client honours whatever security contract the administrator has
        configured.  The user part is stripped; only the host (and optional
        port) is kept, per RFC 3261 §10.2.

        Examples::

            sip:alice@example.com   →  sip:example.com
            sips:alice@example.com  →  sips:example.com
        """
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
        """Send a REGISTER request to the registrar, optionally with credentials.

        The REGISTER Request-URI is the registrar URI derived from `aor`
        (RFC 3261 §10.2).  When an `outbound_proxy` is configured, the
        request is sent over the existing TLS/TCP connection to that proxy,
        which routes it to the registrar on our behalf.
        """
        self.cseq += 1
        if self.outbound_proxy:
            logger.debug(
                "Sending REGISTER via outbound proxy %s:%s to registrar %s (CSeq %s)",
                self.outbound_proxy[0],
                self.outbound_proxy[1],
                self.registrar_uri,
                self.cseq,
            )
        else:
            logger.debug(
                "Sending REGISTER to registrar %s (CSeq %s)",
                self.registrar_uri,
                self.cseq,
            )
        branch = f"{self.VIA_BRANCH_PREFIX}{secrets.token_hex(16)}"
        logger.debug("REGISTER Via branch: %s", branch)
        # Extract SIP user part from AOR (e.g. "sips:alice@example.com" -> "alice")
        aor_rest = self.aor.partition(":")[2] if self.aor else ""
        user = aor_rest.partition("@")[0] if "@" in aor_rest else aor_rest
        headers = {
            "Via": f"SIP/2.0/{'TLS' if self._is_tls else 'TCP'} {self.local_address[0]}:{self.local_address[1]};rport;branch={branch}",
            "From": self.aor,
            "To": self.aor,
            "Call-ID": self.call_id,
            "CSeq": f"{self.cseq} REGISTER",
            "Contact": self._build_contact(user),
            "Expires": "3600",  # 1 hour
            "Max-Forwards": "70",
        }
        if authorization is not None:
            headers["Authorization"] = authorization
        if proxy_authorization is not None:
            headers["Proxy-Authorization"] = proxy_authorization
        self.send(
            Request(method="REGISTER", uri=self.registrar_uri, headers=headers),
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

    #: Map from `DigestAlgorithm` to the hashlib name.
    _DIGEST_HASH_NAME: typing.ClassVar[dict[str, str]] = {
        DigestAlgorithm.MD5: "md5",
        DigestAlgorithm.MD5_SESS: "md5",
        DigestAlgorithm.SHA_256: "sha256",
        DigestAlgorithm.SHA_256_SESS: "sha256",
        DigestAlgorithm.SHA_512_256: "sha512_256",
        DigestAlgorithm.SHA_512_256_SESS: "sha512_256",
    }

    @classmethod
    def digest_response(
        cls,
        *,
        username: str,
        password: str,
        realm: str,
        nonce: str,
        method: str,
        uri: str,
        algorithm: str = DigestAlgorithm.SHA_256,
        qop: str | None = None,
        nc: str = "00000001",
        cnonce: str | None = None,
    ) -> str:
        """Compute a SIP digest response per RFC 3261 §22 and RFC 8760.

        RFC 8760 deprecates MD5 and mandates support for SHA-256 and
        SHA-512-256.  The ``algorithm`` parameter selects the hash function;
        it defaults to ``SHA-256``.

        Raises:
            ValueError: If ``algorithm`` is not a recognised `DigestAlgorithm`,
                or if a ``*-sess`` algorithm is requested without a ``cnonce``.
        """
        try:
            hash_name = cls._DIGEST_HASH_NAME[algorithm]
        except KeyError:
            raise ValueError(f"Unsupported digest algorithm: {algorithm!r}") from None
        is_sess = algorithm.endswith("-sess")
        if is_sess and cnonce is None:
            raise ValueError(f"algorithm={algorithm!r} requires a cnonce value")

        def h(data: str) -> str:
            return hashlib.new(hash_name, data.encode()).hexdigest()

        ha1 = h(f"{username}:{realm}:{password}")
        if is_sess:
            ha1 = h(f"{ha1}:{nonce}:{cnonce}")
        ha2 = h(f"{method}:{uri}")
        if qop in (DigestQoP.AUTH, DigestQoP.AUTH_INT):
            return h(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}")
        return h(f"{ha1}:{nonce}:{ha2}")

    def connection_lost(self, exc: Exception | None) -> None:
        """Handle a lost TLS/TCP connection."""
        if exc is not None:
            logger.exception("Connection lost", exc_info=exc)
        self.transport = None


#: Short alias for `SessionInitiationProtocol`.
SIP = SessionInitiationProtocol
