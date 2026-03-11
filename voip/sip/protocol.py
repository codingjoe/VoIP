"""
SIP asyncio protocol handler.

See also: https://datatracker.ietf.org/doc/html/rfc3261
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import logging
import re
import secrets
import uuid
from typing import TYPE_CHECKING

from voip.sdp.messages import SessionDescription
from voip.sdp.types import Attribute, ConnectionData, MediaDescription, Origin, Timing
from voip.stun import stun_discover
from voip.types import DigestQoP

from .messages import Message, Request, Response
from .types import Status

if TYPE_CHECKING:
    from voip.rtp import RealtimeTransportProtocol

logger = logging.getLogger(__name__)

__all__ = ["SIP", "SessionInitiationProtocol"]


class SessionInitiationProtocol(asyncio.DatagramProtocol):
    """SIP session handler (RFC 3261).

    Handles incoming calls and, optionally, carrier registration with digest
    auth (RFC 3261 §22) and STUN-based NAT traversal (RFC 5389).

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
    VIA_BRANCH_PREFIX = "z9hG4bK"

    #: RFC 3261 §11 – methods supported by this UA (used in Allow header).
    ALLOW = "INVITE, ACK, BYE, CANCEL, OPTIONS"

    def __init__(
        self,
        server_address: tuple[str, int] | None = None,
        aor: str | None = None,
        username: str | None = None,
        password: str | None = None,
        stun_server_address: tuple[str, int] | None = None,
    ) -> None:
        super().__init__()
        #: Pending INVITE addresses keyed by Call-ID.
        self._request_addrs: dict[str, tuple[str, int]] = {}
        #: Call-IDs for which a 200 OK has already been sent.
        self._answered_calls: set[str] = set()
        #: To tags keyed by Call-ID (RFC 3261 §8.2.6.2).
        self._to_tags: dict[str, str] = {}
        self.server_address = server_address
        self.aor = aor
        self.username = username
        self.password = password
        self.call_id = str(uuid.uuid4())
        self.cseq = 0
        self.stun_server_address = stun_server_address
        self.public_address: tuple[str, int] | None = None

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        """Store the transport; if registration params are set, send REGISTER."""
        logger.debug("SIP transport connected")
        self._transport = transport
        if self.server_address is not None:
            if self.stun_server_address:
                asyncio.ensure_future(self._stun_then_register())
            else:
                self.register()

    async def _stun_then_register(self) -> None:
        """Discover the public address via STUN, then send REGISTER."""
        try:
            self.public_address = await stun_discover(*self.stun_server_address)
            logger.info(
                "STUN: public address is %s:%s",
                self.public_address[0],
                self.public_address[1],
            )
        except (TimeoutError, OSError, RuntimeError) as exc:
            logger.warning(
                "STUN discovery failed (%s), continuing with local address", exc
            )
        self.register()

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
        call_id = request.headers.get("Call-ID", "")
        match request.method:
            case "INVITE":
                logger.info("INVITE received from %s", addr[0])
                if call_id in self._answered_calls:
                    logger.debug(
                        "Ignoring INVITE retransmission for Call-ID %r", call_id
                    )
                    return
                # Mark immediately (before async answering) so retransmissions
                # that arrive while STUN/RTP setup is in progress are suppressed.
                self._answered_calls.add(call_id)
                self._request_addrs[call_id] = addr
                self._to_tags[call_id] = secrets.token_hex(8)
                self.call_received(request)
            case "ACK":
                self._answered_calls.discard(call_id)
                self.ack_received(request)
            case "BYE":
                self._answered_calls.discard(call_id)
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
                self.bye_received(request)
            case "CANCEL":
                logger.info("CANCEL received for Call-ID %r", call_id)
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
                self._answered_calls.discard(call_id)
                self._to_tags.pop(call_id, None)
                self.cancel_received(request)
            case _:
                raise NotImplementedError(
                    f"Unsupported SIP request method: {request.method}"
                )

    def response_received(self, response: Response, addr: tuple[str, int]) -> None:
        """Handle REGISTER responses including digest auth challenges (RFC 3261 §22).

        Only processes responses when registration parameters are configured.
        """
        if self.server_address is None:
            return  # no registration active; ignore unsolicited responses
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
                self.register(proxy_authorization=auth_value)
            else:
                self.register(authorization=auth_value)
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

    def answer(
        self, request: Request, *, call_class: type[RealtimeTransportProtocol]
    ) -> None:
        """Answer an incoming call by setting up RTP and sending 200 OK with SDP.

        Schedules the asynchronous RTP setup and SIP response without blocking.

        Args:
            request: The SIP INVITE request (from :meth:`call_received`).
            call_class: The :class:`~voip.rtp.RealtimeTransportProtocol` subclass
                to instantiate for the call.
        """
        asyncio.get_running_loop().create_task(self._answer(request, call_class))

    async def _answer(
        self, request: Request, call_class: type[RealtimeTransportProtocol]
    ) -> None:
        """Perform the asynchronous part of answering: set up RTP, send 200 OK."""
        call_id = request.headers.get("Call-ID", "")
        addr = self._request_addrs.pop(call_id, None)
        if addr is None:
            logger.error("No address found for INVITE with Call-ID %r", call_id)
            return
        caller = request.headers.get("From", "")
        logger.info("Answering call from %s", caller)
        loop = asyncio.get_running_loop()
        remote_audio = next(
            (
                m
                for m in (request.body.media if request.body else [])
                if m.media == "audio"
            ),
            None,
        )
        # Delegate codec negotiation to the call class so that each subclass
        # can declare its own codec preferences independently of the SIP layer.
        selected: tuple[str, str | None, int] | None = None
        if remote_audio:
            selected = call_class.negotiate_codec(remote_audio)
        selected_fmt = selected[0] if selected else None
        selected_rtpmap = selected[1] if selected else None
        selected_pt = int(selected[0]) if selected else 0
        selected_sample_rate = selected[2] if selected else 8000
        rtp_transport, rtp_protocol = await loop.create_datagram_endpoint(
            lambda: call_class(
                caller=caller,
                payload_type=selected_pt,
                sample_rate=selected_sample_rate,
            ),
            local_addr=("0.0.0.0", 0),  # noqa: S104
        )
        local_addr = rtp_transport.get_extra_info("sockname")
        # Discover the public RTP IP via a dedicated STUN socket when a STUN
        # server is configured. The RTP port comes from the locally bound socket.
        sdp_ip = local_addr[0]
        sdp_port = local_addr[1]
        if self.stun_server_address:
            try:
                public_ip, _ = await stun_discover(*self.stun_server_address)
                sdp_ip = public_ip
                logger.info("RTP STUN: public IP is %s, port %s", sdp_ip, sdp_port)
            except (TimeoutError, OSError, RuntimeError) as exc:
                logger.warning(
                    "RTP STUN discovery failed (%s), using local address", exc
                )
        elif self.public_address:
            sdp_ip = self.public_address[0]
        logger.debug("RTP listening on %s:%s", local_addr[0], local_addr[1])
        sip_local_addr = self._transport.get_extra_info("sockname") or ("0.0.0.0", 5060)  # noqa: S104
        sip_contact_addr = self.public_address or sip_local_addr
        record_route = request.headers.get("Record-Route")
        sess_id = str(secrets.randbelow(2**32) + 1)
        attributes = [
            Attribute(name="sendrecv"),
            *(
                [Attribute(name="rtpmap", value=selected_rtpmap)]
                if selected_rtpmap
                else []
            ),
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
                    "Contact": f"<sip:{sip_contact_addr[0]}:{sip_contact_addr[1]}>",
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
                            fmt=[selected_fmt] if selected_fmt else ["0"],
                            attributes=attributes,
                        )
                    ],
                ),
            ),
            addr,
        )
        self._answered_calls.add(call_id)
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
        logger.info(
            "Sending Ringing to %s",
            request.headers.get("From", "unknown"),
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

    def register(
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
        local_address = self._transport.get_extra_info("sockname") or ("0.0.0.0", 5060)  # noqa: S104
        branch = f"{self.VIA_BRANCH_PREFIX}{secrets.token_hex(16)}"
        logger.debug("REGISTER Via branch: %s", branch)
        # Extract SIP user part from AOR (e.g. "sip:alice@example.com" -> "alice")
        aor_rest = self.aor.partition(":")[2] if self.aor else ""
        user = aor_rest.partition("@")[0] if "@" in aor_rest else aor_rest
        # Use the public (STUN-discovered) address in Contact for inbound routing
        contact_address = self.public_address or local_address
        headers = {
            "Via": f"SIP/2.0/UDP {local_address[0]}:{local_address[1]};rport;branch={branch}",
            "From": self.aor,
            "To": self.aor,
            "Call-ID": self.call_id,
            "CSeq": f"{self.cseq} REGISTER",
            "Contact": f"<sip:{user}@{contact_address[0]}:{contact_address[1]}>",
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
        """Handle a transport-level error."""
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            logger.exception("Blocking IO error", exc_info=exc)
        else:
            raise exc

    def connection_lost(self, exc: Exception | None) -> None:
        """Handle a lost connection."""
        if exc is not None:
            logger.exception("Connection lost", exc_info=exc)
        self._transport = None


#: Short alias for :class:`SessionInitiationProtocol`.
SIP = SessionInitiationProtocol
