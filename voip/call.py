from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import secrets
import uuid
from collections.abc import Callable

from voip.rtp import RTPProtocol
from voip.sip.messages import Request, Response
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.types import SIPStatus, SIPStatusCode
from voip.stun import STUNProtocol
from voip.types import DigestQoP

logger = logging.getLogger(__name__)


class IncomingCall(RTPProtocol):
    """An inbound SIP call: answers or rejects the INVITE and receives Opus audio via RTP."""

    def __init__(
        self,
        request: Request,
        addr: tuple[str, int],
        send: Callable[..., None],
        *,
        contact_ip: str | None = None,
    ) -> None:
        self._request = request
        self._addr = addr
        self._send = send
        self._contact_ip = contact_ip
        logger.debug(
            "Incoming call from %s via %s", request.headers.get("From", "unknown"), addr
        )

    @property
    def caller(self) -> str:
        """Return the caller's SIP address."""
        return self._request.headers.get("From", "")

    async def answer(self) -> None:
        """Answer the call and start receiving Opus audio via RTP (RFC 7587)."""
        logger.info("Answering call from %s", self.caller)
        loop = asyncio.get_running_loop()
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: self,
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
        self._send(
            Response(
                status_code=SIPStatus.OK.status_code,
                reason=SIPStatus.OK.reason,
                headers={
                    **{
                        key: value
                        for key, value in self._request.headers.items()
                        if key in ("Via", "To", "From", "Call-ID", "CSeq")
                    },
                    "Content-Type": "application/sdp",
                },
                body=sdp,
            ),
            self._addr,
        )

    def reject(
        self,
        status_code: int = SIPStatus.BUSY_HERE.status_code,
        reason: str = SIPStatus.BUSY_HERE.reason,
    ) -> None:
        """Reject the call."""
        logger.info(
            "Rejecting call from %s with %s %s", self.caller, status_code, reason
        )
        self._send(
            Response(
                status_code=status_code,
                reason=reason,
                headers={
                    key: value
                    for key, value in self._request.headers.items()
                    if key in ("Via", "To", "From", "Call-ID", "CSeq")
                },
            ),
            self._addr,
        )


class IncomingCallProtocol(SessionInitiationProtocol):
    """SIP protocol with incoming call (INVITE) support."""

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
        super().datagram_received(data, addr)

    def send(self, message: Response | Request, addr: tuple[str, int]) -> None:
        """Serialize and send a SIP message to the given address."""
        logger.debug("Sending %r to %r", message, addr)
        self._transport.sendto(bytes(message), addr)

    def request_received(self, request: Request, addr: tuple[str, int]) -> None:
        """Dispatch an INVITE request to invite_received."""
        match request.method:
            case "INVITE":
                logger.info("INVITE received from %s", addr[0])
                self.invite_received(self.create_call(request, addr), addr)
            case _:
                raise NotImplementedError(
                    f"Unsupported SIP request method: {request.method}"
                )

    def create_call(self, request: Request, addr: tuple[str, int]) -> IncomingCall:
        """Create an IncomingCall for an INVITE. Override to use a custom call class."""
        return IncomingCall(request, addr, self.send, contact_ip=self._contact_ip)

    @property
    def _contact_ip(self) -> str | None:
        """Return the IP address to advertise in SDP (None if not available)."""
        return None

    def invite_received(self, call: IncomingCall, addr: tuple[str, int]) -> None:
        """Handle an incoming call. Override in subclasses to process calls."""


class RegisterProtocol(STUNProtocol, IncomingCallProtocol):
    """SIP UAC: registers with a carrier via digest auth and handles inbound calls."""

    #: RFC 3261 §8.1.1.7 Via branch magic cookie (indicates RFC 3261 compliance).
    VIA_BRANCH_PREFIX = "z9hG4bK"

    def __init__(
        self,
        server_address: tuple[str, int],
        aor: str,
        username: str,
        password: str,
        stun_server_address: tuple[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.server_address = server_address
        self.aor = aor
        self.username = username
        self.password = password
        self.call_id = str(uuid.uuid4())
        self.cseq = 0
        self.stun_server_address = stun_server_address
        self.public_address: tuple[str, int] | None = None

    @property
    def _contact_ip(self) -> str | None:
        """Return the STUN-discovered public IP for use in SDP, or None if not available."""
        return self.public_address[0] if self.public_address else None

    @property
    def registrar_uri(self) -> str:
        """Registrar Request-URI derived from the AOR (e.g. sip:example.com)."""
        scheme, _, rest = self.aor.partition(":")
        _, _, hostport = rest.partition("@")
        return f"{scheme}:{hostport}"

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

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        """Store the transport; discover public address via STUN then send REGISTER."""
        super().connection_made(transport)
        if self.stun_server_address:
            asyncio.ensure_future(self._connect())
        else:
            self.register()

    async def _connect(self) -> None:
        """Discover the public address via STUN, then send REGISTER."""
        try:
            self.public_address = await self.stun_discover(*self.stun_server_address)
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
        """Multiplex STUN and SIP messages on the same UDP socket (RFC 7983)."""
        if data and data[0] < 4:  # STUN: first byte is 0–3
            self.handle_stun(data, addr)
        else:
            super().datagram_received(data, addr)

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
        aor_rest = self.aor.partition(":")[2]
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

    def response_received(self, response: Response, addr: tuple[str, int]) -> None:
        """Handle REGISTER responses including digest auth challenges (RFC 3261 §22)."""
        if (
            response.status_code == SIPStatusCode.OK
            and "REGISTER" in response.headers.get("CSeq", "")
        ):
            logger.info("Registration successful")
            self.registered()
            return
        if response.status_code in (
            SIPStatusCode.UNAUTHORIZED,
            SIPStatusCode.PROXY_AUTHENTICATION_REQUIRED,
        ):
            logger.debug(
                "Auth challenge received (%s), retrying with credentials",
                response.status_code,
            )
            is_proxy = (
                response.status_code == SIPStatusCode.PROXY_AUTHENTICATION_REQUIRED
            )
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

    def registered(self) -> None:
        """Handle a confirmed carrier registration. Override to react."""
