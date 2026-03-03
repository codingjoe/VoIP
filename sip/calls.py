"""SIP call handling."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import uuid
from collections.abc import Callable

from .aio import SessionInitiationProtocol
from .messages import Message, Request, Response

__all__ = ["IncomingCall", "IncomingCallProtocol", "RegisterProtocol", "RTPProtocol"]

logger = logging.getLogger(__name__)


def _parse_auth_challenge(header: str) -> dict[str, str]:
    """Parse Digest challenge parameters from a WWW-Authenticate/Proxy-Authenticate header."""
    _, _, params_str = header.partition(" ")
    params = {}
    for part in re.split(r",\s*(?=[a-zA-Z])", params_str):
        key, _, value = part.partition("=")
        if key.strip():
            params[key.strip()] = value.strip().strip('"')
    return params


def _digest_response(
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
    if qop in ("auth", "auth-int"):
        return hashlib.md5(  # noqa: S324
            f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}".encode()
        ).hexdigest()
    return hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()  # noqa: S324


class RTPProtocol(asyncio.DatagramProtocol):
    """An asyncio DatagramProtocol for receiving RTP audio streams (RFC 3550)."""

    rtp_header_size = 12

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Strip the fixed-size RTP header and forward the audio payload."""
        if len(data) > self.rtp_header_size:
            self.audio_received(data[self.rtp_header_size :])

    def audio_received(self, data: bytes) -> None:
        """Handle a decoded RTP audio payload. Override in subclasses."""
        return NotImplemented


class IncomingCall(RTPProtocol):
    """An inbound SIP call: answers or rejects the INVITE and receives Opus audio via RTP."""

    def __init__(
        self,
        request: Request,
        addr: tuple[str, int],
        send: Callable[..., None],
    ) -> None:
        self._request = request
        self._addr = addr
        self._send = send
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
        logger.debug("RTP listening on %s:%s", local_addr[0], local_addr[1])
        sdp = (
            f"v=0\r\n"
            f"c=IN IP4 {local_addr[0]}\r\n"
            f"m=audio {local_addr[1]} RTP/AVP 111\r\n"
            f"a=rtpmap:111 opus/48000/2\r\n"
        ).encode()
        self._send(
            Response(
                status_code=200,
                reason="OK",
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

    def reject(self, status_code: int = 486, reason: str = "Busy Here") -> None:
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

    def send(self, message: Message, addr: tuple[str, int]) -> None:
        """Serialize and send a SIP message to the given address."""
        logger.debug("Sending %s to %s:%s", type(message).__name__, addr[0], addr[1])
        self._transport.sendto(bytes(message), addr)

    def request_received(self, request: Request, addr: tuple[str, int]) -> None:
        """Dispatch an INVITE request to invite_received."""
        match request.method:
            case "INVITE":
                logger.info("INVITE received from %s", addr[0])
                self.invite_received(self.create_call(request, addr), addr)
            case _:
                return NotImplemented

    def create_call(self, request: Request, addr: tuple[str, int]) -> IncomingCall:
        """Create an IncomingCall for an INVITE. Override to use a custom call class."""
        return IncomingCall(request, addr, self.send)

    def invite_received(self, call: IncomingCall, addr: tuple[str, int]) -> None:
        """Handle an incoming call. Override in subclasses to process calls."""
        return NotImplemented


class RegisterProtocol(IncomingCallProtocol):
    """SIP UAC: registers with a carrier via digest auth and handles inbound calls."""

    def __init__(
        self,
        server_addr: tuple[str, int],
        aor: str,
        username: str,
        password: str,
    ) -> None:
        self._server_addr = server_addr
        self._aor = aor
        self._username = username
        self._password = password
        self._call_id = str(uuid.uuid4())
        self._cseq = 0

    @property
    def _registrar_uri(self) -> str:
        """Registrar Request-URI derived from the AOR (e.g. sip:example.com)."""
        scheme, _, rest = self._aor.partition(":")
        _, _, hostport = rest.partition("@")
        return f"{scheme}:{hostport}"

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        """Store the transport and send the initial REGISTER request."""
        super().connection_made(transport)
        self.register()

    def register(
        self,
        authorization: str | None = None,
        proxy_authorization: str | None = None,
    ) -> None:
        """Send a REGISTER request to the carrier, optionally with credentials."""
        self._cseq += 1
        logger.debug(
            "Sending REGISTER to %s:%s (CSeq %s)",
            self._server_addr[0],
            self._server_addr[1],
            self._cseq,
        )
        local_address = self._transport.get_extra_info("sockname") or ("0.0.0.0", 5060)  # noqa: S104
        branch = f"z9hG4bK{uuid.uuid4().hex}"
        logger.debug("REGISTER Via branch: %s", branch)
        headers = {
            "Via": f"SIP/2.0/UDP {local_address[0]}:{local_address[1]};rport;branch={branch}",
            "From": self._aor,
            "To": self._aor,
            "Call-ID": self._call_id,
            "CSeq": f"{self._cseq} REGISTER",
            "Contact": f"<{self._aor}>",
            "Expires": "3600",
            "Max-Forwards": "70",
        }
        if authorization is not None:
            headers["Authorization"] = authorization
        if proxy_authorization is not None:
            headers["Proxy-Authorization"] = proxy_authorization
        self.send(
            Request(method="REGISTER", uri=self._registrar_uri, headers=headers),
            self._server_addr,
        )

    def response_received(self, response: Response, addr: tuple[str, int]) -> None:
        """Handle REGISTER responses including digest auth challenges (RFC 3261 §22)."""
        if response.status_code == 200 and "REGISTER" in response.headers.get(
            "CSeq", ""
        ):
            logger.info("Registration successful")
            self.registered()
            return
        if response.status_code in (401, 407):
            logger.debug(
                "Auth challenge received (%s), retrying with credentials",
                response.status_code,
            )
            is_proxy = response.status_code == 407
            challenge_key = "Proxy-Authenticate" if is_proxy else "WWW-Authenticate"
            params = _parse_auth_challenge(response.headers.get(challenge_key, ""))
            realm = params.get("realm", "")
            nonce = params.get("nonce", "")
            opaque = params.get("opaque")
            qop_options = params.get("qop", "")
            qop = "auth" if "auth" in qop_options.split(",") else None
            nc = "00000001"
            cnonce = os.urandom(8).hex() if qop else None
            digest = _digest_response(
                username=self._username,
                password=self._password,
                realm=realm,
                nonce=nonce,
                method="REGISTER",
                uri=self._registrar_uri,
                qop=qop,
                nc=nc,
                cnonce=cnonce,
            )
            auth_value = (
                f'Digest username="{self._username}", realm="{realm}", '
                f'nonce="{nonce}", uri="{self._registrar_uri}", '
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
        return NotImplemented

    def registered(self) -> None:
        """Handle a confirmed carrier registration. Override to react."""
        return NotImplemented
