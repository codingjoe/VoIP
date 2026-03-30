"""SIP transaction layer (RFC 3261 §17)."""

import asyncio
import dataclasses
import datetime
import hashlib
import ipaddress
import logging
import re
import secrets
import typing
import uuid

import voip
from voip.rtp import Session
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

from ..types import NetworkAddress
from . import messages, types
from .messages import Dialog, Request, Response, SIPHeaderDict
from .types import (
    CallerID,
    DigestAlgorithm,
    DigestQoP,
    SIPMethod,
    SIPStatus,
)

if typing.TYPE_CHECKING:
    from .protocol import SessionInitiationProtocol

logger = logging.getLogger("voip.sip")

__all__ = [
    "InviteTransaction",
    "RegistrationTransaction",
]


@dataclasses.dataclass(kw_only=True, slots=True)
class Transaction:
    """
    Initiated by a request, completed by any number of responses.

    Args:
        dialog: The SIP dialog this transaction belongs to.
        branch: Unique identifier for the transaction, must start with "z9hG4bK".
        cseq: The CSeq number for this transaction, starting at 1 and incremented
            for each new request sent within the transaction.
    """

    branch_prefix: typing.ClassVar[str] = "z9hG4bK"

    method: SIPMethod
    branch: str = dataclasses.field(
        default_factory=lambda: f"{Transaction.branch_prefix}-{uuid.uuid4()}"
    )
    cseq: int
    sip: SessionInitiationProtocol
    request: messages.Request | None = None
    responses: list[messages.Response] = dataclasses.field(
        init=False, default_factory=list
    )
    dialog: Dialog = None

    created: datetime.datetime = dataclasses.field(
        init=False, default_factory=datetime.datetime.now
    )

    def __post_init__(self):
        if not self.branch.startswith(self.branch_prefix):
            raise ValueError(f"Branch parameter must start with {self.branch_prefix!r}")

    @property
    def headers(self) -> dict[str, str]:
        """Return a dict of headers for this transaction."""
        return {
            "Via": f"SIP/2.0/{self.sip.aor.transport} {self.sip.local_address};rport;branch={self.branch}",
            "CSeq": f"{self.cseq} {self.method}",
        }

    def response_received(self, response: messages.Response):
        """Send a response to this transaction."""

    def send_response(self, response: messages.Response):
        """Send a response to this transaction."""
        self.sip.send(response)

    @classmethod
    def from_request(
        cls,
        *,
        request: messages.Request,
        sip: SessionInitiationProtocol,
    ):
        try:
            dialog = sip.dialogs[request.remote_tag, request.local_tag]
        except KeyError:
            dialog = Dialog.from_request(request)
        return cls(
            sip=sip,
            dialog=dialog,
            method=request.method,
            branch=request.branch,
            request=request,
            cseq=request.sequence,
        )


@dataclasses.dataclass(kw_only=True, slots=True)
class RegistrationTransaction(Transaction):
    """SIP REGISTER client transaction [RFC 3261 §10]."""

    #: Map from `DigestAlgorithm` to the hashlib name.
    DIGEST_HASH_NAME: typing.ClassVar[dict[str, str]] = {
        DigestAlgorithm.MD5: "md5",
        DigestAlgorithm.MD5_SESS: "md5",
        DigestAlgorithm.SHA_256: "sha256",
        DigestAlgorithm.SHA_256_SESS: "sha256",
        DigestAlgorithm.SHA_512_256: "sha512_256",
        DigestAlgorithm.SHA_512_256_SESS: "sha512_256",
    }

    authorization: str | None = None
    proxy_authorization: str | None = None
    cseq: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.dialog = self.dialog or Dialog(uac=self.sip.aor)
        headers = (
            self.headers
            | self.dialog.headers
            | {
                "Contact": self.sip.contact,
                "Expires": "3600",
                "Max-Forwards": "70",
                "Supported": "outbound",
            }
        )
        if self.authorization is not None:
            headers["Authorization"] = self.authorization
        if self.proxy_authorization is not None:
            headers["Proxy-Authorization"] = self.proxy_authorization
        self.request = Request.from_dialog(
            dialog=self.dialog,
            method=SIPMethod.REGISTER,
            uri=types.SipUri(host=self.sip.aor.host, scheme=self.sip.aor.scheme),
            headers=headers,
        )

        self.sip.send(self.request)

    def response_received(self, response: Response) -> None:
        """Handle a REGISTER response including digest auth challenges (RFC 3261 §22).

        Args:
            response: The parsed SIP response.
        """
        self.sip.transactions.pop(self.branch)
        match response.status_code:
            case SIPStatus.OK:
                logger.info("Registration successful")
                self.sip.on_registered()
                return
            case SIPStatus.UNAUTHORIZED | SIPStatus.PROXY_AUTHENTICATION_REQUIRED:
                logger.debug(
                    "Auth challenge received (%s), retrying with credentials",
                    response.status_code,
                )
                is_proxy = (
                    response.status_code == SIPStatus.PROXY_AUTHENTICATION_REQUIRED
                )
                challenge_key = "Proxy-Authenticate" if is_proxy else "WWW-Authenticate"
                params = self.parse_auth_challenge(
                    response.headers[challenge_key] or ""
                )
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
                    username=self.sip.aor.user,
                    password=self.sip.aor.password,
                    realm=realm,
                    nonce=nonce,
                    method=SIPMethod.REGISTER,
                    uri=self.sip.aor.host,
                    algorithm=algorithm,
                    qop=qop,
                    nc=nc,
                    cnonce=cnonce,
                )
                auth_value = (
                    f'Digest username="{self.sip.aor.user}", realm="{realm}", '
                    f'nonce="{nonce}", uri="{self.sip.aor.host}", '
                    f'response="{digest}", algorithm="{algorithm}"'
                )
                if qop:
                    auth_value += f', qop={qop}, nc={nc}, cnonce="{cnonce}"'
                if opaque:
                    auth_value += f', opaque="{opaque}"'
                if is_proxy:
                    tx = RegistrationTransaction(
                        sip=self.sip,
                        dialog=self.dialog,
                        cseq=2,
                        method=self.method,
                        proxy_authorization=auth_value,
                    )
                else:
                    tx = RegistrationTransaction(
                        sip=self.sip,
                        dialog=self.dialog,
                        cseq=2,
                        method=self.method,
                        authorization=auth_value,
                    )
                self.sip.transactions[tx.branch] = tx
            case _:
                raise NotImplementedError(
                    f"Unknown SIP status code: {response.status_code}"
                )

    @staticmethod
    def parse_auth_challenge(header: str) -> dict[str, str]:
        """Parse Digest challenge parameters from a WWW-Authenticate/Proxy-Authenticate header.

        Args:
            header: The raw ``WWW-Authenticate`` or ``Proxy-Authenticate`` header value.

        Returns:
            A dict mapping parameter names to their unquoted values.
        """
        _, _, params_str = header.partition(" ")
        params = {}
        for part in re.split(r",\s*(?=[a-zA-Z])", params_str):
            key, _, value = part.partition("=")
            if key.strip():
                params[key.strip()] = value.strip().strip('"')
        return params

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

        Args:
            username: SIP username (AOR user part).
            password: SIP password.
            realm: Digest realm from the challenge.
            nonce: Digest nonce from the challenge.
            method: SIP method string (e.g. ``"REGISTER"``).
            uri: Request-URI string used in the digest.
            algorithm: Digest algorithm identifier (default: ``"SHA-256"``).
            qop: Quality-of-protection value, or ``None``.
            nc: Nonce count hex string (default: ``"00000001"``).
            cnonce: Client nonce, required for ``*-sess`` algorithms and ``qop``.

        Returns:
            Hex-encoded digest response string.

        Raises:
            ValueError: If ``algorithm`` is not a recognised `DigestAlgorithm`,
                or if a ``*-sess`` algorithm is requested without a ``cnonce``.
        """
        try:
            hash_name = cls.DIGEST_HASH_NAME[algorithm]
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


@dataclasses.dataclass(kw_only=True, slots=True)
class InviteTransaction(Transaction):
    """SIP INVITE server transaction [RFC 3261 §17.2].

    Encapsulates the state and behavior of a single INVITE dialog.
    The SIP layer creates one instance per incoming INVITE, keyed by
    Via branch (RFC 3261 §17.1.3).

    Override `call_received` in a subclass to react to the call without
    subclassing
    [`SessionInitiationProtocol`][voip.sip.protocol.SessionInitiationProtocol]:

    ```python
    class MyTransaction(Transaction):
        def call_received(self) -> None:
            asyncio.create_task(self.answer(call_class=MyCall))
    ```

    Register the subclass on the session:

    ```python
    class MySession(SessionInitiationProtocol):
        transaction_class = MyTransaction
    ```

    [RFC 3261 §17.2]: https://datatracker.ietf.org/doc/html/rfc3261#section-17.2
    """

    pending_call_class: type[Session] | None = dataclasses.field(
        default=None, repr=False
    )
    pending_call_kwargs: dict[str, typing.Any] = dataclasses.field(
        default_factory=dict, repr=False
    )

    def invite_received(self, request: Request) -> None:
        """Handle the incoming call.

        Override in subclasses to decide whether to answer, ring, or reject.
        Implementations are expected to produce a response by calling
        `ringing`, `answer`, or `reject`. The base implementation is a no-op.

        Args:
            request: The SIP INVITE request.
        """

    def ack_received(self, request: Request) -> None:
        """Handle an ACK confirming dialog establishment (RFC 3261 §17.2.1).

        Removes the INVITE server transaction from the registry.  Override
        in subclasses to react to the ACK.

        Args:
            request: The SIP ACK request.
        """
        self.sip.transactions.pop(self.branch)

    def bye_received(self, request: Request) -> None:
        """Handle a BYE terminating a dialog.

        Override in subclasses to tear down the call.

        Args:
            request: The SIP BYE request.
        """
        self.sip.dialogs.pop((self.dialog.remote_tag, self.dialog.local_tag))
        self.send_response(
            Response.from_request(
                request,
                dialog=self.dialog,
                status_code=SIPStatus.OK,
                phrase=SIPStatus.OK.phrase,
            )
        )

    def cancel_received(self, request: Request) -> None:
        """Handle a CANCEL request for a pending INVITE.

        Override in subclasses to react to caller cancellation before the call
        is answered.

        Args:
            request: The SIP CANCEL request.
        """
        self.sip.transactions.pop(self.branch)
        self.sip.dialogs.pop((self.dialog.remote_tag, self.dialog.local_tag))
        self.send_response(
            Response.from_request(
                request,
                dialog=self.dialog,
                status_code=SIPStatus.OK,
                phrase=SIPStatus.OK.phrase,
            )
        )

    def ringing(self) -> None:
        """Send a 180 Ringing provisional response [RFC 3261 §21.1.2].

        Call before `answer` to notify the caller that the UA is alerting the
        user.

        [RFC 3261 §21.1.2]: https://datatracker.ietf.org/doc/html/rfc3261#section-21.1.2
        """
        self.send_response(
            Response.from_request(
                self.request,
                dialog=self.dialog,
                status_code=SIPStatus.RINGING,
                phrase=SIPStatus.RINGING.phrase,
                headers=self.headers,
            )
        )

    def reject(self, status_code: SIPStatus = SIPStatus.BUSY_HERE) -> None:
        """Reject the incoming call.

        Args:
            status_code: SIP response status code (default: 486 Busy Here).
        """
        self.send_response(
            Response.from_request(
                self.request,
                dialog=self.dialog,
                status_code=status_code,
                phrase=status_code.phrase,
                headers=self.headers,
            )
        )

    def answer(self, *, call_class: type[Session], **call_kwargs: typing.Any) -> None:
        """Answer the call by setting up RTP and sending 200 OK with SDP.

        Example:
            Call from within `call_received`:

            ```python
            def call_received(self) -> None:
                asyncio.create_task(self.answer(call_class=MyCall))
            ```

        Args:
            call_class: Session implementation that will be initialized.
            **call_kwargs: Additional keyword arguments forwarded to the
                call class constructor.

        Raises:
            NotImplementedError: When `negotiate_codec` raises (no supported
                codec in the remote SDP offer).
        """
        peer = (
            self.sip.transport.get_extra_info("peername")
            if self.sip.transport
            else None
        )
        caller = CallerID(self.request.headers.get("From", ""))
        remote_audio = next(
            (
                m
                for m in (self.request.body.media if self.request.body else [])
                if m.media == "audio"
            ),
            None,
        )
        if remote_audio is not None:
            negotiated_media = call_class.negotiate_codec(remote_audio)
        else:
            negotiated_media = MediaDescription(
                media="audio",
                port=0,
                proto="RTP/SAVP",
                fmt=[RTPPayloadFormat.from_pt(0)],
            )

        use_srtp = negotiated_media.proto == "RTP/SAVP"
        srtp_session = SRTPSession.generate() if use_srtp else None

        dialog = Dialog.from_request(self.request)
        dialog.local_party = f"{self.request.headers['To']};tag={dialog.remote_tag}"
        dialog.remote_party = str(self.request.headers["From"])
        self.sip.dialogs[dialog.remote_tag, dialog.local_tag] = dialog

        call_handler = call_class(
            rtp=self.sip.rtp,
            sip=self.sip,
            caller=caller,
            media=negotiated_media,
            srtp=srtp_session,
            dialog=dialog,
            **call_kwargs,
        )
        if remote_audio is not None and remote_audio.port != 0:
            media_connection = remote_audio.connection
            session_connection = (
                self.request.body.connection if self.request.body else None
            )
            connection = media_connection or session_connection
            if connection is not None:
                remote_ip = connection.connection_address
            else:
                remote_ip = peer[0] if peer else "0.0.0.0"  # noqa: S104
            remote_rtp_address: NetworkAddress | None = NetworkAddress(
                remote_ip, remote_audio.port
            )
        else:
            remote_rtp_address = None
        self.sip.rtp.register_call(remote_rtp_address, call_handler)

        if remote_rtp_address is not None:
            self.sip.rtp.send(b"\x00", remote_rtp_address)

        record_route = self.request.headers.get("Record-Route")
        session_id = str(secrets.randbelow(2**32) + 1)
        rtp_public = self.sip.rtp.public_address
        sdp_media_attributes = [Attribute(name="sendrecv")]
        if srtp_session is not None:
            sdp_media_attributes.append(
                Attribute(name="crypto", value=srtp_session.sdes_attribute)
            )
        self.send_response(
            Response.from_request(
                request=self.request,
                dialog=dialog,
                status_code=SIPStatus.OK,
                phrase=SIPStatus.OK.phrase,
                headers={
                    **({"Record-Route": record_route} if record_route else {}),
                    "Contact": self.sip.contact,
                    "Allow": self.sip.allow_header,
                    "Supported": "replaces",
                    "Content-Type": "application/sdp",
                },
                body=SessionDescription(
                    origin=Origin(
                        username="-",
                        sess_id=session_id,
                        sess_version=session_id,
                        nettype="IN",
                        addrtype="IP6"
                        if isinstance(rtp_public[0], ipaddress.IPv6Address)
                        else "IP4",
                        unicast_address=str(rtp_public[0]),
                    ),
                    timings=[Timing(start_time=0, stop_time=0)],
                    connection=ConnectionData(
                        nettype="IN",
                        addrtype="IP6"
                        if isinstance(rtp_public[0], ipaddress.IPv6Address)
                        else "IP4",
                        connection_address=str(rtp_public[0]),
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
            )
        )

    async def make_call(
        self,
        target: str,
        *,
        call_class: type[Session],
        **call_kwargs: typing.Any,
    ) -> Request:
        """Initiate an outgoing call to `target`.

        Builds an SDP offer using `call_class.sdp_formats`, sends an INVITE,
        and registers this transaction to handle the response. When the callee
        answers (200 OK), `_accept_call` completes the setup, sends the ACK,
        and registers the RTP call handler.

        Args:
            target: SIP URI of the callee (e.g. ``"sip:+15551234567@carrier.com"``).
            call_class: Session implementation that will be initialized for the call.
            **call_kwargs: Additional keyword arguments forwarded to the
                call class constructor.

        Returns:
            The INVITE [`Request`][voip.sip.messages.Request] that was sent.
        """
        self.pending_call_class = call_class
        self.pending_call_kwargs = call_kwargs

        target_uri = types.SipUri.parse(target)
        dialog = Dialog(uac=self.sip.aor)
        self.dialog = dialog

        rtp_public = self.sip.rtp.public_address
        session_id = str(secrets.randbelow(2**32) + 1)
        sdp_offer = SessionDescription(
            origin=Origin(
                username="-",
                sess_id=session_id,
                sess_version=session_id,
                nettype="IN",
                addrtype=(
                    "IP6" if isinstance(rtp_public[0], ipaddress.IPv6Address) else "IP4"
                ),
                unicast_address=str(rtp_public[0]),
            ),
            timings=[Timing(start_time=0, stop_time=0)],
            connection=ConnectionData(
                nettype="IN",
                addrtype=(
                    "IP6" if isinstance(rtp_public[0], ipaddress.IPv6Address) else "IP4"
                ),
                connection_address=str(rtp_public[0]),
            ),
            media=[
                MediaDescription(
                    media="audio",
                    port=rtp_public[1],
                    proto="RTP/AVP",
                    fmt=call_class.sdp_formats(),
                    attributes=[Attribute(name="sendrecv")],
                )
            ],
        )
        self.request = Request(
            method=SIPMethod.INVITE,
            uri=target_uri,
            headers={
                "Max-Forwards": "70",
                **self.headers,
                "From": dialog.from_header,
                "To": str(target_uri),
                "Contact": self.sip.contact,
                "Call-ID": dialog.call_id,
                "Route": f"<sip:{str(rtp_public[0])}:5060;transport=tcp;lr>",
                "Allow": self.sip.allow_header,
                "User-Agent": f"python/voip/{voip.__version__}",
                "Content-Type": "application/sdp",
            },
            body=sdp_offer,
        )
        self.sip.transactions[self.branch] = self
        self.sip.send(self.request)
        return self.request

    def response_received(self, response: Response) -> None:
        """Handle responses to an outbound INVITE.

        Dispatches provisional (1xx), successful (2xx), and failure (4xx–6xx)
        responses.  On 200 OK the call setup is completed asynchronously via
        `_accept_call`.

        Args:
            response: The parsed SIP response.
        """
        match response.status_code // 100:
            case 1:
                pass
            case 2:
                try:
                    asyncio.get_running_loop().create_task(self._accept_call(response))
                except RuntimeError:
                    logger.debug(
                        "response_received called outside of an async context; "
                        "200 OK will not be processed"
                    )
            case _:
                self.sip.transactions.pop(self.branch, None)
                logger.warning(
                    "Outbound call failed: %s %s",
                    response.status_code,
                    response.phrase,
                )

    async def _accept_call(self, response: Response) -> None:
        """Complete call setup after a 200 OK is received.

        Negotiates the codec from the remote SDP answer, creates the call
        handler, registers it with the RTP mux, updates the dialog, and
        sends the ACK.

        Args:
            response: The 200 OK SIP response containing the remote SDP answer.
        """
        peer = (
            self.sip.transport.get_extra_info("peername")
            if self.sip.transport
            else None
        )
        remote_audio = next(
            (
                m
                for m in (response.body.media if response.body else [])
                if m.media == "audio"
            ),
            None,
        )
        if remote_audio is not None and self.pending_call_class is not None:
            negotiated_media = self.pending_call_class.negotiate_codec(remote_audio)
        else:
            negotiated_media = MediaDescription(
                media="audio",
                port=0,
                proto="RTP/AVP",
                fmt=[RTPPayloadFormat.from_pt(0)],
            )

        if self.pending_call_class is not None:
            call_handler = self.pending_call_class(
                rtp=self.sip.rtp,
                sip=self.sip,
                caller=CallerID(str(self.sip.aor)),
                media=negotiated_media,
                srtp=None,
                dialog=self.dialog,
                **self.pending_call_kwargs,
            )
            if remote_audio is not None and remote_audio.port != 0:
                media_connection = remote_audio.connection
                session_connection = response.body.connection if response.body else None
                connection = media_connection or session_connection
                remote_ip = (
                    connection.connection_address
                    if connection is not None
                    else peer[0]
                    if peer
                    else None
                )
                remote_rtp_address: NetworkAddress | None = (
                    NetworkAddress(remote_ip, remote_audio.port)
                    if remote_ip is not None
                    else None
                )
            else:
                remote_rtp_address = None
            self.sip.rtp.register_call(remote_rtp_address, call_handler)
            if remote_rtp_address is not None:
                self.sip.rtp.send(b"\x00", remote_rtp_address)

        # Update the dialog with remote tag from 200 OK then store it.
        # The To-tag in the 200 OK is the callee's tag (remote).  The From-tag
        # is our original local tag, which must become dialog.remote_tag so
        # that subsequent in-dialog BYE lookups (keyed by
        # (request.remote_tag, request.local_tag) = (our_tag, callee_tag))
        # resolve correctly via `sip.dialogs[(dialog.remote_tag, dialog.local_tag)]`.
        our_tag = response.local_tag
        callee_tag = response.remote_tag
        self.dialog.remote_tag = our_tag
        self.dialog.local_tag = callee_tag
        self.sip.dialogs[(our_tag, callee_tag)] = self.dialog

        ack_branch = f"{Transaction.branch_prefix}-{uuid.uuid4()}"
        contact = response.headers.get("Contact")
        ack_uri = (
            contact.strip("<>").split(";")[0] if contact else str(self.request.uri)
        )

        # Store BYE-ready dialog state now that dialog tags are finalised.
        self.dialog.local_party = str(response.headers["From"])
        self.dialog.remote_party = str(response.headers["To"])
        self.dialog.remote_contact = ack_uri
        self.dialog.outbound_cseq = self.cseq + 1
        ack_headers: SIPHeaderDict = SIPHeaderDict(
            {
                "Via": (
                    f"SIP/2.0/{self.sip.aor.transport}"
                    f" {self.sip.rtp.public_address};rport;branch={ack_branch};alias"
                ),
                "Max-Forwards": "70",
                "From": response.headers["From"],
                "To": response.headers["To"],
                "Call-ID": self.dialog.call_id,
                "CSeq": f"{self.cseq} {SIPMethod.ACK}",
                "Content-Length": 0,
            }
        )
        for record_route in response.headers.getlist("Record-Route"):
            ack_headers.add("Route", record_route)
        self.sip.send(
            Request(
                method=SIPMethod.ACK,
                uri=ack_uri,
                headers=ack_headers,
            )
        )
        self.sip.transactions.pop(self.branch, None)
