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
from .messages import Request, Response, SIPHeaderDict
from .types import (
    CallerID,
    DigestAlgorithm,
    DigestQoP,
    SIPMethod,
    SIPStatus,
)

if typing.TYPE_CHECKING:
    from .dialog import Dialog
    from .protocol import SessionInitiationProtocol

logger = logging.getLogger("voip.sip")

__all__ = [
    "ByeTransaction",
    "InviteTransaction",
    "RegisterTransaction",
]


@dataclasses.dataclass(kw_only=True, slots=True)
class Transaction(asyncio.Future):
    """
    Initiated by a request, completed by any number of responses.

    Transactions are awaitable: `await tx` suspends until the transaction
    reaches its terminal state and resolves to the dialog.

    Args:
        dialog: The SIP dialog this transaction belongs to.
        branch: Unique identifier for the transaction, must start with "z9hG4bK".
        cseq: The CSeq sequence number for this transaction.
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
        asyncio.Future.__init__(self)
        if not self.branch.startswith(self.branch_prefix):
            raise ValueError(f"Branch parameter must start with {self.branch_prefix!r}")

    @property
    def headers(self) -> dict[str, str]:
        """Return a dict of headers for this transaction."""
        return {
            "Via": f"SIP/2.0/{self.sip.aor.transport} {self.sip.public_address};rport;branch={self.branch}",
            "CSeq": f"{self.cseq} {self.method}",
        }

    def response_received(self, response: messages.Response):
        """Send a response to this transaction."""

    def send_response(self, response: messages.Response):
        """Send a response to this transaction."""
        self.sip.send(response)

    def complete(self) -> None:
        """Resolve the transaction with its dialog if not already complete."""
        if not self.done():
            self.set_result(self.dialog)

    @classmethod
    async def receive(
        cls,
        *,
        request: Request,
        sip: SessionInitiationProtocol,
    ): ...

    @classmethod
    async def send(
        cls,
        *,
        sip: SessionInitiationProtocol,
        **kwargs: typing.Any,
    ): ...

    @classmethod
    def from_request(
        cls,
        *,
        request: messages.Request,
        sip: SessionInitiationProtocol,
    ):
        try:
            dialog = sip._dialogs[request.remote_tag, request.local_tag]
        except KeyError:
            dialog = sip.dialog_class.from_request(request)
        return cls(
            sip=sip,
            dialog=dialog,
            method=request.method,
            branch=request.branch,
            request=request,
            cseq=request.sequence,
        )


class DigestAuthMixin:
    """Digest authentication for SIP client transactions (RFC 3261 §22, RFC 8760).

    Mix into a [Transaction][voip.sip.transactions.Transaction] subclass whose
    requests may receive a `401 Unauthorized` or `407 Proxy Authentication
    Required` challenge.  The mixin supplies the shared digest machinery
    (challenge parsing, response computation, retry chaining); the host
    transaction only implements
    [retry_with_auth][voip.sip.transactions.DigestAuthMixin.retry_with_auth]
    to rebuild and resend its request carrying the computed credentials.

    A challenge is answered by calling
    [handle_auth_challenge][voip.sip.transactions.DigestAuthMixin.handle_auth_challenge]
    from the host's `response_received`.  The caller must drop the original
    transaction from the registry first (e.g. via `ack` or `drop_transaction`);
    `handle_auth_challenge` then registers a fresh retry transaction and chains
    its result back to the original via
    [forward_result][voip.sip.transactions.DigestAuthMixin.forward_result].
    """

    __slots__ = ()

    #: Map from `DigestAlgorithm` to the hashlib name.
    DIGEST_HASH_NAME: typing.ClassVar[dict[str, str]] = {
        DigestAlgorithm.MD5: "md5",
        DigestAlgorithm.MD5_SESS: "md5",
        DigestAlgorithm.SHA_256: "sha256",
        DigestAlgorithm.SHA_256_SESS: "sha256",
        DigestAlgorithm.SHA_512_256: "sha512_256",
        DigestAlgorithm.SHA_512_256_SESS: "sha512_256",
    }

    def digest_uri(self) -> str:
        """Return the Request-URI used for the digest `uri` parameter.

        Defaults to the request's Request-URI (RFC 3261 §22.1).  Override to
        customise per transaction.
        """
        return str(self.request.uri)  # type: ignore[attr-defined]

    def handle_auth_challenge(self, response: Response) -> bool:
        """Answer a 401/407 by resending the request with digest credentials.

        Parses the challenge, computes the digest response, and delegates to
        [retry_with_auth][voip.sip.transactions.DigestAuthMixin.retry_with_auth]
        to resend.  The original transaction must already be dropped by the
        caller.

        Args:
            response: The `401`/`407` challenge response.

        Returns:
            `True` if a retry transaction was started.
        """
        is_proxy = response.status_code == SIPStatus.PROXY_AUTHENTICATION_REQUIRED
        challenge_key = "Proxy-Authenticate" if is_proxy else "WWW-Authenticate"
        params = self.parse_auth_challenge(response.headers.get(challenge_key, ""))
        realm = params.get("realm", "")
        nonce = params.get("nonce", "")
        opaque = params.get("opaque")
        algorithm = params.get("algorithm", DigestAlgorithm.MD5)
        qop_options = params.get("qop", "")
        qop = (
            DigestQoP.AUTH.value
            if DigestQoP.AUTH.value in qop_options.split(",")
            else None
        )
        nc = "00000001"
        cnonce = secrets.token_hex(8) if qop else None
        uri = self.digest_uri()
        digest = self.digest_response(
            username=self.sip.aor.user,  # type: ignore[attr-defined]
            password=self.sip.aor.password,  # type: ignore[attr-defined]
            realm=realm,
            nonce=nonce,
            method=self.method,  # type: ignore[attr-defined]
            uri=uri,
            algorithm=algorithm,
            qop=qop,
            nc=nc,
            cnonce=cnonce,
        )
        auth_value = (
            f'Digest username="{self.sip.aor.user}", realm="{realm}", '  # type: ignore[attr-defined]
            f'nonce="{nonce}", uri="{uri}", '
            f'response="{digest}", algorithm="{algorithm}"'
        )
        if qop:
            auth_value += f', qop={qop}, nc={nc}, cnonce="{cnonce}"'
        if opaque:
            auth_value += f', opaque="{opaque}"'
        return self.retry_with_auth(response, auth_value, is_proxy)

    def retry_with_auth(
        self, response: Response, auth_value: str, is_proxy: bool
    ) -> bool:
        """Rebuild and resend this request carrying *auth_value*.

        Override in host transactions: construct a new transaction with a fresh
        branch and incremented CSeq, attach `auth_value` as `Authorization`
        (or `Proxy-Authorization` when *is_proxy*), register it, and chain its
        result back to the original via
        [forward_result][voip.sip.transactions.DigestAuthMixin.forward_result].

        Returns:
            `True` if a retry was started.
        """
        raise NotImplementedError

    def forward_result(self, fut: asyncio.Future) -> None:
        """Forward the result of *fut* to this transaction (auth retry chaining)."""
        if not self.done():
            if fut.cancelled():
                self.cancel()
            elif exc := fut.exception():
                self.set_exception(exc)
            else:
                self.set_result(fut.result())

    @staticmethod
    def parse_auth_challenge(header: str) -> dict[str, str]:
        """Parse Digest challenge parameters from a WWW/Proxy-Authenticate header.

        Args:
            header: The raw `WWW-Authenticate` or `Proxy-Authenticate` value.

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
        SHA-512-256.  The `algorithm` parameter selects the hash function;
        it defaults to `SHA-256`.

        Args:
            username: SIP username (AOR user part).
            password: SIP password.
            realm: Digest realm from the challenge.
            nonce: Digest nonce from the challenge.
            method: SIP method string (e.g. `"REGISTER"`).
            uri: Request-URI string used in the digest.
            algorithm: Digest algorithm identifier (default: `"SHA-256"`).
            qop: Quality-of-protection value, or `None`.
            nc: Nonce count hex string (default: `"00000001"`).
            cnonce: Client nonce, required for `*-sess` algorithms and `qop`.

        Returns:
            Hex-encoded digest response string.

        Raises:
            ValueError: If `algorithm` is not a recognised `DigestAlgorithm`,
                or if a `*-sess` algorithm is requested without a `cnonce`.
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
class RegisterTransaction(DigestAuthMixin, Transaction):
    """SIP REGISTER client transaction [RFC 3261 §10]."""

    authorization: str | None = None
    proxy_authorization: str | None = None
    cseq: int = 1

    def __post_init__(self):
        super().__post_init__()
        from .dialog import Dialog

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
            uri=types.SipURI(host=self.sip.aor.host, scheme=self.sip.aor.scheme),
            headers=headers,
        )

        self.sip.send(self.request)

    def response_received(self, response: Response) -> None:
        """Handle a REGISTER response including digest auth challenges (RFC 3261 §22).

        Args:
            response: The parsed SIP response.
        """
        self.sip.drop_transaction(self)
        match response.status_code:
            case SIPStatus.OK:
                logger.info("Registration successful")
                self.set_result(self.dialog)
            case SIPStatus.UNAUTHORIZED | SIPStatus.PROXY_AUTHENTICATION_REQUIRED:
                logger.debug(
                    "Auth challenge received (%s), retrying with credentials",
                    response.status_code,
                )
                self.handle_auth_challenge(response)
            case _:
                raise NotImplementedError(f"Unexpected SIP response: {response!r}")

    def retry_with_auth(
        self, response: Response, auth_value: str, is_proxy: bool
    ) -> bool:
        """Resend the REGISTER with credentials (RFC 3261 §22)."""
        tx = RegisterTransaction(
            sip=self.sip,
            dialog=self.dialog,
            cseq=self.cseq + 1,
            method=self.method,
            proxy_authorization=auth_value if is_proxy else None,
            authorization=None if is_proxy else auth_value,
        )
        self.sip.register_transaction(tx)
        tx.add_done_callback(self.forward_result)
        return True

    def digest_uri(self) -> str:
        """Use the registrar host as the digest URI for REGISTER."""
        return str(self.sip.aor.host)


@dataclasses.dataclass(kw_only=True, slots=True)
class InviteTransaction(DigestAuthMixin, Transaction):
    """SIP INVITE transaction for inbound and outbound calls [RFC 3261 §17].

    Handles the SIP signaling state machine for a single INVITE dialog.  The
    SIP layer creates one instance per incoming INVITE, keyed by Via branch
    (RFC 3261 §17.1.3).

    For inbound call handling, subclass [Dialog][voip.sip.Dialog]
    and override [call_received][voip.sip.Dialog.call_received]:

    ```python
    class MyDialog(Dialog):
        def call_received(self) -> None:
            self.ringing()
            self.accept(session_class=MyCall)

    class MySession(SessionInitiationProtocol):
        dialog_class = MyDialog
    ```

    [RFC 3261 §17]: https://datatracker.ietf.org/doc/html/rfc3261#section-17
    """

    pending_call_class: type[Session] | None = dataclasses.field(
        default=None, repr=False
    )
    pending_call_kwargs: dict[str, typing.Any] = dataclasses.field(
        default_factory=dict, repr=False
    )
    #: Offer SRTP (`RTP/SAVP` + SDES `a=crypto:`) in the outbound INVITE and
    #: fall back to plain RTP on rejection.  Defaults to `True`; flipped to
    #: `False` internally by the RTP fallback retry.
    offer_srtp: bool = True
    #: SRTP send session generated for the current offer; reused as the
    #: call's send `srtp` once the answer confirms SRTP.  `None` for RTP.
    srtp_offer: SRTPSession | None = dataclasses.field(default=None, repr=False)

    @classmethod
    async def receive(
        cls,
        *,
        request: Request,
        sip: SessionInitiationProtocol,
    ) -> Dialog:
        """Handle an incoming INVITE [RFC 3261 §13.3].

        Registers the transaction, notifies the dialog, and resolves when the
        ACK is received.

        Args:
            request: The incoming SIP INVITE request.
            sip: The SIP session receiving the request.

        Returns:
            The dialog once the call is established (ACK received).

        [RFC 3261 §13.3]: https://datatracker.ietf.org/doc/html/rfc3261#section-13.3
        """
        tx = cls.from_request(request=request, sip=sip)
        sip.register_transaction(tx)
        tx.dialog.invite_transaction = tx
        tx.dialog.sip = sip
        tx.dialog.call_received()
        return await tx

    def ack_received(self, request: Request) -> None:
        """Handle an ACK confirming dialog establishment (RFC 3261 §17.2.1).

        Removes the INVITE server transaction from the registry and resolves
        the transaction future with the dialog.

        Args:
            request: The SIP ACK request.
        """
        self.sip.drop_transaction(self)
        self.complete()

    def cancel_received(self, request: Request) -> None:
        """Handle a CANCEL request for a pending INVITE.

        Args:
            request: The SIP CANCEL request.
        """
        self.sip.drop_transaction(self)
        self.sip.drop_dialog(self.dialog)
        self.send_response(
            Response.from_request(
                request,
                dialog=self.dialog,
                status_code=SIPStatus.OK,
                phrase=SIPStatus.OK.phrase,
            )
        )
        if not self.done():
            self.cancel()

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

    def answer(
        self, *, session_class: type[Session], **session_kwargs: typing.Any
    ) -> None:
        """Answer the call by setting up RTP and sending 200 OK with SDP.

        Example:
            Call from within [Dialog.call_received][voip.sip.Dialog.call_received]
            via [Dialog.accept][voip.sip.Dialog.accept]:

            ```python
            class MyDialog(Dialog):
                def call_received(self) -> None:
                    self.accept(call_class=MyCall)
            ```

        Args:
            session_class: Session implementation that will be initialized.
            **session_kwargs: Additional keyword arguments forwarded to the
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
            negotiated_media = session_class.negotiate_codec(remote_audio)
        else:
            negotiated_media = MediaDescription(
                media="audio",
                port=0,
                proto="RTP/SAVP",
                fmt=[RTPPayloadFormat.from_pt(0)],
            )

        # SRTP when the offer is `RTP/SAVP` (or `SAVPF`) and carries an SDES
        # `a=crypto:` key.  We generate a fresh send session (its key goes into
        # our 200-OK `a=crypto:`) and parse the offer's crypto to decrypt the
        # caller's media — SDES keys each direction independently (RFC 4568).
        is_srtp = negotiated_media.proto.startswith("RTP/SAVP")
        srtp_send = SRTPSession.generate() if is_srtp else None
        offer_crypto = (
            next(
                (
                    attr
                    for attr in remote_audio.attributes
                    if attr.name == "crypto" and attr.value
                ),
                None,
            )
            if remote_audio is not None
            else None
        )
        srtp_recv = (
            SRTPSession.from_sdes(offer_crypto.value)
            if is_srtp and offer_crypto is not None
            else None
        )

        self.dialog.local_party = (
            f"{self.request.headers['To']};tag={self.dialog.local_tag}"
        )
        self.dialog.remote_party = str(self.request.headers["From"])
        self.dialog.route_set = list(self.request.headers.getlist("Record-Route"))
        self.sip.register_dialog(self.dialog)

        session = session_class(
            rtp=self.sip.rtp,
            caller=caller,
            media=negotiated_media,
            srtp=srtp_send,
            srtp_recv=srtp_recv,
            dialog=self.dialog,
            **session_kwargs,
        )
        self.dialog.session = session
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
        self.sip.rtp.register_call(remote_rtp_address, session)

        if remote_rtp_address is not None:
            self.sip.rtp.send(b"\x00", remote_rtp_address)

        session_id = str(secrets.randbelow(2**32) + 1)
        rtp_public = self.sip.rtp.public_address.result()
        sdp_media_attributes = [Attribute(name="sendrecv")]
        if srtp_send is not None:
            sdp_media_attributes.append(
                Attribute(name="crypto", value=srtp_send.sdes_attribute)
            )
        self.send_response(
            Response.from_request(
                request=self.request,
                dialog=self.dialog,
                status_code=SIPStatus.OK,
                phrase=SIPStatus.OK.phrase,
                headers={
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

    @classmethod
    async def send(
        cls,
        *,
        sip: SessionInitiationProtocol,
        target: types.SipURI,
        dialog: Dialog,
        session_class: type[Session],
        **session_kwargs: typing.Any,
    ) -> Dialog:
        """Initiate an outgoing call to *target* [RFC 3261 §13.1].

        Offers SRTP (`RTP/SAVP` + SDES `a=crypto:`) and falls back to plain RTP
        if the far end rejects it (488/606/415) or answers with `RTP/AVP`.

        Args:
            sip: The SIP session to send from.
            target: SIP or tel URI of the callee (e.g. `"sip:+15551234567@carrier.com"` or `"tel:+15551234567"`).
            dialog: The dialog to associate with this call.
            session_class: Session implementation that will be initialized for the call.
            **session_kwargs: Additional keyword arguments forwarded to the
                call class constructor.

        Returns:
            The dialog once the call is established (ACK sent).

        [RFC 3261 §13.1]: https://datatracker.ietf.org/doc/html/rfc3261#section-13.1
        """
        if dialog.uac is None:
            dialog.uac = sip.aor
        dialog.sip = sip

        tx = cls(
            sip=sip,
            method=SIPMethod.INVITE,
            cseq=dialog.outbound_cseq,
            dialog=dialog,
        )
        tx.pending_call_class = session_class
        tx.pending_call_kwargs = session_kwargs
        tx.request = tx._build_invite_request(target)
        sip.register_transaction(tx)
        sip.send(tx.request)
        try:
            return await tx
        except asyncio.CancelledError:
            sip.drop_transaction(tx)
            raise

    def _build_invite_request(self, target: types.SipURI) -> messages.Request:
        """Build the outbound INVITE request (with SDP offer) for *target*.

        Factored out of [send][voip.sip.transactions.InviteTransaction.send]
        so that an auth retry can rebuild the request with a fresh branch and
        incremented CSeq via [retry_with_auth][voip.sip.transactions.InviteTransaction.retry_with_auth].

        When [offer_srtp][voip.sip.transactions.InviteTransaction.offer_srtp]
        is set (default) the offer advertises `RTP/SAVP` with an SDES
        `a=crypto:` attribute; otherwise it advertises plain `RTP/AVP`.
        """
        rtp_public = self.sip.rtp.public_address.result()
        session_id = str(secrets.randbelow(2**32) + 1)
        addrtype = "IP6" if isinstance(rtp_public[0], ipaddress.IPv6Address) else "IP4"
        if self.offer_srtp:
            proto = "RTP/SAVP"
            self.srtp_offer = SRTPSession.generate()
            attributes = [
                Attribute(name="sendrecv"),
                Attribute(name="crypto", value=self.srtp_offer.sdes_attribute),
            ]
        else:
            proto = "RTP/AVP"
            self.srtp_offer = None
            attributes = [Attribute(name="sendrecv")]
        sdp_offer = SessionDescription(
            origin=Origin(
                username="-",
                sess_id=session_id,
                sess_version=session_id,
                nettype="IN",
                addrtype=addrtype,
                unicast_address=str(rtp_public[0]),
            ),
            timings=[Timing(start_time=0, stop_time=0)],
            connection=ConnectionData(
                nettype="IN",
                addrtype=addrtype,
                connection_address=str(rtp_public[0]),
            ),
            media=[
                MediaDescription(
                    media="audio",
                    port=rtp_public[1],
                    proto=proto,
                    fmt=self.pending_call_class.sdp_formats(),
                    attributes=attributes,
                )
            ],
        )
        return Request(
            method=SIPMethod.INVITE,
            uri=target,
            headers={
                "Max-Forwards": "70",
                **self.headers,
                "From": self.dialog.from_header,
                "To": str(target),
                "Contact": self.sip.contact,
                "Call-ID": self.dialog.call_id,
                "Allow": self.sip.allow_header,
                "Content-Type": "application/sdp",
            },
            body=sdp_offer,
        )

    def retry_with_auth(
        self, response: Response, auth_value: str, is_proxy: bool
    ) -> bool:
        """Resend the INVITE with credentials after a 401/407 challenge.

        The original transaction is dropped by the caller (via `ack`); this
        builds a fresh INVITE (new branch, incremented CSeq) carrying the
        computed credentials and chains its result back to the original.  The
        SRTP/RTP offer mode is preserved across the retry.
        """
        header = "Proxy-Authorization" if is_proxy else "Authorization"
        return self._retry_invite(
            offer_srtp=self.offer_srtp, auth_headers={header: auth_value}
        )

    def retry_with_rtp(self, response: Response) -> bool:
        """Fall back from SRTP to plain RTP after the far end rejects SRTP.

        The original (SRTP-offering) transaction is dropped by the caller (via
        `ack`); this builds a fresh INVITE (new branch, incremented CSeq)
        advertising `RTP/AVP` with no `a=crypto:`, carrying over any
        `Authorization`/`Proxy-Authorization` already obtained, and chains its
        result back to the original.  Triggered for `488`, `606` and `415`
        responses to an SRTP offer.
        """
        auth_headers = {
            header: self.request.headers[header]
            for header in ("Authorization", "Proxy-Authorization")
            if header in self.request.headers
        }
        return self._retry_invite(offer_srtp=False, auth_headers=auth_headers)

    def _retry_invite(self, *, offer_srtp: bool, auth_headers: dict[str, str]) -> bool:
        """Build, register and send a fresh INVITE retry, chaining its result.

        Shared by [retry_with_auth][voip.sip.transactions.InviteTransaction.retry_with_auth]
        (which preserves the current SRTP/RTP mode) and
        [retry_with_rtp][voip.sip.transactions.InviteTransaction.retry_with_rtp]
        (which switches to plain RTP).  The new transaction carries *auth_headers*
        (e.g. `Authorization`/`Proxy-Authorization`) so credentials obtained on a
        prior attempt are not lost when falling back.
        """
        tx = type(self)(
            sip=self.sip,
            method=SIPMethod.INVITE,
            cseq=self.cseq + 1,
            dialog=self.dialog,
        )
        tx.pending_call_class = self.pending_call_class
        tx.pending_call_kwargs = self.pending_call_kwargs
        tx.offer_srtp = offer_srtp
        tx.request = tx._build_invite_request(self.request.uri)
        for header, value in auth_headers.items():
            tx.request.headers[header] = value
        self.sip.register_transaction(tx)
        self.sip.send(tx.request)
        tx.add_done_callback(self.forward_result)
        return True

    def response_received(self, response: Response) -> None:
        """Dispatch responses to an outbound INVITE."""
        if response.status_code in (
            SIPStatus.UNAUTHORIZED,
            SIPStatus.PROXY_AUTHENTICATION_REQUIRED,
        ):
            # Acknowledge the challenge (RFC 3261 §13.2.2.4); ack() drops this
            # transaction, then retry_with_auth registers a fresh INVITE with
            # credentials and chains its result back here via forward_result.
            self.ack(response)
            self.handle_auth_challenge(response)
            return
        match response.status_code // 100:
            case 1:  # trying/ringing
                return
            case 2:  # OK
                self._start_call(response)
        # SRTP offer rejected as "not acceptable" → fall back to plain RTP.
        if self.offer_srtp and response.status_code in (
            SIPStatus.NOT_ACCEPTABLE_HERE,  # 488
            SIPStatus.NOT_ACCEPTABLE_ANYWHERE,  # 606
            SIPStatus.UNSUPPORTED_MEDIA_TYPE,  # 415
        ):
            self.ack(response)
            self.retry_with_rtp(response)
            return
        self.ack(response)
        self.complete()

    def _start_call(self, response: Response) -> None:
        """Complete call setup after a 200 OK is received.

        Negotiates the codec from the remote SDP answer, creates the call
        handler, registers it with the RTP mux, updates the dialog.

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

        # The answer confirms SRTP only when it mirrors our `RTP/SAVP` offer
        # and supplies its own SDES `a=crypto:` key.  Our generated offer
        # session keys the outbound (send) media; the remote crypto keys the
        # inbound (recv) media.  A downgrade to `RTP/AVP` yields plain RTP.
        remote_crypto = (
            next(
                (
                    attr
                    for attr in remote_audio.attributes
                    if attr.name == "crypto" and attr.value
                ),
                None,
            )
            if remote_audio is not None
            else None
        )
        is_srtp = (
            remote_audio is not None
            and remote_audio.proto.startswith("RTP/SAVP")
            and remote_crypto is not None
            and self.srtp_offer is not None
        )
        srtp_send = self.srtp_offer if is_srtp else None
        srtp_recv = (
            SRTPSession.from_sdes(remote_crypto.value)  # type: ignore[arg-type]
            if is_srtp and remote_crypto is not None
            else None
        )

        if self.pending_call_class is not None:
            self.dialog.session = self.pending_call_class(
                rtp=self.sip.rtp,
                caller=CallerID(str(self.sip.aor)),
                media=negotiated_media,
                srtp=srtp_send,
                srtp_recv=srtp_recv,
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
            self.sip.rtp.register_call(remote_rtp_address, self.dialog.session)
            if remote_rtp_address is not None:
                self.sip.rtp.send(b"\x00", remote_rtp_address)

    def ack(self, response: Response) -> None:
        """Send an ACK after a terminal response to the INVITE.

        For 2xx responses the dialog is established (remote tag, parties, route
        set, remote target) and registered with the protocol; the ACK opens a
        fresh transaction with a new Via branch and is routed along the dialog
        route set (reversed `Record-Route`, RFC 3261 §12.1.2).

        For non-2xx final responses the ACK is the transactional ACK of the
        INVITE client transaction (RFC 3261 §17.1.1.3): it reuses the INVITE's
        Via header (same branch) and mirrors the INVITE's `Route` header
        values and Request-URI, so the proxy holding the INVITE server
        transaction matches and absorbs it (stopping retransmissions) rather
        than loose-routing it onward.  It must NOT derive routes from the
        response's `Record-Route` — that forms a dialog route set only on 2xx.
        """
        if is_success := response.status_code // 100 == 2:
            # RFC 3261 §12.1.2: UAC dialog route set is Record-Route reversed.
            routes = list(reversed(list(response.headers.getlist("Record-Route"))))
            self.dialog.remote_tag = response.remote_tag
            self.dialog.local_party = str(response.headers["From"])
            self.dialog.remote_party = str(response.headers["To"])
            self.dialog.outbound_cseq = self.cseq + 1
            self.dialog.route_set = routes
            self.dialog.remote_contact = response.headers.get("Contact") or str(
                self.request.uri
            )
            self.sip.register_dialog(self.dialog)
        else:
            # RFC 3261 §17.1.1.3: transactional ACK mirrors the INVITE's Route.
            routes = list(self.request.headers.getlist("Route"))
        self.sip.drop_transaction(self)

        # Non-2xx ACK: same Request-URI as the INVITE.  2xx ACK: the remote
        # target (Contact), falling back to the INVITE Request-URI.
        contact = response.headers.get("Contact")
        ack_uri = (
            contact.split(";")[0].strip("<>")
            if (is_success and contact)
            else str(self.request.uri)
        )
        if is_success:
            via = (
                f"SIP/2.0/{self.sip.aor.transport}"
                f" {self.sip.public_address};rport"
                f";branch={Transaction.branch_prefix}-{uuid.uuid4()}"
            )
        else:
            # Reuse the INVITE's Via (same branch + rport) for transaction
            # matching at the proxy (RFC 3261 §17.1.1.3).
            via = self.request.headers.getlist("Via")[0]
        ack_headers: SIPHeaderDict = SIPHeaderDict(
            {
                "Via": via,
                "Max-Forwards": "70",
                "From": response.headers["From"],
                "To": response.headers["To"],
                "Call-ID": self.dialog.call_id,
                "CSeq": f"{self.cseq} {SIPMethod.ACK}",
                "Content-Length": "0",
            }
        )
        for route in routes:
            ack_headers.add("Route", route)
        self.sip.send(
            Request(
                method=SIPMethod.ACK,
                uri=ack_uri,
                headers=ack_headers,
            )
        )


@dataclasses.dataclass(kw_only=True, slots=True)
class ByeTransaction(Transaction):
    """BYE transaction for terminating a dialog [RFC 3261 §15, §17.1.2].

    Use [send][voip.sip.transactions.ByeTransaction.send] to terminate a
    dialog from the local side, or
    [receive][voip.sip.transactions.ByeTransaction.receive] to handle a
    BYE sent by the remote party.

    [RFC 3261 §15]: https://datatracker.ietf.org/doc/html/rfc3261#section-15
    [RFC 3261 §17.1.2]: https://datatracker.ietf.org/doc/html/rfc3261#section-17.1.2
    """

    method: SIPMethod = SIPMethod.BYE

    @classmethod
    async def send(
        cls,
        *,
        sip: SessionInitiationProtocol,
        dialog: Dialog,
    ) -> Dialog:
        """Send a BYE request and wait for the 200 OK [RFC 3261 §15.1.1].

        Args:
            sip: The SIP session to send from.
            dialog: The dialog to terminate.

        Returns:
            The dialog once the BYE is acknowledged.

        [RFC 3261 §15.1.1]: https://datatracker.ietf.org/doc/html/rfc3261#section-15.1.1
        """
        cseq = dialog.outbound_cseq
        dialog.outbound_cseq += 1
        tx = cls(sip=sip, dialog=dialog, cseq=cseq)
        request_uri = str(dialog.remote_contact).strip("<>").split(";")[0]
        headers: SIPHeaderDict = SIPHeaderDict(
            {
                "Via": (
                    f"SIP/2.0/{sip.aor.transport}"
                    f' {sip.rtp.public_address.result()};oc-algo="loss";oc;rport;branch={tx.branch}'
                ),
                "Max-Forwards": "70",
                "From": dialog.local_party,
                "To": dialog.remote_party,
                "Call-ID": dialog.call_id,
                "CSeq": f"{cseq} {SIPMethod.BYE}",
                "Content-Length": "0",
            }
        )
        for route in dialog.route_set:
            headers.add("Route", route)
        tx.request = Request(method=SIPMethod.BYE, uri=request_uri, headers=headers)
        sip.register_transaction(tx)
        sip.send(tx.request)
        try:
            return await tx
        except asyncio.CancelledError:
            sip.drop_transaction(tx)
            raise

    @classmethod
    async def receive(
        cls,
        *,
        request: Request,
        sip: SessionInitiationProtocol,
    ) -> Dialog:
        """Handle an incoming BYE from the remote party [RFC 3261 §15.1.2].

        Sends 200 OK, removes the dialog, and notifies the application via
        [hangup_received][voip.sip.Dialog.hangup_received].

        Args:
            request: The incoming SIP BYE request.
            sip: The SIP session receiving the request.

        Returns:
            The terminated dialog.

        [RFC 3261 §15.1.2]: https://datatracker.ietf.org/doc/html/rfc3261#section-15.1.2
        """
        tx = cls.from_request(request=request, sip=sip)
        sip.drop_dialog(tx.dialog)
        tx.send_response(
            Response.from_request(
                request,
                dialog=tx.dialog,
                status_code=SIPStatus.OK,
                phrase=SIPStatus.OK.phrase,
            )
        )
        tx.dialog.hangup_received()
        tx.set_result(tx.dialog)
        return await tx

    def response_received(self, response: Response) -> None:
        """Handle the 200 OK for an outgoing BYE [RFC 3261 §15.1.1].

        Args:
            response: The parsed SIP response to our BYE request.
        """
        if response.status_code >= 200:
            self.sip.drop_transaction(self)
            self.complete()
            logger.debug(
                "BYE acknowledged: %s %s", response.status_code, response.phrase
            )
