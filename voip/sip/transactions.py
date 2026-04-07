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
    "RegistrationTransaction",
]


@dataclasses.dataclass(kw_only=True, slots=True)
class Transaction(asyncio.Future):
    """
    Initiated by a request, completed by any number of responses.

    Transactions are awaitable: ``await tx`` suspends until the transaction
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
                    response.headers.get(challenge_key, "")
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
                self.sip.register_transaction(tx)
                tx.add_done_callback(self.forward_result)
            case _:
                raise NotImplementedError(
                    f"Unknown SIP status code: {response.status_code}"
                )

    def forward_result(self, fut: asyncio.Future) -> None:
        """Forward the result of *fut* to this transaction (used for auth retry chaining)."""
        if not self.done():
            if fut.cancelled():
                self.cancel()
            elif exc := fut.exception():
                self.set_exception(exc)
            else:
                self.set_result(fut.result())

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

        use_srtp = negotiated_media.proto == "RTP/SAVP"
        srtp_session = SRTPSession.generate() if use_srtp else None

        self.dialog.local_party = (
            f"{self.request.headers['To']};tag={self.dialog.local_tag}"
        )
        self.dialog.remote_party = str(self.request.headers["From"])
        self.dialog.route_set = list(self.request.headers.getlist("Record-Route"))
        self.sip.register_dialog(self.dialog)

        call_handler = session_class(
            rtp=self.sip.rtp,
            caller=caller,
            media=negotiated_media,
            srtp=srtp_session,
            dialog=self.dialog,
            **session_kwargs,
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
        rtp_public = self.sip.public_address
        sdp_media_attributes = [Attribute(name="sendrecv")]
        if srtp_session is not None:
            sdp_media_attributes.append(
                Attribute(name="crypto", value=srtp_session.sdes_attribute)
            )
        self.send_response(
            Response.from_request(
                request=self.request,
                dialog=self.dialog,
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

        Args:
            sip: The SIP session to send from.
            target: SIP or tel URI of the callee (e.g. ``"sip:+15551234567@carrier.com"`` or ``"tel:+15551234567"``).
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

        rtp_public = sip.rtp.public_address
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
                    fmt=session_class.sdp_formats(),
                    attributes=[Attribute(name="sendrecv")],
                )
            ],
        )
        tx.request = Request(
            method=SIPMethod.INVITE,
            uri=target,
            headers={
                "Max-Forwards": "70",
                **tx.headers,
                "From": dialog.from_header,
                "To": str(target),
                "Contact": sip.contact,
                "Call-ID": dialog.call_id,
                "Route": f"<sip:{str(rtp_public[0])}:5060;transport=tcp;lr>",
                "Allow": sip.allow_header,
                "Content-Type": "application/sdp",
            },
            body=sdp_offer,
        )
        sip.register_transaction(tx)
        sip.send(tx.request)
        try:
            return await tx
        except asyncio.CancelledError:
            sip.drop_transaction(tx)
            raise

    def response_received(self, response: Response) -> None:
        """Dispatch responses to an outbound INVITE."""
        match response.status_code // 100:
            case 1:  # trying/ringing
                return
            case 2:  # OK
                self._start_call(response)
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

        if self.pending_call_class is not None:
            call_handler = self.pending_call_class(
                rtp=self.sip.rtp,
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

    def ack(self, response: Response) -> None:
        """Send an ACK after receiving a terminal response.

        For 2xx responses, establishes the dialog and registers it with the
        protocol.
        """
        if response.status_code // 100 == 2:
            self.dialog.remote_tag = response.remote_tag
            self.dialog.local_party = str(response.headers["From"])
            self.dialog.remote_party = str(response.headers["To"])
            self.dialog.outbound_cseq = self.cseq + 1
            # RFC 3261 §12.1.2: UAC route set is Record-Route in reverse order.
            self.dialog.route_set = list(
                reversed(list(response.headers.getlist("Record-Route")))
            )
            self.sip.register_dialog(self.dialog)
        self.sip.drop_transaction(self)

        ack_branch = f"{Transaction.branch_prefix}-{uuid.uuid4()}"
        contact = response.headers.get("Contact")
        ack_uri = (
            contact.split(";")[0].strip("<>") if contact else str(self.request.uri)
        )
        self.dialog.remote_contact = ack_uri
        ack_headers: SIPHeaderDict = SIPHeaderDict(
            {
                "Via": (
                    f"SIP/2.0/{self.sip.aor.transport}"
                    f" {self.sip.public_address};rport;branch={ack_branch};alias"
                ),
                "Max-Forwards": "70",
                "From": response.headers["From"],
                "To": response.headers["To"],
                "Call-ID": self.dialog.call_id,
                "CSeq": f"{self.cseq} {SIPMethod.ACK}",
                "Content-Length": "0",
            }
        )
        for route in self.dialog.route_set:
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
                    f' {sip.rtp.public_address};oc-algo="loss";oc;rport;branch={tx.branch}'
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
