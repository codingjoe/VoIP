import asyncio
import dataclasses
import datetime
import logging
import socket
import typing
import uuid

import voip
from voip.sip import messages, transactions, types
from voip.sip.types import SipUri

logger = logging.getLogger("voip.sip")


@dataclasses.dataclass(kw_only=True, slots=True)
class Dialog:
    """
    Peer-to-peer SIP relationship between two user agents.

    A dialog is identified by the tuple of (Call-ID, From tag, To tag) and
    established by a non-final response to the INVITE, see also: [RFC 3261 §12].

    Subclass `Dialog` to implement inbound call handling.  Override
    [`call_received`][voip.sip.messages.Dialog.call_received] and call
    [`accept`][voip.sip.messages.Dialog.accept] or
    [`reject`][voip.sip.messages.Dialog.reject] from within it.  Register the
    subclass as `dialog_class` on the SIP session:

    ```python
    class MyDialog(Dialog):
        def call_received(self) -> None:
            self.ringing()
            self.accept(call_class=MyCall)

    class MySession(SessionInitiationProtocol):
        dialog_class = MyDialog
    ```

    For outbound calls, create a `Dialog` with the SIP session set and call
    [`dial`][voip.sip.messages.Dialog.dial]:

    ```python
    dialog = Dialog(sip=my_sip_session)
    await dialog.dial("sip:bob@biloxi.com", call_class=MyCall)
    ```

    [RFC 3261 §12]: https://datatracker.ietf.org/doc/html/rfc3261#section-12

    Args:
        uac: The user agent that initiated the dialog.
        call_id: The Call-ID header value for this dialog.
        local_tag: The From-header tag parameter value for this dialog.
        remote_tag: The To-header tag parameter value for this dialog.
        local_party: Raw ``From:`` header value (URI + tag) to use in
            outbound in-dialog requests such as BYE.  Populated by the
            transaction layer when the dialog is confirmed.
        remote_party: Raw ``To:`` header value (URI + tag) to use in
            outbound in-dialog requests such as BYE.  Populated by the
            transaction layer when the dialog is confirmed.
        outbound_cseq: CSeq sequence number for the *next* outbound
            in-dialog request.  Defaults to ``1`` for the UAS side
            (no prior outbound request) and is set to ``cseq + 1`` on the
            UAC side after the INVITE is confirmed.
        sip: The SIP session that owns this dialog.  Set by the transaction
            layer when the dialog is confirmed.
        invite_tx: The [`InviteTransaction`][voip.sip.transactions.InviteTransaction]
            for an inbound INVITE.  Set before
            [`call_received`][voip.sip.messages.Dialog.call_received] is called
            so that [`accept`][voip.sip.messages.Dialog.accept],
            [`reject`][voip.sip.messages.Dialog.reject], and
            [`ringing`][voip.sip.messages.Dialog.ringing] can delegate to it.
    """

    BYE_ACK_TIMEOUT: typing.ClassVar[float] = 32.0
    """Seconds to wait for a 200 OK from the remote party after sending BYE.

    Defaults to 64×T1 = 32 s — the standard non-INVITE transaction timeout
    from [RFC 3261 §17.1.2].  The timeout lives on `Dialog` (rather than on
    [`ByeTransaction`][voip.sip.transactions.ByeTransaction]) so that
    application subclasses can configure it in one place alongside the other
    call lifecycle hooks.  Override in subclasses to change the timeout.

    [RFC 3261 §17.1.2]: https://datatracker.ietf.org/doc/html/rfc3261#section-17.1.2
    """

    uac: SipUri | None = None
    call_id: str = dataclasses.field(
        default_factory=lambda: f"{uuid.uuid4()}@{socket.gethostname()}",
        compare=False,
    )
    local_tag: str = dataclasses.field(
        default_factory=lambda: str(uuid.uuid4()), compare=True
    )
    remote_tag: str | None = dataclasses.field(default=None, compare=True)
    remote_contact: SipUri | None = dataclasses.field(default=None, compare=True)
    route_set: list[SipUri] = dataclasses.field(default_factory=list)
    local_party: str | None = dataclasses.field(default=None, compare=False)
    remote_party: str | None = dataclasses.field(default=None, compare=False)
    outbound_cseq: int = dataclasses.field(default=1, compare=False)
    sip: transactions.SessionInitiationProtocol | None = dataclasses.field(
        default=None, compare=False, repr=False
    )
    invite_tx: transactions.InviteTransaction | None = dataclasses.field(
        default=None, compare=False, repr=False
    )

    created: datetime.datetime = dataclasses.field(
        init=False, default_factory=datetime.datetime.now
    )

    @property
    def from_header(self) -> str:
        """The logical sender of a request."""
        return f"{self.uac.scheme}:{self.uac.user}@{self.uac.host};tag={self.local_tag}"

    @property
    def to_header(self) -> str:
        """The logical recipient of a request."""
        part = f"{self.uac.scheme}:{self.uac.user}@{self.uac.host}:{self.uac.port};transport={self.uac.parameters.get('transport', 'TLS')}"
        if self.remote_tag:
            part += f";tag={self.remote_tag}"
        return part

    @property
    def headers(self) -> dict[str, str]:
        """Return a dict of headers for this dialog."""
        return {
            "From": self.from_header,
            "To": self.to_header,
            "Call-ID": self.call_id,
        }

    def call_received(self) -> None:
        """Handle an incoming INVITE.

        Called by the SIP layer after the dialog is created from the INVITE
        request.  The base implementation rejects the call with ``486 Busy
        Here``.  Override in subclasses to answer, ring, or reject the call
        using [`accept`][voip.sip.messages.Dialog.accept],
        [`ringing`][voip.sip.messages.Dialog.ringing], and
        [`reject`][voip.sip.messages.Dialog.reject].
        """
        self.reject()

    def hangup_received(self) -> None:
        """Handle an inbound BYE (remote party hanging up).

        Called by the SIP layer after the 200 OK response has been sent for
        the BYE.  The base implementation is a no-op.  Override in subclasses
        to perform teardown, e.g. closing the SIP transport for single-shot
        outbound sessions.
        """

    def ringing(self) -> None:
        """Send a 180 Ringing provisional response [RFC 3261 §21.1.2].

        Delegates to the [`InviteTransaction`][voip.sip.transactions.InviteTransaction]
        set on [`invite_tx`][voip.sip.messages.Dialog.invite_tx].

        [RFC 3261 §21.1.2]: https://datatracker.ietf.org/doc/html/rfc3261#section-21.1.2
        """
        if self.invite_tx is not None:
            self.invite_tx.ringing()

    def accept(self, *, call_class: type, **call_kwargs: typing.Any) -> None:
        """Accept the inbound call by answering with 200 OK and SDP.

        Delegates to
        [`InviteTransaction.answer`][voip.sip.transactions.InviteTransaction.answer].

        Args:
            call_class: Session subclass to create for this call.
            **call_kwargs: Extra keyword arguments forwarded to `call_class`.
        """
        if self.invite_tx is not None:
            self.invite_tx.answer(call_class=call_class, **call_kwargs)

    def reject(self, status_code: types.SIPStatus = types.SIPStatus.BUSY_HERE) -> None:
        """Reject the inbound call.

        Delegates to
        [`InviteTransaction.reject`][voip.sip.transactions.InviteTransaction.reject].

        Args:
            status_code: SIP response status code (default: 486 Busy Here).
        """
        if self.invite_tx is not None:
            self.invite_tx.reject(status_code)

    async def bye(self) -> None:
        """Terminate the dialog by sending a SIP BYE request [RFC 3261 §15].

        Constructs and sends a BYE request, removes this dialog from the SIP
        session's registry, and awaits the remote party's 200 OK
        acknowledgment.  The standard non-INVITE transaction timeout of
        [`BYE_ACK_TIMEOUT`][voip.sip.messages.Dialog.BYE_ACK_TIMEOUT] seconds
        applies; a warning is logged if no acknowledgment arrives in time.

        This is a no-op when [`sip`][voip.sip.messages.Dialog.sip] is not set,
        or when [`local_party`][voip.sip.messages.Dialog.local_party],
        [`remote_party`][voip.sip.messages.Dialog.remote_party], or
        [`remote_contact`][voip.sip.messages.Dialog.remote_contact] are
        missing (call not yet fully established).

        [RFC 3261 §15]: https://datatracker.ietf.org/doc/html/rfc3261#section-15
        """
        if self.sip is None:
            return
        if self.local_party is None or self.remote_party is None:
            logger.warning(
                "Cannot BYE dialog %s: local or remote party not set",
                self.call_id,
            )
            return
        if self.remote_contact is None:
            logger.warning(
                "Cannot BYE dialog %s: remote contact not known",
                self.call_id,
            )
            return

        from voip.sip.transactions import ByeTransaction  # noqa: PLC0415
        from voip.sip.types import SIPMethod  # noqa: PLC0415

        request_uri = str(self.remote_contact).strip("<>").split(";")[0]
        tx = ByeTransaction(
            sip=self.sip,
            method=SIPMethod.BYE,
            cseq=self.outbound_cseq,
            dialog=self,
        )
        bye_request = messages.Request(
            method=SIPMethod.BYE,
            uri=request_uri,
            headers={
                "Via": (
                    f"SIP/2.0/{self.sip.aor.transport}"
                    f' {self.sip.rtp.public_address};oc-algo="loss";oc;rport;branch={tx.branch}'
                ),
                "Max-Forwards": "70",
                "From": self.local_party,
                "To": self.remote_party,
                "Call-ID": self.call_id,
                "CSeq": f"{self.outbound_cseq} {SIPMethod.BYE}",
                "Route": self.route_set[0] if self.route_set else None,
                "User-Agent": f"python/vuoip/{voip.__version__}",
                "Content-Length": "0",
            },
        )
        self.sip.transactions[tx.branch] = tx
        self.sip.send(bye_request)
        self.outbound_cseq += 1
        self.sip.dialogs.pop((self.remote_tag, self.local_tag), None)
        try:
            await asyncio.wait_for(tx, timeout=self.BYE_ACK_TIMEOUT)
        except TimeoutError:
            logger.warning(
                "BYE for dialog %s was not acknowledged within %.0f s",
                self.call_id,
                self.BYE_ACK_TIMEOUT,
            )

    async def dial(
        self,
        target: str,
        *,
        call_class: type,
        **call_kwargs: typing.Any,
    ) -> None:
        """Initiate an outbound call to *target* [RFC 3261 §13.1].

        Requires [`sip`][voip.sip.messages.Dialog.sip] to be set.  Sets
        [`uac`][voip.sip.messages.Dialog.uac] from the SIP session's AOR when
        not already provided.

        Args:
            target: SIP URI of the callee (e.g. ``"sip:+15551234567@carrier.com"``).
            call_class: Session subclass to create for this call.
            **call_kwargs: Extra keyword arguments forwarded to `call_class`.

        [RFC 3261 §13.1]: https://datatracker.ietf.org/doc/html/rfc3261#section-13.1
        """
        from voip.sip.transactions import InviteTransaction  # noqa: PLC0415
        from voip.sip.types import SIPMethod  # noqa: PLC0415

        if self.uac is None and self.sip is not None:
            self.uac = self.sip.aor
        tx = InviteTransaction(
            sip=self.sip,
            method=SIPMethod.INVITE,
            cseq=1,
            dialog=self,
        )
        await tx.make_call(target, call_class=call_class, **call_kwargs)

    @classmethod
    def from_request(cls, request: messages.Request) -> Dialog:
        """Create a dialog from a request, extracting relevant headers."""
        return cls(
            call_id=request.headers["Call-ID"],
            local_tag=request.local_tag,
            remote_tag=request.remote_tag or str(uuid.uuid4()),
            remote_contact=request.headers.get("Contact"),
        )
