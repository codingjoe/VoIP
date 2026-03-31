import asyncio
import dataclasses
import datetime
import logging
import socket
import typing
import uuid

from voip.sip import messages, transactions, types
from voip.sip.types import SipUri

logger = logging.getLogger("voip.sip")


@dataclasses.dataclass(kw_only=True, slots=True)
class Dialog:
    """Peer-to-peer SIP relationship between two user agents [RFC 3261 §12].

    Subclass `Dialog` to implement call handling.  Set the subclass as
    `dialog_class` on the SIP session for inbound calls:

    ```python
    class MyDialog(Dialog):
        def call_received(self) -> None:
            self.ringing()
            self.accept(call_class=MyCall)

    class MySession(SessionInitiationProtocol):
        dialog_class = MyDialog
    ```

    For outbound calls:

    ```python
    dialog = Dialog(sip=my_sip_session)
    await dialog.dial("sip:bob@biloxi.com", call_class=MyCall)
    ```

    [RFC 3261 §12]: https://datatracker.ietf.org/doc/html/rfc3261#section-12
    """

    BYE_ACK_TIMEOUT: typing.ClassVar[float] = 32.0
    """Seconds to wait for a 200 OK after sending BYE (64×T1, [RFC 3261 §17.1.2]).

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
        """Called when an inbound INVITE arrives.

        Override in subclasses to accept, ring, or reject the call using
        [`accept`][voip.sip.dialog.Dialog.accept],
        [`ringing`][voip.sip.dialog.Dialog.ringing], and
        [`reject`][voip.sip.dialog.Dialog.reject].
        The base implementation rejects with 486 Busy Here.
        """
        self.reject()

    def hangup_received(self) -> None:
        """Called when the remote party sends a BYE.

        Override in subclasses to perform teardown.
        """

    def ringing(self) -> None:
        """Send a 180 Ringing provisional response [RFC 3261 §21.1.2].

        [RFC 3261 §21.1.2]: https://datatracker.ietf.org/doc/html/rfc3261#section-21.1.2
        """
        if self.invite_tx is not None:
            self.invite_tx.ringing()

    def accept(self, *, call_class: type, **call_kwargs: typing.Any) -> None:
        """Accept the inbound call and answer with 200 OK.

        Args:
            call_class: Session subclass to create for this call.
            **call_kwargs: Extra keyword arguments forwarded to `call_class`.
        """
        if self.invite_tx is not None:
            self.invite_tx.answer(call_class=call_class, **call_kwargs)

    def reject(self, status_code: types.SIPStatus = types.SIPStatus.BUSY_HERE) -> None:
        """Reject the inbound call.

        Args:
            status_code: SIP response status code (default: 486 Busy Here).
        """
        if self.invite_tx is not None:
            self.invite_tx.reject(status_code)

    async def bye(self) -> None:
        """Terminate the dialog by sending a SIP BYE request [RFC 3261 §15].

        [RFC 3261 §15]: https://datatracker.ietf.org/doc/html/rfc3261#section-15
        """
        from voip.sip.transactions import ByeTransaction  # noqa: PLC0415
        from voip.sip.types import SIPMethod  # noqa: PLC0415

        tx = ByeTransaction(sip=self.sip, method=SIPMethod.BYE, dialog=self)
        try:
            await asyncio.wait_for(tx.wait(), timeout=self.BYE_ACK_TIMEOUT)
        except TimeoutError:
            logger.warning(
                "BYE for dialog %s was not acknowledged within %.0f s",
                self.call_id,
                self.BYE_ACK_TIMEOUT,
            )
        self.sip.dialogs.pop((self.remote_tag, self.local_tag), None)

    async def dial(
        self,
        target: str,
        *,
        call_class: type,
        **call_kwargs: typing.Any,
    ) -> None:
        """Initiate an outbound call to *target* [RFC 3261 §13.1].

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
