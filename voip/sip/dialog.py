import asyncio
import dataclasses
import datetime
import logging
import socket
import typing
import uuid

from voip.sip import messages, transactions, types
from voip.sip.types import SipURI

if typing.TYPE_CHECKING:
    from voip.rtp import Session

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
            self.answer(session_class=MyCall)

    class MySession(SessionInitiationProtocol):
        dialog_class = MyDialog
    ```

    For outbound calls:

    ```python
    dialog = Dialog(sip=my_sip_session)
    await dialog.dial("sip:bob@biloxi.com", session_class=MyCall)
    ```


    [RFC 3261 §12]: https://datatracker.ietf.org/doc/html/rfc3261#section-12

    Args:
        sip: The parent protocol the session belongs to.
    """

    # https://datatracker.ietf.org/doc/html/rfc3261#section-17.1.1
    T1: typing.ClassVar[datetime.timedelta] = datetime.timedelta(milliseconds=500)

    # https://datatracker.ietf.org/doc/html/rfc3261#section-17.1.2
    BYE_ACK_TIMEOUT: typing.ClassVar[datetime.timedelta] = 64 * T1

    uac: SipURI = None
    call_id: str = dataclasses.field(
        default_factory=lambda: f"{uuid.uuid4()}@{socket.gethostname()}",
        compare=False,
    )
    local_tag: str = dataclasses.field(
        default_factory=lambda: str(uuid.uuid4()), compare=True
    )
    remote_tag: str | None = dataclasses.field(default=None, compare=True)
    remote_contact: SipURI | None = dataclasses.field(default=None, compare=True)
    route_set: list[SipURI] = dataclasses.field(default_factory=list)
    local_party: str | None = dataclasses.field(default=None, compare=False)
    remote_party: str | None = dataclasses.field(default=None, compare=False)
    outbound_cseq: int = dataclasses.field(default=1, compare=False)
    sip: transactions.SessionInitiationProtocol | None = dataclasses.field(
        default=None, compare=False, repr=False
    )
    invite_transaction: transactions.InviteTransaction | None = dataclasses.field(
        default=None, compare=False, repr=False
    )

    session: Session | None = dataclasses.field(default=None, init=False, compare=False)
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
        """
        Called when an INVITE is received from the remote party.

        Override in subclasses to [answer][voip.sip.Dialog.answer],
        [ring][voip.sip.Dialog.ringing],
        or [reject][voip.sip.Dialog.reject] the call.

        The base implementation rejects with a busy signal.
        """  # noqa: D401
        self.reject()

    def hangup_received(self) -> None:
        """
        Called when the remote party sends a BYE.

        Override in subclasses to perform teardown.
        """  # noqa: D401

    def ringing(self) -> None:
        """
        Send a ringing signal to the remote party.

        This is optional but recommended for good user experience.
        If not called, the caller will hear silence until the call is accepted or rejected.
        """
        if self.invite_transaction is not None:
            self.invite_transaction.ringing()

    def answer(
        self, *, session_class: type[Session], **session_kwargs: typing.Any
    ) -> None:
        """
        Accept the inbound call and start a multimedia session.

        Args:
            session_class: Session subclass to create for this call.
            **session_kwargs: Extra keyword arguments forwarded to `session_class`.
        """
        if self.invite_transaction is not None:
            self.invite_transaction.answer(
                session_class=session_class, **session_kwargs
            )

    def reject(self, status_code: types.SIPStatus = types.SIPStatus.BUSY_HERE) -> None:
        """
        Reject the inbound call with the given status code.

        Common status codes include:
            - [BUSY_HERE][voip.sip.types.SIPStatus.BUSY_HERE]: The remote party will hear a busy signal.
            - [DECLINE][voip.sip.types.SIPStatus.DECLINE]: The remote party will hear a decline signal.
            - [DOES_NOT_EXIST_ANYWHERE][voip.sip.types.SIPStatus.DOES_NOT_EXIST_ANYWHERE]: The remote party will hear a "The person you are trying to reach…" message.

        Args:
            status_code: SIP response status code (default: 486 Busy Here).
        """
        if self.invite_transaction is not None:
            self.invite_transaction.reject(status_code)

    async def bye(self) -> None:
        """End the call and terminate the dialog and multimedia session."""
        from voip.sip.transactions import ByeTransaction  # noqa: PLC0415

        try:
            await asyncio.wait_for(
                ByeTransaction.send(sip=self.sip, dialog=self),
                timeout=self.BYE_ACK_TIMEOUT.total_seconds(),
            )
        except TimeoutError:
            logger.warning(
                "BYE for dialog %s was not acknowledged within %r",
                self.call_id,
                self.BYE_ACK_TIMEOUT,
            )
        self.sip.drop_dialog(self)

    async def dial(
        self,
        target: SipURI,
        *,
        session_class: type[Session],
        **session_kwargs: typing.Any,
    ) -> None:
        """
        Initiate an outbound call to *target*.

        Args:
            target: SIP or tel URI of the remote party (e.g. ``"sip:+15551234567@carrier.com"`` or ``"tel:+15551234567"``).
            session_class: Session subclass to create for this call.
            **session_kwargs: Extra keyword arguments forwarded to `session_class`.

        [RFC 3261 §13.1]: https://datatracker.ietf.org/doc/html/rfc3261#section-13.1
        """
        from voip.sip.transactions import InviteTransaction  # noqa: PLC0415

        await InviteTransaction.send(
            sip=self.sip,
            target=target,
            dialog=self,
            session_class=session_class,
            **session_kwargs,
        )

    @classmethod
    def from_request(cls, request: messages.Request, **kwargs) -> Dialog:
        """Create a dialog from a request, extracting relevant headers."""
        return cls(
            call_id=request.headers["Call-ID"],
            local_tag=request.remote_tag or str(uuid.uuid4()),
            remote_tag=request.local_tag,
            remote_contact=request.headers.get("Contact"),
            **kwargs,
        )
