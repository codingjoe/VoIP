"""T.38 FAX over SIP/UDPTL session.

Implements SIP signaling and UDPTL media transport for sending and receiving
fax documents over IP using the T.38 protocol.

[RFC 3362]: https://datatracker.ietf.org/doc/html/rfc3362
"""

import asyncio
import dataclasses
import logging
from typing import ClassVar

from voip.rtp import Session
from voip.sdp.types import Attribute, MediaDescription, RTPPayloadFormat
from voip.types import NetworkAddress

__all__ = ["FaxSession", "OutboundFaxSession", "InboundFaxSession"]

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class FaxSession(Session):
    """T.38 FAX over SIP/UDPTL session [RFC 3362].

    Attributes:
        T38_VERSION: T.38 protocol version advertised in SDP.
        T38_MAX_BIT_RATE: Maximum fax bit rate in bits per second.

    [RFC 3362]: https://datatracker.ietf.org/doc/html/rfc3362
    """

    media_type: ClassVar[str] = "image"
    T38_VERSION: ClassVar[int] = 0
    T38_MAX_BIT_RATE: ClassVar[int] = 14400

    def data_received(self, data: bytes, address: NetworkAddress) -> None:
        self.document_received(data)

    def document_received(self, data: bytes) -> None:
        """Handle received FAX document data.

        Override in subclasses to process the received T.38 UDPTL data.

        Args:
            data: Raw T.38 UDPTL data.
        """

    def send_document(self, data: bytes) -> None:
        """Send a fax document as T.38 UDPTL data.

        Args:
            data: Raw document data to send.
        """
        if remote_address := next(
            (address for address, call in self.rtp.calls.items() if call is self),
            None,
        ):
            self.rtp.send(data, remote_address)
        else:
            logger.warning("No remote address for FAX call; dropping document data")

    @classmethod
    def negotiate_codec(cls, remote_media: MediaDescription) -> MediaDescription:
        """Negotiate T.38 from a remote SDP `m=image` offer.

        Args:
            remote_media: The SDP `m=image` section from the remote INVITE.

        Returns:
            A T.38 media description for the response SDP.

        Raises:
            NotImplementedError: When the remote offer does not include T.38.
        """
        if any(str(fmt.payload_type).lower() == "t38" for fmt in remote_media.fmt):
            return cls.sdp_media_description(port=remote_media.port)
        raise NotImplementedError("Remote SDP offer does not include T.38")

    @classmethod
    def sdp_formats(cls) -> list[RTPPayloadFormat]:
        return [RTPPayloadFormat(payload_type="t38")]

    @classmethod
    def sdp_media_description(cls, port: int = 0) -> MediaDescription:
        return MediaDescription(
            media="image",
            port=port,
            proto="udptl",
            fmt=[RTPPayloadFormat(payload_type="t38")],
            attributes=[
                Attribute(name="T38FaxVersion", value=str(cls.T38_VERSION)),
                Attribute(name="T38MaxBitRate", value=str(cls.T38_MAX_BIT_RATE)),
                Attribute(name="T38FaxRateManagement", value="transferredTCF"),
            ],
        )


@dataclasses.dataclass(kw_only=True, slots=True)
class OutboundFaxSession(FaxSession):
    """Dial a number, send a FAX document, and hang up.

    Attributes:
        document: Raw document bytes to transmit as a T.38 FAX.
    """

    document: bytes

    def __post_init__(self) -> None:
        asyncio.create_task(self.transmit())

    async def transmit(self) -> None:
        """Send the document and hang up when transmission completes."""
        self.send_document(self.document)
        await self.hang_up()
        if self.dialog is not None and self.dialog.sip is not None:
            self.dialog.sip.close()


@dataclasses.dataclass(kw_only=True, slots=True)
class InboundFaxSession(FaxSession):
    """Collect incoming T.38 UDPTL packets into a single document buffer.

    Attributes:
        document: Accumulated T.38 UDPTL data received so far.
    """

    document: bytes = dataclasses.field(default=b"", init=False)

    def document_received(self, data: bytes) -> None:
        self.document += data
