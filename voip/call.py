"""Call handler hierarchy for RTP/SIP sessions.

This module provides the :class:`Call` base dataclass that represents an
individual call leg managed by the RTP multiplexer.  Audio-specific
subclasses live in :mod:`voip.audio`.

Relationship to the rest of the stack::

    SessionInitiationProtocol   (SIP signalling)
            │
            │  creates and registers
            ▼
    RealtimeTransportProtocol   (shared UDP socket / mux)
            │
            │  routes packets to
            ▼
         Call                   (one per active call leg)
            │
         AudioCall  (audio.py)  (audio buffering + codec decode)
            │
         WhisperCall (audio.py) (speech-to-text via Whisper)
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from voip.rtp import RealtimeTransportProtocol
from voip.sdp.types import MediaDescription
from voip.sip.types import CallerID
from voip.srtp import SRTPSession

if TYPE_CHECKING:
    from voip.sip.protocol import SessionInitiationProtocol

__all__ = ["Call"]


@dataclasses.dataclass
class Call:
    """Handle basic IO and call functions.

    A call handler is associated with one SIP dialog and receives RTP traffic
    delivered by the shared :class:`~voip.rtp.RealtimeTransportProtocol`
    multiplexer.  Subclass and override :meth:`datagram_received` to process
    incoming media.

    The :attr:`rtp` and :attr:`sip` back-references allow the handler to send
    data back to the caller and to terminate the call via SIP BYE.

    Subclass :class:`~voip.audio.AudioCall` for audio calls with codec
    negotiation, buffering, and decoding.

    Attributes:
        rtp: Shared RTP multiplexer socket that delivers packets to this handler.
        sip: SIP session that answered this call (used for BYE etc.).
        caller: Caller identifier as received in the SIP From header.
        media: Negotiated SDP media description for this call leg.
    """

    rtp: RealtimeTransportProtocol
    sip: SessionInitiationProtocol
    #: Caller identifier as received in the SIP From header.
    caller: CallerID = dataclasses.field(default_factory=lambda: CallerID(""))
    #: Negotiated SDP media description for this call leg.
    media: MediaDescription | None = None
    #: SRTP session for encrypting and decrypting media (set by the SIP layer).
    srtp: SRTPSession | None = None

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Handle a raw RTP datagram.  Override in subclasses to process media."""

    def send_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        """Send a datagram through the shared RTP socket.

        Encrypts the datagram with the call's SRTP session when one is set.

        Args:
            data: Raw RTP bytes to send.
            addr: Destination ``(host, port)``.
        """
        if self.srtp is not None:
            data = self.srtp.encrypt(data)
        self.rtp.send(data, addr)

    async def hang_up(self) -> None:
        """Terminate the call by sending a SIP BYE request.

        Raises:
            NotImplementedError: Not yet implemented; the call_id and remote
                SIP address need to be stored per call to make this work.
        """
        raise NotImplementedError("hang_up is not yet implemented")

    @classmethod
    def negotiate_codec(cls, remote_media: MediaDescription) -> MediaDescription:
        """Negotiate a media codec from the remote SDP offer.

        Override in subclasses to implement codec selection.  The SIP layer
        calls this before sending a 200 OK; if the method raises the exception
        propagates and the call is not answered.

        Args:
            remote_media: The SDP ``m=audio`` section from the remote INVITE.

        Returns:
            A :class:`~voip.sdp.types.MediaDescription` with the chosen codec.

        Raises:
            NotImplementedError: When not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement negotiate_codec. "
            "Override this classmethod in a subclass (e.g. AudioCall) to "
            "support codec negotiation."
        )
