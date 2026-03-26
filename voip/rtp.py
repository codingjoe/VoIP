"""
Real-time Transport Protocol (RTP) implementation of [RFC 3550].

[RFC 3550]: https://datatracker.ietf.org/doc/html/rfc3550#section-5
"""

import asyncio
import dataclasses
import enum
import json
import logging
import struct
import typing
from typing import TYPE_CHECKING

from voip.sdp.types import MediaDescription
from voip.srtp import SRTPSession
from voip.stun import STUNProtocol
from voip.types import ByteSerializableObject, NetworkAddress

if TYPE_CHECKING:
    from voip.sip.protocol import SessionInitiationProtocol
    from voip.sip.types import CallerID

__all__ = ["RTP", "Session", "RTPPacket", "RTPPayloadType", "RealtimeTransportProtocol"]

logger = logging.getLogger(__name__)


class RTPPayloadType(enum.IntEnum):
    """Common RTP payload types, aligned with SDP media format identifiers.

    Static payload types (0–95) are defined by RFC 3551.
    Dynamic payload types (96–127) are negotiated via SDP.
    Opus uses payload type 111 per RFC 7587.
    """

    PCMU = 0  # G.711 µ-law (RFC 3551)
    PCMA = 8  # G.711 A-law (RFC 3551)
    G722 = 9  # G.722 (RFC 3551)
    OPUS = 111  # RFC 7587 (dynamic)


@dataclasses.dataclass(frozen=True, slots=True)
class RTPPacket(ByteSerializableObject):
    """
    RTP data packet [RFC 3550 §5.1].

    [RFC 3550 §5.1]: https://datatracker.ietf.org/doc/html/rfc3550#section-5.1
    """

    payload_type: RTPPayloadType | int
    sequence_number: int
    timestamp: int
    ssrc: int
    payload: bytes
    header_size: typing.ClassVar[int] = 12

    @classmethod
    def parse(cls, data: bytes) -> RTPPacket:
        if len(data) < cls.header_size:
            raise ValueError(f"RTP packet too short: {len(data)} bytes")
        payload_type = data[1] & 0x7F
        sequence_number = (data[2] << 8) | data[3]
        timestamp = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
        ssrc = (data[8] << 24) | (data[9] << 16) | (data[10] << 8) | data[11]
        return cls(
            payload_type=payload_type,
            sequence_number=sequence_number,
            timestamp=timestamp,
            ssrc=ssrc,
            payload=data[cls.header_size :],
        )

    def __bytes__(self) -> bytes:
        return (
            struct.pack(
                ">BBHII",
                0x80,  # V=2, P=0, X=0, CC=0
                self.payload_type,
                self.sequence_number & 0xFFFF,
                self.timestamp & 0xFFFFFFFF,
                self.ssrc,
            )
            + self.payload
        )


@dataclasses.dataclass
class Session:
    """One call leg managed by the RTP multiplexer.

    Associates a SIP dialog with the `RealtimeTransportProtocol` media
    stream. Subclass and override `packet_received` to process incoming
    media, and use `send_packet` to transmit outbound media.

    The `rtp` and `sip` back-references allow the handler to send data
    back to the caller and to terminate the call via SIP BYE.

    Subclass `voip.audio.AudioCall` for audio calls with codec
    negotiation, buffering, and decoding.

    Attributes:
        rtp: Shared RTP multiplexer socket that delivers packets to this handler.
        sip: SIP session that answered this call (used for BYE etc.).
        caller: Caller identifier as received in the SIP From header.
        media: Negotiated SDP media description for this call leg.
        srtp: Optional SRTP session for encrypting and decrypting media.
    """

    rtp: RealtimeTransportProtocol
    sip: SessionInitiationProtocol
    media: MediaDescription
    caller: CallerID
    srtp: SRTPSession | None = None

    def packet_received(self, packet: RTPPacket, addr: NetworkAddress) -> None:
        """Handle a parsed RTP packet. Override in subclasses to process media.

        Args:
            packet: Parsed RTP packet.
            addr: Remote ``(host, port)`` the packet arrived from.
        """

    def send_packet(self, packet: RTPPacket, addr: NetworkAddress) -> None:
        """Serialize *packet* and send it via the shared RTP socket.

        Encrypts the packet with the call's SRTP session when one is set.

        Args:
            packet: RTP packet to send.
            addr: Destination ``(host, port)``.
        """
        data = bytes(packet)
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

        Override in subclasses to implement codec selection. The SIP layer
        calls this before sending a 200 OK; if the method raises the exception
        propagates and the call is not answered.

        Args:
            remote_media: The SDP ``m=audio`` section from the remote INVITE.

        Returns:
            A `MediaDescription` with the chosen codec.

        Raises:
            NotImplementedError: When not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement negotiate_codec. "
            "Override this classmethod in a subclass (e.g. AudioCall) to "
            "support codec negotiation."
        )


@dataclasses.dataclass(kw_only=True, slots=True)
class RealtimeTransportProtocol(STUNProtocol):
    """RTP multiplexer: routes incoming datagrams to per-call handlers (RFC 3550).

    One instance manages multiple simultaneous calls on a single UDP socket.
    Register per-call `Call` handlers with
    `register_call`; each incoming datagram is dispatched to the
    matching handler's `datagram_received` method by
    remote source address.

    Use ``addr=None`` in `register_call` as a wildcard catch-all for
    calls whose remote RTP address is not known in advance (no SDP in INVITE).
    """

    rtp_header_size: typing.ClassVar[int] = 12
    calls: dict[tuple[str, int] | None, Session] = dataclasses.field(
        init=False, default_factory=dict
    )
    public_address: NetworkAddress | None = dataclasses.field(init=False, default=None)

    def stun_connection_made(
        self,
        transport: asyncio.DatagramTransport,
        addr: NetworkAddress,
    ) -> None:
        self.public_address = addr

    def register_call(
        self,
        addr: NetworkAddress | None,
        handler: Session,
    ) -> None:
        """Register *handler* for RTP traffic arriving from *addr*.

        Use ``addr=None`` as a wildcard to handle traffic from any source that
        has no dedicated routing entry (useful when the caller's RTP address is
        not known in advance from the INVITE SDP).

        Args:
            addr: Remote ``(ip, port)`` as it will appear in incoming datagrams,
                or ``None`` to register a wildcard catch-all handler.
            handler: A `Call` instance whose
                `datagram_received` will be called for
                matching packets.
        """
        logger.info(
            json.dumps(
                {
                    "event": "rtp_call_registered",
                    "addr": list(addr) if addr else None,
                    "handler": type(handler).__name__,
                }
            ),
            extra={"addr": addr},
        )
        self.calls[addr] = handler

    def unregister_call(self, addr: NetworkAddress | None) -> None:
        """Remove the handler registered for *addr*.

        Args:
            addr: The same key that was passed to `register_call`.
                Silently ignored when no handler is registered for *addr*.
        """
        if addr in self.calls:
            logger.info(
                json.dumps(
                    {
                        "event": "rtp_call_unregistered",
                        "addr": list(addr) if addr else None,
                    }
                ),
                extra={"addr": addr},
            )
            self.calls.pop(addr)

    def packet_received(self, data: bytes, addr: NetworkAddress) -> None:
        """Route an incoming SRTP datagram to the matching per-call handler.

        Looks up *addr* in the call registry.  Falls back to the wildcard
        ``None`` handler when no exact match exists.  Drops the packet with a
        debug log when no handler is registered at all.

        When the matched handler carries an SRTP session the packet is
        authenticated and decrypted before being forwarded; packets that fail
        authentication are logged at WARNING level and discarded.
        """
        handler = self.calls.get(addr)
        if handler is None:
            handler = self.calls.get(None)
        if handler is not None:
            if handler.srtp is not None:
                decrypted = handler.srtp.decrypt(data)
                if decrypted is None:
                    logger.warning(
                        "SRTP authentication failed for packet from %s:%s, discarding",
                        addr[0],
                        addr[1],
                    )
                    return
                data = decrypted
            try:
                handler.packet_received(RTPPacket.parse(data), addr)
            except ValueError:
                logger.warning(
                    "Malformed RTP packet from %s:%s, discarding",
                    addr[0],
                    addr[1],
                )
        else:
            logger.debug(
                "No call handler registered for %s:%s, dropping RTP packet",
                addr[0],
                addr[1],
            )


RTP = RealtimeTransportProtocol
