"""
Real-time Transport Protocol (RTP) implementation of RFC 3550.

See also: https://datatracker.ietf.org/doc/html/rfc3550#section-5
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging

from voip.stun import STUNProtocol

__all__ = ["RTP", "RTPPacket", "RTPPayloadType", "RealtimeTransportProtocol"]

logger = logging.getLogger(__name__)


class RTPPayloadType(enum.IntEnum):
    """Common RTP payload types.

    Static payload types (0–95) are defined by RFC 3551.
    Dynamic payload types (96–127) are negotiated via SDP.
    Opus uses payload type 111 per RFC 7587.
    """

    PCMU = 0  # G.711 µ-law (RFC 3551)
    PCMA = 8  # G.711 A-law (RFC 3551)
    G722 = 9  # G.722 (RFC 3551)
    OPUS = 111  # RFC 7587 (dynamic)


@dataclasses.dataclass
class RTPPacket:
    """A parsed RTP packet (RFC 3550 §5.1)."""

    payload_type: int
    sequence_number: int
    timestamp: int
    ssrc: int
    payload: bytes

    #: Fixed RTP header size in bytes (RFC 3550 §5.1).
    header_size: int = dataclasses.field(default=12, init=False, repr=False)

    @classmethod
    def parse(cls, data: bytes) -> RTPPacket:
        """Parse raw RTP bytes into an RTPPacket."""
        if len(data) < 12:
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
            payload=data[12:],
        )


class RealtimeTransportProtocol(STUNProtocol, asyncio.DatagramProtocol):
    """Base class for RTP audio call handlers (RFC 3550).

    Subclass this and override :meth:`audio_received` to process incoming audio::

        class MyCall(RealtimeTransportProtocol):
            def audio_received(self, packet: RTPPacket) -> None:
                ...  # process audio payload

    Instances are used directly as asyncio datagram protocols, so they handle
    their own RTP header parsing before calling :meth:`audio_received`.

    The :class:`~voip.stun.STUNProtocol` mixin enables STUN-based NAT traversal
    (RFC 5389) so that the session can discover the public IP:port of the RTP
    socket and advertise it correctly in the SDP offer (RFC 7983 multiplexing).
    """

    #: Fixed RTP header size in bytes (RFC 3550 §5.1).
    rtp_header_size: int = 12

    def __init__(self, caller: str = "", payload_type: int = 0) -> None:
        super().__init__()
        #: The SIP address of the caller (from the From header of the INVITE).
        self.caller = caller
        #: The negotiated RTP payload type for this call.
        self.payload_type = payload_type

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        """Store the transport for STUN discovery and outbound sends."""
        self._transport = transport

    def datagram_received(self, data: bytes, address: tuple[str, int]) -> None:
        """Multiplex STUN and RTP per RFC 7983.

        STUN messages (first byte 0–3) are routed to :meth:`~voip.stun.STUNProtocol.handle_stun`.
        Valid RTP packets are parsed and forwarded to :meth:`audio_received`.
        """
        if data and data[0] < 4:  # STUN: first byte is 0-3 (RFC 7983 multiplexing)
            self.handle_stun(data, address)
            return
        try:
            packet = RTPPacket.parse(data)
        except ValueError:
            return
        if not packet.payload:
            return
        self.audio_received(packet)

    def audio_received(self, packet: RTPPacket) -> None:
        """Handle an RTP packet. Override in subclasses."""


#: Short alias for :class:`RealtimeTransportProtocol`.
RTP = RealtimeTransportProtocol
