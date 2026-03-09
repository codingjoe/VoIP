"""Real-time Transport Protocol (RTP) implementation of RFC 3550."""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging

__all__ = ["RTPPacket", "RTPPayloadType", "RealtimeTransportProtocol"]

logger = logging.getLogger(__name__)


class RTPPayloadType(enum.IntEnum):
    """Common RTP payload types.

    Dynamic payload types (96–127) are negotiated via SDP (RFC 3551).
    Opus uses payload type 111 per RFC 7587.
    """

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


class RealtimeTransportProtocol(asyncio.DatagramProtocol):
    """
    Real-time Transport Protocol (RTP) handler for receiving audio streams (RFC 3550).

    See also: https://datatracker.ietf.org/doc/html/rfc3550#section-5
    """

    rtp_header_size = 12

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Strip the fixed RTP header and forward the audio payload."""
        if len(data) > self.rtp_header_size:
            self.audio_received(data[self.rtp_header_size :])

    def audio_received(self, data: bytes) -> None:
        """Handle a decoded RTP audio payload. Override in subclasses."""


RTP = RealtimeTransportProtocol
