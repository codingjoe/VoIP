"""
Real-time Transport Protocol (RTP) implementation of RFC 3550.

See also: https://datatracker.ietf.org/doc/html/rfc3550#section-5
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
from typing import ClassVar

from voip.sdp.types import MediaDescription, RTPPayloadFormat

__all__ = ["RTP", "RTPPacket", "RTPPayloadType", "RealtimeTransportProtocol"]

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
    """Base class for RTP audio call handlers (RFC 3550).

    Subclass and override :meth:`audio_received` to process incoming audio.
    Override :meth:`negotiate_codec` to customise codec selection.
    """

    #: Fixed RTP header size in bytes (RFC 3550 §5.1).
    rtp_header_size: int = 12

    #: Preferred codecs, highest to lowest priority.
    PREFERRED_CODECS: ClassVar[list[RTPPayloadFormat]] = [
        RTPPayloadFormat(
            payload_type=RTPPayloadType.OPUS,
            encoding_name="opus",
            sample_rate=48000,
            channels=2,
        ),
        RTPPayloadFormat(
            payload_type=RTPPayloadType.G722, encoding_name="G722", sample_rate=8000
        ),
        RTPPayloadFormat(
            payload_type=RTPPayloadType.PCMA, encoding_name="PCMA", sample_rate=8000
        ),
        RTPPayloadFormat(
            payload_type=RTPPayloadType.PCMU, encoding_name="PCMU", sample_rate=8000
        ),
    ]

    def __init__(self, caller: str = "", media: MediaDescription | None = None) -> None:
        super().__init__()
        self.caller = caller
        self.media = media
        if media is not None and media.fmt:
            self.payload_type: int = media.fmt[0].payload_type
            self.sample_rate: int = media.fmt[0].sample_rate
        else:
            self.payload_type: int = 0
            self.sample_rate: int = 8000

    @classmethod
    def negotiate_codec(cls, remote_media: MediaDescription) -> MediaDescription:
        """Select the best codec from the offered SDP and return a negotiated MediaDescription.

        Iterates :attr:`PREFERRED_CODECS` in priority order, matching by payload
        type or encoding name.  Raises :exc:`NotImplementedError` when no codec
        matches.
        """
        if not remote_media.fmt:
            raise NotImplementedError("Remote SDP offer contains no audio formats")

        remote_fmts = {f.payload_type for f in remote_media.fmt}
        for preferred in cls.PREFERRED_CODECS:
            # Match by payload type number.
            if preferred.payload_type in remote_fmts:
                remote_fmt = remote_media.get_format(preferred.payload_type)
                codec = (
                    remote_fmt if remote_fmt and remote_fmt.encoding_name else preferred
                )
                return MediaDescription(
                    media="audio",
                    port=0,
                    proto="RTP/AVP",
                    fmt=[codec],
                )
            # Match by encoding name for dynamic payload types.
            for remote_fmt in remote_media.fmt:
                if (
                    remote_fmt.encoding_name is not None
                    and remote_fmt.encoding_name.lower()
                    == preferred.encoding_name.lower()
                ):
                    return MediaDescription(
                        media="audio",
                        port=0,
                        proto="RTP/AVP",
                        fmt=[remote_fmt],
                    )

        raise NotImplementedError(
            f"No supported codec found in remote offer "
            f"{[f.payload_type for f in remote_media.fmt]!r}. "
            f"Supported: {[c.encoding_name for c in cls.PREFERRED_CODECS]!r}"
        )

    def datagram_received(self, data: bytes, address: tuple[str, int]) -> None:
        """Parse and forward incoming RTP packets to :meth:`audio_received`."""
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
