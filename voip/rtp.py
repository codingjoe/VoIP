"""
Real-time Transport Protocol (RTP) implementation of RFC 3550.

See also: https://datatracker.ietf.org/doc/html/rfc3550#section-5
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging

from voip.sdp.types import MediaDescription

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

    Subclass this and override :meth:`audio_received` to process incoming audio::

        class MyCall(RealtimeTransportProtocol):
            def audio_received(self, packet: RTPPacket) -> None:
                ...  # process audio payload

    Instances are used directly as asyncio datagram protocols, so they handle
    their own RTP header parsing before calling :meth:`audio_received`.

    Override :meth:`negotiate_codec` to customise codec selection when answering
    incoming calls.
    """

    #: Fixed RTP header size in bytes (RFC 3550 §5.1).
    rtp_header_size: int = 12

    #: Codec preference list (fmt, rtpmap value, clock rate Hz). Highest priority first.
    #: Opus > G.722 > PCMA (G.711 A-law) > PCMU (G.711 µ-law).
    PREFERRED_CODECS: list[tuple[str, str, int]] = [
        ("111", "opus/48000/2", 48000),
        ("9", "G722/8000", 8000),
        ("8", "PCMA/8000", 8000),
        ("0", "PCMU/8000", 8000),
    ]

    def __init__(
        self, caller: str = "", payload_type: int = 0, sample_rate: int = 8000
    ) -> None:
        super().__init__()
        #: The SIP address of the caller (from the From header of the INVITE).
        self.caller = caller
        #: The negotiated RTP payload type for this call.
        self.payload_type = payload_type
        #: The clock rate (Hz) of the negotiated codec, as declared in the SDP rtpmap.
        self.sample_rate = sample_rate

    @classmethod
    def negotiate_codec(
        cls, remote_media: MediaDescription
    ) -> tuple[str, str | None, int] | None:
        """Select the best codec from the offered SDP MediaDescription.

        Iterates :attr:`PREFERRED_CODECS` in priority order and returns the
        first match found in the remote offer.

        Args:
            remote_media: The ``m=audio`` :class:`~voip.sdp.types.MediaDescription`
                from the INVITE SDP body.

        Returns:
            A ``(fmt, rtpmap_value, sample_rate)`` tuple for the selected codec,
            or ``None`` if the remote offer contains no audio formats.
            *rtpmap_value* may be ``None`` for static payload types that carry no
            ``a=rtpmap`` attribute.
        """
        remote_fmts = set(remote_media.fmt)
        remote_rtpmaps: dict[str, str] = {}
        remote_name_to_fmt: dict[str, str] = {}
        for attribute in remote_media.attributes:
            if attribute.name == "rtpmap" and attribute.value:
                pt, _, codec_str = attribute.value.partition(" ")
                remote_rtpmaps[pt.strip()] = codec_str.strip()
                remote_name_to_fmt[codec_str.strip().lower()] = pt.strip()

        for our_fmt, our_rtpmap, sample_rate in cls.PREFERRED_CODECS:
            if our_fmt in remote_fmts:
                return (our_fmt, f"{our_fmt} {our_rtpmap}", sample_rate)
            if our_rtpmap.lower() in remote_name_to_fmt:
                remote_fmt = remote_name_to_fmt[our_rtpmap.lower()]
                return (remote_fmt, f"{remote_fmt} {our_rtpmap}", sample_rate)

        # Fallback: accept the first format the remote side offered.
        if remote_media.fmt:
            fmt = remote_media.fmt[0]
            sample_rate = 8000  # default for unknown/static codecs
            rtpmap = None
            if fmt in remote_rtpmaps:
                rtpmap_str = remote_rtpmaps[fmt]
                rtpmap = f"{fmt} {rtpmap_str}"
                # Parse the clock rate from "codec/clockrate[/channels]"
                parts = rtpmap_str.split("/")
                if len(parts) >= 2:
                    try:
                        sample_rate = int(parts[1])
                    except ValueError:
                        pass
            return (fmt, rtpmap, sample_rate)

        return None

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
