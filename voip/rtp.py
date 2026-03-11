"""
Real-time Transport Protocol (RTP) implementation of RFC 3550.

See also: https://datatracker.ietf.org/doc/html/rfc3550#section-5
"""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
from typing import ClassVar

from voip.sdp.types import MediaDescription, RTPPayloadFormat
from voip.sip.types import CallerID
from voip.stun import STUNProtocol

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


class RealtimeTransportProtocol(STUNProtocol):
    """RTP multiplexer and call handler base class (RFC 3550).

    One instance of this class can manage multiple simultaneous calls on a
    single UDP socket.  Per-call handlers — subclass instances created by
    :meth:`~voip.sip.protocol.SessionInitiationProtocol.answer` — are
    registered with :meth:`register_call` and routed by the remote address
    reported in each datagram.

    Subclass and override :meth:`audio_received` to process incoming audio::

        class MyCall(RealtimeTransportProtocol):
            def audio_received(self, packets: list[bytes]) -> None:
                save(packets)

    Override :meth:`negotiate_codec` to customise codec selection.

    Set :attr:`chunk_duration` to buffer multiple packets before each
    :meth:`audio_received` call.  The default of ``0`` passes each packet
    individually as a single-element list.
    """

    #: Fixed RTP header size in bytes (RFC 3550 §5.1).
    rtp_header_size: int = 12

    #: Seconds of audio to buffer before emitting an :meth:`audio_received` event.
    #: ``0`` (default) emits one event per RTP packet.
    chunk_duration: ClassVar[int] = 0

    #: Preferred codecs, highest to lowest priority.
    PREFERRED_CODECS: ClassVar[list[RTPPayloadFormat]] = [
        RTPPayloadFormat(
            payload_type=RTPPayloadType.OPUS,
            encoding_name="opus",
            sample_rate=48000,
            channels=2,
        ),
        RTPPayloadFormat(payload_type=RTPPayloadType.G722),
        RTPPayloadFormat(payload_type=RTPPayloadType.PCMA),
        RTPPayloadFormat(payload_type=RTPPayloadType.PCMU),
    ]

    def __init__(self, caller: str = "", media: MediaDescription | None = None) -> None:
        super().__init__()
        self.caller = caller
        self.media = media
        if media is not None and media.fmt:
            fmt = media.fmt[0]
            self.payload_type: int = fmt.payload_type
            self.sample_rate: int = fmt.sample_rate or 8000
            caller_id = CallerID(self.caller)
            logger.info(
                json.dumps(
                    {
                        "event": "call_started",
                        "caller": repr(caller_id),
                        "codec": fmt.encoding_name or "unknown",
                        "sample_rate": fmt.sample_rate or 0,
                        "channels": fmt.channels,
                        "payload_type": fmt.payload_type,
                    }
                ),
                extra={
                    "caller": repr(caller_id),
                    "codec": fmt.encoding_name,
                    "payload_type": fmt.payload_type,
                },
            )
            frame_size = fmt.frame_size
        else:
            self.payload_type: int = 0
            self.sample_rate: int = 8000
            frame_size = 160

        self._audio_buffer: list[bytes] = []
        self._packet_threshold: int = (
            self.sample_rate * self.chunk_duration // frame_size
            if self.chunk_duration
            else 1
        )
        #: Per-call handlers keyed by the remote ``(ip, port)`` address.
        #: ``None`` is used as a wildcard key for calls whose remote address is
        #: not known in advance (e.g. when the INVITE contains no SDP).
        self._calls: dict[tuple[str, int] | None, RealtimeTransportProtocol] = {}

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

    def register_call(
        self,
        addr: tuple[str, int] | None,
        handler: RealtimeTransportProtocol,
    ) -> None:
        """Register *handler* for RTP traffic arriving from *addr*.

        Use ``addr=None`` as a wildcard to handle traffic from any source that
        has no dedicated routing entry (useful when the caller's RTP address is
        not known in advance).

        Args:
            addr: Remote ``(ip, port)`` as it will appear in incoming datagrams,
                or ``None`` to register a wildcard catch-all handler.
            handler: A :class:`RealtimeTransportProtocol` instance whose
                :meth:`audio_received` will be called for matching packets.
        """
        self._calls[addr] = handler

    def unregister_call(self, addr: tuple[str, int] | None) -> None:
        """Remove the handler registered for *addr*.

        Args:
            addr: The same key that was passed to :meth:`register_call`.
                Silently ignored when no handler is registered for *addr*.
        """
        self._calls.pop(addr, None)

    def packet_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Route an incoming RTP datagram to the appropriate per-call handler.

        Looks up *addr* in the call registry.  Falls back to the wildcard
        ``None`` handler when no exact match exists.  If neither is found the
        packet is processed by this instance (stand-alone / mux-level mode).
        """
        handler = self._calls.get(addr)
        if handler is None:
            handler = self._calls.get(None)
        if handler is not None:
            handler._process_rtp(data)
        else:
            self._process_rtp(data)

    def _process_rtp(self, data: bytes) -> None:
        """Parse *data* as an RTP packet and buffer/deliver audio to :meth:`audio_received`."""
        try:
            packet = RTPPacket.parse(data)
        except ValueError:
            return
        if not packet.payload:
            return
        self._audio_buffer.append(packet.payload)
        while len(self._audio_buffer) >= self._packet_threshold:
            packets = self._audio_buffer[: self._packet_threshold]
            self._audio_buffer = self._audio_buffer[self._packet_threshold :]
            self.audio_received(packets)

    def audio_received(self, packets: list[bytes]) -> None:
        """Handle a buffered audio frame. Override in subclasses.

        Called with a list of :attr:`~RealtimeTransportProtocol._packet_threshold`
        raw RTP payloads representing one audio chunk.
        """


#: Short alias for :class:`RealtimeTransportProtocol`.
RTP = RealtimeTransportProtocol
