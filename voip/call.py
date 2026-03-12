"""Call handler hierarchy for RTP/SIP sessions.

This module provides the :class:`Call` and :class:`AudioCall` dataclasses
that represent individual call legs managed by the RTP multiplexer.

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
         AudioCall              (audio buffering + codec negotiation)
            │
         WhisperCall            (speech-to-text via Whisper)
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING, ClassVar

from voip.rtp import RealtimeTransportProtocol, RTPPacket, RTPPayloadType
from voip.sdp.types import MediaDescription, RTPPayloadFormat
from voip.sip.types import CallerID

if TYPE_CHECKING:
    from voip.sip.protocol import SessionInitiationProtocol

__all__ = ["AudioCall", "Call"]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Call:
    """Handle basic IO and call functions.

    A call handler is associated with one SIP dialog and receives RTP traffic
    delivered by the shared :class:`~voip.rtp.RealtimeTransportProtocol`
    multiplexer.  Subclass and override :meth:`datagram_received` to process
    incoming media.

    The :attr:`rtp` and :attr:`sip` back-references allow the handler to send
    data back to the caller and to terminate the call via SIP BYE.

    Subclass :class:`AudioCall` for audio calls with codec negotiation and
    buffering.
    """

    #: Shared RTP multiplexer socket that delivers packets to this handler.
    rtp: RealtimeTransportProtocol
    #: SIP session that answered this call (used for BYE etc.).
    sip: SessionInitiationProtocol
    #: Caller identifier as received in the SIP From header.
    caller: str = ""
    #: Negotiated SDP media description for this call leg.
    media: MediaDescription | None = None

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Handle a raw RTP datagram.  Override in subclasses to process media."""

    async def send_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        """Send a datagram through the shared RTP socket.

        Args:
            data: Raw bytes to send.
            addr: Destination ``(host, port)``.
        """
        self.rtp.send(data, addr)

    async def hang_up(self) -> None:
        """Terminate the call by sending a SIP BYE request.

        Raises:
            NotImplementedError: Not yet implemented; the call_id and remote
                SIP address need to be stored per call to make this work.
        """
        raise NotImplementedError("hang_up is not yet implemented")


@dataclasses.dataclass
class AudioCall(Call):
    """RTP call handler with audio buffering and codec negotiation.

    Subclass and override :meth:`audio_received` to process buffered audio::

        class MyCall(AudioCall):
            def audio_received(self, packets: list[bytes]) -> None:
                save(packets)

    Override :attr:`PREFERRED_CODECS` (class variable) to change the codec
    priority list, and :attr:`chunk_duration` to accumulate multiple packets
    before each :meth:`audio_received` call.
    """

    #: Seconds of audio to buffer before emitting :meth:`audio_received`.
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

    _payload_type: int = dataclasses.field(init=False, default=0, repr=False)
    _sample_rate: int = dataclasses.field(init=False, default=8000, repr=False)
    _audio_buffer: list[bytes] = dataclasses.field(
        init=False, default_factory=list, repr=False
    )
    _packet_threshold: int = dataclasses.field(init=False, default=1, repr=False)

    def __post_init__(self) -> None:
        frame_size = 160  # default for PCMU/PCMA (RFC 3551)
        if self.media is not None and self.media.fmt:
            fmt = self.media.fmt[0]
            self._payload_type = fmt.payload_type
            self._sample_rate = fmt.sample_rate or 8000
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
        self._packet_threshold = (
            self._sample_rate * self.chunk_duration // frame_size
            if self.chunk_duration
            else 1
        )

    @property
    def payload_type(self) -> int:
        """Negotiated RTP payload type number."""
        return self._payload_type

    @property
    def sample_rate(self) -> int:
        """Negotiated audio sample rate in Hz."""
        return self._sample_rate

    @classmethod
    def negotiate_codec(cls, remote_media: MediaDescription) -> MediaDescription:
        """Select the best codec from the offered SDP and return a negotiated MediaDescription.

        Iterates :attr:`PREFERRED_CODECS` in priority order, matching by payload
        type or encoding name.  Raises :exc:`NotImplementedError` when no codec
        matches.

        Args:
            remote_media: The SDP ``m=audio`` section from the remote INVITE.

        Returns:
            A :class:`~voip.sdp.types.MediaDescription` describing the chosen
            codec with port set to ``0`` (the SIP layer fills in the real port).

        Raises:
            NotImplementedError: When no offered codec is in :attr:`PREFERRED_CODECS`.
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

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Parse *data* as an RTP packet and deliver audio to :meth:`audio_received`."""
        self._process_rtp(data)

    def _process_rtp(self, data: bytes) -> None:
        """Parse *data* and buffer/deliver audio payload to :meth:`audio_received`."""
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
        """Handle a buffered audio chunk.  Override in subclasses.

        Args:
            packets: A list of raw RTP payload bytes representing one audio
                chunk of :attr:`chunk_duration` seconds (or a single packet
                when ``chunk_duration == 0``).
        """
