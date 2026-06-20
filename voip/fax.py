"""T.38 and G.711 FAX over SIP.

Implements SIP signaling for sending and receiving fax documents over IP using
either T.38 UDPTL (RFC 3362) or G.711 pass-through (audio/PCMU).

`DualFaxSession` offers both transports in a single SDP so the remote endpoint
picks whichever it supports.

[RFC 3362]: https://datatracker.ietf.org/doc/html/rfc3362
"""

import asyncio
import dataclasses
import logging
import secrets
from typing import ClassVar

import numpy as np

from voip.codecs.pcmu import PCMU
from voip.rtp import RTPPacket, RTPPayloadType, Session
from voip.sdp.types import Attribute, MediaDescription, RTPPayloadFormat
from voip.t30.messages import T30MessageType
from voip.t30.modem import T30Modem, T30Receiver
from voip.types import NetworkAddress

__all__ = [
    "FaxSession",
    "OutboundFaxSession",
    "InboundFaxSession",
    "G711FaxSession",
    "OutboundG711FaxSession",
    "InboundG711FaxSession",
    "DualFaxSession",
    "OutboundDualFaxSession",
    "InboundDualFaxSession",
    "render_pdf_to_bitonal",
]

logger = logging.getLogger(__name__)

#: Safe UDP payload size for T.38 UDPTL datagrams.  The typical Ethernet MTU is
#: 1500 bytes; 1400 leaves headroom for IP/UDP headers and network overhead.
MAX_UDPTL_DATAGRAM_SIZE: int = 1400

#: G.711 PCMU samples per 20 ms RTP frame at 8 kHz (RFC 3551).
PCMU_FRAME_SIZE: int = 160

#: RTP timestamp increment per 20 ms PCMU frame at the 8 kHz clock.
PCMU_TIMESTAMP_INCREMENT: int = 160

#: Wall-clock duration of one 20 ms G.711 RTP frame, in seconds.
PCMU_FRAME_DURATION_SECONDS: float = 0.02

#: Sample rate for all generated fax audio (Hz).
SAMPLE_RATE_HZ: int = 8000

#: Standard fax A4 width in pixels at 200 DPI (T.4).
A4_WIDTH_PX: int = 1728

#: Standard fax A4 height in lines at standard resolution (3.85 lines/mm).
A4_HEIGHT_STANDARD: int = 1078

#: Threshold for binarizing grayscale pixels (0–255).
BITONAL_THRESHOLD: int = 128


def generate_ssrc() -> int:
    """Generate a cryptographically random 32-bit SSRC for an outbound RTP stream.

    Returns:
        A random 32-bit integer suitable for use as an RTP SSRC.
    """
    return secrets.randbits(32)


def render_pdf_to_bitonal(pdf_bytes: bytes) -> list[np.ndarray]:
    """Render a PDF document into a list of bitonal fax images.

    Each PDF page is rendered at standard fax resolution (1728 pixels wide,
    A4 width) and binarized to 0 (white) / 1 (black) using a fixed threshold.

    Requires [PyMuPDF](https://pymupdf.readthedocs.io/) (`fitz`) as an
    optional dependency (install the `fax` extra).

    Args:
        pdf_bytes: Raw PDF file bytes.

    Returns:
        A list of 2-D NumPy arrays (one per page), each of shape
        `(height, A4_WIDTH_PX)` with dtype `uint8`, values 0 (white) or 1 (black).

    Raises:
        ImportError: When PyMuPDF is not installed.
    """
    import fitz  # noqa: PLC0415

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[np.ndarray] = []
    for page in document:
        # Render at a DPI that produces ~1728 pixels across A4 (215 mm).
        # 1728 px / 215 mm * 25.4 mm/in ≈ 204 DPI.
        zoom = A4_WIDTH_PX / page.rect.width
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
        grayscale = np.frombuffer(pixmap.samples, dtype=np.uint8)
        grayscale = grayscale.reshape(pixmap.height, pixmap.width)
        # Binarize: pixels below threshold become black (1), above become white (0).
        bitonal = np.where(grayscale < BITONAL_THRESHOLD, 1, 0).astype(np.uint8)
        # Pad or crop width to exactly A4_WIDTH_PX.
        match bitonal.shape[1]:
            case width if width < A4_WIDTH_PX:
                padding = np.zeros(
                    (bitonal.shape[0], A4_WIDTH_PX - width), dtype=np.uint8
                )
                bitonal = np.concatenate([bitonal, padding], axis=1)
            case width if width > A4_WIDTH_PX:
                bitonal = bitonal[:, :A4_WIDTH_PX]
        images.append(bitonal)
    document.close()
    return images


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

    async def send_document(self, data: bytes) -> None:
        """Send a fax document as T.38 UDPTL datagrams.

        The document is split into MTU-safe chunks (see `MAX_UDPTL_DATAGRAM_SIZE`)
        so that each UDP datagram stays below the path MTU and does not raise
        ``[Errno 40] Message too long``.

        Args:
            data: Raw document data to send.
        """
        remote_address = next(
            (address for address, call in self.rtp.calls.items() if call is self),
            None,
        )
        if remote_address is None:
            logger.warning("No remote address for FAX call; dropping document data")
            return
        total_datagrams = (
            len(data) + MAX_UDPTL_DATAGRAM_SIZE - 1
        ) // MAX_UDPTL_DATAGRAM_SIZE
        logger.info(
            "FaxSession.send_document: remote=%s, data=%d bytes, datagrams=%d",
            remote_address,
            len(data),
            total_datagrams,
        )
        for offset in range(0, len(data), MAX_UDPTL_DATAGRAM_SIZE):
            self.rtp.send(
                data[offset : offset + MAX_UDPTL_DATAGRAM_SIZE], remote_address
            )
            await asyncio.sleep(0)
        logger.info(
            "FaxSession.send_document: sent %d datagrams, done", total_datagrams
        )

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
        mime_type: MIME type of the document,
            e.g. `"application/pdf"` or `"text/plain"`.
    """

    document: bytes
    mime_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        asyncio.create_task(self.transmit())

    async def transmit(self) -> None:
        """Send the document and hang up when transmission completes."""
        logger.info(
            "OutboundFaxSession.transmit: starting, document=%d bytes",
            len(self.document),
        )
        await self.send_document(self.document)
        logger.info("OutboundFaxSession.transmit: send_document returned, hanging up")
        await self.hang_up()
        logger.info("OutboundFaxSession.transmit: hang_up done, closing SIP")
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


@dataclasses.dataclass(kw_only=True)
class G711FaxSession(Session):
    """FAX over G.711 pass-through — fax modem tones as regular audio.

    Treats the fax machine's analog modem audio (T.30 CNG/CED/training/image
    tones) as G.711 PCMU RTP packets.  Works with any SIP provider that
    supports voice calls, unlike T.38 which requires explicit support.

    Trade-off: higher bandwidth (~64 kbps vs ~9.6 kbps for T.38) and no
    packet-loss redundancy.
    """

    media_type: ClassVar[str] = "audio"

    #: T.30 receiver for demodulating the called machine's responses.
    #: Set by subclasses that need to listen for CED/DIS/CFR/MCF/DCN.
    receiver: T30Receiver | None = None

    def data_received(self, data: bytes, address: NetworkAddress) -> None:
        self.document_received(data)

    def document_received(self, data: bytes) -> None:
        """Handle received G.711 audio data.

        Override in subclasses to process the received audio.
        """

    def packet_received(self, packet: RTPPacket, address: NetworkAddress) -> None:
        """Decode incoming PCMU RTP and feed the T.30 receiver.

        The called fax machine's responses (CED, DIS, CFR, MCF, DCN) arrive
        as G.711 PCMU RTP packets.  This method decodes them to PCM and feeds
        the audio to the [T30Receiver][voip.t30.modem.T30Receiver] so the
        outbound transmit state machine can wait for each response.

        Args:
            packet: Incoming RTP packet.
            address: Source (host, port).
        """
        if self.receiver is None:
            return
        pcm = PCMU.decode(packet.payload, SAMPLE_RATE_HZ)
        messages = self.receiver.feed_audio(pcm)
        if messages or self.receiver.ced_detected:
            event = getattr(self, "message_event", None)
            if event is not None:
                event.set()

    async def send_document(self, data: bytes) -> None:
        """Send a fax document as G.711 PCMU RTP packets with real-time pacing.

        When the document is a PDF file (detected by the `%PDF-` magic bytes),
        a [T30Modem][voip.t30.modem.T30Modem] renders the PDF pages into
        bitonal images, T.4-compresses them, and modulates the full T.30
        exchange (CNG, DIS, DCS, V.29 training, image data, EOP, DCN) into
        audio tones that the remote fax machine can demodulate.

        When the document is raw audio (not a PDF), the bytes are interpreted
        as raw G.711 µ-law samples and sent directly, split into 20 ms frames
        (160 bytes each at 8 kHz, per RFC 3551).

        Each frame is wrapped in an RTP packet (payload type 0) and sent via
        the shared RTP socket, which handles SRTP encryption when active.
        Frames are paced at 20 ms intervals (``PCMU_FRAME_DURATION_SECONDS``)
        so the receiver's jitter buffer and fax modem can process them in real
        time instead of receiving a sub-second burst.

        Args:
            data: PDF document bytes (rendered via T.30) or raw µ-law audio.
        """
        remote_address = next(
            (address for address, call in self.rtp.calls.items() if call is self),
            None,
        )
        if remote_address is None:
            logger.warning("No remote address for FAX call; dropping document data")
            return
        audio = self.render_document_audio(data)
        await self.send_audio(audio, remote_address)

    def render_document_audio(self, data: bytes) -> np.ndarray:
        """Render document bytes into float32 PCM audio for RTP transmission.

        PDF documents are rendered via the T.30 modem (CNG + DCS +
        V.29 training + T.4 image + EOP + DCN).  Raw non-PDF bytes are
        interpreted as pre-encoded µ-law audio and passed through directly.

        Args:
            data: PDF document bytes or raw µ-law audio.

        Returns:
            Float32 PCM samples in [-1, 1] at 8 kHz.
        """
        match data[:5]:
            case b"%PDF-":
                images = render_pdf_to_bitonal(data)
                modem = T30Modem()
                audio_segments = [modem.generate_audio(img) for img in images]
                return (
                    np.concatenate(audio_segments)
                    if audio_segments
                    else np.zeros(0, dtype=np.float32)
                )
            case _:
                ulaw = np.frombuffer(data, dtype=np.uint8)
                return PCMU.decode(ulaw.tobytes(), SAMPLE_RATE_HZ)

    async def send_audio(
        self, audio: np.ndarray, remote_address: NetworkAddress
    ) -> None:
        """Encode float32 PCM audio as µ-law RTP frames and send with pacing.

        Splits the audio into 20 ms frames (160 samples at 8 kHz), encodes
        each via [PCMU.encode][voip.codecs.pcmu.PCMU.encode], wraps in an RTP
        packet, and sends at real-time pace.

        Args:
            audio: Float32 PCM samples in [-1, 1] at 8 kHz.
            remote_address: Destination (host, port) for RTP packets.
        """
        total_frames = max(1, (len(audio) + PCMU_FRAME_SIZE - 1) // PCMU_FRAME_SIZE)
        logger.info(
            "G711FaxSession.send_audio: remote=%s, audio=%d samples, frames=%d",
            remote_address,
            len(audio),
            total_frames,
        )
        sequence_number = 0
        timestamp = 0
        ssrc = generate_ssrc()
        for offset in range(0, len(audio), PCMU_FRAME_SIZE):
            chunk = audio[offset : offset + PCMU_FRAME_SIZE]
            payload = PCMU.encode(chunk)
            packet = RTPPacket(
                payload_type=RTPPayloadType.PCMU,
                sequence_number=sequence_number,
                timestamp=timestamp,
                ssrc=ssrc,
                payload=payload,
            )
            self.send_packet(packet, remote_address)
            sequence_number = (sequence_number + 1) & 0xFFFF
            timestamp = (timestamp + PCMU_TIMESTAMP_INCREMENT) & 0xFFFFFFFF
            await asyncio.sleep(PCMU_FRAME_DURATION_SECONDS)
        logger.info(
            "G711FaxSession.send_audio: sent %d RTP packets, done",
            total_frames,
        )

    @classmethod
    def negotiate_codec(cls, remote_media: MediaDescription) -> MediaDescription:
        """Negotiate G.711 PCMU from a remote SDP `m=audio` offer.

        Args:
            remote_media: The SDP `m=audio` section from the remote INVITE.

        Returns:
            A G.711 media description for the response SDP.

        Raises:
            NotImplementedError: When the remote offer does not include PCMU.
        """
        from voip.sdp.types import StaticPayloadType  # noqa: PLC0415

        for fmt in remote_media.fmt:
            if fmt.payload_type == StaticPayloadType.PCMU.pt:
                return cls.sdp_media_description(port=remote_media.port)
        raise NotImplementedError("Remote SDP offer does not include G.711 PCMU")

    @classmethod
    def sdp_formats(cls) -> list[RTPPayloadFormat]:
        from voip.sdp.types import StaticPayloadType  # noqa: PLC0415

        return [RTPPayloadFormat.from_pt(StaticPayloadType.PCMU.pt)]

    @classmethod
    def sdp_media_description(cls, port: int = 0) -> MediaDescription:
        from voip.sdp.types import StaticPayloadType  # noqa: PLC0415

        return MediaDescription(
            media="audio",
            port=port,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat.from_pt(StaticPayloadType.PCMU.pt)],
            attributes=[Attribute(name="sendrecv")],
        )


@dataclasses.dataclass(kw_only=True, slots=True)
class OutboundG711FaxSession(G711FaxSession):
    """Dial a number, send a FAX document via G.711 pass-through, and hang up.

    Uses the [T30Modem][voip.t30.modem.T30Modem] phased transmit path: sends
    CNG, waits for the called machine's CED and DIS, sends DCS, trains V.29,
    waits for CFR, sends image data + EOP, waits for MCF, sends DCN — timing
    each phase to the remote fax machine's actual responses.

    Attributes:
        document: Raw document bytes to transmit as G.711 audio.
        mime_type: MIME type of the document.
        modem: T.30 modem instance used for modulation.
        message_event: Event set when a T.30 message or tone is received.
    """

    document: bytes
    mime_type: str = "application/octet-stream"
    modem: T30Modem = dataclasses.field(init=False)
    message_event: asyncio.Event = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.receiver = T30Receiver()
        self.modem = T30Modem()
        self.message_event = asyncio.Event()
        asyncio.create_task(self.transmit())

    async def transmit(self) -> None:
        """Run the phased T.30 exchange and hang up when complete.

        For PDF documents, uses the [T30Modem][voip.t30.modem.T30Modem] phased
        transmit path that listens for the called machine's responses.  For
        raw non-PDF bytes, sends the audio directly without a T.30 handshake
        (the caller is responsible for providing pre-modulated modem audio).
        """
        remote_address = next(
            (address for address, call in self.rtp.calls.items() if call is self),
            None,
        )
        if remote_address is None:
            logger.warning("OutboundG711FaxSession: no remote address, cannot transmit")
            await self.hang_up()
            return
        match self.document[:5]:
            case b"%PDF-":
                await self._transmit_pdf(remote_address)
            case _:
                await self._transmit_raw_audio(remote_address)
        await self.hang_up()
        if self.dialog is not None and self.dialog.sip is not None:
            self.dialog.sip.close()

    async def _transmit_pdf(self, remote_address: NetworkAddress) -> None:
        """Run the phased T.30 exchange for a PDF document."""
        images = render_pdf_to_bitonal(self.document)
        if not images:
            logger.warning("OutboundG711FaxSession: no pages to send")
            return

        async def send_audio(pcm: np.ndarray) -> None:
            await self.send_audio(pcm, remote_address)

        async def wait_for_message(
            msg_type: T30MessageType | None,
            timeout_seconds: float,
        ) -> bool:
            return await self._wait_for_message(msg_type, timeout_seconds)

        for image in images:
            await self.modem.transmit(image, send_audio, wait_for_message)

    async def _transmit_raw_audio(self, remote_address: NetworkAddress) -> None:
        """Send raw bytes as pre-encoded µ-law audio without a T.30 handshake."""
        audio = self.render_document_audio(self.document)
        await self.send_audio(audio, remote_address)

    async def _wait_for_message(
        self,
        msg_type: T30MessageType | None,
        timeout_seconds: float,
    ) -> bool:
        """Wait until the receiver decodes `msg_type` or the timeout expires.

        Args:
            msg_type: The T.30 message to wait for, or `None` to wait for the
                CED tone.
            timeout_seconds: Maximum time to wait.

        Returns:
            `True` if the message/tone was received, `False` on timeout.
        """
        if self.receiver is None:
            return False
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while True:
            if msg_type is None:
                if self.receiver.ced_detected:
                    return True
            elif self.receiver.has_message(msg_type):
                return True
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return False
            self.message_event.clear()
            try:
                await asyncio.wait_for(self.message_event.wait(), timeout=remaining)
            except TimeoutError:
                return False


@dataclasses.dataclass(kw_only=True, slots=True)
class InboundG711FaxSession(G711FaxSession):
    """Collect incoming G.711 audio into a single document buffer.

    Attributes:
        document: Accumulated G.711 audio data received so far.
    """

    document: bytes = dataclasses.field(default=b"", init=False)

    def document_received(self, data: bytes) -> None:
        self.document += data


@dataclasses.dataclass(kw_only=True)
class DualFaxSession(Session):
    """FAX session that offers both T.38 and G.711 in a single SDP.

    The dual offer lets the remote endpoint pick whichever transport it
    supports: T.38 (efficient, reliable) when available, G.711 (universal)
    as fallback.  After the answer arrives,
    [select_session_class][voip.fax.DualFaxSession.select_session_class]
    resolves to the concrete session subclass matching the negotiated media.
    """

    media_type: ClassVar[str] = "image"

    @classmethod
    def sdp_formats(cls) -> list[RTPPayloadFormat]:
        return FaxSession.sdp_formats()

    @classmethod
    def sdp_media_description(cls, port: int = 0) -> MediaDescription:
        return FaxSession.sdp_media_description(port)

    @classmethod
    def sdp_media_descriptions(cls, port: int) -> list[MediaDescription]:
        """Offer T.38 (`m=image`) first, then G.711 (`m=audio`) as fallback."""
        return [
            FaxSession.sdp_media_description(port),
            G711FaxSession.sdp_media_description(port),
        ]

    @classmethod
    def select_session_class(cls, remote_media: MediaDescription) -> type[Session]:
        """Resolve to `FaxSession` or `G711FaxSession` based on the answer.

        Args:
            remote_media: The `m=` section from the remote SDP answer.

        Returns:
            `FaxSession` for `m=image`/T.38 answers,
            `G711FaxSession` for `m=audio`/G.711 answers.
        """
        if remote_media.media == "image":
            return FaxSession
        if remote_media.media == "audio":
            return G711FaxSession
        msg = f"Unexpected media type in FAX answer: {remote_media.media!r}"
        raise NotImplementedError(msg)

    @classmethod
    def negotiate_codec(cls, remote_media: MediaDescription) -> MediaDescription:
        """Negotiate the codec for whichever media type the remote answered with."""
        if remote_media.media == "image":
            return FaxSession.negotiate_codec(remote_media)
        if remote_media.media == "audio":
            return G711FaxSession.negotiate_codec(remote_media)
        msg = f"Unexpected media type in FAX offer: {remote_media.media!r}"
        raise NotImplementedError(msg)


@dataclasses.dataclass(kw_only=True, slots=True)
class OutboundDualFaxSession(DualFaxSession):
    """Outbound FAX offering both T.38 and G.711; sends via whichever was accepted.

    Attributes:
        document: Raw document bytes to transmit.
        mime_type: MIME type of the document.
    """

    document: bytes
    mime_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        asyncio.create_task(self.transmit())

    async def transmit(self) -> None:
        """Send the document and hang up when transmission completes."""
        logger.info(
            "OutboundDualFaxSession.transmit: starting, document=%d bytes, "
            "resolved media=%r",
            len(self.document),
            self.media.media if self.media else None,
        )
        await self.send_document(self.document)
        logger.info(
            "OutboundDualFaxSession.transmit: send_document returned, hanging up"
        )
        await self.hang_up()
        logger.info("OutboundDualFaxSession.transmit: hang_up done, closing SIP")
        if self.dialog is not None and self.dialog.sip is not None:
            self.dialog.sip.close()

    @classmethod
    def select_session_class(cls, remote_media: MediaDescription) -> type[Session]:
        """Resolve to the outbound variant matching the negotiated media."""
        if remote_media.media == "image":
            return OutboundFaxSession
        if remote_media.media == "audio":
            return OutboundG711FaxSession
        msg = f"Unexpected media type in FAX answer: {remote_media.media!r}"
        raise NotImplementedError(msg)

    async def send_document(self, data: bytes) -> None:
        """Send a fax document via the negotiated transport.

        Delegates to the transport-specific `send_document` of the session
        class resolved from the negotiated media (`self.media`), so that
        T.38 answers chunk into MTU-safe UDPTL datagrams and G.711 answers
        packetize into RTP/PCMU packets with real-time pacing.  In normal
        operation the SIP layer resolves `OutboundDualFaxSession` to the
        concrete subclass before instantiation, so this method is only reached
        when the dual session is used directly.

        Args:
            data: Raw document data to send.
        """
        resolved = self.select_session_class(self.media)
        await resolved.send_document(self, data)


@dataclasses.dataclass(kw_only=True, slots=True)
class InboundDualFaxSession(DualFaxSession):
    """Inbound FAX accepting both T.38 and G.711; collects whichever arrives.

    Attributes:
        document: Accumulated data received so far.
    """

    document: bytes = dataclasses.field(default=b"", init=False)

    @classmethod
    def select_session_class(cls, remote_media: MediaDescription) -> type[Session]:
        """Resolve to the inbound variant matching the negotiated media."""
        if remote_media.media == "image":
            return InboundFaxSession
        if remote_media.media == "audio":
            return InboundG711FaxSession
        msg = f"Unexpected media type in FAX answer: {remote_media.media!r}"
        raise NotImplementedError(msg)

    def data_received(self, data: bytes, address: NetworkAddress) -> None:
        self.document_received(data)

    def document_received(self, data: bytes) -> None:
        self.document += data
