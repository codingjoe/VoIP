"""Tests for FAX sessions (voip.fax) — T.38, G.711, and dual offer."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from voip.fax import (
    DualFaxSession,
    FaxSession,
    G711FaxSession,
    InboundDualFaxSession,
    InboundFaxSession,
    InboundG711FaxSession,
    OutboundDualFaxSession,
    OutboundFaxSession,
    OutboundG711FaxSession,
)
from voip.rtp import RealtimeTransportProtocol, Session
from voip.sdp.types import MediaDescription, RTPPayloadFormat
from voip.sip.dialog import Dialog
from voip.sip.types import CallerID
from voip.types import NetworkAddress


def make_fax_session(**kwargs) -> FaxSession:
    """Create a `FaxSession` with sensible mock defaults."""
    defaults: dict = {
        "rtp": MagicMock(spec=RealtimeTransportProtocol),
        "dialog": MagicMock(spec=Dialog),
        "media": FaxSession.sdp_media_description(),
        "caller": CallerID(""),
    }
    defaults.update(kwargs)
    return FaxSession(**defaults)


class TestFaxSessionAttributes:
    def test_media_type__is_image(self) -> None:
        """media_type class variable is 'image' for T.38."""
        assert FaxSession.media_type == "image"

    def test_t38_version__is_zero(self) -> None:
        """T38_VERSION defaults to 0."""
        assert FaxSession.T38_VERSION == 0

    def test_t38_max_bit_rate__is_14400(self) -> None:
        """T38_MAX_BIT_RATE defaults to 14400 bps."""
        assert FaxSession.T38_MAX_BIT_RATE == 14_400


class TestSdpFormats:
    def test_sdp_formats__returns_t38_format(self) -> None:
        """sdp_formats returns a single T.38 payload format."""
        formats = FaxSession.sdp_formats()
        assert len(formats) == 1
        assert formats[0].payload_type == "t38"


class TestSdpMediaDescription:
    def test_sdp_media_description__media_is_image(self) -> None:
        """sdp_media_description produces an m=image section."""
        assert FaxSession.sdp_media_description(port=5004).media == "image"

    def test_sdp_media_description__proto_is_udptl(self) -> None:
        """sdp_media_description uses udptl transport."""
        assert FaxSession.sdp_media_description().proto == "udptl"

    def test_sdp_media_description__port_is_set(self) -> None:
        """sdp_media_description includes the provided port."""
        assert FaxSession.sdp_media_description(port=9876).port == 9876

    def test_sdp_media_description__includes_t38_fax_version(self) -> None:
        """sdp_media_description includes the T38FaxVersion attribute."""
        attributes = FaxSession.sdp_media_description().attributes
        assert any(a.name == "T38FaxVersion" for a in attributes)

    def test_sdp_media_description__includes_t38_max_bit_rate(self) -> None:
        """sdp_media_description includes the T38MaxBitRate attribute."""
        attributes = FaxSession.sdp_media_description().attributes
        assert any(a.name == "T38MaxBitRate" for a in attributes)

    def test_sdp_media_description__includes_rate_management(self) -> None:
        """sdp_media_description includes T38FaxRateManagement."""
        attributes = FaxSession.sdp_media_description().attributes
        assert any(a.name == "T38FaxRateManagement" for a in attributes)

    def test_sdp_media_description__default_port_is_zero(self) -> None:
        """sdp_media_description uses port 0 when no port is given."""
        assert FaxSession.sdp_media_description().port == 0


class TestNegotiateCodec:
    def test_negotiate_codec__accepts_t38_offer(self) -> None:
        """negotiate_codec returns T.38 MediaDescription for a valid T.38 offer."""
        offer = MediaDescription(
            media="image",
            port=5004,
            proto="udptl",
            fmt=[RTPPayloadFormat(payload_type="t38")],
        )
        result = FaxSession.negotiate_codec(offer)
        assert result.media == "image"
        assert result.proto == "udptl"

    def test_negotiate_codec__raises_for_non_t38_offer(self) -> None:
        """negotiate_codec raises NotImplementedError when T.38 is absent."""
        offer = MediaDescription(
            media="image",
            port=5004,
            proto="udptl",
            fmt=[RTPPayloadFormat(payload_type=0)],
        )
        with pytest.raises(NotImplementedError, match="T.38"):
            FaxSession.negotiate_codec(offer)

    def test_negotiate_codec__uses_remote_port(self) -> None:
        """negotiate_codec returns a description with the remote offer's port."""
        offer = MediaDescription(
            media="image",
            port=7070,
            proto="udptl",
            fmt=[RTPPayloadFormat(payload_type="t38")],
        )
        assert FaxSession.negotiate_codec(offer).port == 7070


class TestDataReceived:
    def test_data_received__delegates_to_document_received(self) -> None:
        """data_received forwards the raw data to document_received."""
        received: list[bytes] = []
        session = make_fax_session()
        session.document_received = received.append
        session.data_received(b"t38 packet", NetworkAddress("127.0.0.1", 5004))
        assert received == [b"t38 packet"]


class TestDocumentReceived:
    def test_document_received__is_noop(self) -> None:
        """document_received is a no-op in the base class."""
        make_fax_session().document_received(b"data")  # must not raise


class TestSendDocument:
    async def test_send_document__sends_to_remote_address(self) -> None:
        """send_document transmits data to the registered remote address."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        session = make_fax_session(rtp=mock_rtp)
        remote_address = ("127.0.0.1", 5004)
        mock_rtp.calls = {remote_address: session}
        await session.send_document(b"fax content")
        mock_rtp.send.assert_called_once_with(b"fax content", remote_address)

    async def test_send_document__logs_warning_when_no_remote_address(
        self, caplog
    ) -> None:
        """send_document logs a warning and does nothing when not registered."""
        import logging  # noqa: PLC0415

        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        session = make_fax_session(rtp=mock_rtp)
        mock_rtp.calls = {}
        with caplog.at_level(logging.WARNING, logger="voip.fax"):
            await session.send_document(b"fax content")
        mock_rtp.send.assert_not_called()
        assert any("No remote address" in r.message for r in caplog.records)


class TestOutboundFaxSession:
    def test_mime_type__defaults_to_octet_stream(self) -> None:
        """mime_type defaults to application/octet-stream when not specified."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        mock_dialog.sip = None

        with patch("asyncio.create_task"):
            session = OutboundFaxSession(
                rtp=mock_rtp,
                dialog=mock_dialog,
                media=FaxSession.sdp_media_description(),
                caller=CallerID(""),
                document=b"data",
            )

        assert session.mime_type == "application/octet-stream"

    def test_mime_type__can_be_set(self) -> None:
        """mime_type can be provided explicitly."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        mock_dialog.sip = None

        with patch("asyncio.create_task"):
            session = OutboundFaxSession(
                rtp=mock_rtp,
                dialog=mock_dialog,
                media=FaxSession.sdp_media_description(),
                caller=CallerID(""),
                document=b"%PDF-1.4",
                mime_type="application/pdf",
            )

        assert session.mime_type == "application/pdf"

    async def test_transmit__sends_document_and_hangs_up(self) -> None:
        """Transmit sends the document, hangs up, and closes the SIP connection."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        mock_sip = MagicMock()
        mock_dialog.sip = mock_sip
        remote_address = ("127.0.0.1", 5004)

        with patch("asyncio.create_task"):
            session = OutboundFaxSession(
                rtp=mock_rtp,
                dialog=mock_dialog,
                media=FaxSession.sdp_media_description(),
                caller=CallerID(""),
                document=b"pdf bytes",
            )
        mock_rtp.calls = {remote_address: session}

        with patch.object(session, "hang_up", new_callable=AsyncMock) as mock_hang_up:
            await session.transmit()

        mock_rtp.send.assert_called_once_with(b"pdf bytes", remote_address)
        mock_hang_up.assert_awaited_once()
        mock_sip.close.assert_called_once()

    async def test_transmit__skips_sip_close_when_no_sip(self) -> None:
        """Transmit does not raise when dialog.sip is None."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        mock_dialog.sip = None
        mock_rtp.calls = {}

        with patch("asyncio.create_task"):
            session = OutboundFaxSession(
                rtp=mock_rtp,
                dialog=mock_dialog,
                media=FaxSession.sdp_media_description(),
                caller=CallerID(""),
                document=b"pdf",
            )

        with patch.object(session, "hang_up", new_callable=AsyncMock):
            await session.transmit()  # must not raise

    def test_post_init__creates_transmit_task(self) -> None:
        """__post_init__ schedules transmit() as an asyncio task."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        mock_dialog.sip = None

        with patch("asyncio.create_task") as mock_create_task:
            OutboundFaxSession(
                rtp=mock_rtp,
                dialog=mock_dialog,
                media=FaxSession.sdp_media_description(),
                caller=CallerID(""),
                document=b"pdf",
            )

        mock_create_task.assert_called_once()


class TestInboundFaxSession:
    def test_document_received__accumulates_data(self) -> None:
        """document_received appends each packet to the document buffer."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        session = InboundFaxSession(
            rtp=mock_rtp,
            dialog=mock_dialog,
            media=FaxSession.sdp_media_description(),
            caller=CallerID(""),
        )
        session.document_received(b"page1")
        session.document_received(b"page2")
        assert session.document == b"page1page2"

    def test_document__starts_empty(self) -> None:
        """Document buffer is empty bytes before any data is received."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        session = InboundFaxSession(
            rtp=mock_rtp,
            dialog=mock_dialog,
            media=FaxSession.sdp_media_description(),
            caller=CallerID(""),
        )
        assert session.document == b""


# ---------------------------------------------------------------------------
# G.711 pass-through FAX
# ---------------------------------------------------------------------------


class TestG711FaxSessionAttributes:
    def test_media_type__is_audio(self) -> None:
        assert G711FaxSession.media_type == "audio"


class TestG711SdpFormats:
    def test_sdp_formats__returns_pcmu(self) -> None:
        formats = G711FaxSession.sdp_formats()
        assert len(formats) == 1
        assert formats[0].payload_type == 0
        assert formats[0].encoding_name == "PCMU"


class TestG711SdpMediaDescription:
    def test_media__is_audio(self) -> None:
        assert G711FaxSession.sdp_media_description(port=5004).media == "audio"

    def test_proto__is_rtp_avp(self) -> None:
        assert G711FaxSession.sdp_media_description().proto == "RTP/AVP"

    def test_port__is_set(self) -> None:
        assert G711FaxSession.sdp_media_description(port=9876).port == 9876

    def test_includes_sendrecv(self) -> None:
        attributes = G711FaxSession.sdp_media_description().attributes
        assert any(a.name == "sendrecv" for a in attributes)

    def test_default_port_is_zero(self) -> None:
        assert G711FaxSession.sdp_media_description().port == 0


class TestG711NegotiateCodec:
    def test_accepts_pcmu_offer(self) -> None:
        offer = MediaDescription(
            media="audio",
            port=5004,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=0)],
        )
        result = G711FaxSession.negotiate_codec(offer)
        assert result.media == "audio"
        assert result.proto == "RTP/AVP"

    def test_raises_for_non_pcmu_offer(self) -> None:
        offer = MediaDescription(
            media="audio",
            port=5004,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=8)],
        )
        with pytest.raises(NotImplementedError, match="G.711 PCMU"):
            G711FaxSession.negotiate_codec(offer)

    def test_uses_remote_port(self) -> None:
        offer = MediaDescription(
            media="audio",
            port=7070,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=0)],
        )
        assert G711FaxSession.negotiate_codec(offer).port == 7070


class TestG711DataReceived:
    def test_delegates_to_document_received(self) -> None:
        received: list[bytes] = []
        session = G711FaxSession(
            rtp=MagicMock(spec=RealtimeTransportProtocol),
            dialog=MagicMock(spec=Dialog),
            media=G711FaxSession.sdp_media_description(),
            caller=CallerID(""),
        )
        session.document_received = received.append
        session.data_received(b"audio", NetworkAddress("127.0.0.1", 5004))
        assert received == [b"audio"]


class TestOutboundG711FaxSession:
    def test_mime_type__defaults_to_octet_stream(self) -> None:
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        mock_dialog.sip = None
        with patch("asyncio.create_task"):
            session = OutboundG711FaxSession(
                rtp=mock_rtp,
                dialog=mock_dialog,
                media=G711FaxSession.sdp_media_description(),
                caller=CallerID(""),
                document=b"data",
            )
        assert session.mime_type == "application/octet-stream"

    async def test_transmit__sends_document_and_hangs_up(self) -> None:
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        mock_sip = MagicMock()
        mock_dialog.sip = mock_sip
        remote_address = ("127.0.0.1", 5004)
        with patch("asyncio.create_task"):
            session = OutboundG711FaxSession(
                rtp=mock_rtp,
                dialog=mock_dialog,
                media=G711FaxSession.sdp_media_description(),
                caller=CallerID(""),
                document=b"pdf bytes",
            )
        mock_rtp.calls = {remote_address: session}
        with patch.object(session, "hang_up", new_callable=AsyncMock) as mock_hang_up:
            await session.transmit()
        # G.711 pass-through sends RTP/PCMU packets.
        assert mock_rtp.send.call_count >= 1
        sent_data, sent_addr = mock_rtp.send.call_args.args
        assert sent_addr == remote_address
        assert sent_data[0] == 0x80  # V=2, P=0, X=0, CC=0
        assert sent_data[1] == 0  # payload type 0 (PCMU)
        mock_hang_up.assert_awaited_once()
        mock_sip.close.assert_called_once()

    def test_post_init__creates_transmit_task(self) -> None:
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        mock_dialog.sip = None
        with patch("asyncio.create_task") as mock_create_task:
            OutboundG711FaxSession(
                rtp=mock_rtp,
                dialog=mock_dialog,
                media=G711FaxSession.sdp_media_description(),
                caller=CallerID(""),
                document=b"pdf",
            )
        mock_create_task.assert_called_once()


class TestInboundG711FaxSession:
    def test_document_received__accumulates_data(self) -> None:
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        session = InboundG711FaxSession(
            rtp=mock_rtp,
            dialog=mock_dialog,
            media=G711FaxSession.sdp_media_description(),
            caller=CallerID(""),
        )
        session.document_received(b"page1")
        session.document_received(b"page2")
        assert session.document == b"page1page2"


# ---------------------------------------------------------------------------
# Dual offer (T.38 + G.711)
# ---------------------------------------------------------------------------


class TestDualFaxSessionSdp:
    def test_sdp_media_descriptions__returns_two_lines(self) -> None:
        descs = DualFaxSession.sdp_media_descriptions(port=5004)
        assert len(descs) == 2
        assert descs[0].media == "image"
        assert descs[1].media == "audio"

    def test_sdp_media_descriptions__first_is_t38(self) -> None:
        descs = DualFaxSession.sdp_media_descriptions(port=5004)
        assert descs[0].proto == "udptl"
        assert str(descs[0].fmt[0].payload_type) == "t38"

    def test_sdp_media_descriptions__second_is_g711(self) -> None:
        descs = DualFaxSession.sdp_media_descriptions(port=5004)
        assert descs[1].proto == "RTP/AVP"
        assert descs[1].fmt[0].payload_type == 0

    def test_sdp_media_descriptions__both_use_same_port(self) -> None:
        descs = DualFaxSession.sdp_media_descriptions(port=1234)
        assert descs[0].port == 1234
        assert descs[1].port == 1234


class TestDualFaxSelectSessionClass:
    def test_image_answer__selects_t38(self) -> None:
        remote = MediaDescription(
            media="image",
            port=5004,
            proto="udptl",
            fmt=[RTPPayloadFormat(payload_type="t38")],
        )
        assert DualFaxSession.select_session_class(remote) is FaxSession

    def test_audio_answer__selects_g711(self) -> None:
        remote = MediaDescription(
            media="audio",
            port=5004,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=0)],
        )
        assert DualFaxSession.select_session_class(remote) is G711FaxSession

    def test_unknown_media__raises(self) -> None:
        remote = MediaDescription(
            media="video",
            port=5004,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=0)],
        )
        with pytest.raises(NotImplementedError, match="Unexpected media type"):
            DualFaxSession.select_session_class(remote)


class TestDualFaxNegotiateCodec:
    def test_image__delegates_to_t38(self) -> None:
        offer = MediaDescription(
            media="image",
            port=7070,
            proto="udptl",
            fmt=[RTPPayloadFormat(payload_type="t38")],
        )
        result = DualFaxSession.negotiate_codec(offer)
        assert result.media == "image"
        assert result.port == 7070

    def test_audio__delegates_to_g711(self) -> None:
        offer = MediaDescription(
            media="audio",
            port=7070,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=0)],
        )
        result = DualFaxSession.negotiate_codec(offer)
        assert result.media == "audio"
        assert result.port == 7070


class TestOutboundDualFaxSession:
    def test_post_init__creates_transmit_task(self) -> None:
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        mock_dialog.sip = None
        with patch("asyncio.create_task") as mock_create_task:
            OutboundDualFaxSession(
                rtp=mock_rtp,
                dialog=mock_dialog,
                media=DualFaxSession.sdp_media_description(),
                caller=CallerID(""),
                document=b"pdf",
            )
        mock_create_task.assert_called_once()

    async def test_transmit__sends_document_and_hangs_up(self) -> None:
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        mock_sip = MagicMock()
        mock_dialog.sip = mock_sip
        remote_address = ("127.0.0.1", 5004)
        with patch("asyncio.create_task"):
            session = OutboundDualFaxSession(
                rtp=mock_rtp,
                dialog=mock_dialog,
                media=DualFaxSession.sdp_media_description(),
                caller=CallerID(""),
                document=b"pdf bytes",
            )
        mock_rtp.calls = {remote_address: session}
        with patch.object(session, "hang_up", new_callable=AsyncMock) as mock_hang_up:
            await session.transmit()
        # DualFaxSession.sdp_media_description defaults to T.38 (m=image), so
        # send_document delegates to FaxSession.send_document which chunks the
        # document into MTU-safe UDPTL datagrams.
        mock_rtp.send.assert_called_once_with(b"pdf bytes", remote_address)
        mock_hang_up.assert_awaited_once()
        mock_sip.close.assert_called_once()


class TestInboundDualFaxSession:
    def test_document_received__accumulates_data(self) -> None:
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        session = InboundDualFaxSession(
            rtp=mock_rtp,
            dialog=mock_dialog,
            media=DualFaxSession.sdp_media_description(),
            caller=CallerID(""),
        )
        session.document_received(b"page1")
        session.document_received(b"page2")
        assert session.document == b"page1page2"

    def test_data_received__delegates_to_document_received(self) -> None:
        received: list[bytes] = []
        session = InboundDualFaxSession(
            rtp=MagicMock(spec=RealtimeTransportProtocol),
            dialog=MagicMock(spec=Dialog),
            media=DualFaxSession.sdp_media_description(),
            caller=CallerID(""),
        )
        session.document_received = received.append
        session.data_received(b"packet", NetworkAddress("127.0.0.1", 5004))
        assert received == [b"packet"]


# ---------------------------------------------------------------------------
# Session base class: sdp_media_descriptions / select_session_class defaults
# ---------------------------------------------------------------------------


class TestSessionDefaults:
    def test_sdp_media_descriptions__defaults_to_single(self) -> None:
        descs = Session.sdp_media_descriptions(5004)
        assert len(descs) == 1

    def test_select_session_class__defaults_to_cls(self) -> None:
        remote = MediaDescription(
            media="audio",
            port=5004,
            proto="RTP/AVP",
            fmt=[RTPPayloadFormat(payload_type=0)],
        )
        assert Session.select_session_class(remote) is Session


# ---------------------------------------------------------------------------
# Regression: oversized document must not exceed UDP MTU (Errno 40)
# ---------------------------------------------------------------------------


class TestFaxSendDocumentMtu:
    """T.38 send_document must chunk oversized documents below the UDP MTU."""

    async def test_large_document__chunked_below_mtu(self) -> None:
        from voip.fax import MAX_UDPTL_DATAGRAM_SIZE

        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        session = FaxSession(
            rtp=mock_rtp,
            dialog=MagicMock(spec=Dialog),
            media=FaxSession.sdp_media_description(),
            caller=CallerID(""),
        )
        remote_address = ("127.0.0.1", 5004)
        mock_rtp.calls = {remote_address: session}
        # 10 KB document — far exceeds a single UDP datagram.
        document = b"\x00" * 10_000
        await session.send_document(document)
        assert mock_rtp.send.call_count >= 2
        for call in mock_rtp.send.call_args_list:
            chunk, addr = call.args
            assert addr == remote_address
            assert len(chunk) <= MAX_UDPTL_DATAGRAM_SIZE
        # All chunks concatenated must reconstruct the original document.
        sent = b"".join(call.args[0] for call in mock_rtp.send.call_args_list)
        assert sent == document

    async def test_small_document__single_datagram(self) -> None:
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        session = FaxSession(
            rtp=mock_rtp,
            dialog=MagicMock(spec=Dialog),
            media=FaxSession.sdp_media_description(),
            caller=CallerID(""),
        )
        remote_address = ("127.0.0.1", 5004)
        mock_rtp.calls = {remote_address: session}
        await session.send_document(b"tiny")
        mock_rtp.send.assert_called_once_with(b"tiny", remote_address)


class TestG711SendDocumentRtp:
    """G.711 send_document must packetize into RTP/PCMU frames."""

    async def test_multiframe_document__multiple_rtp_packets(self) -> None:
        from voip.fax import PCMU_FRAME_SIZE

        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        session = G711FaxSession(
            rtp=mock_rtp,
            dialog=MagicMock(spec=Dialog),
            media=G711FaxSession.sdp_media_description(),
            caller=CallerID(""),
        )
        remote_address = ("127.0.0.1", 5004)
        mock_rtp.calls = {remote_address: session}
        # Generate float32 PCM audio: 2 full frames + a partial third.
        audio = np.zeros(PCMU_FRAME_SIZE * 2 + 50, dtype=np.float32)
        await session.send_audio(audio, remote_address)
        assert mock_rtp.send.call_count == 3
        # Each sent datagram is an RTP packet with PT=0 (PCMU).
        for call in mock_rtp.send.call_args_list:
            data, addr = call.args
            assert addr == remote_address
            assert data[0] == 0x80  # V=2
            assert data[1] == 0  # PCMU
            # 12-byte RTP header precedes the payload.

    async def test_rtp_sequence_and_timestamp_increment(self) -> None:
        from voip.fax import PCMU_FRAME_SIZE, PCMU_TIMESTAMP_INCREMENT

        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        session = G711FaxSession(
            rtp=mock_rtp,
            dialog=MagicMock(spec=Dialog),
            media=G711FaxSession.sdp_media_description(),
            caller=CallerID(""),
        )
        remote_address = ("127.0.0.1", 5004)
        mock_rtp.calls = {remote_address: session}
        audio = np.zeros(PCMU_FRAME_SIZE * 3, dtype=np.float32)
        await session.send_audio(audio, remote_address)
        seqs = []
        timestamps = []
        for call in mock_rtp.send.call_args_list:
            data = call.args[0]
            seq = (data[2] << 8) | data[3]
            ts = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
            seqs.append(seq)
            timestamps.append(ts)
        assert seqs == [0, 1, 2]
        assert timestamps == [0, PCMU_TIMESTAMP_INCREMENT, PCMU_TIMESTAMP_INCREMENT * 2]

    async def test_pdf_document__uses_t30_modem(self, caplog) -> None:
        """send_document renders PDF via the T.30 modem instead of raw passthrough."""
        import logging  # noqa: PLC0415

        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        session = G711FaxSession(
            rtp=mock_rtp,
            dialog=MagicMock(spec=Dialog),
            media=G711FaxSession.sdp_media_description(),
            caller=CallerID(""),
        )
        remote_address = ("127.0.0.1", 5004)
        mock_rtp.calls = {remote_address: session}
        # Mock render_pdf_to_bitonal to avoid needing a real PDF.
        fake_image = np.zeros((10, 1728), dtype=np.uint8)
        with (
            caplog.at_level(logging.INFO, logger="voip.t30.modem"),
            patch("voip.fax.render_pdf_to_bitonal", return_value=[fake_image]),
        ):
            await session.send_document(b"%PDF-1.4\n" + b"\x00" * 160)
        # T30Modem logs image dimensions.
        assert any("compressed" in r.message for r in caplog.records)
