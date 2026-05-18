"""Tests for the T.38 FAX session (voip.fax)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voip.fax import FaxCall, FaxSession
from voip.rtp import RealtimeTransportProtocol
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


# ---------------------------------------------------------------------------
# FaxSession class attributes
# ---------------------------------------------------------------------------


class TestFaxSessionAttributes:
    def test_media_type__is_image(self) -> None:
        """media_type class variable is 'image' for T.38."""
        assert FaxSession.media_type == "image"

    def test_t38_version__is_zero(self) -> None:
        """T38_VERSION defaults to 0."""
        assert FaxSession.T38_VERSION == 0

    def test_t38_max_bit_rate__is_14400(self) -> None:
        """T38_MAX_BIT_RATE defaults to 14400 bps."""
        assert FaxSession.T38_MAX_BIT_RATE == 14400


# ---------------------------------------------------------------------------
# FaxSession.sdp_formats
# ---------------------------------------------------------------------------


class TestSdpFormats:
    def test_sdp_formats__returns_t38_format(self) -> None:
        """sdp_formats returns a single T.38 payload format."""
        formats = FaxSession.sdp_formats()
        assert len(formats) == 1
        assert formats[0].payload_type == "t38"


# ---------------------------------------------------------------------------
# FaxSession.sdp_media_description
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FaxSession.negotiate_codec
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# FaxSession.data_received
# ---------------------------------------------------------------------------


class TestDataReceived:
    def test_data_received__delegates_to_document_received(self) -> None:
        """data_received forwards the raw data to document_received."""
        received: list[bytes] = []
        session = make_fax_session()
        session.document_received = received.append
        session.data_received(b"t38 packet", NetworkAddress("127.0.0.1", 5004))
        assert received == [b"t38 packet"]


# ---------------------------------------------------------------------------
# FaxSession.document_received
# ---------------------------------------------------------------------------


class TestDocumentReceived:
    def test_document_received__is_noop(self) -> None:
        """document_received is a no-op in the base class."""
        make_fax_session().document_received(b"data")  # must not raise


# ---------------------------------------------------------------------------
# FaxSession.send_document
# ---------------------------------------------------------------------------


class TestSendDocument:
    def test_send_document__sends_to_remote_address(self) -> None:
        """send_document transmits data to the registered remote address."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        session = make_fax_session(rtp=mock_rtp)
        remote_address = ("127.0.0.1", 5004)
        mock_rtp.calls = {remote_address: session}
        session.send_document(b"fax content")
        mock_rtp.send.assert_called_once_with(b"fax content", remote_address)

    def test_send_document__logs_warning_when_no_remote_address(
        self, caplog
    ) -> None:
        """send_document logs a warning and does nothing when not registered."""
        import logging  # noqa: PLC0415

        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        session = make_fax_session(rtp=mock_rtp)
        mock_rtp.calls = {}
        with caplog.at_level(logging.WARNING, logger="voip.fax"):
            session.send_document(b"fax content")
        mock_rtp.send.assert_not_called()
        assert any("No remote address" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# FaxCall
# ---------------------------------------------------------------------------


class TestFaxCall:
    async def test_transmit__sends_document_and_hangs_up(self) -> None:
        """transmit sends the document, hangs up, and closes the SIP connection."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        mock_sip = MagicMock()
        mock_dialog.sip = mock_sip
        remote_address = ("127.0.0.1", 5004)

        session = FaxCall(
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
        """transmit does not raise when dialog.sip is None."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_dialog = MagicMock(spec=Dialog)
        mock_dialog.sip = None
        mock_rtp.calls = {}

        session = FaxCall(
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
            FaxCall(
                rtp=mock_rtp,
                dialog=mock_dialog,
                media=FaxSession.sdp_media_description(),
                caller=CallerID(""),
                document=b"pdf",
            )

        mock_create_task.assert_called_once()
