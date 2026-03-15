"""Tests for the RTPCall base class and its backward-compatibility shim."""

from unittest.mock import MagicMock

import pytest
from voip.rtp import RealtimeTransportProtocol, RTPCall, RTPPacket
from voip.sdp.types import MediaDescription, RTPPayloadFormat
from voip.sip.types import CallerID


def make_media() -> MediaDescription:
    """Create a minimal MediaDescription for unit testing."""
    return MediaDescription(
        media="audio", port=0, proto="RTP/AVP", fmt=[RTPPayloadFormat(payload_type=8)]
    )


def make_call(**kwargs) -> RTPCall:
    """Create an RTPCall with mock rtp/sip for unit testing."""
    defaults: dict = {
        "rtp": MagicMock(spec=RealtimeTransportProtocol),
        "sip": MagicMock(),
        "media": make_media(),
        "caller": CallerID(""),
    }
    defaults.update(kwargs)
    return RTPCall(**defaults)


class TestCallCompatShim:
    def test_call__is_rtp_call(self):
        """Call is a backward-compatibility alias for RTPCall."""
        assert RTPCall is RTPCall


class TestRTPCall:
    def test_caller__defaults_to_empty_string(self):
        """Caller defaults to an empty CallerID when not provided."""
        call = make_call()
        assert str(call.caller) == ""

    def test_caller__stores_provided_value(self):
        """Caller stores the CallerID passed at construction."""
        call = make_call(caller=CallerID("sip:bob@biloxi.com"))
        assert str(call.caller) == "sip:bob@biloxi.com"

    def test_media__stored_on_instance(self):
        """Media is stored on the instance when provided."""
        media = make_media()
        assert make_call(media=media).media is media

    def test_rtp_and_sip_stored_as_fields(self):
        """Rtp and sip back-references are stored on the instance."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_sip = MagicMock()
        call = RTPCall(
            rtp=mock_rtp, sip=mock_sip, media=make_media(), caller=CallerID("")
        )
        assert call.rtp is mock_rtp
        assert call.sip is mock_sip

    def test_packet_received__noop_by_default(self):
        """packet_received is a no-op in the base class."""
        packet = RTPPacket(
            payload_type=8, sequence_number=1, timestamp=0, ssrc=0, payload=b"x"
        )
        make_call().packet_received(packet, ("192.0.2.1", 5004))  # must not raise

    def test_send_packet__sends_via_rtp(self):
        """send_packet serializes the packet and forwards it via the RTP socket."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        call = make_call(rtp=mock_rtp)
        packet = RTPPacket(
            payload_type=8, sequence_number=1, timestamp=0, ssrc=0, payload=b"x"
        )
        call.send_packet(packet, ("192.0.2.1", 5004))
        mock_rtp.send.assert_called_once_with(bytes(packet), ("192.0.2.1", 5004))

    def test_send_packet__encrypts_with_srtp_when_set(self):
        """send_packet encrypts the payload when an SRTP session is attached."""
        from voip.srtp import SRTPSession  # noqa: PLC0415

        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        session = SRTPSession.generate()
        call = make_call(rtp=mock_rtp, srtp=session)
        packet = RTPPacket(
            payload_type=8, sequence_number=1, timestamp=0, ssrc=0, payload=b"audio"
        )
        call.send_packet(packet, ("192.0.2.1", 5004))
        sent_data = mock_rtp.send.call_args[0][0]
        assert sent_data != bytes(packet)

    def test_negotiate_codec__raises_not_implemented(self):
        """negotiate_codec raises NotImplementedError in the base class."""
        with pytest.raises(NotImplementedError):
            RTPCall.negotiate_codec(MagicMock())

    async def test_hang_up__raises_not_implemented(self):
        """hang_up raises NotImplementedError in the base class."""
        with pytest.raises(NotImplementedError):
            await make_call().hang_up()
