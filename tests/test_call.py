"""Tests for the Call base class (voip/call.py)."""

from unittest.mock import MagicMock

import pytest
from voip.call import Call
from voip.rtp import RealtimeTransportProtocol
from voip.sip.types import CallerID


def make_call(**kwargs) -> Call:
    """Create a Call with mock rtp/sip for unit testing."""
    defaults: dict = {
        "rtp": MagicMock(spec=RealtimeTransportProtocol),
        "sip": MagicMock(),
    }
    defaults.update(kwargs)
    return Call(**defaults)


class TestCall:
    def test_caller__defaults_to_empty_string(self):
        """Caller defaults to an empty CallerID when not provided."""
        call = make_call()
        assert str(call.caller) == ""

    def test_caller__stores_provided_value(self):
        """Caller stores the CallerID passed at construction."""
        call = make_call(caller=CallerID("sip:bob@biloxi.com"))
        assert str(call.caller) == "sip:bob@biloxi.com"

    def test_media__defaults_to_none(self):
        """Media is None when not provided."""
        assert make_call().media is None

    def test_rtp_and_sip_stored_as_fields(self):
        """Rtp and sip back-references are stored on the instance."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_sip = MagicMock()
        call = Call(rtp=mock_rtp, sip=mock_sip)
        assert call.rtp is mock_rtp
        assert call.sip is mock_sip

    def test_datagram_received__noop(self):
        """datagram_received is a no-op in the base class."""
        make_call().datagram_received(b"data", ("192.0.2.1", 5004))  # must not raise

    def test_send_datagram__delegates_to_rtp(self):
        """send_datagram forwards data through the shared RTP socket."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        call = make_call(rtp=mock_rtp)
        call.send_datagram(b"audio", ("192.0.2.1", 5004))
        mock_rtp.send.assert_called_once_with(b"audio", ("192.0.2.1", 5004))

    def test_negotiate_codec__raises_not_implemented(self):
        """negotiate_codec raises NotImplementedError in the base class."""
        with pytest.raises(NotImplementedError):
            Call.negotiate_codec(MagicMock())

    async def test_hang_up__raises_not_implemented(self):
        """hang_up raises NotImplementedError in the base class."""
        with pytest.raises(NotImplementedError):
            await make_call().hang_up()
