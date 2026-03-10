"""Compatibility shim — use voip.sip and voip.rtp instead.

.. deprecated::
    Import from :mod:`voip.sip` and :mod:`voip.rtp` directly.
"""

from voip.rtp import RTP as IncomingCall
from voip.sip.protocol import SIP as IncomingCallProtocol
from voip.sip.session import RegisterSIP as RegisterProtocol

__all__ = ["IncomingCall", "IncomingCallProtocol", "RegisterProtocol"]
