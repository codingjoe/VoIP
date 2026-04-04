"""Shared fixtures for SIP tests."""

import dataclasses
import ipaddress

import pytest
from voip.rtp import RealtimeTransportProtocol, Session
from voip.sdp.types import MediaDescription, RTPPayloadFormat
from voip.sip.dialog import Dialog
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.types import SipURI
from voip.types import NetworkAddress


@dataclasses.dataclass
class FakeTransport:
    """Minimal asyncio.Transport stub that records written data."""

    _local_address: tuple = ("127.0.0.1", 5061)
    _peer_address: tuple = ("192.0.2.1", 5061)
    _ssl: bool = True
    sent: list[bytes] = dataclasses.field(default_factory=list)
    closed: bool = False

    def write(self, data: bytes) -> None:
        """Record outgoing data."""
        self.sent.append(data)

    def close(self) -> None:
        """Mark transport as closed."""
        self.closed = True

    def get_extra_info(self, key: str, default=None):
        """Return socket metadata."""
        match key:
            case "sockname":
                return self._local_address
            case "peername":
                return self._peer_address
            case "ssl_object":
                return object() if self._ssl else None
            case _:
                return default


class CallFixture(Session):
    """Minimal Session subclass for testing codec negotiation."""

    @classmethod
    def negotiate_codec(cls, remote_media: MediaDescription) -> MediaDescription:
        """Return the first format from the offered media."""
        return MediaDescription(
            media="audio",
            port=5004,
            proto="RTP/AVP",
            fmt=remote_media.fmt[:1] or [RTPPayloadFormat.from_pt(0)],
        )


@pytest.fixture
def fake_transport() -> FakeTransport:
    """Return a fresh FakeTransport with TLS."""
    return FakeTransport()


@pytest.fixture
def rtp() -> RealtimeTransportProtocol:
    """Return a RealtimeTransportProtocol with a pre-set public address."""
    mux = RealtimeTransportProtocol()
    mux.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
    return mux


@pytest.fixture
async def sip(
    fake_transport: FakeTransport, rtp: RealtimeTransportProtocol
) -> SessionInitiationProtocol:
    """Return a connected SIP session with keepalive cancelled."""
    session = SessionInitiationProtocol(
        aor=SipURI.parse("sips:alice:secret@example.com:5061"),
        rtp=rtp,
        dialog_class=Dialog,
    )
    session.connection_made(fake_transport)
    if session.keepalive_task is not None:
        session.keepalive_task.cancel()
        session.keepalive_task = None
    return session


#: A minimal incoming INVITE request as raw bytes.
INVITE_BYTES = (
    b"INVITE sip:alice@example.com SIP/2.0\r\n"
    b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKabc123\r\n"
    b"From: sip:bob@biloxi.com;tag=from-tag-1\r\n"
    b"To: sip:alice@example.com\r\n"
    b"Call-ID: test-call-id@biloxi.com\r\n"
    b"CSeq: 1 INVITE\r\n"
    b"\r\n"
)

#: INVITE bytes that include an SDP body with audio media.
INVITE_WITH_SDP_BYTES = (
    b"INVITE sip:alice@example.com SIP/2.0\r\n"
    b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKsdp456\r\n"
    b"From: sip:bob@biloxi.com;tag=from-tag-2\r\n"
    b"To: sip:alice@example.com\r\n"
    b"Call-ID: test-call-id-sdp@biloxi.com\r\n"
    b"CSeq: 1 INVITE\r\n"
    b"Content-Type: application/sdp\r\n"
    b"\r\n"
    b"v=0\r\n"
    b"o=- 1 1 IN IP4 192.0.2.1\r\n"
    b"s=-\r\n"
    b"c=IN IP4 192.0.2.1\r\n"
    b"t=0 0\r\n"
    b"m=audio 5004 RTP/AVP 0\r\n"
    b"a=rtpmap:0 PCMU/8000\r\n"
)
