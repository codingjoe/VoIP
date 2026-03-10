"""Integration tests requiring network access."""

import asyncio

import pytest

pytest.importorskip("ffmpeg")

from voip.stun import STUNProtocol


class _STUNOnlyProtocol(STUNProtocol, asyncio.DatagramProtocol):
    """Minimal datagram protocol that only handles STUN messages."""

    def connection_made(self, transport):
        self._transport = transport

    def datagram_received(self, data, addr):
        if data and data[0] < 4:
            self.handle_stun(data, addr)


@pytest.mark.integration
async def test_stun_discover__public_stun_server():
    """Discover public address via STUN against stun.l.google.com:19302."""
    loop = asyncio.get_running_loop()
    protocol = _STUNOnlyProtocol()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: protocol,
        local_addr=("0.0.0.0", 0),  # noqa: S104 – ephemeral port for outbound STUN discovery
    )
    try:
        host, port = await protocol.stun_discover("stun.l.google.com", 19302)
    finally:
        transport.close()
    assert isinstance(host, str)
    assert len(host) > 0
    assert isinstance(port, int)
    assert 1 <= port <= 65535
