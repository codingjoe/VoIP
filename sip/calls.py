"""SIP call handling."""

from __future__ import annotations

import asyncio

from .messages import Request, Response

__all__ = ["IncomingCall"]

_RTP_HEADER_SIZE = 12


class _RTPProtocol(asyncio.DatagramProtocol):
    """Receive and dispatch RTP audio packets to the call handler."""

    def __init__(self, call: IncomingCall) -> None:
        self._call = call

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Strip RTP header and forward audio payload to the call handler."""
        if len(data) > _RTP_HEADER_SIZE:
            self._call.handle(data[_RTP_HEADER_SIZE:])


class IncomingCall:
    """An incoming SIP call."""

    def __init__(
        self,
        request: Request,
        addr: tuple[str, int],
        transport: asyncio.DatagramTransport,
    ) -> None:
        self._request = request
        self._addr = addr
        self._transport = transport

    @property
    def caller(self) -> str:
        """Return the caller's SIP address."""
        return self._request.headers.get("From", "")

    async def answer(self) -> None:
        """Answer the call and start receiving audio via RTP."""
        loop = asyncio.get_running_loop()
        rtp_transport, _ = await loop.create_datagram_endpoint(
            lambda: _RTPProtocol(self),
            local_addr=("0.0.0.0", 0),  # noqa: S104
        )
        local_addr = rtp_transport.get_extra_info("sockname")
        sdp = (
            f"v=0\r\n"
            f"c=IN IP4 {local_addr[0]}\r\n"
            f"m=audio {local_addr[1]} RTP/AVP 0\r\n"
        ).encode()
        self._transport.sendto(
            bytes(
                Response(
                    status_code=200,
                    reason="OK",
                    headers={
                        **{
                            key: value
                            for key, value in self._request.headers.items()
                            if key in ("Via", "To", "From", "Call-ID", "CSeq")
                        },
                        "Content-Type": "application/sdp",
                        "Content-Length": str(len(sdp)),
                    },
                    body=sdp,
                )
            ),
            self._addr,
        )

    def reject(self, status_code: int = 486, reason: str = "Busy Here") -> None:
        """Reject the call."""
        self._transport.sendto(
            bytes(
                Response(
                    status_code=status_code,
                    reason=reason,
                    headers={
                        key: value
                        for key, value in self._request.headers.items()
                        if key in ("Via", "To", "From", "Call-ID", "CSeq")
                    },
                )
            ),
            self._addr,
        )

    def handle(self, audio: bytes) -> None:
        """Handle incoming audio data."""
        return NotImplemented
