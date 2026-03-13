"""
Real-time Transport Protocol (RTP) implementation of RFC 3550.

See also: https://datatracker.ietf.org/doc/html/rfc3550#section-5
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import logging
import typing
from typing import TYPE_CHECKING

from voip.stun import STUNProtocol

if TYPE_CHECKING:
    from voip.call import Call

__all__ = ["RTP", "RTPPacket", "RTPPayloadType", "RealtimeTransportProtocol"]

logger = logging.getLogger(__name__)


class RTPPayloadType(enum.IntEnum):
    """Common RTP payload types, aligned with SDP media format identifiers.

    Static payload types (0–95) are defined by RFC 3551.
    Dynamic payload types (96–127) are negotiated via SDP.
    Opus uses payload type 111 per RFC 7587.
    """

    PCMU = 0  # G.711 µ-law (RFC 3551)
    PCMA = 8  # G.711 A-law (RFC 3551)
    G722 = 9  # G.722 (RFC 3551)
    OPUS = 111  # RFC 7587 (dynamic)


@dataclasses.dataclass
class RTPPacket:
    """A parsed RTP packet (RFC 3550 §5.1)."""

    payload_type: int
    sequence_number: int
    timestamp: int
    ssrc: int
    payload: bytes

    #: Fixed RTP header size in bytes (RFC 3550 §5.1).
    header_size: int = dataclasses.field(default=12, init=False, repr=False)

    @classmethod
    def parse(cls, data: bytes) -> RTPPacket:
        """Parse raw RTP bytes into an RTPPacket."""
        if len(data) < 12:
            raise ValueError(f"RTP packet too short: {len(data)} bytes")
        payload_type = data[1] & 0x7F
        sequence_number = (data[2] << 8) | data[3]
        timestamp = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
        ssrc = (data[8] << 24) | (data[9] << 16) | (data[10] << 8) | data[11]
        return cls(
            payload_type=payload_type,
            sequence_number=sequence_number,
            timestamp=timestamp,
            ssrc=ssrc,
            payload=data[12:],
        )


@dataclasses.dataclass(kw_only=True, slots=True)
class RealtimeTransportProtocol(STUNProtocol):
    """RTP multiplexer: routes incoming datagrams to per-call handlers (RFC 3550).

    One instance manages multiple simultaneous calls on a single UDP socket.
    Register per-call :class:`~voip.call.Call` handlers with
    :meth:`register_call`; each incoming datagram is dispatched to the
    matching handler's :meth:`~voip.call.Call.datagram_received` method by
    remote source address.

    Use ``addr=None`` in :meth:`register_call` as a wildcard catch-all for
    calls whose remote RTP address is not known in advance (no SDP in INVITE).
    """

    rtp_header_size: typing.ClassVar[int] = 12
    calls: dict[tuple[str, int] | None, Call] = dataclasses.field(
        init=False, default_factory=dict
    )
    public_address: asyncio.Future[tuple[str, int]] = dataclasses.field(
        init=False, default_factory=asyncio.Future
    )

    def stun_connection_made(self, transport, addr):
        self.public_address.set_result(addr)

    def register_call(
        self,
        addr: tuple[str, int] | None,
        handler: Call,
    ) -> None:
        """Register *handler* for RTP traffic arriving from *addr*.

        Use ``addr=None`` as a wildcard to handle traffic from any source that
        has no dedicated routing entry (useful when the caller's RTP address is
        not known in advance from the INVITE SDP).

        Args:
            addr: Remote ``(ip, port)`` as it will appear in incoming datagrams,
                or ``None`` to register a wildcard catch-all handler.
            handler: A :class:`~voip.call.Call` instance whose
                :meth:`~voip.call.Call.datagram_received` will be called for
                matching packets.
        """
        logger.info(
            json.dumps(
                {
                    "event": "rtp_call_registered",
                    "addr": list(addr) if addr else None,
                    "handler": type(handler).__name__,
                }
            ),
            extra={"addr": addr},
        )
        self.calls[addr] = handler

    def unregister_call(self, addr: tuple[str, int] | None) -> None:
        """Remove the handler registered for *addr*.

        Args:
            addr: The same key that was passed to :meth:`register_call`.
                Silently ignored when no handler is registered for *addr*.
        """
        if addr in self.calls:
            logger.info(
                json.dumps(
                    {
                        "event": "rtp_call_unregistered",
                        "addr": list(addr) if addr else None,
                    }
                ),
                extra={"addr": addr},
            )
            self.calls.pop(addr)

    def packet_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Route an incoming SRTP datagram to the matching per-call handler.

        Looks up *addr* in the call registry.  Falls back to the wildcard
        ``None`` handler when no exact match exists.  Drops the packet with a
        debug log when no handler is registered at all.

        When the matched handler carries an SRTP session the packet is
        authenticated and decrypted before being forwarded; packets that fail
        authentication are logged at WARNING level and discarded.
        """
        handler = self.calls.get(addr)
        if handler is None:
            handler = self.calls.get(None)
        if handler is not None:
            if handler.srtp is not None:
                decrypted = handler.srtp.decrypt(data)
                if decrypted is None:
                    logger.warning(
                        "SRTP authentication failed for packet from %s:%s, discarding",
                        addr[0],
                        addr[1],
                    )
                    return
                data = decrypted
            logger.debug(
                "Routing RTP packet from %s:%s to %s",
                addr[0],
                addr[1],
                type(handler).__name__,
            )
            handler.datagram_received(data, addr)
        else:
            logger.debug(
                "No call handler registered for %s:%s, dropping RTP packet",
                addr[0],
                addr[1],
            )


RTP = RealtimeTransportProtocol
