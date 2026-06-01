"""Tests for MSRP plain-text sending."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from voip.msrp import MSRPURI, MessageSessionRelayProtocol


class TestMSRPURI:
    def test_parse__roundtrip(self) -> None:
        """Parse and serialize MSRP URI without data loss."""
        uri = MSRPURI.parse("msrps://chat.example.com:2855/session-id;tcp")
        assert uri.scheme == "msrps"
        assert uri.host == "chat.example.com"
        assert uri.port == 2855
        assert uri.session_id == "session-id"
        assert uri.transport == "tcp"
        assert str(uri) == "msrps://chat.example.com:2855/session-id;tcp"

    def test_parse__raise_value_error_for_invalid_scheme(self) -> None:
        """Raise ValueError when URI scheme is not MSRP/MSRPS."""
        with pytest.raises(ValueError, match="Invalid MSRP URI"):
            MSRPURI.parse("https://chat.example.com/session-id;tcp")


class TestMessageSessionRelayProtocol:
    def test_build_send_request__builds_plain_text_frame(self) -> None:
        """Build a valid MSRP SEND frame for plain text."""
        target = MSRPURI.parse("msrp://chat.example.com:2855/session-id;tcp")
        sender = MSRPURI.parse("msrp://client.example.com:2855/local-id;tcp")

        frame = MessageSessionRelayProtocol.build_send_request(
            transaction_id="abc123",
            message_id="msg123",
            to_path=target,
            from_path=sender,
            text="hello",
        )

        assert b"MSRP abc123 SEND\r\n" in frame
        assert b"To-Path: msrp://chat.example.com:2855/session-id;tcp\r\n" in frame
        assert (
            b"From-Path: msrp://client.example.com:2855/local-id;tcp\r\n" in frame
        )
        assert b"Message-ID: msg123\r\n" in frame
        assert b"Byte-Range: 1-5/5\r\n" in frame
        assert b"Content-Type: text/plain\r\n" in frame
        assert frame.endswith(b"\r\n-------abc123$\r\n")

    async def test_send_text__uses_tls_for_msrps(self) -> None:
        """Use TLS when sending to MSRPS URI."""
        protocol = MessageSessionRelayProtocol(no_verify_tls=True)
        target = MSRPURI.parse("msrps://chat.example.com:2855/session-id;tcp")
        sender = MSRPURI.parse("msrps://client.example.com:2855/local-id;tcp")

        reader = AsyncMock()
        reader.read = AsyncMock(
            side_effect=[
                b"MSRP deadbeef 200 OK\r\n\r\n-------deadbeef$\r\n",
                b"",
            ]
        )
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.wait_closed = AsyncMock()

        with patch("voip.msrp.secrets.token_hex", side_effect=["deadbeef", "messageid"]):
            with patch(
                "voip.msrp.asyncio.open_connection",
                new_callable=AsyncMock,
                return_value=(reader, writer),
            ) as open_connection:
                response = await protocol.send_text(
                    target=target,
                    sender=sender,
                    text="hello",
                )

        _, kwargs = open_connection.call_args
        assert kwargs["host"] == "chat.example.com"
        assert kwargs["port"] == 2855
        assert kwargs["ssl"] is not None
        assert response.status_code == 200

    async def test_send_text__raise_runtime_error_for_failed_delivery(self) -> None:
        """Raise RuntimeError when MSRP endpoint returns a failure status."""
        protocol = MessageSessionRelayProtocol()
        target = MSRPURI.parse("msrp://chat.example.com:2855/session-id;tcp")
        sender = MSRPURI.parse("msrp://client.example.com:2855/local-id;tcp")

        reader = AsyncMock()
        reader.read = AsyncMock(
            side_effect=[
                b"MSRP deadbeef 500 Failure\r\n\r\n-------deadbeef$\r\n",
                b"",
            ]
        )
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.wait_closed = AsyncMock()

        with patch("voip.msrp.secrets.token_hex", side_effect=["deadbeef", "messageid"]):
            with patch(
                "voip.msrp.asyncio.open_connection",
                new_callable=AsyncMock,
                return_value=(reader, writer),
            ):
                with pytest.raises(RuntimeError, match="500"):
                    await protocol.send_text(
                        target=target,
                        sender=sender,
                        text="hello",
                    )
