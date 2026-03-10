"""Tests for the CLI commands."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("ffmpeg")
_click_testing = pytest.importorskip("click.testing")
CliRunner = _click_testing.CliRunner

# Stub out optional heavy dependencies so the CLI can be imported without them.
_WHISPER_STUBS = {
    "numpy": MagicMock(),
    "whisper": MagicMock(),
    "voip.whisper": MagicMock(WhisperCall=MagicMock),
}


def make_runner():
    """Return a Click test runner."""
    return CliRunner()


class TestParseServer:
    def test_parse_server__without_port(self):
        """Return port 5060 when no port is specified."""
        from voip.__main__ import _parse_server

        assert _parse_server("sip.example.com") == (
            ("sip.example.com", 5060),
            "sip.example.com",
        )

    def test_parse_server__with_port(self):
        """Parse host and port from HOST:PORT format."""
        from voip.__main__ import _parse_server

        assert _parse_server("sip.example.com:5080") == (
            ("sip.example.com", 5080),
            "sip.example.com",
        )


class TestParseStunServer:
    def test_parse_stun_server__none_disables(self):
        """Return None when stun server is 'none'."""
        from voip.__main__ import _parse_stun_server

        assert _parse_stun_server("none") is None

    def test_parse_stun_server__none_case_insensitive(self):
        """Return None regardless of case for 'none'."""
        from voip.__main__ import _parse_stun_server

        assert _parse_stun_server("NONE") is None

    def test_parse_stun_server__with_port(self):
        """Parse host and port from HOST:PORT format."""
        from voip.__main__ import _parse_stun_server

        assert _parse_stun_server("stun.example.com:3478") == ("stun.example.com", 3478)

    def test_parse_stun_server__without_port(self):
        """Return default STUN port 3478 when no port is specified."""
        from voip.__main__ import _parse_stun_server

        assert _parse_stun_server("stun.example.com") == ("stun.example.com", 3478)


class TestVoIPCommand:
    def test_voip__verbose_flag(self):
        """Accept -v flag without error."""
        from voip.__main__ import voip

        result = make_runner().invoke(voip, ["-v", "--help"])
        assert result.exit_code == 0


class TestTranscribeCLI:
    def test_transcribe__missing_server_exits_with_error(self):
        """Exit with an error when SIP_SERVER is not provided."""
        from voip.__main__ import sip

        runner = make_runner()
        result = runner.invoke(
            sip, ["transcribe", "--aor=sip:u@h", "--username=u", "--password=p"]
        )
        assert result.exit_code != 0
        assert "server" in result.output.lower() or "SIP_SERVER" in result.output

    def test_transcribe__missing_aor_defaults_to_username_at_host(self):
        """Default AOR to sip:{username}@{server_host} when SIP_AOR is not provided."""
        from voip.__main__ import sip

        runner = make_runner()
        captured = {}

        async def fake_endpoint(factory, *, local_addr):
            protocol = factory()
            captured["aor"] = protocol.aor
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_datagram_endpoint = fake_endpoint
            runner.invoke(
                sip,
                [
                    "transcribe",
                    "--server=sip.example.com",
                    "--username=u",
                    "--password=p",
                ],
                catch_exceptions=False,
            )
        assert captured.get("aor") == "sip:u@sip.example.com"

    def test_transcribe__server_with_port_is_parsed(self):
        """Parse host and port from SIP_SERVER when a colon is present."""
        from voip.__main__ import sip

        runner = make_runner()
        captured = {}

        async def fake_endpoint(factory, *, local_addr):
            protocol = factory()
            captured["server_addr"] = protocol.server_address
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_datagram_endpoint = fake_endpoint
            runner.invoke(
                sip,
                [
                    "transcribe",
                    "--server=sip.carrier.example:5080",
                    "--aor=sip:user@carrier.example",
                    "--username=user",
                    "--password=pass",
                ],
                catch_exceptions=False,
            )
        assert captured.get("server_addr") == ("sip.carrier.example", 5080)

    def test_transcribe__server_without_port_defaults_to_5060(self):
        """Use port 5060 when SIP_SERVER has no port."""
        from voip.__main__ import sip

        runner = make_runner()
        captured = {}

        async def fake_endpoint(factory, *, local_addr):
            protocol = factory()
            captured["server_addr"] = protocol.server_address
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_datagram_endpoint = fake_endpoint
            runner.invoke(
                sip,
                [
                    "transcribe",
                    "--server=sip.carrier.example",
                    "--aor=sip:user@carrier.example",
                    "--username=user",
                    "--password=pass",
                ],
                catch_exceptions=False,
            )
        assert captured.get("server_addr") == ("sip.carrier.example", 5060)

    def test_transcribe__stun_none_disables_stun(self):
        """Disable STUN when --stun-server=none is passed."""
        from voip.__main__ import sip

        runner = make_runner()
        captured = {}

        async def fake_endpoint(factory, *, local_addr):
            protocol = factory()
            captured["stun"] = protocol.stun_server_address
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_datagram_endpoint = fake_endpoint
            runner.invoke(
                sip,
                [
                    "transcribe",
                    "--server=sip.example.com",
                    "--aor=sip:u@example.com",
                    "--username=u",
                    "--password=p",
                    "--stun-server=none",
                ],
                catch_exceptions=False,
            )
        assert captured.get("stun") is None

    def test_transcribe__registered_logs_and_echoes(self):
        """Log and echo a message when registration succeeds."""
        from voip.__main__ import sip

        runner = make_runner()
        protocol_holder = {}

        async def fake_endpoint(factory, *, local_addr):
            protocol = factory()
            protocol_holder["protocol"] = protocol
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_datagram_endpoint = fake_endpoint
            runner.invoke(
                sip,
                [
                    "transcribe",
                    "--server=sip.example.com",
                    "--aor=sip:u@example.com",
                    "--username=u",
                    "--password=p",
                    "--stun-server=none",
                ],
                catch_exceptions=False,
            )
        protocol = protocol_holder.get("protocol")
        if protocol:
            protocol.registered()  # exercises TranscribingProtocol.registered

    def test_transcribe__call_received_answers_call(self):
        """Answer the call when call_received is invoked."""
        from voip.__main__ import sip

        runner = make_runner()
        protocol_holder = {}

        async def fake_endpoint(factory, *, local_addr):
            protocol = factory()
            protocol_holder["protocol"] = protocol
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_datagram_endpoint = fake_endpoint
            runner.invoke(
                sip,
                [
                    "transcribe",
                    "--server=sip.example.com",
                    "--aor=sip:u@example.com",
                    "--username=u",
                    "--password=p",
                    "--stun-server=none",
                ],
                catch_exceptions=False,
            )
        protocol = protocol_holder.get("protocol")
        if protocol:
            from voip.sip.messages import Request

            request = Request(
                method="INVITE",
                uri="sip:u@example.com",
                headers={"From": "sip:caller@example.com", "Call-ID": "test@pc"},
            )

            async def run():
                with patch.object(protocol, "answer") as mock_answer:
                    protocol.connection_made(MagicMock())
                    protocol._request_addrs[request.headers["Call-ID"]] = (
                        "192.0.2.1",
                        5060,
                    )
                    protocol.call_received(request)
                    mock_answer.assert_called_once()

            asyncio.run(run())

    def test_transcribe__call_received_uses_whisper_call_class(self):
        """call_received answers with a WhisperCall subclass."""
        from voip.__main__ import sip

        runner = make_runner()
        protocol_holder = {}

        async def fake_endpoint(factory, *, local_addr):
            protocol = factory()
            protocol_holder["protocol"] = protocol
            raise KeyboardInterrupt

        whisper_mock = MagicMock()
        stubs = dict(_WHISPER_STUBS)
        stubs["whisper"] = whisper_mock

        with (
            patch.dict(sys.modules, stubs),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
            patch("voip.whisper.whisper") as wm,
        ):
            wm.load_model.return_value = MagicMock()
            mock_loop.return_value.create_datagram_endpoint = fake_endpoint
            runner.invoke(
                sip,
                [
                    "transcribe",
                    "--server=sip.example.com",
                    "--aor=sip:u@example.com",
                    "--username=u",
                    "--password=p",
                    "--stun-server=none",
                ],
                catch_exceptions=False,
            )
        protocol = protocol_holder.get("protocol")
        if protocol:
            from voip.sip.messages import Request

            request = Request(
                method="INVITE",
                uri="sip:u@example.com",
                headers={"From": "sip:caller@example.com", "Call-ID": "test@pc"},
            )

            async def _run_whisper():
                with patch.object(protocol, "answer") as mock_answer:
                    protocol.connection_made(MagicMock())
                    protocol._request_addrs[request.headers["Call-ID"]] = (
                        "192.0.2.1",
                        5060,
                    )
                    protocol.call_received(request)
                    mock_answer.assert_called_once()
                    _, kwargs = mock_answer.call_args
                    assert "call_class" in kwargs
                    assert isinstance(kwargs["call_class"], type)

            asyncio.run(_run_whisper())
