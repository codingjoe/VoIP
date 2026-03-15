"""Tests for the CLI commands."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("click")
_click_testing = pytest.importorskip("click.testing")
CliRunner = _click_testing.CliRunner

# Stub out optional heavy dependencies so the CLI can be imported without them.
_WHISPER_STUBS = {
    "numpy": MagicMock(),
    "whisper": MagicMock(),
    "av": MagicMock(),
    "faster_whisper": MagicMock(),
    "ollama": MagicMock(),
    "pocket_tts": MagicMock(),
    "voip.audio": MagicMock(),
    "voip.ai": MagicMock(TranscribeCall=MagicMock, AgentCall=MagicMock),
}


def make_runner():
    """Return a Click test runner."""
    return CliRunner()


class TestParseAOR:
    def test_parse_aor__sips_no_port(self):
        """Parse scheme, user and host from a sips URI without port."""
        from voip.__main__ import _parse_aor

        assert _parse_aor("sips:alice@example.com") == (
            "sips",
            "alice",
            "example.com",
            None,
        )

    def test_parse_aor__sip_with_port(self):
        """Parse scheme, user, host and port from a sip URI with port."""
        from voip.__main__ import _parse_aor

        assert _parse_aor("sip:alice@example.com:5060") == (
            "sip",
            "alice",
            "example.com",
            5060,
        )

    def test_parse_aor__sips_with_port(self):
        """Parse all components including an explicit port."""
        from voip.__main__ import _parse_aor

        assert _parse_aor("sips:+15551234567@carrier.com:5061") == (
            "sips",
            "+15551234567",
            "carrier.com",
            5061,
        )

    def test_parse_aor__invalid_no_at(self):
        """Raise BadParameter when user@host part is missing."""
        import click
        from voip.__main__ import _parse_aor

        with pytest.raises(click.BadParameter):
            _parse_aor("sip:example.com")

    def test_parse_aor__invalid_no_scheme(self):
        """Raise BadParameter when scheme is missing."""
        import click
        from voip.__main__ import _parse_aor

        with pytest.raises(click.BadParameter):
            _parse_aor("alice@example.com")


class TestParseHostport:
    def test_parse_hostport__without_port(self):
        """Return default port 5061 when no port is specified."""
        from voip.__main__ import _parse_hostport

        assert _parse_hostport(None, None, "sip.example.com") == (
            "sip.example.com",
            5061,
        )

    def test_parse_hostport__with_port(self):
        """Parse host and port from HOST:PORT format."""
        from voip.__main__ import _parse_hostport

        assert _parse_hostport(None, None, "sip.example.com:5080") == (
            "sip.example.com",
            5080,
        )


class TestParseStunServer:
    def test_parse_stun_server__none_disables_stun(self):
        """Return None when STUN server is not configured."""
        from voip.__main__ import _parse_stun_server

        assert _parse_stun_server(None, None, None) is None

    def test_parse_stun_server__string_none_disables_stun(self):
        """Return None when STUN server is explicitly disabled."""
        from voip.__main__ import _parse_stun_server

        assert _parse_stun_server(None, None, "none") is None

    def test_parse_stun_server__without_port_uses_stun_default(self):
        """Return port 3478 when no port is specified for STUN."""
        from voip.__main__ import _parse_stun_server

        assert _parse_stun_server(None, None, "stun.example.com") == (
            "stun.example.com",
            3478,
        )


class TestVoIPCommand:
    def test_voip__verbose_flag(self):
        """Accept -v flag without error."""
        from voip.__main__ import voip

        result = make_runner().invoke(voip, ["-v", "--help"])
        assert result.exit_code == 0


class TestTranscribeCLI:
    def test_transcribe__sips_aor_uses_tls(self):
        """sips: AOR without explicit port defaults to TLS on port 5061."""
        from voip.__main__ import voip

        captured = {}

        async def fake_connection(factory, *, host, port, ssl):
            captured["host"] = host
            captured["port"] = port
            captured["ssl"] = ssl
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=secret",
                    "sips:alice@sip.example.com",
                    "transcribe",
                ],
                catch_exceptions=False,
            )
        assert captured.get("host") == "sip.example.com"
        assert captured.get("port") == 5061
        assert captured.get("ssl") is not None

    def test_transcribe__port_5060_uses_tcp(self):
        """Port 5060 in the AOR triggers plain TCP (no TLS)."""
        from voip.__main__ import voip

        captured = {}

        async def fake_connection(factory, *, host, port, ssl):
            captured["ssl"] = ssl
            captured["port"] = port
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=secret",
                    "sip:alice@example.com:5060",
                    "transcribe",
                ],
                catch_exceptions=False,
            )
        assert captured.get("ssl") is None
        assert captured.get("port") == 5060

    def test_transcribe__sip_aor_defaults_to_port_5060(self):
        """sip: AOR without explicit port defaults to port 5060 and plain TCP."""
        from voip.__main__ import voip

        captured = {}

        async def fake_connection(factory, *, host, port, ssl):
            captured["ssl"] = ssl
            captured["port"] = port
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=secret",
                    "sip:alice@example.com",
                    "transcribe",
                ],
                catch_exceptions=False,
            )
        assert captured.get("ssl") is None
        assert captured.get("port") == 5060

    def test_transcribe__no_tls_forces_tcp_on_sips_aor(self):
        """--no-tls forces plain TCP even when the AOR uses sips:."""
        from voip.__main__ import voip

        captured = {}

        async def fake_connection(factory, *, host, port, ssl):
            captured["ssl"] = ssl
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=secret",
                    "--no-tls",
                    "sips:alice@example.com",
                    "transcribe",
                ],
                catch_exceptions=False,
            )
        assert captured.get("ssl") is None

    def test_transcribe__aor_sets_protocol_aor(self):
        """The AOR positional argument sets the normalized aor on the protocol."""
        from voip.__main__ import voip

        captured = {}

        async def fake_connection(factory, *, host, port, ssl):
            protocol = factory()
            captured["aor"] = protocol.aor
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=p",
                    "sips:alice@carrier.example.com",
                    "transcribe",
                ],
                catch_exceptions=False,
            )
        # AOR stored on protocol must NOT include port (RFC 3261 §10)
        assert captured.get("aor") == "sips:alice@carrier.example.com"

    def test_transcribe__username_override(self):
        """--username overrides the user part from the AOR."""
        from voip.__main__ import voip

        captured = {}

        async def fake_connection(factory, *, host, port, ssl):
            protocol = factory()
            captured["aor"] = protocol.aor
            captured["username"] = protocol.username
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=p",
                    "--username=bob",
                    "sip:alice@carrier.example.com",
                    "transcribe",
                ],
                catch_exceptions=False,
            )
        assert captured.get("username") == "bob"
        assert captured.get("aor") == "sip:bob@carrier.example.com"

    def test_transcribe__proxy_overrides_outbound_proxy(self):
        """--proxy overrides the outbound proxy address derived from AOR."""
        from voip.__main__ import voip

        captured = {}

        async def fake_connection(factory, *, host, port, ssl):
            protocol = factory()
            captured["proxy"] = protocol.outbound_proxy
            captured["host"] = host
            captured["port"] = port
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=p",
                    "--proxy=proxy.carrier.com:5061",
                    "sips:alice@carrier.com",
                    "transcribe",
                ],
                catch_exceptions=False,
            )
        assert captured.get("proxy") == ("proxy.carrier.com", 5061)
        assert captured.get("host") == "proxy.carrier.com"
        assert captured.get("port") == 5061

    def test_transcribe__aor_with_port_parsed_as_outbound_proxy(self):
        """Port in AOR sets the outbound proxy port on the protocol."""
        from voip.__main__ import voip

        captured = {}

        async def fake_connection(factory, *, host, port, ssl):
            protocol = factory()
            captured["proxy"] = protocol.outbound_proxy
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=p",
                    "sips:alice@carrier.example.com:5080",
                    "transcribe",
                ],
                catch_exceptions=False,
            )
        assert captured.get("proxy") == ("carrier.example.com", 5080)

    def test_transcribe__stun_none_disables_stun(self):
        """Disable RTP STUN when --stun-server=none is passed."""
        from voip.__main__ import voip

        captured = {}

        async def fake_connection(factory, *, host, port, ssl):
            protocol = factory()
            captured["stun"] = protocol.rtp_stun_server_address
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=p",
                    "--stun-server=none",
                    "sips:alice@example.com",
                    "transcribe",
                ],
                catch_exceptions=False,
            )
        assert captured.get("stun") is None

    def test_transcribe__registered_logs_and_echoes(self):
        """Log and echo a message when registration succeeds."""
        from voip.__main__ import voip

        protocol_holder = {}

        async def fake_connection(factory, *, host, port, ssl):
            protocol = factory()
            protocol_holder["protocol"] = protocol
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=p",
                    "--stun-server=none",
                    "sips:alice@example.com",
                    "transcribe",
                ],
                catch_exceptions=False,
            )
        protocol = protocol_holder["protocol"]
        protocol.registered()  # exercises TranscribingProtocol.registered

    def test_transcribe__call_received_answers_call(self):
        """Answer the call when call_received is invoked."""
        from voip.__main__ import voip

        protocol_holder = {}

        async def fake_connection(factory, *, host, port, ssl):
            protocol = factory()
            protocol_holder["protocol"] = protocol
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=p",
                    "--stun-server=none",
                    "sips:alice@example.com",
                    "transcribe",
                ],
                catch_exceptions=False,
            )
        protocol = protocol_holder["protocol"]
        from voip.sip.messages import Request

        request = Request(
            method="INVITE",
            uri="sip:u@example.com",
            headers={"From": "sip:caller@example.com", "Call-ID": "test@pc"},
        )

        async def run():
            with patch.object(protocol, "answer") as mock_answer:
                protocol.connection_made(MagicMock())
                protocol._pending_invites.add(request.headers["Call-ID"])
                protocol.call_received(request)
                mock_answer.assert_called_once()

        asyncio.run(run())

    def test_transcribe__call_received_uses_whisper_call_class(self):
        """call_received answers with a WhisperCall subclass."""
        from voip.__main__ import voip

        protocol_holder = {}

        async def fake_connection(factory, *, host, port, ssl):
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
            patch("voip.audio.whisper") as wm,
        ):
            wm.load_model.return_value = MagicMock()
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=p",
                    "--stun-server=none",
                    "sips:alice@example.com",
                    "transcribe",
                ],
                catch_exceptions=False,
            )
        protocol = protocol_holder["protocol"]
        from voip.sip.messages import Request

        request = Request(
            method="INVITE",
            uri="sip:u@example.com",
            headers={"From": "sip:caller@example.com", "Call-ID": "test@pc"},
        )

        async def _run_whisper():
            with patch.object(protocol, "answer") as mock_answer:
                protocol.connection_made(MagicMock())
                protocol._pending_invites.add(request.headers["Call-ID"])
                protocol.call_received(request)
                mock_answer.assert_called_once()
                _, kwargs = mock_answer.call_args
                assert "call_class" in kwargs
                assert isinstance(kwargs["call_class"], type)

        asyncio.run(_run_whisper())


class TestAgentCLI:
    def test_agent__sips_aor_uses_tls(self):
        """sips: AOR without explicit port defaults to TLS on port 5061."""
        from voip.__main__ import voip

        captured = {}

        async def fake_connection(factory, *, host, port, ssl):
            captured["host"] = host
            captured["port"] = port
            captured["ssl"] = ssl
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=secret",
                    "sips:alice@sip.example.com",
                    "agent",
                ],
                catch_exceptions=False,
            )
        assert captured.get("host") == "sip.example.com"
        assert captured.get("port") == 5061
        assert captured.get("ssl") is not None

    def test_agent__port_5060_uses_tcp(self):
        """Port 5060 in the AOR triggers plain TCP (no TLS)."""
        from voip.__main__ import voip

        captured = {}

        async def fake_connection(factory, *, host, port, ssl):
            captured["ssl"] = ssl
            captured["port"] = port
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=secret",
                    "sip:alice@example.com:5060",
                    "agent",
                ],
                catch_exceptions=False,
            )
        assert captured.get("ssl") is None
        assert captured.get("port") == 5060

    def test_agent__call_received_uses_agent_call_class(self):
        """call_received answers with an AgentCall subclass."""
        from voip.__main__ import voip

        protocol_holder = {}

        async def fake_connection(factory, *, host, port, ssl):
            protocol = factory()
            protocol_holder["protocol"] = protocol
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=p",
                    "--stun-server=none",
                    "sips:alice@example.com",
                    "agent",
                ],
                catch_exceptions=False,
            )
        protocol = protocol_holder["protocol"]
        from voip.sip.messages import Request

        request = Request(
            method="INVITE",
            uri="sip:u@example.com",
            headers={"From": "sip:caller@example.com", "Call-ID": "test@pc"},
        )

        async def _run_agent():
            with patch.object(protocol, "answer") as mock_answer:
                protocol.connection_made(MagicMock())
                protocol._pending_invites.add(request.headers["Call-ID"])
                protocol.call_received(request)
                mock_answer.assert_called_once()
                _, kwargs = mock_answer.call_args
                assert "call_class" in kwargs
                assert isinstance(kwargs["call_class"], type)

        asyncio.run(_run_agent())

    def test_agent__ollama_model_option(self):
        """--ollama-model sets the ollama_model on the call class."""
        from voip.__main__ import voip

        protocol_holder = {}

        async def fake_connection(factory, *, host, port, ssl):
            protocol = factory()
            protocol_holder["protocol"] = protocol
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=p",
                    "--stun-server=none",
                    "sips:alice@example.com",
                    "agent",
                    "--ollama-model=mistral",
                ],
                catch_exceptions=False,
            )
        protocol = protocol_holder["protocol"]
        from voip.sip.messages import Request

        request = Request(
            method="INVITE",
            uri="sip:u@example.com",
            headers={"From": "sip:caller@example.com", "Call-ID": "test@pc"},
        )

        async def _run_ollama():
            with patch.object(protocol, "answer") as mock_answer:
                protocol.connection_made(MagicMock())
                protocol._pending_invites.add(request.headers["Call-ID"])
                protocol.call_received(request)
                _, kwargs = mock_answer.call_args
                call_cls = kwargs.get("call_class")
                if call_cls:
                    assert call_cls.ollama_model == "mistral"

        asyncio.run(_run_ollama())

    def test_agent__voice_option(self):
        """--voice sets the voice on the call class."""
        from voip.__main__ import voip

        protocol_holder = {}

        async def fake_connection(factory, *, host, port, ssl):
            protocol = factory()
            protocol_holder["protocol"] = protocol
            raise KeyboardInterrupt

        with (
            patch.dict(sys.modules, _WHISPER_STUBS),
            patch("asyncio.get_event_loop"),
            patch("voip.__main__.asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value.create_connection = fake_connection
            make_runner().invoke(
                voip,
                [
                    "sip",
                    "--password=p",
                    "--stun-server=none",
                    "sips:alice@example.com",
                    "agent",
                    "--voice=ellie",
                ],
                catch_exceptions=False,
            )
        protocol = protocol_holder["protocol"]
        from voip.sip.messages import Request

        request = Request(
            method="INVITE",
            uri="sip:u@example.com",
            headers={"From": "sip:caller@example.com", "Call-ID": "test@pc"},
        )

        async def _run_voice():
            with patch.object(protocol, "answer") as mock_answer:
                protocol.connection_made(MagicMock())
                protocol._pending_invites.add(request.headers["Call-ID"])
                protocol.call_received(request)
                _, kwargs = mock_answer.call_args
                call_cls = kwargs.get("call_class")
                if call_cls:
                    assert call_cls.voice == "ellie"

        asyncio.run(_run_voice())
