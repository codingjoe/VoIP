"""Tests for the CLI commands."""

from __future__ import annotations

from unittest.mock import patch

import pytest

_click_testing = pytest.importorskip("click.testing")
CliRunner = _click_testing.CliRunner


def make_runner():
    return CliRunner()


class TestTranscribeCLI:
    def test_transcribe__missing_server_exits_with_error(self):
        """Exit with an error when SIP_SERVER is not provided."""
        from sip.__main__ import sip

        runner = make_runner()
        result = runner.invoke(
            sip, ["transcribe", "--aor=sip:u@h", "--username=u", "--password=p"]
        )
        assert result.exit_code != 0
        assert "server" in result.output.lower() or "SIP_SERVER" in result.output

    def test_transcribe__missing_aor_exits_with_error(self):
        """Exit with an error when SIP_AOR is not provided."""
        from sip.__main__ import sip

        runner = make_runner()
        result = runner.invoke(
            sip,
            ["transcribe", "--server=sip.example.com", "--username=u", "--password=p"],
        )
        assert result.exit_code != 0
        assert "aor" in result.output.lower() or "SIP_AOR" in result.output

    def test_transcribe__server_with_port_is_parsed(self):
        """Parse host and port from SIP_SERVER when a colon is present."""
        from sip.__main__ import sip

        runner = make_runner()
        captured = {}

        async def fake_endpoint(factory, *, local_addr):
            protocol = factory()
            captured["server_addr"] = protocol._server_addr
            raise KeyboardInterrupt

        with (
            patch("asyncio.get_event_loop"),
            patch("sip.__main__.asyncio.get_running_loop") as mock_loop,
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
        from sip.__main__ import sip

        runner = make_runner()
        captured = {}

        async def fake_endpoint(factory, *, local_addr):
            protocol = factory()
            captured["server_addr"] = protocol._server_addr
            raise KeyboardInterrupt

        with (
            patch("asyncio.get_event_loop"),
            patch("sip.__main__.asyncio.get_running_loop") as mock_loop,
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
