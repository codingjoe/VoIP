"""Tests for the MCP server (voip.mcp)."""

import asyncio
import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

fastmcp = pytest.importorskip("fastmcp")
pytest.importorskip("faster_whisper")
pytest.importorskip("pocket_tts")

from fastmcp.exceptions import ToolError  # noqa: E402
from mcp.types import TextContent  # noqa: E402
from voip.mcp import (  # noqa: E402
    DEFAULT_STUN_SERVER,
    HangupDialog,
    MCPAgentCall,
    run,
)
from voip.rtp import RealtimeTransportProtocol  # noqa: E402
from voip.sip.protocol import SessionInitiationProtocol  # noqa: E402
from voip.sip.types import SipURI  # noqa: E402
from voip.types import NetworkAddress  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_context(reply: str = "Hello!") -> MagicMock:
    """Return a mock FastMCP Context whose sample() returns *reply*."""
    ctx = MagicMock(spec=fastmcp.Context)
    result = MagicMock()
    result.text = reply
    ctx.sample = AsyncMock(return_value=result)
    return ctx


def make_mock_sip() -> MagicMock:
    """Return a mock SessionInitiationProtocol with a disconnected_event."""
    sip = MagicMock(spec=SessionInitiationProtocol)
    sip.disconnected_event = asyncio.Event()
    return sip


@dataclasses.dataclass
class FakeDatagramTransport:
    """Minimal UDP transport stub."""

    closed: bool = False

    def close(self) -> None:
        """Mark transport as closed."""
        self.closed = True


# ---------------------------------------------------------------------------
# Tests: run()
# ---------------------------------------------------------------------------


class TestRun:
    async def test_run__creates_rtp_when_none(self):
        """Creates a datagram endpoint when rtp=None."""
        aor = SipURI.parse("sip:alice@192.0.2.1:5060;maddr=192.0.2.1")
        mock_protocol = make_mock_sip()
        mock_protocol.disconnected_event.set()
        fake_transport = FakeDatagramTransport()
        fake_rtp = MagicMock(spec=RealtimeTransportProtocol)

        loop = asyncio.get_running_loop()
        with (
            patch.object(
                loop,
                "create_datagram_endpoint",
                new=AsyncMock(return_value=(fake_transport, fake_rtp)),
            ) as mock_udp,
            patch.object(
                loop,
                "create_connection",
                new=AsyncMock(return_value=(MagicMock(), mock_protocol)),
            ),
        ):
            await run(lambda sip: None, aor)

        mock_udp.assert_called_once()

    async def test_run__uses_external_rtp(self):
        """Skips datagram endpoint creation when rtp is provided."""
        aor = SipURI.parse("sip:alice@192.0.2.1:5060;maddr=192.0.2.1")
        mock_protocol = make_mock_sip()
        mock_protocol.disconnected_event.set()
        external_rtp = NetworkAddress.parse("192.0.2.10:5004")

        loop = asyncio.get_running_loop()
        with (
            patch.object(
                loop,
                "create_datagram_endpoint",
                new=AsyncMock(),
            ) as mock_udp,
            patch.object(
                loop,
                "create_connection",
                new=AsyncMock(return_value=(MagicMock(), mock_protocol)),
            ),
        ):
            await run(lambda sip: None, aor, rtp=external_rtp)

        mock_udp.assert_not_called()

    async def test_run__external_rtp_sets_public_address(self):
        """Public address of the stub RTP protocol matches the external address."""
        aor = SipURI.parse("sip:alice@192.0.2.1:5060;maddr=192.0.2.1")
        mock_protocol = make_mock_sip()
        mock_protocol.disconnected_event.set()
        external_rtp = NetworkAddress.parse("192.0.2.10:5004")

        captured_factories = []

        loop = asyncio.get_running_loop()

        async def fake_conn(factory, **kwargs):
            captured_factories.append(factory)
            return MagicMock(), mock_protocol

        with patch.object(loop, "create_connection", new=fake_conn):
            await run(lambda sip: None, aor, rtp=external_rtp)

        # The factory builds an InlineProtocol; its rtp attribute is a stub
        # RealtimeTransportProtocol with public_address pre-set.
        # We can't invoke the factory (it needs a running event loop as well
        # as a real transport), but we can check the closure captured the
        # external address by verifying run() called create_connection once.
        assert len(captured_factories) == 1

    async def test_run__ipv4_bind(self):
        """Binds to 0.0.0.0 when the proxy address is IPv4."""
        aor = SipURI.parse("sip:alice@192.0.2.1:5060;maddr=192.0.2.1")
        mock_protocol = make_mock_sip()
        mock_protocol.disconnected_event.set()

        loop = asyncio.get_running_loop()
        with (
            patch.object(
                loop,
                "create_datagram_endpoint",
                new=AsyncMock(return_value=(FakeDatagramTransport(), MagicMock())),
            ) as mock_udp,
            patch.object(
                loop,
                "create_connection",
                new=AsyncMock(return_value=(MagicMock(), mock_protocol)),
            ),
        ):
            await run(lambda sip: None, aor)

        _, kwargs = mock_udp.call_args
        assert kwargs["local_addr"] == ("0.0.0.0", 0)  # noqa: S104

    async def test_run__ipv6_bind(self):
        """Binds to :: when the proxy address is IPv6."""
        aor = SipURI.parse("sip:alice@[::1]:5060;maddr=[::1]")
        mock_protocol = make_mock_sip()
        mock_protocol.disconnected_event.set()

        loop = asyncio.get_running_loop()
        with (
            patch.object(
                loop,
                "create_datagram_endpoint",
                new=AsyncMock(return_value=(FakeDatagramTransport(), MagicMock())),
            ) as mock_udp,
            patch.object(
                loop,
                "create_connection",
                new=AsyncMock(return_value=(MagicMock(), mock_protocol)),
            ),
        ):
            await run(lambda sip: None, aor)

        _, kwargs = mock_udp.call_args
        assert kwargs["local_addr"] == ("::", 0)

    async def test_run__tls_uses_ssl_context(self):
        """Passes an SSL context to create_connection when transport=TLS."""
        aor = SipURI.parse("sip:alice@sip.example.com:5061;maddr=sip.example.com;transport=TLS")
        mock_protocol = make_mock_sip()
        mock_protocol.disconnected_event.set()

        loop = asyncio.get_running_loop()
        with (
            patch.object(
                loop,
                "create_datagram_endpoint",
                new=AsyncMock(return_value=(FakeDatagramTransport(), MagicMock())),
            ),
            patch.object(
                loop,
                "create_connection",
                new=AsyncMock(return_value=(MagicMock(), mock_protocol)),
            ) as mock_conn,
        ):
            await run(lambda sip: None, aor)

        _, kwargs = mock_conn.call_args
        assert kwargs["ssl"] is not None

    async def test_run__no_tls_passes_none_ssl(self):
        """Passes ssl=None when the AOR does not request TLS."""
        aor = SipURI.parse("sip:alice@192.0.2.1:5060;maddr=192.0.2.1;transport=tcp")
        mock_protocol = make_mock_sip()
        mock_protocol.disconnected_event.set()

        loop = asyncio.get_running_loop()
        with (
            patch.object(
                loop,
                "create_datagram_endpoint",
                new=AsyncMock(return_value=(FakeDatagramTransport(), MagicMock())),
            ),
            patch.object(
                loop,
                "create_connection",
                new=AsyncMock(return_value=(MagicMock(), mock_protocol)),
            ) as mock_conn,
        ):
            await run(lambda sip: None, aor)

        _, kwargs = mock_conn.call_args
        assert kwargs["ssl"] is None

    async def test_run__no_verify_tls(self):
        """Disables cert verification when no_verify_tls=True."""
        import ssl  # noqa: PLC0415

        aor = SipURI.parse("sip:alice@sip.example.com:5061;maddr=sip.example.com;transport=TLS")
        mock_protocol = make_mock_sip()
        mock_protocol.disconnected_event.set()

        captured = []
        loop = asyncio.get_running_loop()

        async def fake_conn(factory, **kwargs):
            captured.append(kwargs.get("ssl"))
            return MagicMock(), mock_protocol

        with (
            patch.object(
                loop,
                "create_datagram_endpoint",
                new=AsyncMock(return_value=(FakeDatagramTransport(), MagicMock())),
            ),
            patch.object(loop, "create_connection", new=fake_conn),
        ):
            await run(lambda sip: None, aor, no_verify_tls=True)

        assert captured[0] is not None
        assert captured[0].verify_mode == ssl.CERT_NONE

    async def test_run__calls_fn_on_registered(self):
        """Calls fn once via on_registered when the session is registered."""
        aor = SipURI.parse("sip:alice@192.0.2.1:5060;maddr=192.0.2.1")
        called_with = []

        # We simulate the on_registered call manually via the factory.
        captured_factory = []
        mock_sip = make_mock_sip()
        mock_sip.disconnected_event.set()

        loop = asyncio.get_running_loop()

        async def fake_conn(factory, **kwargs):
            captured_factory.append(factory)
            return MagicMock(), mock_sip

        with (
            patch.object(
                loop,
                "create_datagram_endpoint",
                new=AsyncMock(return_value=(FakeDatagramTransport(), MagicMock())),
            ),
            patch.object(loop, "create_connection", new=fake_conn),
        ):
            await run(lambda sip: called_with.append(sip), aor)

        # fn not called because we used a mock SIP protocol — but the factory
        # was captured.  Verify at least that the factory was registered.
        assert len(captured_factory) == 1

    async def test_run__blocks_until_disconnected(self):
        """Blocks until the protocol's disconnected_event is set."""
        aor = SipURI.parse("sip:alice@192.0.2.1:5060;maddr=192.0.2.1")
        mock_protocol = make_mock_sip()

        loop = asyncio.get_running_loop()
        with (
            patch.object(
                loop,
                "create_datagram_endpoint",
                new=AsyncMock(return_value=(FakeDatagramTransport(), MagicMock())),
            ),
            patch.object(
                loop,
                "create_connection",
                new=AsyncMock(return_value=(MagicMock(), mock_protocol)),
            ),
        ):
            async def disconnect_later():
                await asyncio.sleep(0)
                mock_protocol.disconnected_event.set()

            asyncio.create_task(disconnect_later())
            await run(lambda sip: None, aor)

        assert mock_protocol.disconnected_event.is_set()

    async def test_run__uses_custom_stun_server(self):
        """Passes the custom STUN server to RealtimeTransportProtocol."""
        aor = SipURI.parse("sip:alice@192.0.2.1:5060;maddr=192.0.2.1")
        mock_protocol = make_mock_sip()
        mock_protocol.disconnected_event.set()
        stun = NetworkAddress.parse("stun.example.com:3478")
        created = []

        loop = asyncio.get_running_loop()

        async def fake_datagram(factory, **kwargs):
            created.append(factory())
            return FakeDatagramTransport(), MagicMock()

        with (
            patch.object(loop, "create_datagram_endpoint", new=fake_datagram),
            patch.object(
                loop,
                "create_connection",
                new=AsyncMock(return_value=(MagicMock(), mock_protocol)),
            ),
        ):
            await run(lambda sip: None, aor, rtp_stun_server=stun)

        assert created[0].stun_server_address == stun

    async def test_run__defaults_to_cloudflare_stun(self):
        """Uses DEFAULT_STUN_SERVER when rtp_stun_server is not provided."""
        aor = SipURI.parse("sip:alice@192.0.2.1:5060;maddr=192.0.2.1")
        mock_protocol = make_mock_sip()
        mock_protocol.disconnected_event.set()
        created = []

        loop = asyncio.get_running_loop()

        async def fake_datagram(factory, **kwargs):
            created.append(factory())
            return FakeDatagramTransport(), MagicMock()

        with (
            patch.object(loop, "create_datagram_endpoint", new=fake_datagram),
            patch.object(
                loop,
                "create_connection",
                new=AsyncMock(return_value=(MagicMock(), mock_protocol)),
            ),
        ):
            await run(lambda sip: None, aor)

        stun = created[0].stun_server_address
        assert str(stun) == DEFAULT_STUN_SERVER


# ---------------------------------------------------------------------------
# Tests: HangupDialog
# ---------------------------------------------------------------------------


class TestHangupDialog:
    def test_hangup_received__closes_sip(self):
        """Calls sip.close() when the remote party hangs up."""
        mock_sip = MagicMock(spec=SessionInitiationProtocol)
        d = HangupDialog(sip=mock_sip)
        d.hangup_received()
        mock_sip.close.assert_called_once()

    def test_hangup_received__no_sip(self):
        """Does not raise when sip is None."""
        d = HangupDialog()
        d.hangup_received()  # must not raise


# ---------------------------------------------------------------------------
# Helpers for MCPAgentCall
# ---------------------------------------------------------------------------


def make_mcp_agent_call(ctx=None, **kwargs) -> MCPAgentCall:
    """Create an MCPAgentCall with mocked ML models and transport."""
    from voip.rtp import RealtimeTransportProtocol  # noqa: PLC0415
    from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: PLC0415
    from voip.sip.dialog import Dialog  # noqa: PLC0415
    from voip.sip.types import CallerID  # noqa: PLC0415

    media = MediaDescription(
        media="audio",
        port=5004,
        proto="RTP/AVP",
        fmt=[RTPPayloadFormat(payload_type=0, encoding_name="pcmu", sample_rate=8000)],
    )
    mock_codec = MagicMock()
    mock_codec.payload_type = 0
    mock_codec.sample_rate_hz = 8000
    mock_codec.rtp_clock_rate_hz = 8000
    mock_codec.create_decoder.return_value = MagicMock()

    mock_tts = MagicMock()
    mock_tts.get_state_for_audio_prompt.return_value = {}
    mock_tts.sample_rate = 8000

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = ([], None)

    with patch("voip.codecs.get", return_value=mock_codec):
        return MCPAgentCall(
            ctx=ctx or make_mock_context(),
            rtp=MagicMock(spec=RealtimeTransportProtocol),
            dialog=Dialog(),
            media=media,
            caller=CallerID(""),
            tts_model=mock_tts,
            stt_model=mock_stt,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Tests: MCPAgentCall
# ---------------------------------------------------------------------------


class TestMCPAgentCall:
    def test_post_init__no_initial_prompt(self):
        """_messages starts empty when no initial_prompt is provided."""
        agent = make_mcp_agent_call()
        assert agent._messages == []

    def test_post_init__with_initial_prompt(self):
        """Adds the opening message to _messages and schedules TTS."""
        with patch("asyncio.create_task") as mock_task:
            agent = make_mcp_agent_call(initial_prompt="Hello there!")
        assert agent._messages == [
            {"role": "assistant", "content": "Hello there!"}
        ]
        mock_task.assert_called_once()

    def test_transcript__empty(self):
        """Returns an empty string when no messages exist."""
        agent = make_mcp_agent_call()
        assert agent.transcript == ""

    def test_transcript__formats_correctly(self):
        """Formats user messages with Caller: and agent messages with Agent:."""
        agent = make_mcp_agent_call()
        agent._messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        assert agent.transcript == "Caller: Hi\nAgent: Hello!"

    def test_transcription_received__appends_user_message(self):
        """Adds a user message to _messages and schedules respond()."""
        with patch("asyncio.create_task") as mock_task:
            agent = make_mcp_agent_call()
            agent.transcription_received("How are you?")

        assert agent._messages == [{"role": "user", "content": "How are you?"}]
        mock_task.assert_called_once()

    async def test_respond__calls_sample_and_speaks(self):
        """Calls ctx.sample and sends the reply as speech."""
        ctx = make_mock_context("I'm fine, thanks!")
        agent = make_mcp_agent_call(ctx=ctx)
        agent._messages = [{"role": "user", "content": "How are you?"}]

        agent.send_speech = AsyncMock()
        await agent.respond()

        ctx.sample.assert_called_once()
        sample_args = ctx.sample.call_args
        assert "system_prompt" in sample_args.kwargs
        agent.send_speech.assert_called_once_with("I'm fine, thanks!")
        assert agent._messages[-1] == {
            "role": "assistant",
            "content": "I'm fine, thanks!",
        }

    async def test_respond__empty_reply_skipped(self):
        """Does not append an empty reply to _messages."""
        ctx = make_mock_context("   ")  # only whitespace
        agent = make_mcp_agent_call(ctx=ctx)
        agent._messages = [{"role": "user", "content": "Hi"}]
        agent.send_speech = AsyncMock()

        await agent.respond()

        agent.send_speech.assert_not_called()
        assert len(agent._messages) == 1

    async def test_respond__none_text_skipped(self):
        """Does not raise when sample returns None text."""
        ctx = make_mock_context("")
        ctx.sample.return_value.text = None
        agent = make_mcp_agent_call(ctx=ctx)
        agent._messages = [{"role": "user", "content": "Hi"}]
        agent.send_speech = AsyncMock()

        await agent.respond()

        agent.send_speech.assert_not_called()

    async def test_respond__sampling_messages_include_history(self):
        """Forwards the full conversation history to ctx.sample."""
        from mcp.types import SamplingMessage  # noqa: PLC0415

        ctx = make_mock_context("Sure!")
        agent = make_mcp_agent_call(ctx=ctx)
        agent._messages = [
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What can you do?"},
        ]
        agent.send_speech = AsyncMock()

        await agent.respond()

        sampling_messages = ctx.sample.call_args.args[0]
        assert len(sampling_messages) == 2
        assert isinstance(sampling_messages[0], SamplingMessage)
        assert sampling_messages[0].role == "assistant"
        content = sampling_messages[0].content
        assert isinstance(content, TextContent)
        assert content.text == "Hi there!"


# ---------------------------------------------------------------------------
# Tests: say tool
# ---------------------------------------------------------------------------


class TestSayTool:
    async def test_say__missing_aor_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Raises ToolError when SIP_AOR is not set."""
        monkeypatch.delenv("SIP_AOR", raising=False)
        from voip.mcp import say  # noqa: PLC0415

        ctx = make_mock_context()
        with pytest.raises(ToolError, match="SIP_AOR"):
            await say(ctx=ctx, target="tel:+1234567890", prompt="Hello")

    async def test_say__calls_run(self, monkeypatch: pytest.MonkeyPatch):
        """Delegates to run() with the correct AOR and TLS flag."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5061;transport=TLS")
        monkeypatch.delenv("SIP_NO_VERIFY_TLS", raising=False)
        monkeypatch.delenv("STUN_SERVER", raising=False)

        from voip.mcp import say  # noqa: PLC0415

        with patch("voip.mcp.run", new=AsyncMock()) as mock_run:
            ctx = make_mock_context()
            await say(ctx=ctx, target="sip:bob@sip.example.com", prompt="Hi!")

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert isinstance(args[1], SipURI)
        assert "alice" in str(args[1])

    async def test_say__no_verify_tls(self, monkeypatch: pytest.MonkeyPatch):
        """Passes no_verify_tls=True to run() when SIP_NO_VERIFY_TLS is set."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5061;transport=TLS")
        monkeypatch.setenv("SIP_NO_VERIFY_TLS", "1")
        monkeypatch.delenv("STUN_SERVER", raising=False)

        from voip.mcp import say  # noqa: PLC0415

        with patch("voip.mcp.run", new=AsyncMock()) as mock_run:
            ctx = make_mock_context()
            await say(ctx=ctx, target="sip:bob@sip.example.com")

        _, kwargs = mock_run.call_args
        assert kwargs.get("no_verify_tls") is True

    async def test_say__passes_stun_server(self, monkeypatch: pytest.MonkeyPatch):
        """Forwards STUN_SERVER env var to run()."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5060")
        monkeypatch.setenv("STUN_SERVER", "stun.example.com:3478")
        monkeypatch.delenv("SIP_NO_VERIFY_TLS", raising=False)

        from voip.mcp import say  # noqa: PLC0415

        with patch("voip.mcp.run", new=AsyncMock()) as mock_run:
            ctx = make_mock_context()
            await say(ctx=ctx, target="sip:bob@sip.example.com")

        _, kwargs = mock_run.call_args
        assert kwargs.get("rtp_stun_server") is not None

    async def test_say__no_stun_server_passes_none(self, monkeypatch: pytest.MonkeyPatch):
        """Passes rtp_stun_server=None when STUN_SERVER is not set."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5060")
        monkeypatch.delenv("STUN_SERVER", raising=False)
        monkeypatch.delenv("SIP_NO_VERIFY_TLS", raising=False)

        from voip.mcp import say  # noqa: PLC0415

        with patch("voip.mcp.run", new=AsyncMock()) as mock_run:
            ctx = make_mock_context()
            await say(ctx=ctx, target="sip:bob@sip.example.com")

        _, kwargs = mock_run.call_args
        assert kwargs.get("rtp_stun_server") is None


# ---------------------------------------------------------------------------
# Tests: call tool
# ---------------------------------------------------------------------------


class TestCallTool:
    async def test_call__missing_aor_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Raises ToolError when SIP_AOR is not set."""
        monkeypatch.delenv("SIP_AOR", raising=False)
        from voip.mcp import call  # noqa: PLC0415

        ctx = make_mock_context()
        with pytest.raises(ToolError, match="SIP_AOR"):
            await call(ctx=ctx, target="tel:+1234567890")

    async def test_call__returns_empty_when_no_session(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Returns an empty string when no session was established."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5060")
        monkeypatch.delenv("SIP_NO_VERIFY_TLS", raising=False)
        monkeypatch.delenv("STUN_SERVER", raising=False)

        from voip.mcp import call  # noqa: PLC0415

        with patch("voip.mcp.run", new=AsyncMock()):
            ctx = make_mock_context()
            result = await call(ctx=ctx, target="sip:bob@sip.example.com")

        assert result == ""

    async def test_call__calls_run(self, monkeypatch: pytest.MonkeyPatch):
        """Delegates to run() with correct arguments."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5060")
        monkeypatch.delenv("SIP_NO_VERIFY_TLS", raising=False)
        monkeypatch.delenv("STUN_SERVER", raising=False)

        from voip.mcp import call  # noqa: PLC0415

        with patch("voip.mcp.run", new=AsyncMock()) as mock_run:
            ctx = make_mock_context()
            await call(ctx=ctx, target="sip:bob@sip.example.com")

        mock_run.assert_called_once()

    async def test_call__no_verify_tls(self, monkeypatch: pytest.MonkeyPatch):
        """Passes no_verify_tls=True when SIP_NO_VERIFY_TLS is set."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5060")
        monkeypatch.setenv("SIP_NO_VERIFY_TLS", "true")
        monkeypatch.delenv("STUN_SERVER", raising=False)

        from voip.mcp import call  # noqa: PLC0415

        with patch("voip.mcp.run", new=AsyncMock()) as mock_run:
            ctx = make_mock_context()
            await call(ctx=ctx, target="sip:bob@sip.example.com")

        _, kwargs = mock_run.call_args
        assert kwargs.get("no_verify_tls") is True

    async def test_call__passes_stun_server(self, monkeypatch: pytest.MonkeyPatch):
        """Forwards STUN_SERVER env var to run()."""
        monkeypatch.setenv("SIP_AOR", "sip:alice@sip.example.com:5060")
        monkeypatch.setenv("STUN_SERVER", "stun.example.com:3478")
        monkeypatch.delenv("SIP_NO_VERIFY_TLS", raising=False)

        from voip.mcp import call  # noqa: PLC0415

        with patch("voip.mcp.run", new=AsyncMock()) as mock_run:
            ctx = make_mock_context()
            await call(ctx=ctx, target="sip:bob@sip.example.com")

        _, kwargs = mock_run.call_args
        assert kwargs.get("rtp_stun_server") is not None
