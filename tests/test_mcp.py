"""Tests for the MCP server (voip.mcp)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

fastmcp = pytest.importorskip("fastmcp")
pytest.importorskip("faster_whisper")
pytest.importorskip("pocket_tts")


import voip.mcp  # noqa: E402
from voip.mcp import MCPAgentCall, call, connection_pool, run, say  # noqa: E402
from voip.rtp import RealtimeTransportProtocol  # noqa: E402
from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: E402
from voip.sip.dialog import Dialog  # noqa: E402
from voip.sip.protocol import SessionInitiationProtocol  # noqa: E402
from voip.sip.types import CallerID, SipURI  # noqa: E402
from voip.types import NetworkAddress  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_media() -> MediaDescription:
    """Return a minimal MediaDescription for testing."""
    return MediaDescription(
        media="audio", port=0, proto="RTP/AVP", fmt=[RTPPayloadFormat(payload_type=8)]
    )


def make_mock_context(reply: str = "Hello!") -> MagicMock:
    """Return a mock FastMCP Context whose sample() returns *reply*."""
    ctx = MagicMock(spec=fastmcp.Context)
    result = MagicMock()
    result.text = reply
    ctx.sample = AsyncMock(return_value=result)
    return ctx


def make_agent_call(
    ctx: MagicMock | None = None,
    system_prompt: str | None = None,
    salutation: str = "",
    messages: list[dict] | None = None,
) -> MCPAgentCall:
    """Instantiate a MCPAgentCall with mocked heavy dependencies.

    Patches model-loading factories so no real models are downloaded.
    """
    if ctx is None:
        ctx = make_mock_context()
    mock_tts = MagicMock()
    mock_tts.get_state_for_audio_prompt.return_value = {}
    mock_stt = MagicMock()

    kwargs: dict = {
        "rtp": MagicMock(spec=RealtimeTransportProtocol),
        "dialog": MagicMock(spec=Dialog),
        "media": make_media(),
        "caller": CallerID(""),
        "tts_model": mock_tts,
        "stt_model": mock_stt,
        "ctx": ctx,
        "salutation": salutation,
    }
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt

    agent = MCPAgentCall(**kwargs)
    if messages is not None:
        # Inject synthetic _messages (bypasses AgentCall's auto-population).
        object.__setattr__(agent, "_messages", messages)
    return agent


# ---------------------------------------------------------------------------
# MCPAgentCall.transcript
# ---------------------------------------------------------------------------


class TestTranscript:
    def test_transcript__empty(self) -> None:
        """Return empty string when only a system message exists."""
        agent = make_agent_call()
        assert agent.transcript == ""

    def test_transcript__user_and_assistant(self) -> None:
        """Format user/assistant turns; exclude system message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        agent = make_agent_call(messages=messages)
        assert agent.transcript == "Caller: Hi\nAgent: Hello!"

    def test_transcript__skips_system_messages(self) -> None:
        """System messages are excluded from the transcript."""
        messages = [
            {"role": "system", "content": "secret"},
            {"role": "user", "content": "test"},
        ]
        agent = make_agent_call(messages=messages)
        assert "secret" not in agent.transcript
        assert "Caller: test" in agent.transcript


# ---------------------------------------------------------------------------
# MCPAgentCall.transcription_received
# ---------------------------------------------------------------------------


class TestTranscriptionReceived:
    async def test_transcription_received__appends_user_message(self) -> None:
        """Incoming transcription is appended as a user message."""
        agent = make_agent_call()
        with patch.object(agent, "cancel_outbound_audio"):
            with patch("asyncio.create_task"):
                agent.transcription_received("How are you?")

        user_msgs = [m for m in agent._messages if m["role"] == "user"]
        assert any(m["content"] == "How are you?" for m in user_msgs)

    async def test_transcription_received__cancels_pending_task(self) -> None:
        """A pending response task is cancelled before scheduling a new one."""
        agent = make_agent_call()
        old_task = MagicMock(spec=asyncio.Task)
        old_task.done.return_value = False
        object.__setattr__(agent, "_response_task", old_task)

        with patch.object(agent, "cancel_outbound_audio"):
            with patch("asyncio.create_task") as mock_create:
                mock_create.return_value = MagicMock(spec=asyncio.Task)
                agent.transcription_received("test")

        old_task.cancel.assert_called_once()

    async def test_transcription_received__skips_cancel_when_done(self) -> None:
        """A completed response task is not cancelled."""
        agent = make_agent_call()
        old_task = MagicMock(spec=asyncio.Task)
        old_task.done.return_value = True
        object.__setattr__(agent, "_response_task", old_task)

        with patch.object(agent, "cancel_outbound_audio"):
            with patch("asyncio.create_task"):
                agent.transcription_received("test")

        old_task.cancel.assert_not_called()


# ---------------------------------------------------------------------------
# MCPAgentCall.respond
# ---------------------------------------------------------------------------


class TestRespond:
    async def test_respond__speaks_reply(self) -> None:
        """respond() speaks the LLM reply and appends it to _messages."""
        ctx = make_mock_context("Nice to meet you.")
        agent = make_agent_call(ctx=ctx, messages=[
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ])
        with patch.object(agent, "send_speech", new_callable=AsyncMock) as mock_send:
            await agent.respond()

        mock_send.assert_awaited_once_with("Nice to meet you.")
        assert {"role": "assistant", "content": "Nice to meet you."} in agent._messages

    async def test_respond__filters_system_from_sampling(self) -> None:
        """System messages are not forwarded to ctx.sample."""
        ctx = make_mock_context("OK")
        agent = make_agent_call(ctx=ctx, messages=[
            {"role": "system", "content": "secret"},
            {"role": "user", "content": "hello"},
        ])
        with patch.object(agent, "send_speech", new_callable=AsyncMock):
            await agent.respond()

        args, kwargs = ctx.sample.call_args
        sent_messages = args[0] if args else kwargs.get("messages", [])
        roles = [m.role for m in sent_messages]
        assert "system" not in roles

    async def test_respond__uses_system_prompt_kwarg(self) -> None:
        """system_prompt is passed as a keyword arg to ctx.sample."""
        ctx = make_mock_context("OK")
        agent = make_agent_call(ctx=ctx, system_prompt="Act as an assistant.")
        with patch.object(agent, "send_speech", new_callable=AsyncMock):
            await agent.respond()

        _, kwargs = ctx.sample.call_args
        assert kwargs.get("system_prompt") == "Act as an assistant."

    async def test_respond__empty_reply_is_silent(self) -> None:
        """An empty or whitespace-only reply does not call send_speech."""
        ctx = make_mock_context("   ")
        agent = make_agent_call(ctx=ctx, messages=[
            {"role": "user", "content": "hello"},
        ])
        with patch.object(agent, "send_speech", new_callable=AsyncMock) as mock_send:
            await agent.respond()

        mock_send.assert_not_awaited()

    async def test_respond__none_text_is_silent(self) -> None:
        """A None result.text does not call send_speech."""
        ctx = MagicMock(spec=fastmcp.Context)
        result = MagicMock()
        result.text = None
        ctx.sample = AsyncMock(return_value=result)
        agent = make_agent_call(ctx=ctx, messages=[{"role": "user", "content": "hi"}])
        with patch.object(agent, "send_speech", new_callable=AsyncMock) as mock_send:
            await agent.respond()

        mock_send.assert_not_awaited()


# ---------------------------------------------------------------------------
# say tool
# ---------------------------------------------------------------------------


class TestSayTool:
    async def test_say__dials_with_parsed_uri(self) -> None:
        """say() resolves the target via parse_uri relative to the AOR."""
        aor = SipURI.parse("sip:alice@carrier.example;transport=TLS")
        mock_sip = MagicMock(spec=SessionInitiationProtocol)
        mock_sip.aor = aor
        connection_pool.sip = mock_sip

        ctx = make_mock_context()
        target_uri = SipURI.parse("sip:bob@carrier.example")

        with patch("voip.mcp.parse_uri", return_value=target_uri) as mock_parse:
            with patch("voip.mcp.Dialog") as MockDialog:
                mock_dialog = MagicMock(spec=Dialog)
                MockDialog.return_value = mock_dialog
                mock_dialog.dial = AsyncMock()

                await say(ctx=ctx, target="sip:bob@carrier.example", prompt="Hello!")

        mock_parse.assert_called_once_with("sip:bob@carrier.example", aor)
        MockDialog.assert_called_once_with(sip=mock_sip)
        mock_dialog.dial.assert_awaited_once()
        _, kwargs = mock_dialog.dial.call_args
        assert kwargs["session_class"].__name__ == "SayCall"
        assert kwargs["text"] == "Hello!"

    async def test_say__empty_prompt(self) -> None:
        """say() passes an empty string when no prompt is given."""
        aor = SipURI.parse("sip:alice@example.com")
        mock_sip = MagicMock(spec=SessionInitiationProtocol)
        mock_sip.aor = aor
        connection_pool.sip = mock_sip

        ctx = make_mock_context()
        with patch("voip.mcp.parse_uri", return_value=aor):
            with patch("voip.mcp.Dialog") as MockDialog:
                mock_dialog = MagicMock(spec=Dialog)
                MockDialog.return_value = mock_dialog
                mock_dialog.dial = AsyncMock()
                await say(ctx=ctx, target="sip:bob@example.com")

        _, kwargs = mock_dialog.dial.call_args
        assert kwargs["text"] == ""


# ---------------------------------------------------------------------------
# call tool
# ---------------------------------------------------------------------------


class TestCallTool:
    async def test_call__returns_transcript(self) -> None:
        """call() returns dialog.session.transcript after dialing."""
        aor = SipURI.parse("sip:alice@example.com")
        mock_sip = MagicMock(spec=SessionInitiationProtocol)
        mock_sip.aor = aor
        connection_pool.sip = mock_sip

        ctx = make_mock_context()
        target_uri = SipURI.parse("sip:bob@example.com")

        mock_session = MagicMock()
        mock_session.transcript = "Caller: Hi\nAgent: Hello!"

        with patch("voip.mcp.parse_uri", return_value=target_uri):
            with patch("voip.mcp.Dialog") as MockDialog:
                mock_dialog = MagicMock(spec=Dialog)
                mock_dialog.session = mock_session
                MockDialog.return_value = mock_dialog
                mock_dialog.dial = AsyncMock()

                result = await call(
                    ctx=ctx,
                    target="sip:bob@example.com",
                    initial_prompt="Hello!",
                )

        assert result == "Caller: Hi\nAgent: Hello!"
        _, kwargs = mock_dialog.dial.call_args
        assert kwargs["session_class"] is MCPAgentCall
        assert kwargs["ctx"] is ctx
        assert kwargs["salutation"] == "Hello!"
        assert "system_prompt" not in kwargs

    async def test_call__passes_system_prompt_when_given(self) -> None:
        """call() passes system_prompt to MCPAgentCall when explicitly supplied."""
        aor = SipURI.parse("sip:alice@example.com")
        mock_sip = MagicMock(spec=SessionInitiationProtocol)
        mock_sip.aor = aor
        connection_pool.sip = mock_sip

        ctx = make_mock_context()
        with patch("voip.mcp.parse_uri", return_value=aor):
            with patch("voip.mcp.Dialog") as MockDialog:
                mock_dialog = MagicMock(spec=Dialog)
                mock_dialog.session = MagicMock(transcript="")
                MockDialog.return_value = mock_dialog
                mock_dialog.dial = AsyncMock()

                await call(
                    ctx=ctx,
                    target="sip:bob@example.com",
                    system_prompt="Act as a robot.",
                )

        _, kwargs = mock_dialog.dial.call_args
        assert kwargs["system_prompt"] == "Act as a robot."

    async def test_call__default_empty_initial_prompt(self) -> None:
        """call() passes empty string as salutation when initial_prompt is omitted."""
        aor = SipURI.parse("sip:alice@example.com")
        mock_sip = MagicMock(spec=SessionInitiationProtocol)
        mock_sip.aor = aor
        connection_pool.sip = mock_sip

        ctx = make_mock_context()
        with patch("voip.mcp.parse_uri", return_value=aor):
            with patch("voip.mcp.Dialog") as MockDialog:
                mock_dialog = MagicMock(spec=Dialog)
                mock_dialog.session = MagicMock(transcript="")
                MockDialog.return_value = mock_dialog
                mock_dialog.dial = AsyncMock()

                await call(ctx=ctx, target="sip:bob@example.com")

        _, kwargs = mock_dialog.dial.call_args
        assert kwargs["salutation"] == ""


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    async def test_run__sets_connection_pool_sip(self) -> None:
        """run() stores the SIP protocol in connection_pool.sip."""
        aor = SipURI.parse("sip:alice@example.com")
        mock_protocol = MagicMock(spec=SessionInitiationProtocol)

        fn = MagicMock()
        with patch.object(
            SessionInitiationProtocol, "run", new_callable=AsyncMock, return_value=mock_protocol
        ):
            with patch.object(voip.mcp.mcp, "run_async", new_callable=AsyncMock):
                await run(fn, aor)

        assert connection_pool.sip is mock_protocol

    async def test_run__calls_mcp_run_async_with_transport(self) -> None:
        """run() forwards the transport argument to mcp.run_async."""
        aor = SipURI.parse("sip:alice@example.com")
        mock_protocol = MagicMock(spec=SessionInitiationProtocol)

        with patch.object(
            SessionInitiationProtocol, "run", new_callable=AsyncMock, return_value=mock_protocol
        ):
            with patch.object(voip.mcp.mcp, "run_async", new_callable=AsyncMock) as mock_run:
                await run(lambda: None, aor, transport="stdio")

        mock_run.assert_awaited_once_with(transport="stdio")

    async def test_run__passes_no_verify_tls(self) -> None:
        """run() forwards no_verify_tls to SessionInitiationProtocol.run."""
        aor = SipURI.parse("sip:alice@example.com;transport=TLS")
        mock_protocol = MagicMock(spec=SessionInitiationProtocol)

        with patch.object(
            SessionInitiationProtocol, "run", new_callable=AsyncMock, return_value=mock_protocol
        ) as mock_sip_run:
            with patch.object(voip.mcp.mcp, "run_async", new_callable=AsyncMock):
                await run(lambda: None, aor, no_verify_tls=True)

        _, kwargs = mock_sip_run.call_args
        assert kwargs["no_verify_tls"] is True

    async def test_run__passes_stun_server(self) -> None:
        """run() forwards a custom stun_server to SessionInitiationProtocol.run."""
        aor = SipURI.parse("sip:alice@example.com")
        stun = NetworkAddress.parse("stun.example.com:3478")
        mock_protocol = MagicMock(spec=SessionInitiationProtocol)

        with patch.object(
            SessionInitiationProtocol, "run", new_callable=AsyncMock, return_value=mock_protocol
        ) as mock_sip_run:
            with patch.object(voip.mcp.mcp, "run_async", new_callable=AsyncMock):
                await run(lambda: None, aor, stun_server=stun)

        _, kwargs = mock_sip_run.call_args
        assert kwargs["stun_server"] is stun
