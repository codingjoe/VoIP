"""Tests for the VoIP MCP server."""

import asyncio
import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import mcp.types as mcp_types
import numpy as np
from voip.mcp import (
    HangupDialog,
    MCPAgentCall,
    MCPCallProtocol,
    OutboundMessageProtocol,
    VoIPServer,
)
from voip.rtp import RealtimeTransportProtocol
from voip.sdp.types import MediaDescription, RTPPayloadFormat
from voip.sip.dialog import Dialog
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.types import CallerID, SipURI
from voip.types import NetworkAddress


def _make_media() -> MediaDescription:
    """Return a minimal MediaDescription for testing."""
    return MediaDescription(
        media="audio",
        port=5004,
        proto="RTP/AVP",
        fmt=[RTPPayloadFormat(payload_type=0)],
    )


def _make_tts_model() -> MagicMock:
    """Return a minimal TTS model mock."""
    model = MagicMock()
    model.sample_rate = 16000
    model.get_state_for_audio_prompt.return_value = {}
    model.generate_audio.return_value = MagicMock(
        **{"numpy.return_value": np.zeros(100, dtype=np.float32)}
    )
    return model


def _make_agent_call(
    *,
    conversation: list | None = None,
    initial_prompt: str = "",
    system_prompt: str = "",
    mcp_session: object | None = None,
) -> MCPAgentCall:
    """Return a minimal MCPAgentCall for testing."""
    if conversation is None:
        conversation = []
    if mcp_session is None:
        mcp_session = AsyncMock()
    return MCPAgentCall(
        rtp=MagicMock(spec=RealtimeTransportProtocol),
        dialog=Dialog(),
        media=_make_media(),
        caller=CallerID(""),
        tts_model=_make_tts_model(),
        stt_model=MagicMock(),
        mcp_session=mcp_session,
        conversation=conversation,
        initial_prompt=initial_prompt,
        system_prompt=system_prompt,
    )


class TestMCPAgentCall:
    def test_init__no_initial_prompt(self):
        """Create call without initial prompt."""
        call = _make_agent_call()
        assert call.conversation == []

    async def test_init__with_initial_prompt(self):
        """Create call with initial prompt adds it to conversation."""
        call = _make_agent_call(initial_prompt="Hello!")
        # Let the scheduled task run
        await asyncio.sleep(0)
        assert call.conversation == [{"role": "assistant", "content": "Hello!"}]

    async def test_transcription_received__adds_to_conversation(self):
        """Add transcription to conversation and schedule a response task."""
        call = _make_agent_call()
        call.transcription_received("How are you?")
        assert call.conversation == [{"role": "user", "content": "How are you?"}]

    async def test_transcription_received__schedules_respond(self):
        """Schedule a respond task when transcription is received."""
        session = AsyncMock()
        session.create_message.return_value = mcp_types.CreateMessageResult(
            role="assistant",
            content=mcp_types.TextContent(type="text", text="Fine, thanks."),
            model="test-model",
        )
        call = _make_agent_call(mcp_session=session)
        call.send_speech = AsyncMock()
        call.transcription_received("How are you?")
        await asyncio.sleep(0)
        session.create_message.assert_awaited_once()

    async def test_respond__sends_speech(self):
        """Call send_speech with the LLM reply."""
        session = AsyncMock()
        session.create_message.return_value = mcp_types.CreateMessageResult(
            role="assistant",
            content=mcp_types.TextContent(type="text", text="I am fine."),
            model="test-model",
        )
        call = _make_agent_call(
            conversation=[{"role": "user", "content": "How are you?"}],
            mcp_session=session,
        )
        call.send_speech = AsyncMock()
        await call.respond()
        call.send_speech.assert_awaited_once_with("I am fine.")
        assert {"role": "assistant", "content": "I am fine."} in call.conversation

    async def test_respond__empty_reply(self):
        """Skip send_speech when LLM returns an empty reply."""
        session = AsyncMock()
        session.create_message.return_value = mcp_types.CreateMessageResult(
            role="assistant",
            content=mcp_types.TextContent(type="text", text="   "),
            model="test-model",
        )
        call = _make_agent_call(
            conversation=[{"role": "user", "content": "Hello"}],
            mcp_session=session,
        )
        call.send_speech = AsyncMock()
        await call.respond()
        call.send_speech.assert_not_awaited()

    async def test_respond__with_system_prompt(self):
        """Pass system_prompt in the sampling request."""
        session = AsyncMock()
        session.create_message.return_value = mcp_types.CreateMessageResult(
            role="assistant",
            content=mcp_types.TextContent(type="text", text="reply"),
            model="test-model",
        )
        call = _make_agent_call(
            conversation=[{"role": "user", "content": "Hi"}],
            system_prompt="Be helpful.",
            mcp_session=session,
        )
        call.send_speech = AsyncMock()
        await call.respond()
        _, kwargs = session.create_message.await_args
        assert kwargs["system_prompt"] == "Be helpful."

    async def test_respond__no_system_prompt_sends_none(self):
        """Pass None as system_prompt when none is set."""
        session = AsyncMock()
        session.create_message.return_value = mcp_types.CreateMessageResult(
            role="assistant",
            content=mcp_types.TextContent(type="text", text="reply"),
            model="test-model",
        )
        call = _make_agent_call(
            conversation=[{"role": "user", "content": "Hi"}],
            mcp_session=session,
        )
        call.send_speech = AsyncMock()
        await call.respond()
        _, kwargs = session.create_message.await_args
        assert kwargs["system_prompt"] is None


class TestHangupDialog:
    def test_hangup_received__closes_sip(self):
        """Close the SIP connection when hangup is received."""
        mock_sip = MagicMock()
        d = HangupDialog(sip=mock_sip)
        d.hangup_received()
        mock_sip.close.assert_called_once()

    def test_hangup_received__no_sip(self):
        """Handle hangup gracefully when SIP session is absent."""
        d = HangupDialog()
        d.hangup_received()  # Must not raise


class TestOutboundMessageProtocol:
    async def test_on_registered__dials_target(self):
        """Initiate an outbound call after SIP registration."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_rtp.public_address = None
        target = SipURI.parse("sip:bob@192.0.2.1:5060")
        protocol = OutboundMessageProtocol(
            aor=SipURI.parse("sip:alice@192.168.1.1:5060;transport=TCP"),
            rtp=mock_rtp,
            dialog_class=HangupDialog,
            dial_target=target,
            text="Hello there.",
        )
        with patch("voip.mcp.HangupDialog") as mock_dialog_cls:
            mock_dialog = MagicMock()
            mock_dialog.dial = AsyncMock()
            mock_dialog_cls.return_value = mock_dialog
            protocol.on_registered()
            await asyncio.sleep(0)
            mock_dialog.dial.assert_awaited_once()
            _, kwargs = mock_dialog.dial.await_args
            assert kwargs["text"] == "Hello there."


class TestMCPCallProtocol:
    async def test_on_registered__dials_target(self):
        """Initiate an outbound call after SIP registration."""
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_rtp.public_address = None
        target = SipURI.parse("sip:bob@192.0.2.1:5060")
        conversation: list[dict[str, str]] = []
        mock_session = AsyncMock()
        protocol = MCPCallProtocol(
            aor=SipURI.parse("sip:alice@192.168.1.1:5060;transport=TCP"),
            rtp=mock_rtp,
            dialog_class=HangupDialog,
            dial_target=target,
            system_prompt="Be brief.",
            initial_prompt="Hi!",
            mcp_session=mock_session,
            conversation=conversation,
        )
        with patch("voip.mcp.HangupDialog") as mock_dialog_cls:
            mock_dialog = MagicMock()
            mock_dialog.dial = AsyncMock()
            mock_dialog_cls.return_value = mock_dialog
            protocol.on_registered()
            await asyncio.sleep(0)
            mock_dialog.dial.assert_awaited_once()
            _, kwargs = mock_dialog.dial.await_args
            assert kwargs["system_prompt"] == "Be brief."
            assert kwargs["initial_prompt"] == "Hi!"
            assert kwargs["mcp_session"] is mock_session
            assert kwargs["conversation"] is conversation


@dataclasses.dataclass
class FakeTransport:
    """Minimal asyncio.Transport stub."""

    _local_address: tuple = ("127.0.0.1", 5061)
    _peer_address: tuple = ("192.0.2.1", 5061)
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
                return None
            case _:
                return default


def _make_sip_protocol_that_disconnects(
    server: VoIPServer,
) -> SessionInitiationProtocol:
    """Return a SIP protocol stub that immediately signals disconnection."""
    mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
    mock_rtp.public_address = NetworkAddress.parse("192.0.2.1:5060")
    protocol = SessionInitiationProtocol(
        aor=server._sip_aor,
        rtp=mock_rtp,
        dialog_class=HangupDialog,
    )
    fake_transport = FakeTransport()
    protocol.connection_made(fake_transport)
    # Signal immediate disconnection
    protocol.disconnected_event.set()
    return protocol


class TestVoIPServer:
    def test_init__parses_aor(self):
        """Parse the AOR into a SipURI."""
        server = VoIPServer(aor="sip:alice@192.168.1.1:5060;transport=TCP")
        assert server._sip_aor == SipURI.parse(
            "sip:alice@192.168.1.1:5060;transport=TCP"
        )

    def test_init__parses_stun_server(self):
        """Parse the STUN server address."""
        server = VoIPServer(
            aor="sip:alice@192.168.1.1:5060;transport=TCP",
            stun_server="stun.example.com:3478",
        )
        assert str(server._stun_address[0]) == "stun.example.com"

    async def test_init__registers_tools(self):
        """Register outbound_message and make_call tools."""
        server = VoIPServer(aor="sip:alice@192.168.1.1:5060;transport=TCP")
        tools = await server._mcp.list_tools()
        names = {t.name for t in tools}
        assert "outbound_message" in names
        assert "make_call" in names

    async def test_create_rtp__returns_protocol(self):
        """Return an RTP protocol after creating a datagram endpoint."""
        server = VoIPServer(aor="sip:alice@192.168.1.1:5060;transport=TCP")
        mock_transport = MagicMock()
        mock_protocol = MagicMock(spec=RealtimeTransportProtocol)
        with patch.object(
            asyncio.get_event_loop(),
            "create_datagram_endpoint",
            return_value=(mock_transport, mock_protocol),
        ) as mock_create:
            loop = asyncio.get_event_loop()
            mock_create.return_value = (mock_transport, mock_protocol)
            with patch("asyncio.get_running_loop", return_value=loop):
                with patch.object(
                    loop,
                    "create_datagram_endpoint",
                    new=AsyncMock(return_value=(mock_transport, mock_protocol)),
                ):
                    _, protocol = await server._create_rtp()
                    assert protocol is mock_protocol

    async def test_connect_once__waits_for_disconnection(self):
        """Wait until the SIP protocol signals disconnection."""
        server = VoIPServer(aor="sip:alice@192.168.1.1:5060;transport=TCP")
        protocol = _make_sip_protocol_that_disconnects(server)
        loop = asyncio.get_event_loop()
        with patch.object(
            loop,
            "create_connection",
            new=AsyncMock(return_value=(MagicMock(), protocol)),
        ):
            await server._connect_once(lambda: protocol)

    async def test_connect_once__no_tls_for_sip_scheme(self):
        """Skip TLS setup when the AOR uses plain sip: scheme over TCP."""
        server = VoIPServer(aor="sip:alice@192.168.1.1:5060;transport=TCP")
        protocol = _make_sip_protocol_that_disconnects(server)
        loop = asyncio.get_event_loop()
        with patch.object(
            loop,
            "create_connection",
            new=AsyncMock(return_value=(MagicMock(), protocol)),
        ) as mock_connect:
            await server._connect_once(lambda: protocol)
            _, kwargs = mock_connect.await_args
            assert kwargs.get("ssl") is None

    async def test_connect_once__tls_for_sips_scheme(self):
        """Set up a TLS context when the AOR uses sips: scheme."""
        server = VoIPServer(aor="sips:alice@192.168.1.1:5061")
        protocol = _make_sip_protocol_that_disconnects(server)
        loop = asyncio.get_event_loop()
        with patch.object(
            loop,
            "create_connection",
            new=AsyncMock(return_value=(MagicMock(), protocol)),
        ) as mock_connect:
            await server._connect_once(lambda: protocol)
            _, kwargs = mock_connect.await_args
            import ssl

            assert isinstance(kwargs.get("ssl"), ssl.SSLContext)

    async def test_connect_once__no_verify_tls(self):
        """Disable certificate verification when verify_tls is False."""
        import ssl

        server = VoIPServer(aor="sips:alice@192.168.1.1:5061", verify_tls=False)
        protocol = _make_sip_protocol_that_disconnects(server)
        loop = asyncio.get_event_loop()
        with patch.object(
            loop,
            "create_connection",
            new=AsyncMock(return_value=(MagicMock(), protocol)),
        ) as mock_connect:
            await server._connect_once(lambda: protocol)
            _, kwargs = mock_connect.await_args
            ctx = kwargs.get("ssl")
            assert isinstance(ctx, ssl.SSLContext)
            assert not ctx.check_hostname
            assert ctx.verify_mode == ssl.CERT_NONE

    async def test_outbound_message__dials_target(self):
        """Dial the target and send a TTS message."""
        server = VoIPServer(aor="sip:alice@192.168.1.1:5060;transport=TCP")
        protocol = _make_sip_protocol_that_disconnects(server)
        mock_transport = MagicMock()
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        loop = asyncio.get_event_loop()
        with patch.object(
            loop,
            "create_datagram_endpoint",
            new=AsyncMock(return_value=(mock_transport, mock_rtp)),
        ):
            with patch.object(
                loop,
                "create_connection",
                new=AsyncMock(return_value=(MagicMock(), protocol)),
            ) as mock_connect:
                await server.outbound_message(
                    target="sip:bob@192.0.2.1:5060",
                    text="Hello!",
                )
                mock_connect.assert_awaited_once()

    async def test_make_call__returns_empty_when_no_conversation(self):
        """Return an empty string when the call ends with no conversation."""
        server = VoIPServer(aor="sip:alice@192.168.1.1:5060;transport=TCP")
        protocol = _make_sip_protocol_that_disconnects(server)
        mock_transport = MagicMock()
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_ctx = MagicMock()
        mock_ctx.session = AsyncMock()
        loop = asyncio.get_event_loop()
        with patch.object(
            loop,
            "create_datagram_endpoint",
            new=AsyncMock(return_value=(mock_transport, mock_rtp)),
        ):
            with patch.object(
                loop,
                "create_connection",
                new=AsyncMock(return_value=(MagicMock(), protocol)),
            ):
                result = await server.make_call(
                    target="sip:bob@192.0.2.1:5060",
                    ctx=mock_ctx,
                )
                assert result == ""

    async def test_make_call__returns_summary(self):
        """Return a summary after a conversation."""
        server = VoIPServer(aor="sip:alice@192.168.1.1:5060;transport=TCP")
        mock_transport = MagicMock()
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_ctx = MagicMock()
        mock_session = AsyncMock()
        mock_session.create_message.return_value = mcp_types.CreateMessageResult(
            role="assistant",
            content=mcp_types.TextContent(type="text", text="Call summary here."),
            model="test-model",
        )
        mock_ctx.session = mock_session
        loop = asyncio.get_event_loop()

        async def fake_connect(factory, **kwargs):
            """Pre-populate conversation before returning the protocol."""
            sip_protocol = factory()
            # Populate the shared conversation list via the protocol reference
            sip_protocol.conversation.extend(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            )
            sip_protocol.disconnected_event.set()
            return MagicMock(), sip_protocol

        with patch.object(
            loop,
            "create_datagram_endpoint",
            new=AsyncMock(return_value=(mock_transport, mock_rtp)),
        ):
            with patch.object(loop, "create_connection", new=fake_connect):
                result = await server.make_call(
                    target="sip:bob@192.0.2.1:5060",
                    ctx=mock_ctx,
                )
                assert result == "Call summary here."
                mock_session.create_message.assert_awaited_once()

    async def test_make_call__with_initial_and_system_prompt(self):
        """Pass initial_prompt and system_prompt to the call protocol."""
        server = VoIPServer(aor="sip:alice@192.168.1.1:5060;transport=TCP")
        protocol = _make_sip_protocol_that_disconnects(server)
        mock_transport = MagicMock()
        mock_rtp = MagicMock(spec=RealtimeTransportProtocol)
        mock_ctx = MagicMock()
        mock_ctx.session = AsyncMock()
        loop = asyncio.get_event_loop()
        factory_calls = []

        async def capture_factory(factory, **kwargs):
            factory_calls.append(factory)
            protocol.disconnected_event.set()
            return MagicMock(), protocol

        with patch.object(
            loop,
            "create_datagram_endpoint",
            new=AsyncMock(return_value=(mock_transport, mock_rtp)),
        ):
            with patch.object(loop, "create_connection", new=capture_factory):
                await server.make_call(
                    target="sip:bob@192.0.2.1:5060",
                    initial_prompt="Hello!",
                    system_prompt="Be brief.",
                    ctx=mock_ctx,
                )
        assert len(factory_calls) == 1
        created = factory_calls[0]()
        assert isinstance(created, SessionInitiationProtocol)

    def test_run__delegates_to_mcp(self):
        """Delegate to the underlying FastMCP instance."""
        server = VoIPServer(aor="sip:alice@192.168.1.1:5060;transport=TCP")
        with patch.object(server._mcp, "run") as mock_run:
            server.run(transport="stdio")
            mock_run.assert_called_once_with(transport="stdio")
