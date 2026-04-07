"""MCP server for VoIP telephony operations.

Provides a [VoIPServer][voip.mcp.VoIPServer] that exposes two MCP tools:

- [outbound_message][voip.mcp.VoIPServer.outbound_message]: Dial a number, say a
  message using TTS, and hang up.
- [make_call][voip.mcp.VoIPServer.make_call]: Conduct a voice conversation using
  MCP sampling for LLM inference, and return a summary.

Requires the ``mcp`` extra: ``pip install voip[mcp]``.
"""

import asyncio
import collections.abc
import dataclasses
import ipaddress
import ssl

import mcp.types as mcp_types
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from voip.ai import SayCall, TranscribeCall, TTSMixin
from voip.rtp import RealtimeTransportProtocol
from voip.sip import dialog
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.types import SipURI, parse_uri
from voip.types import NetworkAddress

__all__ = ["VoIPServer"]


@dataclasses.dataclass(kw_only=True, slots=True)
class MCPAgentCall(TTSMixin, TranscribeCall):
    """TranscribeCall variant that uses MCP sampling for LLM inference.

    Unlike [AgentCall][voip.ai.AgentCall], no local Ollama instance is
    required. Instead, each transcribed utterance is forwarded to the MCP
    client via a ``sampling/createMessage`` request, allowing the client to
    use its own LLM.

    Args:
        system_prompt: System prompt forwarded to the MCP client on every
            sampling request.
        initial_prompt: Opening message the agent sends when the call connects.
        mcp_session: Active MCP server session used to issue sampling requests.
        conversation: Shared mutable list that accumulates conversation
            messages. The caller populates this list so it can read the
            transcript after the call ends.
    """

    system_prompt: str = ""
    initial_prompt: str = ""
    mcp_session: ServerSession
    conversation: list[dict[str, str]]

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.initial_prompt:
            self.conversation.append(
                {"role": "assistant", "content": self.initial_prompt}
            )
            asyncio.create_task(self.send_speech(self.initial_prompt))

    def transcription_received(self, text: str) -> None:
        """Add a transcription to the conversation and schedule an LLM response.

        Args:
            text: Transcribed speech from the remote party.
        """
        self.conversation.append({"role": "user", "content": text})
        asyncio.create_task(self.respond())

    async def respond(self) -> None:
        """Generate a response via MCP sampling and send it as audio."""
        messages = [
            mcp_types.SamplingMessage(
                role=message["role"],
                content=mcp_types.TextContent(type="text", text=message["content"]),
            )
            for message in self.conversation
            if message["role"] in ("user", "assistant")
        ]
        result = await self.mcp_session.create_message(
            messages=messages,
            system_prompt=self.system_prompt or None,
            max_tokens=200,
        )
        if reply := result.content.text.strip():
            self.conversation.append({"role": "assistant", "content": reply})
            await self.send_speech(reply)


class HangupDialog(dialog.Dialog):
    """Dialog that closes the SIP connection when the remote party hangs up."""

    def hangup_received(self) -> None:
        """Close the SIP connection when a BYE request is received."""
        if self.sip is not None:
            self.sip.close()


@dataclasses.dataclass(kw_only=True, slots=True)
class OutboundMessageProtocol(SessionInitiationProtocol):
    """SIP protocol for outbound TTS message calls.

    Dials the target and plays a TTS message via [SayCall][voip.ai.SayCall].

    Args:
        dial_target: SIP URI to dial.
        text: Text to synthesise and send as audio.
    """

    dial_target: SipURI
    text: str

    def on_registered(self) -> None:
        """Initiate the outbound call after successful SIP registration."""
        d = HangupDialog(sip=self)
        asyncio.create_task(
            d.dial(self.dial_target, session_class=SayCall, text=self.text)
        )


@dataclasses.dataclass(kw_only=True, slots=True)
class MCPCallProtocol(SessionInitiationProtocol):
    """SIP protocol for outbound MCP agent calls.

    Dials the target and conducts a voice conversation using
    [MCPAgentCall][voip.mcp.MCPAgentCall].

    Args:
        dial_target: SIP URI to dial.
        system_prompt: System prompt for the MCP LLM.
        initial_prompt: Opening message for the agent.
        mcp_session: Active MCP server session used for sampling.
        conversation: Shared list that accumulates conversation messages.
    """

    dial_target: SipURI
    system_prompt: str
    initial_prompt: str
    mcp_session: ServerSession
    conversation: list[dict[str, str]]

    def on_registered(self) -> None:
        """Initiate the outbound call after successful SIP registration."""
        d = HangupDialog(sip=self)
        asyncio.create_task(
            d.dial(
                self.dial_target,
                session_class=MCPAgentCall,
                system_prompt=self.system_prompt,
                initial_prompt=self.initial_prompt,
                mcp_session=self.mcp_session,
                conversation=self.conversation,
            )
        )


@dataclasses.dataclass
class VoIPServer:
    """MCP server that exposes VoIP telephony tools.

    Wraps [FastMCP][mcp.server.fastmcp.FastMCP] and registers two tools:
    [outbound_message][voip.mcp.VoIPServer.outbound_message] for sending a
    TTS message, and [make_call][voip.mcp.VoIPServer.make_call] for
    conducting a full voice conversation using client-side LLM inference via
    MCP sampling.

    Configure the server with a SIP Address of Record and optional STUN/TLS
    settings, then call [run][voip.mcp.VoIPServer.run] to start the server.

    Example:
        ```python
        server = VoIPServer(aor="sips:alice:secret@sip.example.com")
        server.run()
        ```

    Args:
        aor: SIP Address of Record (e.g. ``"sips:alice:secret@sip.example.com"``).
        stun_server: STUN server address for RTP NAT traversal.
        verify_tls: Whether to verify TLS certificates when connecting to the
            SIP proxy.
    """

    aor: str
    stun_server: str = "stun.cloudflare.com:3478"
    verify_tls: bool = True

    def __post_init__(self) -> None:
        self._sip_aor = SipURI.parse(self.aor)
        self._stun_address = NetworkAddress.parse(self.stun_server)
        self._mcp = FastMCP("VoIP")
        self._mcp.add_tool(self.outbound_message)
        self._mcp.add_tool(self.make_call)

    async def _create_rtp(
        self,
    ) -> tuple[asyncio.DatagramTransport, RealtimeTransportProtocol]:
        """Create and return an RTP endpoint.

        Returns:
            A tuple of the datagram transport and the RTP protocol instance.
        """
        loop = asyncio.get_running_loop()
        bind = (
            "::"
            if isinstance(self._sip_aor.maddr[0], ipaddress.IPv6Address)
            else "0.0.0.0"  # noqa: S104
        )
        return await loop.create_datagram_endpoint(
            lambda: RealtimeTransportProtocol(stun_server_address=self._stun_address),
            local_addr=(bind, 0),
        )

    async def _connect_once(
        self,
        factory: collections.abc.Callable[[], SessionInitiationProtocol],
    ) -> None:
        """Connect to the SIP proxy once and wait for disconnection.

        Args:
            factory: Callable that returns a new SIP protocol instance.
        """
        loop = asyncio.get_running_loop()
        ssl_context: ssl.SSLContext | None = None
        if self._sip_aor.transport == "TLS":
            ssl_context = ssl.create_default_context()
            if not self.verify_tls:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
        _, protocol = await loop.create_connection(
            factory,
            host=str(self._sip_aor.maddr[0]),
            port=self._sip_aor.maddr[1],
            ssl=ssl_context,
        )
        await protocol.disconnected_event.wait()

    async def outbound_message(self, target: str, text: str) -> None:
        """Run TTS on text and send audio to target.

        Dials *target* via SIP, synthesises *text* using Pocket TTS, transmits
        the resulting audio over RTP, and hangs up once the audio is sent.

        Args:
            target: SIP URI of the call recipient.
            text: Text to synthesise and send as audio.
        """
        target_uri = parse_uri(target, self._sip_aor)
        _, rtp = await self._create_rtp()
        await self._connect_once(
            lambda: OutboundMessageProtocol(
                aor=self._sip_aor,
                rtp=rtp,
                dialog_class=HangupDialog,
                dial_target=target_uri,
                text=text,
            )
        )

    async def make_call(
        self,
        target: str,
        initial_prompt: str = "",
        system_prompt: str = "",
        *,
        ctx: Context,
    ) -> str:
        """Have a voice conversation and return a summary.

        Dials *target* via SIP and conducts a voice conversation. Each
        utterance from the remote party is transcribed and forwarded to the MCP
        client via a ``sampling/createMessage`` request; the client-side LLM
        generates a reply that is synthesised and played back over RTP.

        When the call ends, the full transcript is summarised using a final MCP
        sampling request and the summary is returned.

        Args:
            target: SIP URI of the call recipient.
            initial_prompt: Opening message the agent says when the call connects.
            system_prompt: System prompt forwarded to the MCP client on every
                sampling request.
            ctx: MCP context (injected automatically by FastMCP).

        Returns:
            Summary of the conversation, or an empty string if no conversation
            took place.
        """
        target_uri = parse_uri(target, self._sip_aor)
        _, rtp = await self._create_rtp()
        conversation: list[dict[str, str]] = []
        mcp_session = ctx.session
        await self._connect_once(
            lambda: MCPCallProtocol(
                aor=self._sip_aor,
                rtp=rtp,
                dialog_class=HangupDialog,
                dial_target=target_uri,
                system_prompt=system_prompt,
                initial_prompt=initial_prompt,
                mcp_session=mcp_session,
                conversation=conversation,
            )
        )
        if not conversation:
            return ""
        summary_result = await mcp_session.create_message(
            messages=[
                mcp_types.SamplingMessage(
                    role=message["role"],
                    content=mcp_types.TextContent(type="text", text=message["content"]),
                )
                for message in conversation
                if message["role"] in ("user", "assistant")
            ],
            system_prompt="Summarize this phone call conversation in a few sentences.",
            max_tokens=300,
        )
        return summary_result.content.text

    def run(self, transport: str = "stdio") -> None:
        """Start the MCP, RTP, and SIP server.

        Args:
            transport: MCP transport protocol. One of ``"stdio"``, ``"sse"``,
                or ``"streamable-http"``.
        """
        self._mcp.run(transport=transport)
