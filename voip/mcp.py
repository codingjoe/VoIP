"""MCP server for VoIP actions.

This module exposes two MCP tools — [`say`][voip.mcp.say] and [`call`][voip.mcp.call] —
and a [`run`][voip.mcp.run] helper that handles all transport setup in one call,
mirroring the start-and-block pattern of [`mcp.run`][fastmcp.FastMCP.run].

Requires the ``mcp`` extra: ``pip install voip[mcp]``.
"""

import asyncio
import dataclasses
import threading
import typing

from fastmcp import Context, FastMCP
from mcp.types import SamplingMessage, TextContent

import voip
from voip import ai
from voip.ai import SayCall
from voip.sip import Dialog
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.types import SipURI, parse_uri
from voip.types import NetworkAddress

__all__ = [
    "mcp",
    "run",
    "MCPAgentCall",
]

mcp = FastMCP(
    "VoIP",
    "Provide a set of tools to make phone calls.",
    version=voip.__version__,
    website_url="https://codingjoe.dev/VoIP/",
)

#: Thread-local storage holding the active [`SessionInitiationProtocol`][voip.sip.protocol.SessionInitiationProtocol].
#: Populated by [`run`][voip.mcp.run] before the MCP server starts.
connection_pool = threading.local()


@dataclasses.dataclass(kw_only=True, slots=True)
class MCPAgentCall(ai.AgentCall):
    """Agent call that generates voice responses via MCP sampling.

    Replaces the Ollama backend of [`AgentCall`][voip.ai.AgentCall] with the
    MCP client's language model via [`Context.sample`][fastmcp.Context.sample].
    Transcription and TTS are inherited from
    [`AgentCall`][voip.ai.AgentCall].

    Args:
        ctx: FastMCP [`Context`][fastmcp.Context] used for LLM sampling.
        system_prompt: System instruction forwarded to the language model.
            Defaults to [`AgentCall.system_prompt`][voip.ai.AgentCall].
        salutation: Opening message spoken as soon as the call connects.
            Pass an empty string to suppress the default greeting.
    """

    ctx: Context

    @property
    def transcript(self) -> str:
        """Formatted conversation transcript.

        Returns:
            Each turn on its own line, prefixed with ``Caller:`` or ``Agent:``.
            System messages are excluded.
        """
        return "\n".join(
            f"{'Caller' if msg['role'] == 'user' else 'Agent'}: {msg['content']}"
            for msg in self._messages
            if msg["role"] != "system"
        )

    def transcription_received(self, text: str) -> None:
        """Handle a transcription chunk and schedule an MCP-sampled response.

        Overrides [`AgentCall.transcription_received`][voip.ai.AgentCall] to
        call [`respond`][voip.mcp.MCPAgentCall.respond] instead of the Ollama
        backend.

        Args:
            text: Transcribed speech from the remote party.
        """
        self.cancel_outbound_audio()
        self._messages.append({"role": "user", "content": text})
        if self._response_task is not None and not self._response_task.done():
            self._response_task.cancel()
        self._response_task = asyncio.create_task(self.respond())

    async def respond(self) -> None:
        """Sample the MCP client LLM and speak the reply.

        Filters system messages before forwarding the conversation history to
        [`Context.sample`][fastmcp.Context.sample], then synthesises the reply
        via [`send_speech`][voip.ai.TTSMixin.send_speech].
        """
        sampling_messages = [
            SamplingMessage(
                role=typing.cast(typing.Literal["user", "assistant"], msg["role"]),
                content=TextContent(type="text", text=msg["content"]),
            )
            for msg in self._messages
            if msg["role"] != "system"
        ]
        result = await self.ctx.sample(
            sampling_messages,
            system_prompt=self.system_prompt,
        )
        if result.text and (reply := result.text.strip()):
            self._messages.append({"role": "assistant", "content": reply})
            await self.send_speech(reply)


@mcp.tool
async def say(ctx: Context, target: str, prompt: str = "") -> None:
    """Call a phone number and speak a message.

    Dials *target*, synthesises *prompt* as speech via Pocket TTS, then hangs
    up automatically once the message has been delivered.

    Args:
        ctx: FastMCP context (injected automatically by the framework).
        target: Phone number or SIP URI to call, e.g. ``"tel:+1234567890"``
            or ``"sip:alice@example.com"``.
        prompt: Text to speak during the call.
    """
    if not hasattr(connection_pool, "sip"):
        raise RuntimeError("VoIP not connected: call run() before using tools.")
    target_uri = parse_uri(target, connection_pool.sip.aor)
    dialog = Dialog(sip=connection_pool.sip)
    await dialog.dial(target_uri, session_class=SayCall, text=prompt)


@mcp.tool
async def call(
    ctx: Context,
    target: str,
    initial_prompt: str = "",
    system_prompt: str | None = None,
) -> str:
    """Call a phone number, hold a conversation, and return the transcript.

    Dials *target*, optionally speaks *initial_prompt*, then drives the
    conversation via [`MCPAgentCall`][voip.mcp.MCPAgentCall] (which samples
    the MCP client's language model for each reply).  Returns once the remote
    party hangs up.

    Args:
        ctx: FastMCP context (injected automatically by the framework).
        target: Phone number or SIP URI to call, e.g. ``"tel:+1234567890"``
            or ``"sip:alice@example.com"``.
        initial_prompt: Opening message spoken when the call connects.
            Pass an empty string to suppress the default greeting.
        system_prompt: System instruction passed to the language model.
            Defaults to [`AgentCall.system_prompt`][voip.ai.AgentCall].

    Returns:
        The full conversation transcript with ``Caller:`` / ``Agent:`` prefixes.
    """
    if not hasattr(connection_pool, "sip"):
        raise RuntimeError("VoIP not connected: call run() before using tools.")
    target_uri = parse_uri(target, connection_pool.sip.aor)
    dialog = Dialog(sip=connection_pool.sip)
    kwargs: dict[str, typing.Any] = {"ctx": ctx, "salutation": initial_prompt}
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    await dialog.dial(target_uri, session_class=MCPAgentCall, **kwargs)
    return dialog.session.transcript


async def run(
    fn: typing.Callable[[], None],
    aor: SipURI,
    *,
    no_verify_tls: bool = False,
    stun_server: NetworkAddress | None = None,
    transport: str | None = None,
) -> None:
    """Run the VoIP MCP server.

    Sets up RTP and SIP transports via
    [`SessionInitiationProtocol.run`][voip.sip.protocol.SessionInitiationProtocol.run],
    stores the resulting protocol in [`connection_pool`][voip.mcp.connection_pool],
    calls *fn* (e.g. to schedule initial tasks), and then starts the MCP server.

    This is the main entry point, analogous to [`mcp.run`][fastmcp.FastMCP.run].
    The CLI [`voip mcp`][voip.__main__] calls this function.

    Args:
        fn: Called once the SIP session is registered. May be a no-op
            (``lambda: None``) when no action is needed at startup.
        aor: SIP address-of-record, e.g. ``sip:alice@carrier.example``.
            Transport (TLS/TCP) and proxy address are derived from the URI.
        no_verify_tls: Skip TLS certificate verification. Insecure; for
            testing only.
        stun_server: STUN server for RTP NAT traversal. Defaults to
            ``stun.cloudflare.com:3478`` (from
            [`STUNProtocol`][voip.stun.STUNProtocol]).
        transport: MCP transport to use (e.g. ``"stdio"``). Forwarded to
            [`FastMCP.run_async`][fastmcp.FastMCP.run_async].
    """
    connection_pool.sip = await SessionInitiationProtocol.run(
        fn,
        aor,
        Dialog,
        no_verify_tls=no_verify_tls,
        stun_server=stun_server,
    )
    await mcp.run_async(transport=transport)


if __name__ == "__main__":  # pragma: no cover
    mcp.run()

