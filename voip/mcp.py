"""MCP server for VoIP actions.

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
from voip.msrp import MSRPURI, MessageSessionRelayProtocol
from voip.sip import Dialog
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.types import SipURI, parse_uri
from voip.types import NetworkAddress

__all__ = [
    "mcp",
    "run",
    "MCPAgentCall",
    "send_text",
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
    """
    Agent call that generates voice responses via MCP sampling.

    Replaces the Ollama backend of [`AgentCall`][voip.ai.AgentCall] with the
    MCP client's language model via MCP's sampling API.
    """

    ctx: Context

    @property
    def transcript(self) -> str:
        return "\n".join(
            f"{'Caller' if msg['role'] == 'user' else 'Agent'}: {msg['content']}"
            for msg in self._messages
            if msg["role"] != "system"
        )

    def transcription_received(self, text: str) -> None:
        self.cancel_outbound_audio()
        self._messages.append({"role": "user", "content": text})
        if self._response_task is not None and not self._response_task.done():
            self._response_task.cancel()
        self._response_task = asyncio.create_task(self.respond())

    async def respond(self) -> None:
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

    Dials `target`, synthesises `prompt` as speech, then hangs
    up automatically once the message has been delivered.

    Args:
        ctx: FastMCP context (injected automatically by the framework).
        target: Phone number or SIP URI to call, e.g. `"tel:+1234567890"`
            or `"sip:alice@example.com"`.
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

    Dials `target`, optionally speaks `initial_prompt`, then drives a conversation.

    Returns once the remote party hangs up.

    Args:
        target: Phone number or SIP URI to call, e.g. `"tel:+1234567890"`
            or `"sip:alice@example.com"`.
        initial_prompt: Opening message spoken when the call connects.
            Pass an empty string to suppress the default greeting.
        system_prompt: System instruction passed to the language model.
            Defaults to [`AgentCall.system_prompt`][voip.ai.AgentCall].

    Returns:
        The full conversation transcript with `Caller:` / `Agent:` prefixes.
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


@mcp.tool
async def send_text(ctx: Context, target: str, text: str) -> None:
    """Send plain text via MSRP.

    Args:
        ctx: FastMCP context (injected automatically by the framework).
        target: MSRP endpoint URI, e.g. `"msrps://chat.example.com/abc;tcp"`.
        text: UTF-8 plain text payload.
    """
    if not hasattr(connection_pool, "sip"):
        raise RuntimeError("VoIP not connected: call run() before using tools.")
    target_uri = MSRPURI.parse(target)
    sender_uri = MSRPURI.create(
        host=connection_pool.sip.aor.host,
        secure=target_uri.scheme == "msrps",
    )
    sender = MessageSessionRelayProtocol(
        no_verify_tls=getattr(connection_pool.sip, "no_verify_tls", False)
    )
    await sender.send_text(target=target_uri, sender=sender_uri, text=text)


async def run(
    fn: typing.Callable[[], None],
    aor: SipURI,
    *,
    no_verify_tls: bool = False,
    stun_server: NetworkAddress | None = None,
    transport: str | None = None,
) -> None:
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
