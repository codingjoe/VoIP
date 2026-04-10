"""MCP server for VoIP actions.

This module exposes two MCP tools — [`say`][voip.mcp.say] and [`call`][voip.mcp.call] —
and a [`run`][voip.mcp.run] helper that handles all transport setup in one call,
mirroring the start-and-block pattern of [`mcp.run`][fastmcp.FastMCP.run].

Requires the ``mcp`` extra: ``pip install voip[mcp]``.
"""

import asyncio
import collections.abc
import dataclasses
import ipaddress
import os
import ssl
import typing

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import SamplingMessage, TextContent

import voip
from voip.ai import SayCall, TranscribeCall, TTSMixin
from voip.rtp import RealtimeTransportProtocol
from voip.sip import dialog
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.types import SipURI, parse_uri
from voip.types import NetworkAddress

__all__ = [
    "mcp",
    "run",
    "HangupDialog",
    "MCPAgentCall",
    "DEFAULT_STUN_SERVER",
    "DEFAULT_SYSTEM_PROMPT",
]

#: Default STUN server used when *rtp_stun_server* is not provided to [`run`][voip.mcp.run].
DEFAULT_STUN_SERVER: typing.Final[str] = "stun.cloudflare.com:3478"

#: Default system prompt for [`MCPAgentCall`][voip.mcp.MCPAgentCall].
DEFAULT_SYSTEM_PROMPT: typing.Final[str] = (
    "You are a person on a phone call."
    " Keep your answers very brief and conversational."
    " YOU MUST NEVER USE NON-VERBAL CHARACTERS IN YOUR RESPONSES!"
)

mcp = FastMCP(
    "VoIP",
    "Provide a set of tools to make phone calls.",
    version=voip.__version__,
    website_url="https://codingjoe.dev/VoIP/",
)


class HangupDialog(dialog.Dialog):
    """Dialog that closes the SIP transport when the remote party hangs up.

    When used as *dialog_class* in [`run`][voip.mcp.run], the SIP transport is
    closed after a remote BYE, which unblocks `run` and ends the session.
    """

    def hangup_received(self) -> None:
        """Close the SIP transport on receiving a remote BYE."""
        if self.sip is not None:
            self.sip.close()


async def run(
    fn: collections.abc.Callable[[SessionInitiationProtocol], None],
    aor: SipURI,
    dialog_class: type[dialog.Dialog] = HangupDialog,
    *,
    no_verify_tls: bool = False,
    rtp: NetworkAddress | None = None,
    rtp_stun_server: NetworkAddress | None = None,
) -> None:
    """Run a SIP session and call *fn* once registered.

    This is a start-and-block function, similar to [`mcp.run`][fastmcp.FastMCP.run]:
    it sets up RTP (unless an external address is provided), establishes the SIP/TLS
    connection derived from *aor*, calls *fn* with the registered SIP session, and
    suspends until the transport is closed.

    The transport protocol (TLS vs plain TCP) and proxy address are read from *aor*
    directly — no extra arguments are needed.

    Args:
        fn: Called when the SIP session is registered. Receives the
            [`SessionInitiationProtocol`][voip.sip.protocol.SessionInitiationProtocol]
            instance. May use [`asyncio.create_task`][] for async work.
        aor: SIP Address of Record, e.g. ``sip:alice@carrier.example``. The host,
            port, and ``transport`` parameter are used to connect to the SIP proxy.
        dialog_class: [`Dialog`][voip.sip.Dialog] subclass used for inbound calls.
            Defaults to [`HangupDialog`][voip.mcp.HangupDialog], which closes the SIP
            transport on remote BYE.
        no_verify_tls: Disable TLS certificate verification. Insecure; for testing
            only. Defaults to ``False``.
        rtp: External RTP server address (e.g. for ffmpeg). When ``None`` (default),
            a [`RealtimeTransportProtocol`][voip.rtp.RealtimeTransportProtocol]
            endpoint is created automatically.
        rtp_stun_server: STUN server for RTP NAT traversal. Defaults to
            ``stun.cloudflare.com:3478``. Ignored when *rtp* is supplied.
    """
    loop = asyncio.get_running_loop()

    if rtp is None:
        stun_server = rtp_stun_server or NetworkAddress.parse(DEFAULT_STUN_SERVER)
        rtp_bind_address = "::" if isinstance(aor.maddr[0], ipaddress.IPv6Address) else "0.0.0.0"  # noqa: S104
        _, rtp_protocol = await loop.create_datagram_endpoint(
            lambda: RealtimeTransportProtocol(stun_server_address=stun_server),
            local_addr=(rtp_bind_address, 0),
        )
    else:
        rtp_protocol = RealtimeTransportProtocol()
        rtp_protocol.public_address = rtp

    ssl_context: ssl.SSLContext | None = None
    if aor.transport == "TLS":
        ssl_context = ssl.create_default_context()
        if no_verify_tls:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

    @dataclasses.dataclass(kw_only=True, slots=True)
    class InlineProtocol(SessionInitiationProtocol):
        def on_registered(self) -> None:
            fn(self)

    _, protocol = await loop.create_connection(
        lambda: InlineProtocol(aor=aor, rtp=rtp_protocol, dialog_class=dialog_class),
        host=str(aor.maddr[0]),
        port=aor.maddr[1],
        ssl=ssl_context,
    )
    await protocol.disconnected_event.wait()


@dataclasses.dataclass(kw_only=True, slots=True)
class MCPAgentCall(TTSMixin, TranscribeCall):
    """Agent call that generates voice responses via MCP sampling.

    Transcribes the remote party's speech with
    [Whisper][voip.ai.TranscribeCall], then forwards the conversation
    history to the MCP client's language model via
    [`Context.sample`][fastmcp.Context.sample] and speaks the reply
    using [Pocket TTS][voip.ai.TTSMixin].

    Args:
        ctx: The FastMCP [`Context`][fastmcp.Context] used for LLM sampling.
        system_prompt: System instruction forwarded to the language model.
        initial_prompt: Opening message spoken as soon as the call connects.
    """

    ctx: Context
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    initial_prompt: str = ""

    _messages: list[dict[str, str]] = dataclasses.field(
        init=False, repr=False, default_factory=list
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.initial_prompt:
            self._messages.append(
                {"role": "assistant", "content": self.initial_prompt}
            )
            asyncio.create_task(self.send_speech(self.initial_prompt))

    @property
    def transcript(self) -> str:
        """Formatted conversation transcript.

        Returns:
            Each turn on its own line, prefixed with ``Caller:`` or ``Agent:``.
        """
        lines = []
        for msg in self._messages:
            role = "Caller" if msg["role"] == "user" else "Agent"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def transcription_received(self, text: str) -> None:
        """Handle a transcription chunk and schedule an LLM response.

        Args:
            text: Transcribed speech from the remote party (already stripped).
        """
        self._messages.append({"role": "user", "content": text})
        asyncio.create_task(self.respond())

    async def respond(self) -> None:
        """Sample the MCP client LLM and speak the reply."""
        sampling_messages = [
            SamplingMessage(
                role=typing.cast(
                    typing.Literal["user", "assistant"], msg["role"]
                ),
                content=TextContent(type="text", text=msg["content"]),
            )
            for msg in self._messages
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
    aor_str = os.environ.get("SIP_AOR")
    if not aor_str:
        raise ToolError("SIP_AOR environment variable is not set.")
    aor = SipURI.parse(aor_str)
    no_verify_tls = os.environ.get("SIP_NO_VERIFY_TLS", "").lower() in ("1", "true")
    stun_str = os.environ.get("STUN_SERVER")
    stun_server = NetworkAddress.parse(stun_str) if stun_str else None
    target_uri = parse_uri(target, aor)

    def on_registered(sip: SessionInitiationProtocol) -> None:
        d = HangupDialog(sip=sip)
        asyncio.create_task(d.dial(target_uri, session_class=SayCall, text=prompt))

    await run(on_registered, aor, no_verify_tls=no_verify_tls, rtp_stun_server=stun_server)


@mcp.tool
async def call(
    ctx: Context,
    target: str,
    initial_prompt: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
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
        system_prompt: System instruction passed to the language model.

    Returns:
        The full conversation transcript with ``Caller:`` / ``Agent:`` prefixes.
    """
    aor_str = os.environ.get("SIP_AOR")
    if not aor_str:
        raise ToolError("SIP_AOR environment variable is not set.")
    aor = SipURI.parse(aor_str)
    no_verify_tls = os.environ.get("SIP_NO_VERIFY_TLS", "").lower() in ("1", "true")
    stun_str = os.environ.get("STUN_SERVER")
    stun_server = NetworkAddress.parse(stun_str) if stun_str else None
    target_uri = parse_uri(target, aor)
    sessions: list[MCPAgentCall] = []

    @dataclasses.dataclass(kw_only=True, slots=True)
    class CallSession(MCPAgentCall):
        def __post_init__(self) -> None:
            super().__post_init__()
            sessions.append(self)

    def on_registered(sip: SessionInitiationProtocol) -> None:
        d = HangupDialog(sip=sip)
        asyncio.create_task(
            d.dial(
                target_uri,
                session_class=CallSession,
                ctx=ctx,
                system_prompt=system_prompt,
                initial_prompt=initial_prompt,
            )
        )

    await run(on_registered, aor, no_verify_tls=no_verify_tls, rtp_stun_server=stun_server)
    return sessions[0].transcript if sessions else ""


if __name__ == "__main__":  # pragma: no cover
    mcp.run()
