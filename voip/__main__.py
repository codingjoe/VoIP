#!/usr/bin/env python3
import asyncio
import dataclasses
import ipaddress
import logging
import socket
import ssl
import time

from voip.rtp import RealtimeTransportProtocol
from voip.sip import messages
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.transactions import InviteTransaction
from voip.sip.types import SipUri
from voip.types import NetworkAddress

try:
    import click
    import numpy as np
    from pygments import highlight
    from pygments.formatters import TerminalFormatter  # type: ignore[unresolved-import]

    from voip.sip.lexers import SIPLexer
except ImportError as e:
    raise ImportError(
        "The VoIP CLI requires extra dependencies. Install via `pip install voip[cli]`."
    ) from e


logger = logging.getLogger("voip")


@dataclasses.dataclass(kw_only=True, slots=True)
class ConsoleMessageProtocol(SessionInitiationProtocol):
    """Pretty print SIP messages to stdout using pygments."""

    verbose: int = 0

    def request_received(self, request: messages.Request):
        self.pprint(request)
        super().request_received(request)

    def response_received(self, response: messages.Response):
        self.pprint(response)
        super().response_received(response)

    def send(self, message) -> None:
        """Send a message and print it to stdout."""
        self.pprint(message)
        super().send(message)

    def pprint(self, msg):
        """Pretty print the message.

        Args:
            msg: Message to print.
        """
        if self.verbose >= 3:
            transport = getattr(self, "transport", None)
            addr = transport.get_extra_info("peername") if transport else None
            if addr:
                host = f"[{addr[0]}]" if ":" in addr[0] else addr[0]
                host = click.style(host, fg="green", bold=True)
                port = click.style(str(addr[1]), fg="yellow", bold=True)
                prefix = f"{host}:{port} - - [{time.asctime()}]"
            else:
                prefix = f"[unknown] - - [{time.asctime()}]"
            pretty_msg = highlight(str(msg), SIPLexer(), TerminalFormatter())
            click.echo(f"{prefix} {pretty_msg}")


@click.group()
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
@click.pass_context
def voip(ctx, verbose: int = 0):
    """VoIP CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(
            "%(addr)s - %(levelname)s: [%(asctime)s] (%(name)s) %(message)s",
            defaults={"addr": NetworkAddress(socket.gethostname())},
        )
    )

    logging.basicConfig(
        level=max(10, 10 * (4 - verbose)),
        handlers=[console],
    )
    logging.getLogger("voip").setLevel(max(10, 10 * (3 - verbose)))


@voip.group()
@click.argument("aor", metavar="AOR", envvar="SIP_AOR")
@click.option(
    "--stun-server",
    envvar="STUN_SERVER",
    default="stun.cloudflare.com:3478",
    show_default=True,
    metavar="HOST[:PORT]",
    callback=lambda ctx, param, value: NetworkAddress.parse(value),
    is_eager=False,
    help="STUN server for RTP NAT traversal.",
)
@click.option(
    "--no-verify-tls",
    is_flag=True,
    default=False,
    help="Disable TLS certificate verification (insecure; for testing only).",
)
@click.pass_context
def sip(ctx, aor, stun_server, no_verify_tls):
    """Session Initiation Protocol (SIP)."""
    ctx.ensure_object(dict)
    try:
        parsed_aor = SipUri.parse(aor)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="AOR") from exc

    ctx.obj.update(
        aor=parsed_aor,
        proxy_addr=parsed_aor.maddr,
        stun_server=stun_server,
        no_verify_tls=no_verify_tls,
    )


async def _connect_rtp(
    proxy_addr: NetworkAddress,
    rtp_stun_server_address: NetworkAddress | None,
) -> tuple[asyncio.DatagramTransport, RealtimeTransportProtocol]:
    loop = asyncio.get_running_loop()
    rtp_bind = (
        "::" if isinstance(proxy_addr[0], ipaddress.IPv6Address) else "0.0.0.0"  # noqa: S104
    )
    return await loop.create_datagram_endpoint(
        lambda: RealtimeTransportProtocol(stun_server_address=rtp_stun_server_address),
        local_addr=(rtp_bind, 0),
    )


async def _connect_sip(
    session_factory,
    proxy_addr: NetworkAddress,
    use_tls: bool,
    no_verify_tls: bool,
) -> None:
    loop = asyncio.get_running_loop()
    ssl_context: ssl.SSLContext | None = None
    if use_tls:
        ssl_context = ssl.create_default_context()
        if no_verify_tls:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
    backoff_secs = 1
    while True:
        try:
            _, protocol = await loop.create_connection(
                session_factory,
                host=str(proxy_addr[0]),
                port=proxy_addr[1],
                ssl=ssl_context,
            )
            backoff_secs = 1
            await protocol.disconnected_event.wait()
            logger.info("SIP connection closed; reconnecting in %s s", backoff_secs)
        except (OSError, ssl.SSLError) as exc:
            logger.warning(
                "SIP connection failed (%s); retrying in %s s", exc, backoff_secs
            )
        await asyncio.sleep(backoff_secs)
        backoff_secs = min(backoff_secs * 2, 60)


@sip.command()
@click.pass_context
def echo(ctx):
    """Echo the caller's speech back after they finish speaking."""
    from .audio import EchoCall  # noqa: PLC0415

    obj = ctx.obj
    aor = obj["aor"]

    class EchoInviteTransaction(InviteTransaction):
        def invite_received(self, request: messages.Request) -> None:
            self.ringing()
            self.answer(call_class=EchoCall)

    async def run():
        _, rtp_protocol = await _connect_rtp(
            aor.maddr,
            obj["stun_server"],
        )
        await _connect_sip(
            lambda: ConsoleMessageProtocol(
                verbose=obj.get("verbose", 0),
                transaction_class=EchoInviteTransaction,
                aor=aor,
                rtp=rtp_protocol,
            ),
            aor.maddr,
            aor.transport == "TLS",
            obj["no_verify_tls"],
        )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


@sip.command()
@click.option(
    "--stt-model",
    default="tiny",
    envvar="STT_MODEL",
    show_default=True,
    help="Whisper model size.",
)
@click.pass_context
def transcribe(ctx, stt_model):
    """Transcribe incoming call audio."""
    from faster_whisper import WhisperModel

    from .ai import TranscribeCall  # noqa: PLC0415

    obj = ctx.obj
    aor = obj["aor"]

    @dataclasses.dataclass(kw_only=True, slots=True)
    class TranscribingCall(TranscribeCall):
        """TranscribeCall with the CLI-selected model and console output."""

        def transcription_received(self, text: str) -> None:
            click.echo(click.style(text, fg="green", bold=True))

    class TranscribeInviteTransaction(InviteTransaction):
        def invite_received(self, request: messages.Request) -> None:
            self.ringing()
            self.answer(
                call_class=TranscribingCall,
                stt_model=WhisperModel(stt_model),
            )

    async def run():
        _, rtp_protocol = await _connect_rtp(
            aor.maddr,
            obj["stun_server"],
        )
        await _connect_sip(
            lambda: ConsoleMessageProtocol(
                verbose=obj.get("verbose", 0),
                transaction_class=TranscribeInviteTransaction,
                aor=aor,
                rtp=rtp_protocol,
            ),
            aor.maddr,
            aor.transport == "TLS",
            obj["no_verify_tls"],
        )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


@sip.command()
@click.option(
    "--stt-model",
    default="base",
    envvar="STT_MODEL",
    show_default=True,
    help="Whisper model size.",
)
@click.option(
    "--llm-model",
    default="ministral-3",
    envvar="LLM_MODEL",
    show_default=True,
    help="Ollama language model name.",
)
@click.option(
    "--voice",
    default="marius",
    envvar="TTS_VOICE",
    show_default=True,
    help="Pocket TTS voice name or path to a conditioning audio file.",
)
@click.option(
    "--system-prompt",
    default=(
        "You are a person on a phone call."
        " Keep your answers very brief and conversational."
        " YOU MUST NEVER USE NON-VERBAL CHARACTERS IN YOUR RESPONSES!"
    ),
    envvar="LLM_SYSTEM_PROMPT",
    help=("System prompt for the language model."),
)
@click.pass_context
def agent(ctx, stt_model, llm_model, voice, system_prompt):
    """Register with a SIP carrier and handle calls with an AI voice agent."""
    from faster_whisper import WhisperModel

    from .ai import AgentCall  # noqa: PLC0415

    obj = ctx.obj
    aor = obj["aor"]

    @dataclasses.dataclass(kw_only=True, slots=True)
    class AgentCallWithOutput(AgentCall):
        """AgentCall that echoes the conversation to the console."""

        msg_count: int = dataclasses.field(init=False, default=0)

        def transcription_received(self, text: str) -> None:
            click.echo(click.style(f"User: {text}", fg="blue", bold=True))
            super().transcription_received(text)

        async def send_audio(self, audio: np.ndarray) -> None:
            for msg in self._messages[self.msg_count :]:
                click.echo(
                    click.style(
                        f"Agent: {msg['content']}",
                        fg="magenta",
                        bold=True,
                    )
                )
            await super().send_audio(audio)

        async def respond(self) -> None:
            self.msg_count = len(self._messages)
            await super().respond()

    class AgentInviteTransaction(InviteTransaction):
        def invite_received(self, request: messages.Request) -> None:
            self.ringing()
            self.answer(
                call_class=AgentCallWithOutput,
                stt_model=WhisperModel(stt_model),
                llm_model=llm_model,
                voice=voice,
                system_prompt=system_prompt,
            )

    async def run():
        _, rtp_protocol = await _connect_rtp(
            aor.maddr,
            obj["stun_server"],
        )
        await _connect_sip(
            lambda: ConsoleMessageProtocol(
                verbose=obj.get("verbose", 0),
                transaction_class=AgentInviteTransaction,
                aor=aor,
                rtp=rtp_protocol,
            ),
            aor.maddr,
            aor.transport == "TLS",
            obj["no_verify_tls"],
        )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


main = voip
if __name__ == "__main__":  # pragma: no cover
    main()
