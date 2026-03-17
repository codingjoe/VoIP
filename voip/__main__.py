#!/usr/bin/env python3
import asyncio
import dataclasses
import logging
import ssl
import time

from voip.sip import messages
from voip.sip.protocol import SessionInitiationProtocol

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


#: Standard SIP/TCP port — plain text, no TLS (RFC 3261 §18.2).
SIP_TCP_PORT = 5060
#: Standard SIP/TLS port (RFC 3261 §26.2.2).
SIP_TLS_PORT = 5061


def _parse_aor(value: str) -> tuple[str, str, str, int | None]:
    """Parse a SIP URI into `(scheme, user, host, port)`.

    The port is `None` when not present in the URI.

    Examples:
    ```
    >>> _parse_aor("sip:alice@example.com")
    ('sip', 'alice', 'example.com', None)
    >>> _parse_aor("sips:+15551234567@carrier.com:5061")
    ('sips', '+15551234567', 'carrier.com', 5061)
    ```

    Args:
        value: SIP URI string.

    Returns:
        Tuple of (scheme, user, host, port).

    Raises:
        click.BadParameter: When the URI is malformed.
    """
    scheme, _, rest = value.partition(":")
    if not scheme or not rest:
        raise click.BadParameter(
            f"Invalid SIP URI: {value!r}. Expected sip[s]:user@host[:port]."
        )
    user_part, _, hostport = rest.partition("@")
    if not hostport:
        raise click.BadParameter(f"Invalid SIP URI: {value!r}. Missing user@host part.")
    host, _, port_str = hostport.partition(":")
    if not host:
        raise click.BadParameter(f"Invalid SIP URI: {value!r}. Missing host.")
    port: int | None = int(port_str) if port_str else None
    return scheme, user_part, host, port


def _parse_hostport(
    ctx, param, value: str, default_port: int = 5061
) -> tuple[str, int]:
    """Parse `HOST[:PORT]` into a `(host, port)` tuple.

    Args:
        ctx: Click context.
        param: Click parameter.
        value: Hostport string.
        default_port: Port to use when not specified.

    Returns:
        Tuple of (host, port).

    Raises:
        click.BadParameter: When port is invalid.
    """
    host, _, port_str = value.rpartition(":")
    if not host:
        return value, default_port
    try:
        return host, int(port_str)
    except ValueError:
        raise click.BadParameter(f"Invalid port in {value!r}.", param=param) from None


def _parse_stun_server(ctx, param, value: str | None) -> tuple[str, int] | None:
    """Parse the --stun-server option; return None when the value is 'none'.

    Args:
        ctx: Click context.
        param: Click parameter.
        value: Stun server string or None.

    Returns:
        Tuple of (host, port) or None.
    """
    if value is None or value.lower() == "none":
        return None
    return _parse_hostport(ctx, param, value, default_port=3478)


# Keep the old name as an alias so existing internal callers still work.
_parse_server = _parse_hostport


class ConsoleMessageProtocol(SessionInitiationProtocol):
    """Pretty print SIP messages to stdout using pygments."""

    __slots__ = ("verbose",)

    def request_received(self, request: messages.Request, addr: tuple[str, int]):
        self.pprint(request)
        super().request_received(request, addr)

    def response_received(
        self, response: messages.Response, addr: tuple[str, int] | None
    ):
        self.pprint(response)
        super().response_received(response, addr)

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
    logging.basicConfig(
        level=max(10, 10 * (4 - verbose)),
        format="%(levelname)s: [%(asctime)s] (%(name)s) %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("voip").setLevel(max(10, 10 * (3 - verbose)))


@voip.group()
@click.argument("aor", metavar="AOR", envvar="SIP_AOR")
@click.option(
    "--password",
    envvar="SIP_PASSWORD",
    required=True,
    help="SIP password (not parsed from AOR for security).",
)
@click.option(
    "--username",
    envvar="SIP_USERNAME",
    default=None,
    help="Override SIP username (defaults to user part of AOR).",
)
@click.option(
    "--proxy",
    envvar="SIP_PROXY",
    default=None,
    metavar="HOST[:PORT]",
    help=(
        "Outbound proxy address (RFC 3261 §8.1.2). "
        "Defaults to the host and port from AOR. "
        "Use this when the proxy differs from the registrar domain."
    ),
)
@click.option(
    "--stun-server",
    envvar="STUN_SERVER",
    default="stun.cloudflare.com:3478",
    show_default=True,
    metavar="HOST[:PORT]",
    callback=_parse_stun_server,
    is_eager=False,
    help="STUN server for RTP NAT traversal (use 'none' to disable).",
)
@click.option(
    "--no-tls",
    is_flag=True,
    default=False,
    help=(
        "Force plain TCP — skips TLS. "
        "Auto-selected when port 5060 is used; explicit flag overrides any port."
    ),
)
@click.option(
    "--no-verify-tls",
    is_flag=True,
    default=False,
    help="Disable TLS certificate verification (insecure; for testing only).",
)
@click.pass_context
def sip(ctx, aor, password, username, proxy, stun_server, no_tls, no_verify_tls):
    """Session Initiation Protocol (SIP)."""
    ctx.ensure_object(dict)
    try:
        scheme, aor_user, aor_host, aor_port = _parse_aor(aor)
    except click.BadParameter as exc:
        raise click.BadParameter(str(exc), param_hint="AOR") from exc

    effective_username = username or aor_user

    if proxy is not None:
        proxy_addr = _parse_hostport(ctx, None, proxy)
    else:
        default_port = SIP_TCP_PORT if scheme == "sip" else SIP_TLS_PORT
        port = aor_port if aor_port is not None else default_port
        proxy_addr = (aor_host, port)

    use_tls = not no_tls and proxy_addr[1] != SIP_TCP_PORT
    normalized_aor = f"{scheme}:{effective_username}@{aor_host}"

    ctx.obj.update(
        aor=normalized_aor,
        username=effective_username,
        password=password,
        proxy_addr=proxy_addr,
        stun_server=stun_server,
        use_tls=use_tls,
        no_verify_tls=no_verify_tls,
    )


async def _connect_sip(
    session_factory,
    proxy_addr: tuple[str, int],
    use_tls: bool,
    no_verify_tls: bool,
) -> None:
    """Connect to a SIP proxy and wait indefinitely."""
    loop = asyncio.get_running_loop()
    ssl_context: ssl.SSLContext | None = None
    if use_tls:
        ssl_context = ssl.create_default_context()
        if no_verify_tls:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
    await loop.create_connection(
        session_factory,
        host=proxy_addr[0],
        port=proxy_addr[1],
        ssl=ssl_context,
    )
    await asyncio.Future()


@sip.command()
@click.pass_context
def echo(ctx):
    """Echo the caller's speech back after they finish speaking."""
    from .audio import EchoCall  # noqa: PLC0415

    obj = ctx.obj
    proxy_addr = obj["proxy_addr"]

    class EchoSession(ConsoleMessageProtocol):
        verbose = obj.get("verbose", 0)

        def call_received(self, request) -> None:
            self.ringing(request=request)
            asyncio.create_task(self.answer(request=request, call_class=EchoCall))

    async def run():
        await _connect_sip(
            lambda: EchoSession(
                outbound_proxy=proxy_addr,
                aor=obj["aor"],
                username=obj["username"],
                password=obj["password"],
                rtp_stun_server_address=obj["stun_server"],
            ),
            proxy_addr,
            obj["use_tls"],
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
    proxy_addr = obj["proxy_addr"]

    @dataclasses.dataclass(kw_only=True, slots=True)
    class TranscribingCall(TranscribeCall):
        """TranscribeCall with the CLI-selected model and console output."""

        def transcription_received(self, text: str) -> None:
            click.echo(click.style(text, fg="green", bold=True))

    class TranscribeSession(ConsoleMessageProtocol):
        verbose = obj.get("verbose", 0)

        def call_received(self, request) -> None:
            self.ringing(request=request)
            asyncio.create_task(
                self.answer(
                    request=request,
                    call_class=TranscribingCall,
                    stt_model=WhisperModel(stt_model),
                )
            )

    async def run():
        await _connect_sip(
            lambda: TranscribeSession(
                outbound_proxy=proxy_addr,
                aor=obj["aor"],
                username=obj["username"],
                password=obj["password"],
                rtp_stun_server_address=obj["stun_server"],
            ),
            proxy_addr,
            obj["use_tls"],
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
@click.option(
    "--llm-model",
    default="ministral-3",
    envvar="LLM_MODEL",
    show_default=True,
    help="Ollama language model name.",
)
@click.option(
    "--voice",
    default="azelma",
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
    proxy_addr = obj["proxy_addr"]

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

    class AgentSession(ConsoleMessageProtocol):
        verbose = obj.get("verbose", 0)

        def call_received(self, request) -> None:
            self.ringing(request=request)
            asyncio.create_task(
                self.answer(
                    request=request,
                    call_class=AgentCallWithOutput,
                    stt_model=WhisperModel(stt_model),
                    llm_model=llm_model,
                    voice=voice,
                    system_prompt=system_prompt,
                )
            )

    async def run():
        await _connect_sip(
            lambda: AgentSession(
                outbound_proxy=proxy_addr,
                aor=obj["aor"],
                username=obj["username"],
                password=obj["password"],
                rtp_stun_server_address=obj["stun_server"],
            ),
            proxy_addr,
            obj["use_tls"],
            obj["no_verify_tls"],
        )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


main = voip
if __name__ == "__main__":  # pragma: no cover
    main()
