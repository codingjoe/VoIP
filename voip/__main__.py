#!/usr/bin/env python3
import asyncio
import dataclasses
import logging
import time

from voip.sip import messages

try:
    import click
    from pygments import formatters, highlight

    from voip.sip.lexers import SIPLexer
except ImportError as e:
    raise ImportError(
        "The VoIP CLI requires extra dependencies. Install via `pip install voip[cli]`."
    ) from e


class ConsoleMessageProcessor:
    """Protocol mixin that prints messages to stdout."""

    def request_received(self, request: messages.Request, addr: tuple[str, int]):
        self.pprint(request, addr)
        super().request_received(request, addr)

    def response_received(self, response: messages.Response, addr: tuple[str, int]):
        self.pprint(response, addr)
        super().response_received(response, addr)

    def send(self, message, addr: tuple[str, int]) -> None:
        """Send a message and print it to stdout."""
        self.pprint(message, addr)
        super().send(message, addr)

    @staticmethod
    def pprint(msg, addr):
        """Pretty print the message."""
        host = f"[{addr[0]}]" if ":" in addr[0] else addr[0]
        host = click.style(host, fg="green", bold=True)
        port = click.style(str(addr[1]), fg="yellow", bold=True)
        pretty_msg = highlight(str(msg), SIPLexer(), formatters.TerminalFormatter())
        click.echo(f"{host}:{port} - - [{time.asctime()}] {pretty_msg}")


@click.group()
@click.option("-v", "--verbose", count=True, help="Increase verbosity.")
@click.pass_context
def voip(ctx, verbose):
    """VoIP command line interface."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    logging.basicConfig(
        level=max(10, 10 * (4 - verbose)),
        format="%(levelname)s: [%(asctime)s] (%(name)s) %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("voip").setLevel(max(10, 10 * (3 - verbose)))


@voip.group()
def sip():
    """Session Initiation Protocol (SIP)."""


main = voip


def _parse_server(ctx, param, value: str, default_port=5060) -> tuple[str, int]:
    """Parse 'HOST[:PORT]' option into a (host, port) tuple."""
    try:
        host, port_str = value.rsplit(":", 1)
    except ValueError:
        host, port_str = value, default_port
    return host, int(port_str)


def _parse_stun_server(ctx, param, value: str | None) -> tuple[str, int] | None:
    """Parse the --stun-server option; return None when the value is 'none'."""
    if value is None or value.lower() == "none":
        return None
    return _parse_server(ctx, param, value, default_port=3478)


@sip.command()
@click.option(
    "--model",
    default="base",
    envvar="WHISPER_MODEL",
    show_default=True,
    help="Whisper model size.",
)
@click.option(
    "--server",
    envvar="SIP_SERVER",
    required=True,
    metavar="HOST[:PORT]",
    callback=_parse_server,
    help="SIP server address.",
)
@click.option(
    "--aor",
    envvar="SIP_AOR",
    required=False,
    default=None,
    metavar="SIP_AOR",
    help="SIP Address of Record (defaults to sip:{username}@{server_host}).",
)
@click.option("--username", envvar="SIP_USERNAME", required=True, help="SIP username.")
@click.option("--password", envvar="SIP_PASSWORD", help="SIP password.")
@click.option(
    "--local-port", default=5060, show_default=True, help="Local UDP port to bind."
)
@click.option(
    "--stun-server",
    envvar="STUN_SERVER",
    default="stun.cloudflare.com:3478",
    show_default=True,
    metavar="HOST[:PORT]",
    callback=_parse_stun_server,
    is_eager=False,
    help="STUN server for NAT traversal (use 'none' to disable).",
)
@click.pass_context
def transcribe(ctx, model, server, aor, username, password, local_port, stun_server):
    """Register with a SIP carrier and transcribe incoming calls via Whisper."""
    from voip.sip.protocol import SIP

    from .audio import WhisperCall  # noqa: PLC0415

    server_addr = server
    host = server_addr[0]
    if aor is None:
        aor = f"sip:{username}@{host}"

    verbose = ctx.obj.get("verbose", 0)

    # Capture the CLI model arg as the dataclass field default so that the
    # class can still be passed as a plain type to SIP.answer().
    _model = model

    @dataclasses.dataclass
    class TranscribingCall(WhisperCall):
        """WhisperCall with the CLI-selected model and console output."""

        model: str = _model

        def transcription_received(self, text: str) -> None:
            click.echo(click.style(text, fg="green", bold=True), nl=False)

    # Mix in ConsoleMessageProcessor only at maximum verbosity (-vvv) so that
    # normal operation is not flooded with protocol-level message dumps.
    bases = (ConsoleMessageProcessor, SIP) if verbose >= 3 else (SIP,)

    class TranscribeSession(*bases):
        def call_received(self, request) -> None:
            self.ringing(request=request)
            asyncio.create_task(
                self.answer(request=request, call_class=TranscribingCall)
            )

    async def run():
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(
            lambda: TranscribeSession(
                server_address=server_addr,
                aor=aor,
                username=username,
                password=password,
                stun_server_address=stun_server,
            ),
            local_addr=("0.0.0.0", local_port),  # noqa: S104
        )
        await asyncio.Future()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
