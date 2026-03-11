#!/usr/bin/env python3
import asyncio
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


def _parse_server(ctx, param, value: str) -> tuple[str, int]:
    """Parse 'HOST[:PORT]' option into a (host, port) tuple."""
    if ":" in value:
        host, port_str = value.rsplit(":", 1)
        return (host, int(port_str))
    return (value, 5060)


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
@click.pass_context
def transcribe(ctx, model, server, aor, username, password, local_port):
    """Register with a SIP carrier and transcribe incoming calls via Whisper."""
    from voip.sip.protocol import SIP

    from .audio import WhisperCall  # noqa: PLC0415

    server_addr = server
    host = server_addr[0]
    if aor is None:
        aor = f"sip:{username}@{host}"

    verbose = ctx.obj.get("verbose", 0)

    class TranscribingCall(WhisperCall):
        def __init__(self, caller: str = "", media=None) -> None:
            super().__init__(caller=caller, model=model, media=media)

        def transcription_received(self, text: str) -> None:
            click.echo(text)

    # Mix in ConsoleMessageProcessor only at maximum verbosity (-vvv) so that
    # normal operation is not flooded with protocol-level message dumps.
    bases = (ConsoleMessageProcessor, SIP) if verbose >= 3 else (SIP,)

    class TranscribeSession(*bases):
        def call_received(self, request) -> None:
            self.ringing(request=request)
            self.answer(request=request, call_class=TranscribingCall)

    async def run():
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(
            lambda: TranscribeSession(
                server_addr,
                aor,
                username,
                password,
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
