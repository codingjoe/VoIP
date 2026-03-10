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
def voip(verbose):
    """VoIP command line interface."""
    logging.basicConfig(
        level=max(10, 10 * (2 - verbose)),
        format="%(levelname)s: [%(asctime)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


@voip.group()
def sip():
    """Session Initiation Protocol (SIP)."""


logger = logging.getLogger(__name__)

main = voip


def _parse_server(server: str) -> tuple[tuple[str, int], str]:
    """Parse 'HOST[:PORT]' -> ((host, port), host)."""
    if ":" in server:
        host, port_str = server.rsplit(":", 1)
        return (host, int(port_str)), host
    return (server, 5060), server


def _parse_stun_server(stun_server: str) -> tuple[str, int] | None:
    """Parse a STUN server string into a (host, port) tuple, or None if disabled."""
    if stun_server.lower() == "none":
        return None
    if ":" in stun_server:
        stun_host, stun_port_str = stun_server.rsplit(":", 1)
        return (stun_host, int(stun_port_str))
    return (stun_server, 3478)


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
    default="stun.l.google.com:19302",
    envvar="STUN_SERVER",
    show_default=True,
    help="STUN server for NAT traversal (HOST:PORT or 'none' to disable).",
)
def transcribe(model, server, aor, username, password, local_port, stun_server):
    """Register with a SIP carrier and transcribe incoming calls via Whisper."""
    from voip.sip import SIP

    from .whisper import WhisperCall  # noqa: PLC0415

    server_addr, host = _parse_server(server)
    if aor is None:
        aor = f"sip:{username}@{host}"
    stun = _parse_stun_server(stun_server)

    class TranscribingCall(WhisperCall):
        def __init__(self, caller: str = "") -> None:
            super().__init__(caller=caller, model=model)

        def transcription_received(self, text: str) -> None:
            logger.info("Transcription: %s", text)
            click.echo(text)

    class TranscribeSession(SIP):
        def registered(self) -> None:
            logger.info("Registered with %s — waiting for calls", server)
            click.echo(f"Registered with {server} — waiting for calls", err=True)

        def call_received(self, request) -> None:
            click.echo(
                f"Incoming call from {request.headers.get('From', '')}", err=True
            )
            self.answer(request=request, call_class=TranscribingCall)

    async def run():
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(
            lambda: TranscribeSession(
                server_addr,
                aor,
                username,
                password,
                stun_server_address=stun,
            ),
            local_addr=("0.0.0.0", local_port),  # noqa: S104
        )
        click.echo(f"Listening on port {local_port}…", err=True)
        await asyncio.Future()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
