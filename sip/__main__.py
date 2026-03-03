#!/usr/bin/env python3
import logging
import time

from . import messages

try:
    import click
    from pygments import formatters, highlight

    from .lexers import SIPLexer
except ImportError as e:
    raise ImportError(
        "The SIP CLI requires needs to be installed via `pip install libsip[cli]`."
    ) from e


class ConsoleMessageProcessor:
    """Protocol mixin that prints messages to stdout."""

    def request_received(self, request: messages.Request, addr: tuple[str, int]):
        self.pprint(request, addr)
        super().request_received()

    def response_received(self, response: messages.Response, addr: tuple[str, int]):
        self.pprint(response, addr)
        super().request_received()

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
def sip(verbose):
    """SSDP command line interface."""
    logging.basicConfig(
        level=max(10, 10 * (2 - verbose)),
        format="%(levelname)s: [%(asctime)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


if __name__ == "__main__":  # pragma: no cover
    sip()
