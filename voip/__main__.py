#!/usr/bin/env python3
import asyncio
import dataclasses
import logging
import ssl
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
        self.pprint(request)
        super().request_received(request, addr)

    def response_received(self, response: messages.Response, addr: tuple[str, int]):
        self.pprint(response)
        super().response_received(response, addr)

    def send(self, message) -> None:
        """Send a message and print it to stdout."""
        self.pprint(message)
        super().send(message)

    def pprint(self, msg):
        """Pretty print the message."""
        transport = getattr(self, "transport", None)
        addr = transport.get_extra_info("peername") if transport else None
        if addr:
            host = f"[{addr[0]}]" if ":" in addr[0] else addr[0]
            host = click.style(host, fg="green", bold=True)
            port = click.style(str(addr[1]), fg="yellow", bold=True)
            prefix = f"{host}:{port} - - [{time.asctime()}]"
        else:
            prefix = f"[unknown] - - [{time.asctime()}]"
        pretty_msg = highlight(str(msg), SIPLexer(), formatters.TerminalFormatter())
        click.echo(f"{prefix} {pretty_msg}")


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

#: Standard SIP/TCP port — plain text, no TLS (RFC 3261 §18.2).
SIP_TCP_PORT = 5060
#: Standard SIP/TLS port (RFC 3261 §26.2.2).
SIP_TLS_PORT = 5061


def _parse_aor(value: str) -> tuple[str, str, str, int | None]:
    """Parse a SIP URI into ``(scheme, user, host, port)``.

    The port is ``None`` when not present in the URI.

    Examples::

        >>> _parse_aor("sip:alice@example.com")
        ('sip', 'alice', 'example.com', None)
        >>> _parse_aor("sips:+15551234567@carrier.com:5061")
        ('sips', '+15551234567', 'carrier.com', 5061)
    """
    scheme, _, rest = value.partition(":")
    if not scheme or not rest:
        raise click.BadParameter(f"Invalid SIP URI: {value!r}. Expected sip[s]:user@host[:port].")
    user_part, _, hostport = rest.partition("@")
    if not hostport:
        raise click.BadParameter(
            f"Invalid SIP URI: {value!r}. Missing user@host part."
        )
    host, _, port_str = hostport.partition(":")
    if not host:
        raise click.BadParameter(f"Invalid SIP URI: {value!r}. Missing host.")
    port: int | None = int(port_str) if port_str else None
    return scheme, user_part, host, port


def _parse_hostport(ctx, param, value: str, default_port: int = 5061) -> tuple[str, int]:
    """Parse ``HOST[:PORT]`` into a ``(host, port)`` tuple."""
    host, _, port_str = value.rpartition(":")
    if not host:
        return value, default_port
    try:
        return host, int(port_str)
    except ValueError:
        raise click.BadParameter(f"Invalid port in {value!r}.", param=param) from None


def _parse_stun_server(ctx, param, value: str | None) -> tuple[str, int] | None:
    """Parse the --stun-server option; return None when the value is 'none'."""
    if value is None or value.lower() == "none":
        return None
    return _parse_hostport(ctx, param, value, default_port=3478)


# Keep the old name as an alias so existing internal callers still work.
_parse_server = _parse_hostport


@sip.command()
@click.argument("aor", metavar="AOR", envvar="SIP_AOR")
@click.option(
    "--model",
    default="base",
    envvar="WHISPER_MODEL",
    show_default=True,
    help="Whisper model size.",
)
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
def transcribe(ctx, aor, model, password, username, proxy, stun_server, no_tls, no_verify_tls):
    """Register with a SIP carrier and transcribe incoming calls via Whisper.

    AOR is a SIP Address of Record URI identifying the account to register,
    e.g. ``sips:alice@carrier.example.com`` or ``sip:alice@carrier.example.com:5060``.

    \b
    Transport selection (overridable with --no-tls):
      sips: URI or port 5061  →  TLS (default)
      sip:  URI or port 5060  →  plain TCP

    \b
    Examples:
      voip sip transcribe sips:alice@sip.example.com --password secret
      voip sip transcribe sip:alice@sip.example.com:5060 --password secret
      voip sip transcribe sips:alice@carrier.com --proxy proxy.carrier.com --password secret
    """
    from voip.sip.protocol import SIP

    from .audio import WhisperCall  # noqa: PLC0415

    try:
        scheme, aor_user, aor_host, aor_port = _parse_aor(aor)
    except click.BadParameter as exc:
        raise click.BadParameter(str(exc), param_hint="AOR") from exc

    effective_username = username or aor_user

    # Determine outbound proxy address: --proxy overrides, otherwise use AOR host.
    if proxy is not None:
        proxy_addr = _parse_hostport(ctx, None, proxy)
    else:
        # Default port: SIP_TCP_PORT for sip scheme, SIP_TLS_PORT for sips.
        default_port = SIP_TCP_PORT if scheme == "sip" else SIP_TLS_PORT
        port = aor_port if aor_port is not None else default_port
        proxy_addr = (aor_host, port)

    # Transport: port 5060 (SIP_TCP_PORT) → plain TCP; any other port → TLS.
    # --no-tls always overrides this auto-detection.
    use_tls = not no_tls and proxy_addr[1] != SIP_TCP_PORT

    # The AOR stored in the protocol must NOT include the port
    # (AOR is scheme:user@host per RFC 3261 §10).
    normalized_aor = f"{scheme}:{effective_username}@{aor_host}"

    verbose = ctx.obj.get("verbose", 0)

    # Capture the CLI model arg as the dataclass field default so that the
    # class can still be passed as a plain type to SIP.answer().
    _model = model

    @dataclasses.dataclass
    class TranscribingCall(WhisperCall):
        """WhisperCall with the CLI-selected model and console output."""

        model: str = _model

        def transcription_received(self, text: str) -> None:
            click.echo(click.style(text, fg="green", bold=True))

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
        if use_tls:
            ssl_context = ssl.create_default_context()
            if no_verify_tls:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
        else:
            ssl_context = None
        await loop.create_connection(
            lambda: TranscribeSession(
                outbound_proxy=proxy_addr,
                aor=normalized_aor,
                username=effective_username,
                password=password,
                rtp_stun_server_address=stun_server,
            ),
            host=proxy_addr[0],
            port=proxy_addr[1],
            ssl=ssl_context,
        )
        await asyncio.Future()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
