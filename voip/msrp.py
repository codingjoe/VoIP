"""Message Session Relay Protocol helpers.

Implements minimal plain-text MSRP SEND support from RFC 4975.
"""

import asyncio
import dataclasses
import ipaddress
import re
import secrets
import ssl
import typing
import urllib.parse
import uuid

DEFAULT_PORT: typing.Final[int] = 2855
MSRP_URI_PATTERN: typing.Final[re.Pattern[str]] = re.compile(
    r"^(?P<scheme>msrps?)://"
    r"(?P<host>\[[0-9a-fA-F:]+]|[^/:;]+)"
    r"(?::(?P<port>[0-9]+))?"
    r"/(?P<session_id>[^;/?#]+)"
    r";(?P<transport>[^;/?#]+)$",
    re.IGNORECASE,
)


class MSRPURI(str):
    """Parsed MSRP URI.

    Format: ``msrp://host:port/session-id;tcp`` or ``msrps://...``.
    """

    __slots__ = ("scheme", "host", "port", "session_id", "transport")

    def __new__(
        cls,
        *,
        scheme: str,
        host: str | ipaddress.IPv4Address | ipaddress.IPv6Address,
        session_id: str,
        transport: str = "tcp",
        port: int | None = None,
    ) -> typing.Self:
        if scheme not in {"msrp", "msrps"}:
            raise ValueError(f"Invalid MSRP scheme: {scheme!r}")
        try:
            host = ipaddress.ip_address(host)
        except ValueError:
            pass
        encoded_host = f"[{host}]" if isinstance(host, ipaddress.IPv6Address) else str(host)
        encoded_session = urllib.parse.quote(session_id)
        encoded_transport = urllib.parse.quote(transport)
        instance = super().__new__(
            cls,
            f"{scheme}://{encoded_host}:{port or DEFAULT_PORT}/{encoded_session};{encoded_transport}",
        )
        instance.scheme = scheme
        instance.host = host
        instance.port = port or DEFAULT_PORT
        instance.session_id = session_id
        instance.transport = transport
        return instance

    @classmethod
    def create(
        cls,
        *,
        host: str | ipaddress.IPv4Address | ipaddress.IPv6Address,
        secure: bool,
        port: int | None = None,
    ) -> typing.Self:
        """Create a URI with a random session ID."""
        return cls(
            scheme="msrps" if secure else "msrp",
            host=host,
            port=port,
            session_id=uuid.uuid4().hex,
            transport="tcp",
        )

    @classmethod
    def parse(cls, value: str) -> typing.Self:
        """Parse an MSRP or MSRPS URI."""
        if not (match := MSRP_URI_PATTERN.fullmatch(value)):
            raise ValueError(f"Invalid MSRP URI: {value!r}")
        host = match.group("host").strip("[]")
        return cls(
            scheme=match.group("scheme").lower(),
            host=urllib.parse.unquote(host),
            port=int(match.group("port")) if match.group("port") else None,
            session_id=urllib.parse.unquote(match.group("session_id")),
            transport=urllib.parse.unquote(match.group("transport")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class MSRPResponse:
    """MSRP response status line."""

    transaction_id: str
    status_code: int
    phrase: str


@dataclasses.dataclass(kw_only=True, slots=True)
class MessageSessionRelayProtocol:
    """MSRP text sender."""

    no_verify_tls: bool = False

    @staticmethod
    def parse_response(data: bytes) -> MSRPResponse:
        """Parse an MSRP response start line."""
        first_line = data.split(b"\r\n", 1)[0].decode("utf-8", errors="replace")
        version, transaction_id, status_code, phrase = first_line.split(" ", 3)
        if version != "MSRP":
            raise ValueError(f"Invalid MSRP response version: {version!r}")
        return MSRPResponse(
            transaction_id=transaction_id,
            status_code=int(status_code),
            phrase=phrase,
        )

    @staticmethod
    def build_send_request(
        *,
        transaction_id: str,
        message_id: str,
        to_path: MSRPURI,
        from_path: MSRPURI,
        text: str,
    ) -> bytes:
        """Build a plain-text MSRP SEND request."""
        body = text.encode()
        if not body:
            raise ValueError("MSRP text body must not be empty")
        headers = (
            f"MSRP {transaction_id} SEND\r\n"
            f"To-Path: {to_path}\r\n"
            f"From-Path: {from_path}\r\n"
            f"Message-ID: {message_id}\r\n"
            f"Byte-Range: 1-{len(body)}/{len(body)}\r\n"
            "Content-Type: text/plain\r\n"
            "\r\n"
        ).encode()
        trailer = f"\r\n-------{transaction_id}$\r\n".encode()
        return headers + body + trailer

    async def send_text(
        self,
        *,
        target: MSRPURI,
        sender: MSRPURI,
        text: str,
    ) -> MSRPResponse:
        """Send plain text to an MSRP target."""
        ssl_context: ssl.SSLContext | None = None
        if target.scheme == "msrps":
            ssl_context = ssl.create_default_context()
            if self.no_verify_tls:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

        reader, writer = await asyncio.open_connection(
            host=str(target.host),
            port=target.port,
            ssl=ssl_context,
        )

        transaction_id = secrets.token_hex(8)
        request = self.build_send_request(
            transaction_id=transaction_id,
            message_id=secrets.token_hex(12),
            to_path=target,
            from_path=sender,
            text=text,
        )
        try:
            writer.write(request)
            await writer.drain()

            delimiter = f"-------{transaction_id}$\r\n".encode()
            response_data = bytearray()
            while delimiter not in response_data:
                if not (chunk := await reader.read(4096)):
                    break
                response_data.extend(chunk)

            response = self.parse_response(bytes(response_data))
            if response.status_code >= 400:
                raise RuntimeError(
                    f"MSRP delivery failed with {response.status_code} {response.phrase}"
                )
            return response
        finally:
            writer.close()
            await writer.wait_closed()
