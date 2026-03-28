"""Tests for the VoIP CLI (__main__ module)."""

import asyncio
import ipaddress
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from voip.__main__ import (
    ConsoleMessageProtocol,
    _connect_sip_once,
    _make_outbound_factory,
    _parse_dial_target,
    voip,
)
from voip.rtp import RealtimeTransportProtocol
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.transactions import InviteTransaction
from voip.sip.types import SipUri
from voip.types import NetworkAddress

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_sip_protocol() -> SessionInitiationProtocol:
    """Return a minimal SIP protocol stub whose disconnected_event is pre-set."""
    mux = RealtimeTransportProtocol()
    aor = SipUri.parse("sips:alice:secret@example.com")
    protocol = ConsoleMessageProtocol(
        aor=aor,
        rtp=mux,
        transaction_class=InviteTransaction,
    )
    protocol.disconnected_event.set()
    protocol.local_address = NetworkAddress(ipaddress.ip_address("127.0.0.1"), 5061)
    protocol.is_secure = True
    return protocol


def _fake_transport_get_extra_info(key, default=None):
    """Return standard fake transport metadata for SIP sessions."""
    match key:
        case "sockname":
            return ("127.0.0.1", 5061)
        case "peername":
            return ("192.0.2.1", 5061)
        case "ssl_object":
            return object()
        case _:
            return default


# ---------------------------------------------------------------------------
# _connect_sip_once
# ---------------------------------------------------------------------------


class TestConnectSipOnce:
    async def test_connects_and_returns_after_disconnect(self):
        """_connect_sip_once waits for the session to disconnect, then returns."""
        protocol = _make_fake_sip_protocol()
        proxy = NetworkAddress(ipaddress.ip_address("127.0.0.1"), 5061)

        with patch.object(
            asyncio.get_event_loop(),
            "create_connection",
            new=AsyncMock(return_value=(MagicMock(), protocol)),
        ):
            await _connect_sip_once(lambda: protocol, proxy, False, False)

    async def test_tls_creates_ssl_context(self):
        """_connect_sip_once creates an SSL context when use_tls is True."""
        import ssl

        protocol = _make_fake_sip_protocol()
        proxy = NetworkAddress(ipaddress.ip_address("127.0.0.1"), 5061)
        captured: list = []

        async def fake_connect(factory, *, host, port, ssl=None):
            captured.append(ssl)
            return MagicMock(), protocol

        loop = asyncio.get_event_loop()
        with patch.object(loop, "create_connection", side_effect=fake_connect):
            await _connect_sip_once(lambda: protocol, proxy, True, False)

        assert captured
        assert isinstance(captured[0], ssl.SSLContext)

    async def test_no_verify_tls_disables_certificate_check(self):
        """_connect_sip_once disables cert verification when no_verify_tls is True."""
        import ssl

        protocol = _make_fake_sip_protocol()
        proxy = NetworkAddress(ipaddress.ip_address("127.0.0.1"), 5061)
        captured: list = []

        async def fake_connect(factory, *, host, port, ssl=None):
            captured.append(ssl)
            return MagicMock(), protocol

        loop = asyncio.get_event_loop()
        with patch.object(loop, "create_connection", side_effect=fake_connect):
            await _connect_sip_once(lambda: protocol, proxy, True, True)

        assert captured
        ctx = captured[0]
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.check_hostname is False
        assert ctx.verify_mode == ssl.CERT_NONE


# ---------------------------------------------------------------------------
# ConsoleMessageProtocol
# ---------------------------------------------------------------------------


class TestConsoleMessageProtocol:
    def test_verbose_0_does_not_print(self, capsys):
        """Verbose=0 suppresses all output from pprint."""
        from voip.rtp import RealtimeTransportProtocol
        from voip.sip.messages import Request

        mux = RealtimeTransportProtocol()
        aor = SipUri.parse("sips:alice:secret@example.com")
        proto = ConsoleMessageProtocol(
            aor=aor,
            rtp=mux,
            transaction_class=InviteTransaction,
            verbose=0,
        )
        request = Request(
            method="OPTIONS",
            uri="sip:alice@example.com",
            headers={
                "Via": "SIP/2.0/TLS 127.0.0.1:5061;branch=z9hG4bKtest",
                "From": "sip:alice@example.com;tag=t1",
                "To": "sip:alice@example.com",
                "Call-ID": "c@test",
                "CSeq": "1 OPTIONS",
            },
        )
        proto.pprint(request)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_3_prints_message(self, capsys):
        """Verbose=3 prints the formatted SIP message with peer address."""
        import dataclasses

        from voip.rtp import RealtimeTransportProtocol
        from voip.sip.messages import Request

        mux = RealtimeTransportProtocol()
        aor = SipUri.parse("sips:alice:secret@example.com")
        proto = ConsoleMessageProtocol(
            aor=aor,
            rtp=mux,
            transaction_class=InviteTransaction,
            verbose=3,
        )

        @dataclasses.dataclass
        class FakeTransportStub:
            def get_extra_info(self, key, default=None):
                match key:
                    case "peername":
                        return ("192.0.2.1", 5061)
                    case _:
                        return default

        proto.transport = FakeTransportStub()
        request = Request(
            method="OPTIONS",
            uri="sip:alice@example.com",
            headers={
                "Via": "SIP/2.0/TLS 127.0.0.1:5061;branch=z9hG4bKtest",
                "From": "sip:alice@example.com;tag=t1",
                "To": "sip:alice@example.com",
                "Call-ID": "c2@test",
                "CSeq": "1 OPTIONS",
            },
        )
        proto.pprint(request)
        captured = capsys.readouterr()
        assert "192.0.2.1" in captured.out

    def test_verbose_3_prints_message_ipv6(self, capsys):
        """Verbose=3 formats IPv6 peer address in brackets."""
        import dataclasses

        from voip.rtp import RealtimeTransportProtocol
        from voip.sip.messages import Request

        mux = RealtimeTransportProtocol()
        aor = SipUri.parse("sips:alice:secret@example.com")
        proto = ConsoleMessageProtocol(
            aor=aor,
            rtp=mux,
            transaction_class=InviteTransaction,
            verbose=3,
        )

        @dataclasses.dataclass
        class FakeIPv6Transport:
            def get_extra_info(self, key, default=None):
                match key:
                    case "peername":
                        return ("::1", 5061)
                    case _:
                        return default

        proto.transport = FakeIPv6Transport()
        request = Request(
            method="OPTIONS",
            uri="sip:alice@example.com",
            headers={
                "Via": "SIP/2.0/TLS ::1;branch=z9hG4bKtest6",
                "From": "sip:alice@example.com;tag=t2",
                "To": "sip:alice@example.com",
                "Call-ID": "c3@test",
                "CSeq": "1 OPTIONS",
            },
        )
        proto.pprint(request)
        captured = capsys.readouterr()
        assert "[::1]" in captured.out

    def test_verbose_3_no_transport_prints_unknown(self, capsys):
        """Verbose=3 prints '[unknown]' when no transport is set."""
        from voip.rtp import RealtimeTransportProtocol
        from voip.sip.messages import Request

        mux = RealtimeTransportProtocol()
        aor = SipUri.parse("sips:alice:secret@example.com")
        proto = ConsoleMessageProtocol(
            aor=aor,
            rtp=mux,
            transaction_class=InviteTransaction,
            verbose=3,
        )
        proto.transport = None
        request = Request(
            method="OPTIONS",
            uri="sip:alice@example.com",
            headers={
                "Via": "SIP/2.0/TLS 127.0.0.1:5061;branch=z9hG4bKtest7",
                "From": "sip:alice@example.com;tag=t3",
                "To": "sip:alice@example.com",
                "Call-ID": "c4@test",
                "CSeq": "1 OPTIONS",
            },
        )
        proto.pprint(request)
        captured = capsys.readouterr()
        assert "[unknown]" in captured.out


# ---------------------------------------------------------------------------
# _parse_dial_target
# ---------------------------------------------------------------------------


class TestParseDialTarget:
    def test_none_returns_none(self):
        """_parse_dial_target returns None when no --dial option is provided."""
        assert _parse_dial_target(None) is None

    def test_valid_uri_returns_sip_uri(self):
        """_parse_dial_target returns a parsed SipUri for a valid SIP URI string."""
        result = _parse_dial_target("sip:bob@biloxi.com")
        assert isinstance(result, SipUri)
        assert str(result.user) == "bob"
        assert str(result.host) == "biloxi.com"

    def test_invalid_uri_raises_bad_parameter(self):
        """_parse_dial_target raises click.BadParameter for an invalid SIP URI."""
        import click  # noqa: PLC0415

        with pytest.raises(click.BadParameter):
            _parse_dial_target("not-a-sip-uri")


# ---------------------------------------------------------------------------
# _make_outbound_factory
# ---------------------------------------------------------------------------


class TestMakeOutboundFactory:
    def test_factory_creates_protocol_with_dial_target(self):
        """Factory produces a protocol whose dial_target matches the target URI."""
        mux = RealtimeTransportProtocol()
        aor = SipUri.parse("sips:alice:secret@example.com")
        target = SipUri.parse("sip:bob@biloxi.com")

        from voip.audio import EchoCall  # noqa: PLC0415

        factory = _make_outbound_factory(
            verbose=0,
            aor=aor,
            rtp_protocol=mux,
            target_uri=target,
            call_class=EchoCall,
            call_kwargs={},
        )
        proto = factory()
        assert proto.dial_target == str(target)


# ---------------------------------------------------------------------------
# echo --dial command
# ---------------------------------------------------------------------------


class TestEchoDialCommand:
    def test_echo_dial__invalid_target_raises_bad_parameter(self):
        """Echo --dial raises BadParameter for an invalid SIP URI target."""
        from click.testing import CliRunner  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(
            voip,
            ["sip", "sips:alice:secret@example.com", "echo", "--dial", "not-a-sip-uri"],
        )
        assert result.exit_code != 0
        assert "--dial" in result.output

    def test_echo_dial__initiates_outbound_invite(self):
        """Echo --dial registers, then sends an INVITE to the target."""
        import dataclasses  # noqa: PLC0415

        sent_data: list[bytes] = []

        @dataclasses.dataclass
        class WritingTransport:
            closed: bool = False

            def write(self, data: bytes) -> None:
                sent_data.append(data)

            def close(self) -> None:
                self.closed = True

            def get_extra_info(self, key, default=None):
                return _fake_transport_get_extra_info(key, default)

        transport = WritingTransport()
        mux = RealtimeTransportProtocol()
        mux.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)

        async def fake_connect_rtp(proxy_addr, stun):
            return MagicMock(), mux

        async def fake_connect_sip_once(factory, proxy_addr, use_tls, no_verify_tls):
            proto = factory()
            proto.connection_made(transport)
            if proto.keepalive_task:
                proto.keepalive_task.cancel()
                proto.keepalive_task = None
            await _simulate_register_ok(proto, "reg@example.com")
            await asyncio.sleep(0)

        _run_dial_command(
            fake_connect_rtp,
            fake_connect_sip_once,
            [
                "sip",
                "sips:alice:secret@example.com",
                "echo",
                "--dial",
                "sip:bob@biloxi.com",
            ],
        )

        assert any(b"INVITE" in data for data in sent_data)

    def test_echo_dial__bye_received_closes_session(self):
        """Echo --dial closes the session when BYE arrives."""
        import dataclasses  # noqa: PLC0415

        proto_ref: list = []

        @dataclasses.dataclass
        class ClosingTransport:
            closed: bool = False
            sent: list = dataclasses.field(default_factory=list)

            def write(self, data: bytes) -> None:
                self.sent.append(data)

            def close(self) -> None:
                self.closed = True
                if proto_ref:
                    proto_ref[0].connection_lost(None)

            def get_extra_info(self, key, default=None):
                return _fake_transport_get_extra_info(key, default)

        transport = ClosingTransport()
        mux = RealtimeTransportProtocol()
        mux.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)

        async def fake_connect_rtp(proxy_addr, stun):
            return MagicMock(), mux

        async def fake_connect_sip_once(factory, proxy_addr, use_tls, no_verify_tls):
            proto = factory()
            proto_ref.append(proto)
            proto.connection_made(transport)
            if proto.keepalive_task:
                proto.keepalive_task.cancel()
                proto.keepalive_task = None
            await _simulate_register_ok(proto, "reg2@example.com")
            await asyncio.sleep(0)
            invite_tx = _find_invite_tx(proto)
            if invite_tx is None:
                return
            await _simulate_invite_ok(invite_tx)
            await _simulate_bye(proto)
            await proto.disconnected_event.wait()

        _run_dial_command(
            fake_connect_rtp,
            fake_connect_sip_once,
            [
                "sip",
                "sips:alice:secret@example.com",
                "echo",
                "--dial",
                "sip:bob@biloxi.com",
            ],
        )

        assert transport.closed


# ---------------------------------------------------------------------------
# say command
# ---------------------------------------------------------------------------


class TestSayCommand:
    def test_say__invalid_target_raises_bad_parameter(self):
        """Say raises BadParameter for an invalid SIP URI target."""
        from click.testing import CliRunner  # noqa: PLC0415

        runner = CliRunner()
        result = runner.invoke(
            voip,
            [
                "sip",
                "sips:alice:secret@example.com",
                "say",
                "not-a-sip-uri",
                "Hello",
            ],
        )
        assert result.exit_code != 0
        assert "TARGET" in result.output

    def test_say__initiates_outbound_invite(self):
        """Say registers and sends an INVITE to the target."""
        import dataclasses  # noqa: PLC0415

        sent_data: list[bytes] = []

        @dataclasses.dataclass
        class WritingTransport:
            closed: bool = False

            def write(self, data: bytes) -> None:
                sent_data.append(data)

            def close(self) -> None:
                self.closed = True

            def get_extra_info(self, key, default=None):
                return _fake_transport_get_extra_info(key, default)

        transport = WritingTransport()
        mux = RealtimeTransportProtocol()
        mux.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)

        async def fake_connect_rtp(proxy_addr, stun):
            return MagicMock(), mux

        async def fake_connect_sip_once(factory, proxy_addr, use_tls, no_verify_tls):
            proto = factory()
            proto.connection_made(transport)
            if proto.keepalive_task:
                proto.keepalive_task.cancel()
                proto.keepalive_task = None
            await _simulate_register_ok(proto, "reg-say@example.com")
            await asyncio.sleep(0)

        _run_dial_command(
            fake_connect_rtp,
            fake_connect_sip_once,
            [
                "sip",
                "sips:alice:secret@example.com",
                "say",
                "sip:bob@biloxi.com",
                "Hello!",
            ],
        )

        assert any(b"INVITE" in data for data in sent_data)


# ---------------------------------------------------------------------------
# Helpers used by TestEchoDialCommand / TestSayCommand
# ---------------------------------------------------------------------------


async def _simulate_register_ok(proto, call_id: str) -> None:
    """Send a 200 OK REGISTER response to *proto*."""
    from voip.sip.messages import Message

    reg_branch = list(proto.transactions.keys())[0]
    proto.response_received(
        Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={reg_branch}\r\n"
            f"From: sips:alice@example.com;tag=our-tag\r\n"
            f"To: sips:example.com;tag=rt\r\n"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: 1 REGISTER\r\n"
            f"\r\n".encode()
        )
    )


def _find_invite_tx(proto):
    """Return the first outbound InviteTransaction in *proto.transactions*."""
    return next(
        (tx for tx in proto.transactions.values() if hasattr(tx, "pending_call_class")),
        None,
    )


async def _simulate_invite_ok(invite_tx) -> None:
    """Send a 200 OK INVITE response and complete _accept_call."""
    from voip.sip.messages import Message

    ok_invite = Message.parse(
        f"SIP/2.0 200 OK\r\n"
        f"Via: SIP/2.0/TLS example.com;branch={invite_tx.branch}\r\n"
        f"From: sips:alice@example.com;tag=our-tag\r\n"
        f"To: sip:bob@biloxi.com;tag=callee-tag\r\n"
        f"Call-ID: invite2@example.com\r\n"
        f"CSeq: 1 INVITE\r\n"
        f"\r\n".encode()
    )
    await invite_tx._accept_call(ok_invite)


async def _simulate_bye(proto) -> None:
    """Deliver a BYE request to *proto*."""
    from voip.sip.messages import Message

    proto.request_received(
        Message.parse(
            b"BYE sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKbye999\r\n"
            b"From: sip:bob@biloxi.com;tag=callee-tag\r\n"
            b"To: sips:alice@example.com;tag=our-tag\r\n"
            b"Call-ID: invite2@example.com\r\n"
            b"CSeq: 2 BYE\r\n"
            b"\r\n"
        )
    )


def _run_dial_command(
    fake_connect_rtp,
    fake_connect_sip_once,
    cli_args: list[str],
) -> None:
    """Invoke a dial-capable CLI command with patched transport helpers."""
    import voip.__main__ as main_module  # noqa: PLC0415
    from click.testing import CliRunner  # noqa: PLC0415

    orig_rtp = main_module._connect_rtp
    orig_sip_once = main_module._connect_sip_once
    main_module._connect_rtp = fake_connect_rtp
    main_module._connect_sip_once = fake_connect_sip_once
    try:
        result = CliRunner().invoke(voip, cli_args, catch_exceptions=False)
    finally:
        main_module._connect_rtp = orig_rtp
        main_module._connect_sip_once = orig_sip_once
    assert result.exit_code == 0
