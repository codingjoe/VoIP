"""Tests for the SIP asyncio protocol handler."""

from __future__ import annotations

import asyncio
import datetime

from voip.sip.messages import Message, Response
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.transactions import InviteTransaction
from voip.sip.types import SIPMethod, SipUri

from .conftest import INVITE_BYTES, FakeTransport


class TestSessionInitiationProtocol:
    def test_connection_made__stores_transport(self, fake_transport, rtp):
        """Store transport reference after connection_made."""
        session = SessionInitiationProtocol(
            aor=SipUri.parse("sips:alice:secret@example.com"),
            rtp=rtp,
            transaction_class=InviteTransaction,
        )
        session.connection_made(fake_transport)
        assert session.transport is fake_transport

    def test_connection_made__sets_local_address(self, fake_transport, rtp):
        """Set local_address from the socket's sockname after connection_made."""
        session = SessionInitiationProtocol(
            aor=SipUri.parse("sips:alice:secret@example.com"),
            rtp=rtp,
            transaction_class=InviteTransaction,
        )
        session.connection_made(fake_transport)
        assert str(session.local_address.host) == "127.0.0.1"
        assert session.local_address.port == 5061

    def test_connection_made__sets_is_secure_for_tls(self, rtp):
        """Mark connection as secure when ssl_object is present."""
        transport = FakeTransport(_ssl=True)
        session = SessionInitiationProtocol(
            aor=SipUri.parse("sips:alice:secret@example.com"),
            rtp=rtp,
            transaction_class=InviteTransaction,
        )
        session.connection_made(transport)
        assert session.is_secure is True

    def test_connection_made__is_not_secure_without_ssl(self, rtp):
        """Mark connection as not secure when ssl_object is absent."""
        transport = FakeTransport(_ssl=False)
        session = SessionInitiationProtocol(
            aor=SipUri.parse("sip:alice:secret@example.com"),
            rtp=rtp,
            transaction_class=InviteTransaction,
        )
        session.connection_made(transport)
        assert session.is_secure is False

    async def test_connection_made__sends_register(self, fake_transport, rtp):
        """Send a REGISTER request immediately after connection_made in async context."""
        session = SessionInitiationProtocol(
            aor=SipUri.parse("sips:alice:secret@example.com"),
            rtp=rtp,
            transaction_class=InviteTransaction,
        )
        session.connection_made(fake_transport)
        if session.keepalive_task:
            session.keepalive_task.cancel()
        assert any(b"REGISTER" in data for data in fake_transport.sent)

    async def test_connection_made__creates_keepalive_task(self, fake_transport, rtp):
        """Create a keepalive task in async context."""
        session = SessionInitiationProtocol(
            aor=SipUri.parse("sips:alice:secret@example.com"),
            rtp=rtp,
            transaction_class=InviteTransaction,
        )
        session.connection_made(fake_transport)
        assert session.keepalive_task is not None
        session.keepalive_task.cancel()

    async def test_send_keepalive__sends_ping(self, sip, fake_transport):
        """Send a CRLF CRLF ping after the keepalive interval elapses."""
        sip.keepalive_interval = datetime.timedelta(milliseconds=10)
        task = asyncio.create_task(sip.send_keepalive())
        await asyncio.sleep(0.05)
        task.cancel()
        assert b"\r\n\r\n" in fake_transport.sent

    async def test_send_keepalive__stops_when_transport_is_none(self, sip):
        """Stop the keepalive loop immediately when transport is cleared."""
        sip.transport = None
        sip.keepalive_interval = datetime.timedelta(milliseconds=1)
        await sip.send_keepalive()

    def test_data_received__pong(self, sip):
        r"""Receive a PONG (\r\n keepalive reply) without sending any reply."""
        initial_sent = len(sip.transport.sent)
        sip.data_received(b"\r\n")
        assert len(sip.transport.sent) == initial_sent

    def test_data_received__ping__sends_pong(self, sip, fake_transport):
        r"""Reply with \r\n when a PING (\r\n\r\n) is received."""
        sip.data_received(b"\r\n\r\n")
        assert b"\r\n" in fake_transport.sent

    def test_data_received__sip_request(self, sip):
        """Dispatch a valid SIP request to request_received without error."""
        before = len(sip.transactions)
        sip.data_received(INVITE_BYTES)
        # An InviteTransaction is added to transactions
        assert len(sip.transactions) > before

    def test_data_received__sip_response(self, sip):
        """Dispatch a valid SIP response to response_received."""
        branch = list(sip.transactions.keys())[0]
        response_bytes = (
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={branch}\r\n"
            f"From: sip:alice@example.com;tag=local-tag\r\n"
            f"To: sip:example.com;tag=remote-tag\r\n"
            f"Call-ID: call-id@example.com\r\n"
            f"CSeq: 1 REGISTER\r\n"
            f"\r\n"
        ).encode()
        sip.data_received(response_bytes)

    def test_send__writes_message_bytes(self, sip, fake_transport):
        """Serialize and write a SIP message to the transport."""
        response = Response(status_code=200, phrase="OK")
        sip.send(response)
        assert bytes(response) in fake_transport.sent

    def test_send__with_no_transport(self, sip):
        """Skip writing when transport is None."""
        sip.transport = None
        response = Response(status_code=200, phrase="OK")
        sip.send(response)

    def test_close__closes_transport(self, sip, fake_transport):
        """Close the underlying transport."""
        sip.close()
        assert fake_transport.closed is True

    def test_close__with_no_transport(self, sip):
        """Do nothing when transport is already None."""
        sip.transport = None
        sip.close()

    def test_allowed_methods__includes_options(self, sip):
        """Always include OPTIONS in allowed methods."""
        assert "OPTIONS" in sip.allowed_methods

    def test_allowed_methods__includes_invite_when_transaction_class_has_handler(
        self, sip
    ):
        """Include INVITE when transaction_class defines invite_received."""
        assert SIPMethod.INVITE in sip.allowed_methods

    def test_allow_header__is_comma_separated_string(self, sip):
        """allow_header returns a comma-separated string of supported methods."""
        header = sip.allow_header
        assert "OPTIONS" in header
        assert "," in header

    def test_method_not_allowed__sends_405(self, sip, fake_transport):
        """Send a 405 Method Not Allowed response."""
        request = Message.parse(
            b"PUBLISH sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKpub\r\n"
            b"From: sip:bob@biloxi.com;tag=t1\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: pub-call@biloxi.com\r\n"
            b"CSeq: 1 PUBLISH\r\n"
            b"\r\n"
        )
        sip.method_not_allowed(request)
        assert any(b"405" in data for data in fake_transport.sent)

    def test_request_received__options__sends_200(self, sip, fake_transport):
        """Reply with 200 OK for an OPTIONS request."""
        request = Message.parse(
            b"OPTIONS sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKopt\r\n"
            b"From: sip:bob@biloxi.com;tag=t1\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: opt-call@biloxi.com\r\n"
            b"CSeq: 2 OPTIONS\r\n"
            b"\r\n"
        )
        sip.request_received(request)
        assert any(b"200" in data for data in fake_transport.sent)

    def test_request_received__invite__creates_transaction(self, sip):
        """Create an InviteTransaction for an incoming INVITE."""
        request = Message.parse(INVITE_BYTES)
        sip.request_received(request)
        assert request.branch in sip.transactions

    def test_request_received__unsupported_method__sends_405(self, sip, fake_transport):
        """Send 405 for a method not handled by the transaction class."""
        request = Message.parse(
            b"PUBLISH sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKpub2\r\n"
            b"From: sip:bob@biloxi.com;tag=t2\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: pub2-call@biloxi.com\r\n"
            b"CSeq: 1 PUBLISH\r\n"
            b"\r\n"
        )
        sip.request_received(request)
        assert any(b"405" in data for data in fake_transport.sent)

    def test_request_received__cancel__dispatches_to_existing_transaction(
        self, sip, fake_transport
    ):
        """Dispatch a CANCEL to the matching INVITE transaction."""
        invite = Message.parse(INVITE_BYTES)
        sip.request_received(invite)
        tx = sip.transactions[invite.branch]
        sip.dialogs[(tx.dialog.remote_tag, tx.dialog.local_tag)] = tx.dialog

        cancel = Message.parse(
            b"CANCEL sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKabc123\r\n"
            b"From: sip:bob@biloxi.com;tag=from-tag-1\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: test-call-id@biloxi.com\r\n"
            b"CSeq: 1 CANCEL\r\n"
            b"\r\n"
        )
        sip.request_received(cancel)
        assert any(b"200" in data for data in fake_transport.sent)

    def test_request_received__cancel__gone_when_no_transaction(
        self, sip, fake_transport
    ):
        """Send 410 Gone for a CANCEL with no matching transaction."""
        cancel = Message.parse(
            b"CANCEL sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKnone\r\n"
            b"From: sip:bob@biloxi.com;tag=t3\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: no-tx@biloxi.com\r\n"
            b"CSeq: 1 CANCEL\r\n"
            b"\r\n"
        )
        sip.request_received(cancel)
        assert any(b"410" in data for data in fake_transport.sent)

    async def test_response_received__delegates_to_transaction(self, sip):
        """Delegate a response to the matching transaction by branch."""
        branch = list(sip.transactions.keys())[0]
        response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={branch}\r\n"
            f"From: sip:alice@example.com;tag=local-tag\r\n"
            f"To: sip:example.com;tag=remote-tag\r\n"
            f"Call-ID: call@example.com\r\n"
            f"CSeq: 1 REGISTER\r\n"
            f"\r\n".encode()
        )
        sip.response_received(response)

    def test_contact__sips_aor_produces_sips_contact(self, sip):
        """Build a sips: Contact for a sips: AOR."""
        assert sip.contact.startswith("<sips:")

    def test_contact__sip_aor_with_tls_produces_transport_param(self, rtp):
        """Build a sip: Contact with transport=tls for a plain sip: AOR over TLS."""
        transport = FakeTransport(_ssl=True)
        session = SessionInitiationProtocol(
            aor=SipUri.parse("sip:alice:secret@example.com"),
            rtp=rtp,
            transaction_class=InviteTransaction,
        )
        session.connection_made(transport)
        if session.keepalive_task:
            session.keepalive_task.cancel()
        assert "transport=tls" in session.contact

    def test_contact__sip_aor_without_tls_has_no_transport_param(self, rtp):
        """Build a sip: Contact without transport= for a plain TCP connection."""
        transport = FakeTransport(_ssl=False)
        session = SessionInitiationProtocol(
            aor=SipUri.parse("sip:alice:secret@example.com"),
            rtp=rtp,
            transaction_class=InviteTransaction,
        )
        session.connection_made(transport)
        assert "transport=tls" not in session.contact

    def test_contact__aor_without_user_omits_user(self, rtp):
        """Build a Contact without a user part when AOR has no user."""
        transport = FakeTransport(_ssl=True)
        session = SessionInitiationProtocol(
            aor=SipUri(scheme="sips", host="example.com"),
            rtp=rtp,
            transaction_class=InviteTransaction,
        )
        session.connection_made(transport)
        if session.keepalive_task:
            session.keepalive_task.cancel()
        assert "@" not in session.contact

    async def test_connection_lost__cancels_keepalive_task(self, sip):
        """Cancel and clear the keepalive task on connection loss."""
        sip.keepalive_task = asyncio.create_task(asyncio.sleep(9999))
        sip.connection_lost(None)
        assert sip.keepalive_task is None

    def test_connection_lost__clears_transport(self, sip):
        """Set transport to None on connection loss."""
        sip.connection_lost(None)
        assert sip.transport is None

    def test_connection_lost__sets_disconnected_event(self, sip):
        """Set the disconnected_event on connection loss."""
        sip.connection_lost(None)
        assert sip.disconnected_event.is_set()

    def test_connection_lost__with_exception_logs_error(self, sip, caplog):
        """Log the exception when connection is lost with an error."""
        import logging

        with caplog.at_level(logging.ERROR):
            sip.connection_lost(OSError("Connection reset"))
        assert sip.transport is None

    def test_connection_lost__without_keepalive_task(self, sip):
        """Handle connection loss gracefully when no keepalive task exists."""
        sip.keepalive_task = None
        sip.connection_lost(None)
        assert sip.transport is None
