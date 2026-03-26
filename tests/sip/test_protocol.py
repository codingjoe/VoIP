"""Tests for the SIP asyncio protocol handler."""

import asyncio
import datetime
import ipaddress

from voip.sip.messages import Message, Response
from voip.sip.protocol import PING, PONG, SessionInitiationProtocol
from voip.sip.transactions import InviteTransaction
from voip.sip.types import SIPMethod, SipUri
from voip.types import NetworkAddress

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

    # ------------------------------------------------------------------
    # Helpers shared by extract_frames / dispatch_frame tests
    # ------------------------------------------------------------------

    def _make_session(self, rtp, fake_transport=None):
        """Return a SessionInitiationProtocol wired with a fake transport.

        Does **not** call `connection_made` to avoid triggering
        `RegistrationTransaction.__post_init__` inside a running event loop,
        which raises `TypeError` due to a `super()` / `slots=True` interaction
        in Python 3.13.
        """
        session = SessionInitiationProtocol(
            aor=SipUri.parse("sips:alice@example.com"),
            rtp=rtp,
            transaction_class=InviteTransaction,
        )
        transport = fake_transport or FakeTransport()
        session.transport = transport
        session.local_address = NetworkAddress(ipaddress.ip_address("127.0.0.1"), 5061)
        return session

    # ------------------------------------------------------------------
    # extract_frames
    # ------------------------------------------------------------------

    def test_extract_frames__empty_buffer(self, rtp):
        """Return an empty iterator when the receive buffer is empty."""
        session = self._make_session(rtp)
        assert [bytes(f) for f in session._extract_frames()] == []

    def test_extract_frames__complete_message(self, rtp):
        """Extract a single complete SIP message and clear the buffer."""
        session = self._make_session(rtp)
        session.recv_buffer.extend(INVITE_BYTES)
        frames = [bytes(f) for f in session._extract_frames()]
        assert frames == [INVITE_BYTES]
        assert len(session.recv_buffer) == 0

    def test_extract_frames__partial_headers(self, rtp):
        """Keep partial message bytes in the buffer until complete."""
        session = self._make_session(rtp)
        session.recv_buffer.extend(INVITE_BYTES[:20])
        assert [bytes(f) for f in session._extract_frames()] == []
        assert len(session.recv_buffer) == 20

    def test_extract_frames__two_coalesced_messages(self, rtp):
        """Extract two SIP messages delivered in a single TCP segment."""
        second = (
            b"OPTIONS sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKopt1\r\n"
            b"From: sip:bob@biloxi.com;tag=t99\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: opt-coalesced@biloxi.com\r\n"
            b"CSeq: 1 OPTIONS\r\n"
            b"\r\n"
        )
        session = self._make_session(rtp)
        session.recv_buffer.extend(INVITE_BYTES + second)
        frames = [bytes(f) for f in session._extract_frames()]
        assert len(frames) == 2
        assert frames[0] == INVITE_BYTES
        assert frames[1] == second
        assert len(session.recv_buffer) == 0

    def test_extract_frames__message_with_body(self, rtp):
        """Extract a SIP message that includes a Content-Length body."""
        body = b"v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"
        headers = (
            b"INVITE sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKbody1\r\n"
            b"From: sip:bob@biloxi.com;tag=tb1\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: body-call@biloxi.com\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"Content-Type: application/sdp\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"\r\n"
        )
        message = headers + body
        session = self._make_session(rtp)
        session.recv_buffer.extend(message)
        frames = [bytes(f) for f in session._extract_frames()]
        assert frames == [message]
        assert len(session.recv_buffer) == 0

    def test_extract_frames__incomplete_body(self, rtp):
        """Keep bytes in the buffer when only part of the body has arrived."""
        body = b"v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\n"
        headers = (
            b"INVITE sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKbody2\r\n"
            b"From: sip:bob@biloxi.com;tag=tb2\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: body2-call@biloxi.com\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"\r\n"
        )
        session = self._make_session(rtp)
        session.recv_buffer.extend(headers + body[:5])
        assert [bytes(f) for f in session._extract_frames()] == []
        assert len(session.recv_buffer) == len(headers) + 5

    def test_extract_frames__ping(self, rtp):
        """Extract an RFC 5626 PING (CRLF CRLF) keepalive frame."""
        session = self._make_session(rtp)
        session.recv_buffer.extend(PING)
        frames = [bytes(f) for f in session._extract_frames()]
        assert frames == [PING]
        assert len(session.recv_buffer) == 0

    def test_extract_frames__pong(self, rtp):
        """Extract an RFC 5626 PONG (CRLF) keepalive frame."""
        session = self._make_session(rtp)
        session.recv_buffer.extend(PONG)
        frames = [bytes(f) for f in session._extract_frames()]
        assert frames == [PONG]
        assert len(session.recv_buffer) == 0

    def test_extract_frames__partial_keepalive_wait(self, rtp):
        """Buffer a partial PING (CRLF CR) without dispatching until the 4th byte arrives."""
        session = self._make_session(rtp)
        session.recv_buffer.extend(b"\r\n\r")
        assert [bytes(f) for f in session._extract_frames()] == []
        assert len(session.recv_buffer) == 3

    def test_extract_frames__ping_followed_by_message(self, rtp):
        """Extract a PING and a SIP message from the same buffer."""
        session = self._make_session(rtp)
        session.recv_buffer.extend(PING + INVITE_BYTES)
        frames = [bytes(f) for f in session._extract_frames()]
        assert len(frames) == 2
        assert frames[0] == PING
        assert frames[1] == INVITE_BYTES

    def test_extract_frames__pong_followed_by_message(self, rtp):
        """Extract a PONG and a SIP message coalesced in the same buffer."""
        session = self._make_session(rtp)
        session.recv_buffer.extend(PONG + INVITE_BYTES)
        frames = [bytes(f) for f in session._extract_frames()]
        assert len(frames) == 2
        assert frames[0] == PONG
        assert frames[1] == INVITE_BYTES

    def test_extract_frames__invalid_content_length(self, rtp):
        """Treat an unparseable Content-Length as zero (no body)."""
        message = (
            b"OPTIONS sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKcl0\r\n"
            b"From: sip:bob@biloxi.com;tag=tclx\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: clx-call@biloxi.com\r\n"
            b"CSeq: 1 OPTIONS\r\n"
            b"Content-Length: notanumber\r\n"
            b"\r\n"
        )
        session = self._make_session(rtp)
        session.recv_buffer.extend(message)
        frames = [bytes(f) for f in session._extract_frames()]
        assert len(frames) == 1

    def test_extract_frames__single_cr_waits(self, rtp):
        """Keep a lone CR in the buffer without dispatching (incomplete keepalive)."""
        session = self._make_session(rtp)
        session.recv_buffer.extend(b"\r")
        assert [bytes(f) for f in session._extract_frames()] == []
        assert session.recv_buffer == bytearray(b"\r")

    # ------------------------------------------------------------------
    # dispatch_frame
    # ------------------------------------------------------------------

    def test_dispatch_frame__pong(self, rtp, fake_transport, caplog):
        """Log PONG on receiving a CRLF frame without sending a reply."""
        import logging

        session = self._make_session(rtp, fake_transport)
        with caplog.at_level(logging.INFO):
            session._dispatch_frame(b"\r\n")
        assert "PONG" in caplog.text
        assert fake_transport.sent == []

    def test_dispatch_frame__ping__sends_pong(self, rtp, fake_transport):
        """Reply with a CRLF PONG when a PING (CRLF CRLF) frame is dispatched."""
        session = self._make_session(rtp, fake_transport)
        session._dispatch_frame(b"\r\n\r\n")
        assert b"\r\n" in fake_transport.sent

    def test_dispatch_frame__sip_request(self, rtp, fake_transport):
        """Dispatch a SIP request frame to request_received."""
        session = self._make_session(rtp, fake_transport)
        before = len(session.transactions)
        session._dispatch_frame(INVITE_BYTES)
        assert len(session.transactions) > before

    def test_dispatch_frame__sip_response(self, rtp, fake_transport):
        """Dispatch a SIP response frame to response_received without error."""
        session = self._make_session(rtp, fake_transport)
        branch = "z9hG4bKresp-test"
        session.transactions[branch] = InviteTransaction(
            sip=session,
            method=SIPMethod.INVITE,
            branch=branch,
            cseq=1,
        )
        response_bytes = (
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={branch}\r\n"
            f"From: sip:alice@example.com;tag=local\r\n"
            f"To: sip:example.com;tag=remote\r\n"
            f"Call-ID: resp-test@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"\r\n"
        ).encode()
        session._dispatch_frame(response_bytes)

    # ------------------------------------------------------------------
    # data_received – stream reassembly
    # ------------------------------------------------------------------

    def test_data_received__split_message(self, rtp, fake_transport):
        """Reassemble a SIP request delivered in two TCP segments."""
        session = self._make_session(rtp, fake_transport)
        split = len(INVITE_BYTES) // 2
        before = len(session.transactions)
        session.data_received(INVITE_BYTES[:split])
        assert len(session.transactions) == before  # incomplete – not dispatched yet
        session.data_received(INVITE_BYTES[split:])
        assert len(session.transactions) > before  # now dispatched

    def test_data_received__coalesced_messages(self, rtp, fake_transport):
        """Dispatch two SIP requests coalesced into one TCP segment."""
        second = (
            b"OPTIONS sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKopt2\r\n"
            b"From: sip:bob@biloxi.com;tag=t88\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: opt-coalesced2@biloxi.com\r\n"
            b"CSeq: 2 OPTIONS\r\n"
            b"\r\n"
        )
        session = self._make_session(rtp, fake_transport)
        before = len(session.transactions)
        session.data_received(INVITE_BYTES + second)
        # INVITE creates a transaction; OPTIONS is answered directly (no tx added)
        assert len(session.transactions) > before

    def test_data_received__body_split_across_segments(self, rtp, fake_transport):
        """Reassemble a SIP request with a body split across two TCP segments."""
        body = b"v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n"
        headers = (
            b"INVITE sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKbodysplit\r\n"
            b"From: sip:bob@biloxi.com;tag=tbs\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: body-split@biloxi.com\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"Content-Type: application/sdp\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"\r\n"
        )
        session = self._make_session(rtp, fake_transport)
        before = len(session.transactions)
        session.data_received(headers + body[:5])
        assert len(session.transactions) == before  # body incomplete
        session.data_received(body[5:])
        assert len(session.transactions) > before  # body complete, dispatched

    # ------------------------------------------------------------------
    # send_keepalive (via _make_session to avoid RegistrationTransaction)
    # ------------------------------------------------------------------

    async def test_send_keepalive__sends_ping_without_sip_fixture(self, rtp):
        """send_keepalive writes a PING (CRLF CRLF) after the interval elapses."""
        fake_transport = FakeTransport()
        session = self._make_session(rtp, fake_transport)
        session.keepalive_interval = datetime.timedelta(milliseconds=10)
        task = asyncio.create_task(session.send_keepalive())
        await asyncio.sleep(0.05)
        task.cancel()
        assert b"\r\n\r\n" in fake_transport.sent

    async def test_send_keepalive__stops_when_transport_cleared(self, rtp):
        """send_keepalive exits when transport is set to None."""
        session = self._make_session(rtp)
        session.transport = None
        session.keepalive_interval = datetime.timedelta(milliseconds=1)
        await session.send_keepalive()

    # ------------------------------------------------------------------
    # send / close (via _make_session)
    # ------------------------------------------------------------------

    def test_send__writes_bytes_without_sip_fixture(self, rtp, fake_transport):
        """send() serialises and writes a SIP message to the transport."""
        session = self._make_session(rtp, fake_transport)
        response = Response(status_code=200, phrase="OK")
        session.send(response)
        assert bytes(response) in fake_transport.sent

    def test_send__no_op_when_transport_is_none(self, rtp):
        """send() is a no-op when transport is None."""
        session = self._make_session(rtp)
        session.transport = None
        session.send(Response(status_code=200, phrase="OK"))

    def test_close__closes_transport_without_sip_fixture(self, rtp, fake_transport):
        """close() closes the underlying transport."""
        session = self._make_session(rtp, fake_transport)
        session.close()
        assert fake_transport.closed is True

    def test_close__no_op_when_transport_is_none(self, rtp):
        """close() is a no-op when transport is None."""
        session = self._make_session(rtp)
        session.transport = None
        session.close()

    # ------------------------------------------------------------------
    # allowed_methods / allow_header / method_not_allowed (via _make_session)
    # ------------------------------------------------------------------

    def test_allowed_methods__includes_options_without_sip_fixture(self, rtp):
        """OPTIONS is always included in allowed_methods."""
        session = self._make_session(rtp)
        assert SIPMethod.OPTIONS in session.allowed_methods

    def test_allow_header__is_comma_separated_without_sip_fixture(self, rtp):
        """allow_header returns a comma-separated string of methods."""
        session = self._make_session(rtp)
        header = session.allow_header
        assert "OPTIONS" in header
        assert "," in header

    def test_method_not_allowed__sends_405_without_sip_fixture(
        self, rtp, fake_transport
    ):
        """method_not_allowed() sends a 405 Method Not Allowed response."""
        session = self._make_session(rtp, fake_transport)
        request = Message.parse(
            b"PUBLISH sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKpub3\r\n"
            b"From: sip:bob@biloxi.com;tag=t5\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: pub3-call@biloxi.com\r\n"
            b"CSeq: 1 PUBLISH\r\n"
            b"\r\n"
        )
        session.method_not_allowed(request)
        assert any(b"405" in data for data in fake_transport.sent)

    # ------------------------------------------------------------------
    # request_received (via _make_session)
    # ------------------------------------------------------------------

    def test_request_received__options_sends_200_without_sip_fixture(
        self, rtp, fake_transport
    ):
        """OPTIONS request is answered with 200 OK."""
        session = self._make_session(rtp, fake_transport)
        request = Message.parse(
            b"OPTIONS sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKopt3\r\n"
            b"From: sip:bob@biloxi.com;tag=t6\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: opt3-call@biloxi.com\r\n"
            b"CSeq: 3 OPTIONS\r\n"
            b"\r\n"
        )
        session.request_received(request)
        assert any(b"200" in data for data in fake_transport.sent)

    def test_request_received__invite_creates_transaction_without_sip_fixture(
        self, rtp, fake_transport
    ):
        """INVITE request creates an InviteTransaction."""
        session = self._make_session(rtp, fake_transport)
        request = Message.parse(INVITE_BYTES)
        before = len(session.transactions)
        session.request_received(request)
        assert len(session.transactions) > before

    def test_request_received__unsupported_method_sends_405_without_sip_fixture(
        self, rtp, fake_transport
    ):
        """Unsupported method triggers method_not_allowed (405)."""
        session = self._make_session(rtp, fake_transport)
        request = Message.parse(
            b"PUBLISH sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKpub4\r\n"
            b"From: sip:bob@biloxi.com;tag=t7\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: pub4-call@biloxi.com\r\n"
            b"CSeq: 1 PUBLISH\r\n"
            b"\r\n"
        )
        session.request_received(request)
        assert any(b"405" in data for data in fake_transport.sent)

    def test_request_received__cancel_dispatches_to_transaction_without_sip_fixture(
        self, rtp, fake_transport
    ):
        """CANCEL is forwarded to the matching INVITE transaction."""
        session = self._make_session(rtp, fake_transport)
        invite = Message.parse(INVITE_BYTES)
        session.request_received(invite)
        tx = session.transactions[invite.branch]
        session.dialogs[(tx.dialog.remote_tag, tx.dialog.local_tag)] = tx.dialog
        cancel = Message.parse(
            b"CANCEL sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKabc123\r\n"
            b"From: sip:bob@biloxi.com;tag=from-tag-1\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: test-call-id@biloxi.com\r\n"
            b"CSeq: 1 CANCEL\r\n"
            b"\r\n"
        )
        session.request_received(cancel)
        assert any(b"200" in data for data in fake_transport.sent)

    def test_request_received__cancel_gone_when_no_transaction_without_sip_fixture(
        self, rtp, fake_transport
    ):
        """CANCEL with no matching transaction returns 410 Gone."""
        session = self._make_session(rtp, fake_transport)
        cancel = Message.parse(
            b"CANCEL sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKnone2\r\n"
            b"From: sip:bob@biloxi.com;tag=t8\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: no-tx2@biloxi.com\r\n"
            b"CSeq: 1 CANCEL\r\n"
            b"\r\n"
        )
        session.request_received(cancel)
        assert any(b"410" in data for data in fake_transport.sent)

    # ------------------------------------------------------------------
    # response_received (via _make_session)
    # ------------------------------------------------------------------

    def test_response_received__delegates_to_transaction_without_sip_fixture(
        self, rtp, fake_transport
    ):
        """Response is forwarded to the matching transaction."""
        session = self._make_session(rtp, fake_transport)
        branch = "z9hG4bKdel-test"
        session.transactions[branch] = InviteTransaction(
            sip=session,
            method=SIPMethod.INVITE,
            branch=branch,
            cseq=1,
        )
        response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={branch}\r\n"
            f"From: sip:alice@example.com;tag=lt\r\n"
            f"To: sip:example.com;tag=rt\r\n"
            f"Call-ID: del-test@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"\r\n".encode()
        )
        session.response_received(response)

    def test_response_received__warns_on_unknown_branch_without_sip_fixture(
        self, rtp, fake_transport, caplog
    ):
        """Log a warning when the response branch is not in transactions."""
        import logging

        session = self._make_session(rtp, fake_transport)
        response = Message.parse(
            b"SIP/2.0 200 OK\r\n"
            b"Via: SIP/2.0/TLS example.com;branch=z9hG4bKunknown\r\n"
            b"From: sip:alice@example.com;tag=lt2\r\n"
            b"To: sip:example.com;tag=rt2\r\n"
            b"Call-ID: unknown-branch@example.com\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"\r\n"
        )
        with caplog.at_level(logging.WARNING):
            session.response_received(response)
        assert "unknown branch" in caplog.text

    # ------------------------------------------------------------------
    # connection_lost (via _make_session)
    # ------------------------------------------------------------------

    async def test_connection_lost__cancels_keepalive_without_sip_fixture(self, rtp):
        """connection_lost() cancels and clears the keepalive task."""
        session = self._make_session(rtp)
        session.keepalive_task = asyncio.create_task(asyncio.sleep(9999))
        session.connection_lost(None)
        assert session.keepalive_task is None

    def test_connection_lost__clears_transport_without_sip_fixture(self, rtp):
        """connection_lost() sets transport to None."""
        session = self._make_session(rtp)
        session.connection_lost(None)
        assert session.transport is None

    def test_connection_lost__sets_disconnected_event_without_sip_fixture(self, rtp):
        """connection_lost() sets the disconnected_event."""
        session = self._make_session(rtp)
        session.connection_lost(None)
        assert session.disconnected_event.is_set()

    def test_connection_lost__logs_exception_without_sip_fixture(self, rtp, caplog):
        """connection_lost() logs an error when an exception is provided."""
        import logging

        session = self._make_session(rtp)
        with caplog.at_level(logging.ERROR):
            session.connection_lost(OSError("reset"))
        assert session.transport is None

    def test_connection_lost__no_keepalive_task_without_sip_fixture(self, rtp):
        """connection_lost() is safe when keepalive_task is None."""
        session = self._make_session(rtp)
        session.keepalive_task = None
        session.connection_lost(None)
        assert session.transport is None
