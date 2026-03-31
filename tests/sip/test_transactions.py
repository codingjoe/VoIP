"""Tests for the SIP transaction layer."""

import asyncio
import dataclasses

import pytest
from voip.rtp import RealtimeTransportProtocol
from voip.sip.exceptions import RegistrationError
from voip.sip.messages import Dialog, Message, Response
from voip.sip.transactions import (
    ByeTransaction,
    InviteTransaction,
    RegistrationTransaction,
)
from voip.sip.types import DigestAlgorithm, DigestQoP, SIPMethod, SIPStatus, SipUri

from .conftest import INVITE_BYTES, INVITE_WITH_SDP_BYTES, CallFixture, FakeTransport

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_PASSWORD = "secret"  # noqa: S105

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_sip_session(fake_transport=None, rtp=None):
    """Create a minimal SessionInitiationProtocol without async event loop."""
    from voip.sip.messages import Dialog
    from voip.sip.protocol import SessionInitiationProtocol

    transport = fake_transport or FakeTransport()
    mux = rtp or RealtimeTransportProtocol()
    session = SessionInitiationProtocol(
        aor=SipUri.parse("sips:alice:secret@example.com"),
        rtp=mux,
        dialog_class=Dialog,
    )
    # Set up local_address without triggering async registration
    import ipaddress

    from voip.types import NetworkAddress

    session.transport = transport
    session.local_address = NetworkAddress(ipaddress.ip_address("127.0.0.1"), 5061)
    session.is_secure = True
    return session


# ---------------------------------------------------------------------------
# Transaction base class
# ---------------------------------------------------------------------------


class TestTransaction:
    def test_post_init__valid_branch(self):
        """Accept a branch that starts with the magic cookie."""
        sip = create_sip_session()
        tx = InviteTransaction(
            sip=sip,
            method=SIPMethod.INVITE,
            branch="z9hG4bK-test-branch",
            cseq=1,
        )
        assert tx.branch == "z9hG4bK-test-branch"

    def test_post_init__invalid_branch__raises(self):
        """Raise ValueError when branch does not start with 'z9hG4bK'."""
        sip = create_sip_session()
        with pytest.raises(ValueError):
            RegistrationTransaction(
                sip=sip,
                method=SIPMethod.INVITE,
                branch="invalid-branch",
                cseq=1,
            )

    def test_headers__contains_via_and_cseq(self):
        """Return a dict with Via and CSeq headers."""
        sip = create_sip_session()
        tx = InviteTransaction(
            sip=sip,
            method=SIPMethod.INVITE,
            branch="z9hG4bK-headers-test",
            cseq=7,
        )
        headers = tx.headers
        assert "Via" in headers
        assert "CSeq" in headers
        assert "7 INVITE" in headers["CSeq"]

    def test_response_received__is_noop(self):
        """Base response_received does nothing and returns None."""
        sip = create_sip_session()
        tx = InviteTransaction(
            sip=sip,
            method=SIPMethod.INVITE,
            branch="z9hG4bK-noop",
            cseq=1,
        )
        assert tx.response_received(Response(status_code=200, phrase="OK")) is None

    async def test_await__suspends_until_done_is_set(self):
        """Awaiting a transaction suspends until done.set() is called."""
        sip = create_sip_session()
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        task = asyncio.create_task(asyncio.wait_for(asyncio.shield(tx), timeout=1.0))
        await asyncio.sleep(0)
        assert not task.done()
        tx.done.set()
        await task
        assert task.done()

    def test_send_response__calls_sip_send(self):
        """Send a response through the SIP layer."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        tx = InviteTransaction(
            sip=sip,
            method=SIPMethod.INVITE,
            branch="z9hG4bK-send-resp",
            cseq=1,
        )
        response = Response(status_code=200, phrase="OK")
        tx.send_response(response)
        assert bytes(response) in transport.sent

    def test_from_request__creates_transaction_from_request(self):
        """Create an InviteTransaction from an incoming INVITE request."""
        sip = create_sip_session()
        request = Message.parse(INVITE_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        assert tx.branch == request.branch
        assert tx.method == request.method
        assert tx.cseq == request.sequence

    def test_from_request__uses_existing_dialog(self):
        """Reuse an existing dialog when one matches the request's tags."""
        sip = create_sip_session()
        request = Message.parse(INVITE_BYTES)
        # For an INVITE with no To-tag, remote_tag is None and local_tag is from From header
        existing_dialog = Dialog(
            local_tag=request.local_tag, remote_tag=request.remote_tag
        )
        # The lookup key is (request.remote_tag, request.local_tag)
        sip.dialogs[(request.remote_tag, request.local_tag)] = existing_dialog
        tx = InviteTransaction.from_request(request=request, sip=sip)
        assert tx.dialog is existing_dialog


# ---------------------------------------------------------------------------
# RegistrationTransaction
# ---------------------------------------------------------------------------


class TestRegistrationTransaction:
    def test_post_init__sends_register(self):
        """Send a REGISTER request immediately on creation."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        RegistrationTransaction(sip=sip, method=SIPMethod.REGISTER)
        assert any(b"REGISTER" in data for data in transport.sent)

    def test_post_init__includes_contact_header(self):
        """Include Contact header in the REGISTER request."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        RegistrationTransaction(sip=sip, method=SIPMethod.REGISTER)
        register_data = b"".join(transport.sent)
        assert b"Contact:" in register_data

    def test_post_init__with_authorization(self):
        """Include Authorization header when authorization value is provided."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        RegistrationTransaction(
            sip=sip,
            method=SIPMethod.REGISTER,
            authorization='Digest username="alice"',
        )
        register_data = b"".join(transport.sent)
        assert b"Authorization:" in register_data

    def test_post_init__with_proxy_authorization(self):
        """Include Proxy-Authorization header when proxy_authorization is provided."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        RegistrationTransaction(
            sip=sip,
            method=SIPMethod.REGISTER,
            proxy_authorization='Digest username="alice"',
        )
        register_data = b"".join(transport.sent)
        assert b"Proxy-Authorization:" in register_data

    def test_response_received__200_ok(self):
        """Handle 200 OK without error and remove transaction from registry."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        tx = RegistrationTransaction(sip=sip, method=SIPMethod.REGISTER)
        sip.transactions[tx.branch] = tx
        response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=local-tag\r\n"
            f"To: sip:example.com;tag=remote-tag\r\n"
            f"Call-ID: reg-call@example.com\r\n"
            f"CSeq: 1 REGISTER\r\n"
            f"\r\n".encode()
        )
        tx.response_received(response)
        assert tx.branch not in sip.transactions

    def test_response_received__401_sends_credentials(self):
        """Retry with digest credentials after receiving 401 Unauthorized."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        tx = RegistrationTransaction(sip=sip, method=SIPMethod.REGISTER)
        sip.transactions[tx.branch] = tx
        initial_sent_count = len(transport.sent)

        response = Message.parse(
            f"SIP/2.0 401 Unauthorized\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=local-tag\r\n"
            f"To: sip:example.com;tag=remote-tag\r\n"
            f"Call-ID: reg-call@example.com\r\n"
            f"CSeq: 1 REGISTER\r\n"
            f'WWW-Authenticate: Digest realm="example.com", nonce="abc123", algorithm=SHA-256\r\n'
            f"\r\n".encode()
        )
        tx.response_received(response)
        assert len(transport.sent) > initial_sent_count
        second_register = b"".join(transport.sent[initial_sent_count:])
        assert b"Authorization:" in second_register

    def test_response_received__401_with_qop(self):
        """Retry with digest credentials and qop=auth after 401 with qop option."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        tx = RegistrationTransaction(sip=sip, method=SIPMethod.REGISTER)
        sip.transactions[tx.branch] = tx

        response = Message.parse(
            f"SIP/2.0 401 Unauthorized\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=lt\r\n"
            f"To: sip:example.com;tag=rt\r\n"
            f"Call-ID: qop-call@example.com\r\n"
            f"CSeq: 1 REGISTER\r\n"
            f'WWW-Authenticate: Digest realm="example.com", nonce="nonce1", qop="auth", algorithm=SHA-256\r\n'
            f"\r\n".encode()
        )
        tx.response_received(response)
        second_register = b"".join(transport.sent[1:])
        assert b"qop=auth" in second_register

    def test_response_received__401_with_opaque(self):
        """Include opaque parameter in Authorization when challenge includes opaque."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        tx = RegistrationTransaction(sip=sip, method=SIPMethod.REGISTER)
        sip.transactions[tx.branch] = tx

        response = Message.parse(
            f"SIP/2.0 401 Unauthorized\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=lt\r\n"
            f"To: sip:example.com;tag=rt\r\n"
            f"Call-ID: opaque-call@example.com\r\n"
            f"CSeq: 1 REGISTER\r\n"
            f'WWW-Authenticate: Digest realm="example.com", nonce="nonce2", opaque="myopaque", algorithm=SHA-256\r\n'
            f"\r\n".encode()
        )
        tx.response_received(response)
        second_register = b"".join(transport.sent[1:])
        assert b"opaque=" in second_register

    def test_response_received__407_sends_proxy_credentials(self):
        """Retry with Proxy-Authorization after receiving 407."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        tx = RegistrationTransaction(sip=sip, method=SIPMethod.REGISTER)
        sip.transactions[tx.branch] = tx
        initial_sent_count = len(transport.sent)

        response = Message.parse(
            f"SIP/2.0 407 Proxy Authentication Required\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=lt\r\n"
            f"To: sip:example.com;tag=rt\r\n"
            f"Call-ID: proxy-reg@example.com\r\n"
            f"CSeq: 1 REGISTER\r\n"
            f'Proxy-Authenticate: Digest realm="example.com", nonce="proxy-nonce", algorithm=SHA-256\r\n'
            f"\r\n".encode()
        )
        tx.response_received(response)
        second_register = b"".join(transport.sent[initial_sent_count:])
        assert b"Proxy-Authorization:" in second_register

    def test_response_received__unknown_status__raises(self):
        """Raise NotImplementedError for unrecognised status codes."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        tx = RegistrationTransaction(sip=sip, method=SIPMethod.REGISTER)
        sip.transactions[tx.branch] = tx

        response = Message.parse(
            f"SIP/2.0 500 Server Internal Error\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=lt\r\n"
            f"To: sip:example.com;tag=rt\r\n"
            f"Call-ID: err-call@example.com\r\n"
            f"CSeq: 1 REGISTER\r\n"
            f"\r\n".encode()
        )
        with pytest.raises(NotImplementedError):
            tx.response_received(response)

    def test_parse_auth_challenge__parses_realm_and_nonce(self):
        """Extract realm and nonce from a Digest challenge header."""
        header = 'Digest realm="example.com", nonce="abc123"'
        params = RegistrationTransaction.parse_auth_challenge(header)
        assert params["realm"] == "example.com"
        assert params["nonce"] == "abc123"

    def test_parse_auth_challenge__empty_header(self):
        """Return empty dict for an empty challenge header."""
        assert RegistrationTransaction.parse_auth_challenge("") == {}

    def test_parse_auth_challenge__multiple_params(self):
        """Parse multiple parameters including algorithm and qop."""
        header = 'Digest realm="test.com", nonce="xyz", algorithm=SHA-256, qop="auth"'
        params = RegistrationTransaction.parse_auth_challenge(header)
        assert params["realm"] == "test.com"
        assert params["algorithm"] == "SHA-256"
        assert params["qop"] == "auth"

    def test_digest_response__sha256(self):
        """Compute a deterministic SHA-256 digest response."""
        result = RegistrationTransaction.digest_response(
            username="alice",
            password=TEST_PASSWORD,
            realm="example.com",
            nonce="nonce123",
            method="REGISTER",
            uri="example.com",
            algorithm=DigestAlgorithm.SHA_256,
        )
        assert isinstance(result, str)
        assert len(result) == 64

    def test_digest_response__md5(self):
        """Compute a deterministic MD5 digest response."""
        result = RegistrationTransaction.digest_response(
            username="alice",
            password=TEST_PASSWORD,
            realm="example.com",
            nonce="nonce123",
            method="REGISTER",
            uri="example.com",
            algorithm=DigestAlgorithm.MD5,
        )
        assert len(result) == 32

    def test_digest_response__with_qop_auth(self):
        """Include nc and cnonce in the digest when qop=auth."""
        result = RegistrationTransaction.digest_response(
            username="alice",
            password=TEST_PASSWORD,
            realm="example.com",
            nonce="nonce123",
            method="REGISTER",
            uri="example.com",
            algorithm=DigestAlgorithm.SHA_256,
            qop=DigestQoP.AUTH,
            cnonce="clientnonce",
        )
        assert isinstance(result, str)

    def test_digest_response__sess_algorithm_requires_cnonce(self):
        """Raise ValueError when a -sess algorithm is used without cnonce."""
        with pytest.raises(ValueError, match="cnonce"):
            RegistrationTransaction.digest_response(
                username="alice",
                password=TEST_PASSWORD,
                realm="example.com",
                nonce="nonce123",
                method="REGISTER",
                uri="example.com",
                algorithm=DigestAlgorithm.SHA_256_SESS,
                cnonce=None,
            )

    def test_digest_response__sess_algorithm_with_cnonce(self):
        """Compute a digest with a -sess algorithm when cnonce is provided."""
        result = RegistrationTransaction.digest_response(
            username="alice",
            password=TEST_PASSWORD,
            realm="example.com",
            nonce="nonce123",
            method="REGISTER",
            uri="example.com",
            algorithm=DigestAlgorithm.SHA_256_SESS,
            cnonce="client-cnonce",
        )
        assert isinstance(result, str)

    def test_digest_response__unsupported_algorithm_raises(self):
        """Raise ValueError for unrecognised digest algorithm."""
        with pytest.raises(ValueError, match="Unsupported"):
            RegistrationTransaction.digest_response(
                username="alice",
                password=TEST_PASSWORD,
                realm="example.com",
                nonce="nonce123",
                method="REGISTER",
                uri="example.com",
                algorithm="UNKNOWN-ALG",
            )

    def test_response_received__200_ok__calls_on_registered(self):
        """Successful registration invokes sip.on_registered()."""
        registered_calls: list[bool] = []

        from voip.sip.protocol import SessionInitiationProtocol

        class TrackingSession(SessionInitiationProtocol):
            def on_registered(self) -> None:
                registered_calls.append(True)

        import ipaddress

        from voip.rtp import RealtimeTransportProtocol
        from voip.types import NetworkAddress

        transport = FakeTransport()
        mux = RealtimeTransportProtocol()
        from voip.sip.types import SipUri

        session = TrackingSession(
            aor=SipUri.parse("sips:alice:secret@example.com"),
            rtp=mux,
            dialog_class=Dialog,
        )
        session.transport = transport
        session.local_address = NetworkAddress(ipaddress.ip_address("127.0.0.1"), 5061)
        session.is_secure = True
        tx = RegistrationTransaction(sip=session, method=SIPMethod.REGISTER)
        session.transactions[tx.branch] = tx
        response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=lt\r\n"
            f"To: sip:example.com;tag=rt\r\n"
            f"Call-ID: reg-hook@example.com\r\n"
            f"CSeq: 1 REGISTER\r\n"
            f"\r\n".encode()
        )
        tx.response_received(response)
        assert registered_calls == [True]


class TestInviteTransaction:
    def test_invite_received__delegates_to_dialog(self):
        """invite_received sets dialog.invite_tx, dialog.sip and calls dialog.call_received()."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        request = Message.parse(INVITE_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        # Base Dialog.call_received() rejects with 486.
        tx.invite_received(request)
        assert tx.dialog.invite_tx is tx
        assert tx.dialog.sip is sip
        assert any(b"486" in data for data in transport.sent)

    def test_ack_received__removes_transaction(self):
        """ack_received removes the transaction from sip.transactions."""
        sip = create_sip_session()
        request = Message.parse(INVITE_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        sip.transactions[tx.branch] = tx

        ack = Message.parse(
            b"ACK sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKabc123\r\n"
            b"From: sip:bob@biloxi.com;tag=from-tag-1\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: test-call-id@biloxi.com\r\n"
            b"CSeq: 1 ACK\r\n"
            b"\r\n"
        )
        tx.ack_received(ack)
        assert tx.branch not in sip.transactions
        assert tx.done.is_set()

    def test_bye_received__removes_dialog_and_sends_200(self):
        """bye_received removes the dialog, sends 200 OK, and calls hangup_received."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        request = Message.parse(INVITE_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        sip.dialogs[(tx.dialog.remote_tag, tx.dialog.local_tag)] = tx.dialog

        bye = Message.parse(
            b"BYE sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKbye001\r\n"
            b"From: sip:bob@biloxi.com;tag=from-tag-1\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: test-call-id@biloxi.com\r\n"
            b"CSeq: 2 BYE\r\n"
            b"\r\n"
        )
        tx.bye_received(bye)
        assert any(b"200" in data for data in transport.sent)
        assert (tx.dialog.remote_tag, tx.dialog.local_tag) not in sip.dialogs

    def test_bye_received__calls_hangup_received(self):
        """bye_received calls dialog.hangup_received() after sending 200 OK."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        request = Message.parse(INVITE_BYTES)

        hangup_calls: list[bool] = []

        @dataclasses.dataclass(kw_only=True)
        class TrackingDialog(Dialog):
            def hangup_received(self) -> None:
                hangup_calls.append(True)

        tx = InviteTransaction.from_request(request=request, sip=sip)
        # Replace dialog with TrackingDialog instance that has the same identity fields
        tracking_dialog = TrackingDialog(
            call_id=tx.dialog.call_id,
            local_tag=tx.dialog.local_tag,
            remote_tag=tx.dialog.remote_tag,
            remote_contact=tx.dialog.remote_contact,
        )
        tx.dialog = tracking_dialog
        sip.dialogs[(tracking_dialog.remote_tag, tracking_dialog.local_tag)] = (
            tracking_dialog
        )

        bye = Message.parse(
            b"BYE sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKbye002\r\n"
            b"From: sip:bob@biloxi.com;tag=from-tag-1\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: test-call-id@biloxi.com\r\n"
            b"CSeq: 2 BYE\r\n"
            b"\r\n"
        )
        tx.bye_received(bye)
        assert hangup_calls == [True]

    def test_cancel_received__removes_transaction_and_sends_200(self):
        """cancel_received removes the transaction, the dialog, and sends 200 OK."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        request = Message.parse(INVITE_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        sip.transactions[tx.branch] = tx
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
        tx.cancel_received(cancel)
        assert tx.branch not in sip.transactions
        assert any(b"200" in data for data in transport.sent)

    def test_ringing__sends_180(self):
        """Ringing sends a 180 Ringing provisional response."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        request = Message.parse(INVITE_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        tx.ringing()
        assert any(b"180" in data for data in transport.sent)

    def test_reject__sends_busy_here_by_default(self):
        """Reject sends 486 Busy Here when no status code is specified."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        request = Message.parse(INVITE_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        tx.reject()
        assert any(b"486" in data for data in transport.sent)

    def test_reject__sends_custom_status_code(self):
        """Reject sends the specified status code."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        request = Message.parse(INVITE_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        tx.reject(SIPStatus.NOT_FOUND)
        assert any(b"404" in data for data in transport.sent)

    def test_answer__without_sdp__sends_200_ok(self):
        """Answer sends 200 OK with SDP even when the INVITE has no body."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        request = Message.parse(INVITE_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        tx.answer(call_class=CallFixture)
        assert any(b"200" in data for data in transport.sent)
        assert any(b"application/sdp" in data for data in transport.sent)

    def test_answer__with_sdp__negotiates_codec(self):
        """Answer negotiates a codec from the SDP offer in the INVITE."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        request = Message.parse(INVITE_WITH_SDP_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        tx.answer(call_class=CallFixture)
        assert any(b"200" in data for data in transport.sent)

    def test_answer__stores_dialog(self):
        """Answer stores the dialog in sip.dialogs."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        request = Message.parse(INVITE_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        tx.answer(call_class=CallFixture)
        assert len(sip.dialogs) > 0

    def test_answer__dialog_has_local_and_remote_party(self):
        """Answer populates dialog.local_party and dialog.remote_party for BYE."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        request = Message.parse(INVITE_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        tx.answer(call_class=CallFixture)
        dialog = next(iter(sip.dialogs.values()))
        assert dialog.local_party is not None
        assert dialog.remote_party is not None
        assert "tag=" in dialog.local_party
        assert "tag=" in dialog.remote_party

    def test_answer__call_handler_has_dialog(self):
        """Answer passes the dialog to the call handler."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        request = Message.parse(INVITE_WITH_SDP_BYTES)
        tx = InviteTransaction.from_request(request=request, sip=sip)
        tx.answer(call_class=CallFixture)
        dialog = next(iter(sip.dialogs.values()))
        registered_handler = next(iter(rtp.calls.values()))
        assert registered_handler.dialog is dialog

    def test_answer__with_record_route(self):
        """Include Record-Route in 200 OK when present in the INVITE."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        request = Message.parse(
            b"INVITE sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKrr99\r\n"
            b"From: sip:bob@biloxi.com;tag=rr-tag\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: rr-call@biloxi.com\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"Record-Route: <sip:proxy.example.com;lr>\r\n"
            b"\r\n"
        )
        tx = InviteTransaction.from_request(request=request, sip=sip)
        tx.answer(call_class=CallFixture)
        ok_data = b"".join(transport.sent)
        assert b"Record-Route:" in ok_data

    async def test_make_call__sends_invite(self):
        """make_call sends an INVITE request and registers the transaction."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        request = await tx.make_call("sip:bob@biloxi.com", call_class=CallFixture)
        assert any(b"INVITE" in data for data in transport.sent)
        assert tx.branch in sip.transactions
        assert request.method == SIPMethod.INVITE

    async def test_make_call__sdp_offer_contains_codec(self):
        """make_call includes a non-empty SDP offer body in the INVITE."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        await tx.make_call("sip:bob@biloxi.com", call_class=CallFixture)
        sent_data = b"".join(transport.sent)
        assert b"application/sdp" in sent_data
        assert b"m=audio" in sent_data

    async def test_make_call__with_existing_dialog_reuses_it(self):
        """make_call() with a dialog parameter reuses that dialog instance."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        existing_dialog = Dialog()
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        await tx.make_call(
            "sip:bob@biloxi.com", call_class=CallFixture, dialog=existing_dialog
        )
        assert tx.dialog is existing_dialog
        assert existing_dialog.sip is sip

    def test_response_received__100_is_noop(self):
        """1xx provisional responses are silently ignored."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        sip.transactions[tx.branch] = tx
        response = Message.parse(
            f"SIP/2.0 100 Trying\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=lt\r\n"
            f"To: sip:bob@biloxi.com\r\n"
            f"Call-ID: trying-call@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"\r\n".encode()
        )
        tx.response_received(response)
        assert tx.branch in sip.transactions

    def test_response_received__4xx_removes_transaction(self):
        """4xx responses remove the transaction from the registry."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        sip.transactions[tx.branch] = tx
        response = Message.parse(
            f"SIP/2.0 486 Busy Here\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=lt\r\n"
            f"To: sip:bob@biloxi.com;tag=rt\r\n"
            f"Call-ID: busy-call@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"\r\n".encode()
        )
        tx.response_received(response)
        assert tx.branch not in sip.transactions

    async def test_accept_call__sends_ack_on_200_ok(self):
        """_accept_call sends an ACK after receiving 200 OK."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        sip.transactions[tx.branch] = tx
        await tx.make_call("sip:bob@biloxi.com", call_class=CallFixture)
        ok_response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=our-tag\r\n"
            f"To: sip:bob@biloxi.com;tag=callee-tag\r\n"
            f"Call-ID: out-call@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"Contact: <sip:bob@192.0.2.2>\r\n"
            f"\r\n".encode()
        )
        await tx._accept_call(ok_response)
        sent_data = b"".join(transport.sent)
        assert b"ACK" in sent_data

    async def test_accept_call__with_sdp_registers_rtp_handler(self):
        """_accept_call registers an RTP call handler when remote SDP is present."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        await tx.make_call("sip:bob@biloxi.com", call_class=CallFixture)
        ok_response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=our-tag\r\n"
            f"To: sip:bob@biloxi.com;tag=callee-tag\r\n"
            f"Call-ID: out-sdp@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"Content-Type: application/sdp\r\n"
            f"\r\n"
            f"v=0\r\n"
            f"o=- 1 1 IN IP4 192.0.2.2\r\n"
            f"s=-\r\n"
            f"c=IN IP4 192.0.2.2\r\n"
            f"t=0 0\r\n"
            f"m=audio 5004 RTP/AVP 0\r\n"
            f"a=rtpmap:0 PCMU/8000\r\n".encode()
        )
        await tx._accept_call(ok_response)
        assert len(rtp.calls) > 0

    async def test_accept_call__stores_dialog(self):
        """_accept_call stores the dialog in sip.dialogs after 200 OK."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        await tx.make_call("sip:bob@biloxi.com", call_class=CallFixture)
        ok_response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=our-tag\r\n"
            f"To: sip:bob@biloxi.com;tag=callee-tag\r\n"
            f"Call-ID: out-dialog@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"\r\n".encode()
        )
        await tx._accept_call(ok_response)
        assert len(sip.dialogs) > 0

    async def test_accept_call__dialog_has_bye_fields(self):
        """_accept_call populates dialog.local_party, remote_party, and outbound_cseq."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        await tx.make_call("sip:bob@biloxi.com", call_class=CallFixture)
        ok_response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=our-tag\r\n"
            f"To: sip:bob@biloxi.com;tag=callee-tag\r\n"
            f"Call-ID: bye-fields@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"Contact: <sip:bob@192.0.2.2>\r\n"
            f"\r\n".encode()
        )
        await tx._accept_call(ok_response)
        dialog = next(iter(sip.dialogs.values()))
        assert dialog.local_party == "sip:alice@example.com;tag=our-tag"
        assert dialog.remote_party == "sip:bob@biloxi.com;tag=callee-tag"
        assert dialog.outbound_cseq == 2
        assert dialog.remote_contact == "sip:bob@192.0.2.2"

    async def test_accept_call__call_handler_has_dialog(self):
        """_accept_call passes the dialog to the call handler."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        await tx.make_call("sip:bob@biloxi.com", call_class=CallFixture)
        ok_response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=our-tag\r\n"
            f"To: sip:bob@biloxi.com;tag=callee-tag\r\n"
            f"Call-ID: bye-fields@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"Contact: <sip:bob@192.0.2.2>\r\n"
            f"Content-Type: application/sdp\r\n"
            f"\r\n"
            f"v=0\r\n"
            f"o=- 1 1 IN IP4 192.0.2.2\r\n"
            f"s=-\r\n"
            f"c=IN IP4 192.0.2.2\r\n"
            f"t=0 0\r\n"
            f"m=audio 5004 RTP/AVP 0\r\n"
            f"a=rtpmap:0 PCMU/8000\r\n".encode()
        )
        await tx._accept_call(ok_response)
        dialog = next(iter(sip.dialogs.values()))
        registered_handler = next(iter(rtp.calls.values()))
        assert registered_handler.dialog is dialog

    async def test_accept_call__no_pending_call_class_sends_ack(self):
        """_accept_call sends ACK even when no pending_call_class is set."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        await tx.make_call("sip:bob@biloxi.com", call_class=CallFixture)
        tx.pending_call_class = None
        ok_response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=our-tag\r\n"
            f"To: sip:bob@biloxi.com;tag=callee-tag\r\n"
            f"Call-ID: no-class@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"\r\n".encode()
        )
        await tx._accept_call(ok_response)
        sent_data = b"".join(transport.sent)
        assert b"ACK" in sent_data

    async def test_accept_call__sdp_no_connection_uses_peer(self):
        """_accept_call falls back to transport peer address when SDP has no c= line."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        await tx.make_call("sip:bob@biloxi.com", call_class=CallFixture)
        ok_response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=our-tag\r\n"
            f"To: sip:bob@biloxi.com;tag=callee-tag\r\n"
            f"Call-ID: no-conn@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"Content-Type: application/sdp\r\n"
            f"\r\n"
            f"v=0\r\n"
            f"o=- 1 1 IN IP4 192.0.2.2\r\n"
            f"s=-\r\n"
            f"t=0 0\r\n"
            f"m=audio 5004 RTP/AVP 0\r\n"
            f"a=rtpmap:0 PCMU/8000\r\n".encode()
        )
        await tx._accept_call(ok_response)
        assert len(rtp.calls) > 0

    async def test_accept_call__sdp_zero_port_no_rtp_address(self):
        """_accept_call registers call with None address when audio port is 0."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        await tx.make_call("sip:bob@biloxi.com", call_class=CallFixture)
        ok_response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=our-tag\r\n"
            f"To: sip:bob@biloxi.com;tag=callee-tag\r\n"
            f"Call-ID: zero-port@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"Content-Type: application/sdp\r\n"
            f"\r\n"
            f"v=0\r\n"
            f"o=- 1 1 IN IP4 192.0.2.2\r\n"
            f"s=-\r\n"
            f"c=IN IP4 192.0.2.2\r\n"
            f"t=0 0\r\n"
            f"m=audio 0 RTP/AVP 0\r\n".encode()
        )
        await tx._accept_call(ok_response)
        assert None in rtp.calls

    def test_answer__sdp_without_connection_uses_peer_address(self):
        """Use the transport peer address when SDP has no c= connection line."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        # INVITE SDP with audio port > 0 but no c= connection line
        request = Message.parse(
            b"INVITE sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKnoconn\r\n"
            b"From: sip:bob@biloxi.com;tag=noconn-tag\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: noconn-call@biloxi.com\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"Content-Type: application/sdp\r\n"
            b"\r\n"
            b"v=0\r\n"
            b"o=- 1 1 IN IP4 192.0.2.1\r\n"
            b"s=-\r\n"
            b"t=0 0\r\n"
            b"m=audio 5004 RTP/AVP 0\r\n"
            b"a=rtpmap:0 PCMU/8000\r\n"
        )
        tx = InviteTransaction.from_request(request=request, sip=sip)
        tx.answer(call_class=CallFixture)
        assert any(b"200" in data for data in transport.sent)

    def test_answer__sdp_with_zero_port_uses_none_rtp_address(self):
        """Use None for RTP address when audio port is 0 in SDP offer."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        # INVITE SDP with port=0 (rejected audio)
        request = Message.parse(
            b"INVITE sip:alice@example.com SIP/2.0\r\n"
            b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=z9hG4bKzeroport\r\n"
            b"From: sip:bob@biloxi.com;tag=zero-tag\r\n"
            b"To: sip:alice@example.com\r\n"
            b"Call-ID: zero-call@biloxi.com\r\n"
            b"CSeq: 1 INVITE\r\n"
            b"Content-Type: application/sdp\r\n"
            b"\r\n"
            b"v=0\r\n"
            b"o=- 1 1 IN IP4 192.0.2.1\r\n"
            b"s=-\r\n"
            b"c=IN IP4 192.0.2.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 0 RTP/AVP 0\r\n"
        )
        tx = InviteTransaction.from_request(request=request, sip=sip)
        tx.answer(call_class=CallFixture)
        assert any(b"200" in data for data in transport.sent)

    async def test_accept_call__record_route_adds_route_header(self):
        """_accept_call includes a Route header in the ACK when Record-Route is present."""
        import ipaddress

        from voip.types import NetworkAddress

        transport = FakeTransport()
        rtp = RealtimeTransportProtocol()
        rtp.public_address = NetworkAddress(ipaddress.ip_address("192.0.2.1"), 5004)
        sip = create_sip_session(fake_transport=transport, rtp=rtp)
        tx = InviteTransaction(sip=sip, method=SIPMethod.INVITE, cseq=1)
        await tx.make_call("sip:bob@biloxi.com", call_class=CallFixture)
        ok_response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS example.com;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=our-tag\r\n"
            f"To: sip:bob@biloxi.com;tag=callee-tag\r\n"
            f"Call-ID: rr-call@example.com\r\n"
            f"CSeq: 1 INVITE\r\n"
            f"Contact: <sip:bob@192.0.2.2>\r\n"
            f"Record-Route: <sip:proxy.example.com;lr>\r\n"
            f"\r\n".encode()
        )
        await tx._accept_call(ok_response)
        sent_data = b"".join(transport.sent)
        assert b"ACK" in sent_data
        assert b"Route" in sent_data


class TestByeTransaction:
    def test_bye_transaction__has_default_cseq(self):
        """ByeTransaction.cseq defaults to 1."""
        sip = create_sip_session()
        tx = ByeTransaction(sip=sip, method=SIPMethod.BYE)
        assert tx.cseq == 1

    def test_response_received__removes_transaction_on_200(self):
        """response_received removes the transaction when 200 OK is received."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        tx = ByeTransaction(sip=sip, method=SIPMethod.BYE, cseq=2)
        sip.transactions[tx.branch] = tx
        response = Message.parse(
            f"SIP/2.0 200 OK\r\n"
            f"Via: SIP/2.0/TLS 127.0.0.1:5061;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=our-tag\r\n"
            f"To: sip:bob@biloxi.com;tag=callee-tag\r\n"
            f"Call-ID: bye-call@example.com\r\n"
            f"CSeq: 2 BYE\r\n"
            f"\r\n".encode()
        )
        tx.response_received(response)
        assert tx.branch not in sip.transactions
        assert tx.done.is_set()

    def test_response_received__ignores_provisional_response(self):
        """response_received leaves the transaction in place for 1xx responses."""
        transport = FakeTransport()
        sip = create_sip_session(fake_transport=transport)
        tx = ByeTransaction(sip=sip, method=SIPMethod.BYE, cseq=2)
        sip.transactions[tx.branch] = tx
        response = Message.parse(
            f"SIP/2.0 100 Trying\r\n"
            f"Via: SIP/2.0/TLS 127.0.0.1:5061;branch={tx.branch}\r\n"
            f"From: sip:alice@example.com;tag=our-tag\r\n"
            f"To: sip:bob@biloxi.com\r\n"
            f"Call-ID: bye-call@example.com\r\n"
            f"CSeq: 2 BYE\r\n"
            f"\r\n".encode()
        )
        tx.response_received(response)
        assert tx.branch in sip.transactions
        assert not tx.done.is_set()


class TestRegistrationError:
    def test_is_exception(self):
        """RegistrationError is a subclass of Exception."""
        assert issubclass(RegistrationError, Exception)

    def test_raise(self):
        """RegistrationError can be raised and caught."""
        with pytest.raises(RegistrationError, match="403 Forbidden"):
            raise RegistrationError("403 Forbidden")

    def test___str__(self):
        """RegistrationError stores the message string."""
        err = RegistrationError("500 Server Error")
        assert str(err) == "500 Server Error"
