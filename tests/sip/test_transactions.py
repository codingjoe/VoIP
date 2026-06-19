"""Tests for SIP transaction authentication (RFC 3261 §22, RFC 8760)."""

import asyncio

from voip.sip import messages
from voip.sip.dialog import Dialog
from voip.sip.protocol import SessionInitiationProtocol
from voip.sip.transactions import DigestAuthMixin, InviteTransaction
from voip.sip.types import SIPMethod, SIPStatus, SipURI

from .conftest import CallFixture


def _last_request(sip: SessionInitiationProtocol) -> messages.Request:
    """Return the most recently sent SIP request captured by the fake transport."""
    return messages.Message.parse(sip.transport.sent[-1])  # type: ignore[arg-type]


def _make_challenge_response(
    request: messages.Request, *, status_code: SIPStatus, authenticate: str
) -> messages.Response:
    """Build a 401/407 response echoing *request*'s dialog headers."""
    return messages.Message.parse(  # type: ignore[return-value]
        (
            f"SIP/2.0 {int(status_code)} {status_code.phrase}\r\n"
            f"Via: {request.headers.getlist('Via')[0]}\r\n"
            f"From: {request.headers['From']}\r\n"
            f"To: {request.headers['To']}\r\n"
            f"Call-ID: {request.headers['Call-ID']}\r\n"
            f"CSeq: {request.headers['CSeq']}\r\n"
            f"{'Proxy-Authenticate' if status_code == SIPStatus.PROXY_AUTHENTICATION_REQUIRED else 'WWW-Authenticate'}: {authenticate}\r\n"
            f"Content-Length: 0\r\n"
            "\r\n"
        ).encode()
    )


_OK_SDP = (
    b"v=0\r\n"
    b"o=- 1 1 IN IP4 192.0.2.1\r\n"
    b"s=-\r\n"
    b"c=IN IP4 192.0.2.1\r\n"
    b"t=0 0\r\n"
    b"m=audio 5004 RTP/AVP 0\r\n"
    b"a=rtpmap:0 PCMU/8000\r\n"
)


def _make_ok_response(request: messages.Request) -> messages.Response:
    """Build a 200 OK with an SDP answer echoing *request*'s dialog headers."""
    return messages.Message.parse(  # type: ignore[return-value]
        (
            "SIP/2.0 200 OK\r\n"
            f"Via: {request.headers.getlist('Via')[0]}\r\n"
            f"From: {request.headers['From']}\r\n"
            f"To: {request.headers['To']};tag=remote-tag-1\r\n"
            f"Call-ID: {request.headers['Call-ID']}\r\n"
            f"CSeq: {request.headers['CSeq']}\r\n"
            "Contact: <sip:bob@192.0.2.1:5060>\r\n"
            "Content-Type: application/sdp\r\n"
            "\r\n"
        ).encode()
        + _OK_SDP
    )


async def _complete_invite(sip: SessionInitiationProtocol, send_task: asyncio.Task):
    """Answer the retried INVITE with 200 OK so the outbound call resolves."""
    retry = _last_request(sip)
    sip.response_received(_make_ok_response(retry))
    await asyncio.wait_for(send_task, timeout=1)


def _last_ack(sip: SessionInitiationProtocol) -> messages.Request:
    """Return the most recently sent ACK."""
    for raw in reversed(sip.transport.sent):  # type: ignore[attr-defined]
        request = messages.Message.parse(raw)
        if isinstance(request, messages.Request) and request.method == SIPMethod.ACK:
            return request
    raise AssertionError("no ACK was sent")


class TestInviteAck:
    """The ACK terminating an outbound INVITE must be well-formed."""

    async def test_ack_for_non_2xx_reuses_invite_via_and_mirrors_route(self, sip):
        """A non-2xx final ACK reuses the INVITE Via/branch and mirrors its Route.

        Per RFC 3261 §17.1.1.3 the transactional ACK must mirror the INVITE's
        Route header values (not the response's Record-Route) so the proxy
        holding the INVITE server transaction matches it by branch and absorbs
        it, rather than loose-routing it onward and retransmitting.
        """
        target = SipURI.parse("sip:+15551234567@example.com:5060")
        dialog = Dialog(uac=sip.aor, sip=sip)
        send_task = asyncio.create_task(
            InviteTransaction.send(
                sip=sip, target=target, dialog=dialog, session_class=CallFixture
            )
        )
        await asyncio.sleep(0)
        invite = _last_request(sip)
        invite_via = invite.headers.getlist("Via")[0]

        decline = messages.Message.parse(
            (
                "SIP/2.0 603 Decline\r\n"
                f"Via: {invite_via}\r\n"
                f"From: {invite.headers['From']}\r\n"
                f"To: {invite.headers['To']};tag=decline-tag\r\n"
                f"Call-ID: {invite.headers['Call-ID']}\r\n"
                f"CSeq: {invite.headers['CSeq']}\r\n"
                "Record-Route: <sip:proxy-a.example;lr>\r\n"
                "Record-Route: <sip:proxy-b.example;lr>\r\n"
                "Content-Length: 0\r\n"
                "\r\n"
            ).encode()
        )
        sip.response_received(decline)

        ack = _last_ack(sip)
        # Reuses the INVITE's Via (same branch + rport); no non-standard `alias`.
        assert ack.headers.getlist("Via")[0] == invite_via
        assert "alias" not in ack.headers.getlist("Via")[0]
        # The INVITE carried no Route, so the transactional ACK carries none.
        # Record-Route in the response must NOT leak into the ACK here.
        assert invite.headers.getlist("Route") == []
        assert ack.headers.getlist("Route") == []
        # Non-2xx ACK mirrors the INVITE Request-URI, not a Contact.
        assert str(ack.uri) == str(target)
        # The original transaction completes without establishing a dialog.
        await asyncio.wait_for(send_task, timeout=1)

    async def test_ack_for_2xx_carries_dialog_route_set(self, sip):
        """A 2xx ACK opens a fresh transaction and follows the dialog route set.

        On 2xx the route set is the reversed Record-Route (RFC 3261 §12.1.2)
        and the Request-URI is the Contact.
        """
        target = SipURI.parse("sip:+15551234567@example.com:5060")
        dialog = Dialog(uac=sip.aor, sip=sip)
        send_task = asyncio.create_task(
            InviteTransaction.send(
                sip=sip, target=target, dialog=dialog, session_class=CallFixture
            )
        )
        await asyncio.sleep(0)
        invite = _last_request(sip)
        invite_via = invite.headers.getlist("Via")[0]

        ok = messages.Message.parse(
            (
                "SIP/2.0 200 OK\r\n"
                f"Via: {invite_via}\r\n"
                f"From: {invite.headers['From']}\r\n"
                f"To: {invite.headers['To']};tag=ok-tag\r\n"
                f"Call-ID: {invite.headers['Call-ID']}\r\n"
                f"CSeq: {invite.headers['CSeq']}\r\n"
                "Record-Route: <sip:proxy-a.example;lr>\r\n"
                "Record-Route: <sip:proxy-b.example;lr>\r\n"
                "Contact: <sip:bob@192.0.2.1:5060>\r\n"
                "Content-Type: application/sdp\r\n"
                "\r\n"
            ).encode()
            + _OK_SDP
        )
        sip.response_received(ok)

        ack = _last_ack(sip)
        # 2xx ACK uses a fresh Via branch (different from the INVITE's).
        assert ack.headers.getlist("Via")[0] != invite_via
        # Dialog route set is the reversed Record-Route (first hop first).
        assert ack.headers.getlist("Route") == [
            "<sip:proxy-b.example;lr>",
            "<sip:proxy-a.example;lr>",
        ]
        # 2xx ACK targets the Contact, not the INVITE Request-URI.
        assert str(ack.uri) == "sip:bob@192.0.2.1:5060"
        await asyncio.wait_for(send_task, timeout=1)


class TestInviteAuth:
    """Outbound INVITE must answer 401/407 challenges with credentials."""

    async def test_invite_retries_with_authorization_on_401(self, sip):
        """A 401 challenge triggers a retried INVITE carrying a digest Authorization."""
        target = SipURI.parse("sip:+15551234567@example.com:5060")
        dialog = Dialog(uac=sip.aor, sip=sip)
        send_task = asyncio.create_task(
            InviteTransaction.send(
                sip=sip, target=target, dialog=dialog, session_class=CallFixture
            )
        )
        await asyncio.sleep(0)  # let the INVITE be sent

        invite = _last_request(sip)
        assert invite.method == SIPMethod.INVITE

        challenge = 'Digest realm="example.com", nonce="abc123", algorithm=MD5'
        sip.response_received(
            _make_challenge_response(
                invite, status_code=SIPStatus.UNAUTHORIZED, authenticate=challenge
            )
        )

        # The retry INVITE is the last sent request (after the ACK to the 401).
        retry = _last_request(sip)
        assert retry.method == SIPMethod.INVITE
        assert retry.headers["CSeq"] == "2 INVITE"
        authorization = str(retry.headers["Authorization"])
        assert authorization.startswith("Digest ")
        assert 'algorithm="MD5"' in authorization
        # SipURI canonicalises the `+` in the user part to `%2B`.
        digest_uri = str(target)
        assert f'uri="{digest_uri}"' in authorization

        # The digest response matches the spec computation for the challenge.
        expected = DigestAuthMixin.digest_response(
            username=sip.aor.user,
            password=sip.aor.password,
            realm="example.com",
            nonce="abc123",
            method=SIPMethod.INVITE,
            uri=digest_uri,
            algorithm="MD5",
        )
        assert f'response="{expected}"' in authorization

        # Original transaction is still pending; its result is chained to the retry.
        assert not send_task.done()
        await _complete_invite(sip, send_task)

    async def test_invite_retries_with_proxy_authorization_on_407(self, sip):
        """A 407 challenge yields a Proxy-Authorization header on the retry."""
        target = SipURI.parse("sip:+15551234567@example.com:5060")
        dialog = Dialog(uac=sip.aor, sip=sip)
        send_task = asyncio.create_task(
            InviteTransaction.send(
                sip=sip, target=target, dialog=dialog, session_class=CallFixture
            )
        )
        await asyncio.sleep(0)

        invite = _last_request(sip)
        challenge = 'Digest realm="example.com", nonce="xyz789", algorithm=SHA-256'
        sip.response_received(
            _make_challenge_response(
                invite,
                status_code=SIPStatus.PROXY_AUTHENTICATION_REQUIRED,
                authenticate=challenge,
            )
        )

        retry = _last_request(sip)
        assert "Proxy-Authorization" in retry.headers
        assert "Authorization" not in retry.headers
        assert 'algorithm="SHA-256"' in str(retry.headers["Proxy-Authorization"])

        await _complete_invite(sip, send_task)


class TestRegistrationAuth:
    """Registration digest behaviour preserved after the mixin extraction."""

    async def test_register_retries_with_authorization_on_401(self, sip):
        """A 401 to REGISTER triggers a credentialed retry with an incremented CSeq."""
        # The initial REGISTER was sent by connection_made().
        register = _last_request(sip)
        assert register.method == SIPMethod.REGISTER

        challenge = 'Digest realm="example.com", nonce="reg-nonce", algorithm=MD5'
        sip.response_received(
            _make_challenge_response(
                register, status_code=SIPStatus.UNAUTHORIZED, authenticate=challenge
            )
        )

        retry = _last_request(sip)
        assert retry.method == SIPMethod.REGISTER
        assert retry.headers["CSeq"] == "2 REGISTER"
        assert str(retry.headers["Authorization"]).startswith("Digest ")

        # Digest URI for REGISTER remains the registrar host (unchanged behaviour).
        expected = DigestAuthMixin.digest_response(
            username=sip.aor.user,
            password=sip.aor.password,
            realm="example.com",
            nonce="reg-nonce",
            method=SIPMethod.REGISTER,
            uri=str(sip.aor.host),
            algorithm="MD5",
        )
        assert f'response="{expected}"' in str(retry.headers["Authorization"])
        assert f'uri="{sip.aor.host}"' in str(retry.headers["Authorization"])
