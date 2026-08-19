"""Tests for the SIP protocol's background task handling."""

import asyncio

import pytest
from voip.sip import messages

#: An INVITE request whose Via branch violates the ``z9hG4bK`` prefix, so
#: transaction creation fails with a ``ValueError``.
INVITE_INVALID_BRANCH_BYTES = (
    b"INVITE sip:alice@example.com SIP/2.0\r\n"
    b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=invalid\r\n"
    b"From: sip:bob@biloxi.com;tag=from-tag-1\r\n"
    b"To: sip:alice@example.com\r\n"
    b"Call-ID: test-call-id@biloxi.com\r\n"
    b"CSeq: 1 INVITE\r\n"
    b"\r\n"
)

#: A BYE request whose Via branch violates the ``z9hG4bK`` prefix, so
#: transaction creation fails with a ``ValueError``.
BYE_INVALID_BRANCH_BYTES = (
    b"BYE sip:alice@example.com SIP/2.0\r\n"
    b"Via: SIP/2.0/TLS 192.0.2.1:5061;branch=invalid\r\n"
    b"From: sip:bob@biloxi.com;tag=from-tag-1\r\n"
    b"To: sip:alice@example.com;tag=to-tag-1\r\n"
    b"Call-ID: test-call-id@biloxi.com\r\n"
    b"CSeq: 2 BYE\r\n"
    b"\r\n"
)

#: Message logged by ``log_task_failure`` when a background task raises.
TASK_FAILURE_MESSAGE = "Background task failed"

#: Value error raised when a transaction branch violates the prefix.
INVALID_BRANCH_MESSAGE = "Branch parameter must start with 'z9hG4bK'"


class TestSessionInitiationProtocol:
    async def test_dispatch_frame__raise_invite_value_error(self, sip):
        """Raise the error when the inbound INVITE transaction fails."""
        with pytest.raises(ValueError):
            await sip.dispatch_frame(INVITE_INVALID_BRANCH_BYTES)

    async def test_dispatch_frame__raise_bye_value_error(self, sip):
        """Raise the error when the inbound BYE transaction fails."""
        with pytest.raises(ValueError):
            await sip.dispatch_frame(BYE_INVALID_BRANCH_BYTES)

    async def test_request_received__raise_invite_value_error(self, sip):
        """Raise the error when the inbound INVITE transaction fails."""
        with pytest.raises(ValueError):
            await sip.request_received(
                messages.Message.parse(INVITE_INVALID_BRANCH_BYTES)
            )

    async def test_request_received__raise_bye_value_error(self, sip):
        """Raise the error when the inbound BYE transaction fails."""
        with pytest.raises(ValueError):
            await sip.request_received(messages.Message.parse(BYE_INVALID_BRANCH_BYTES))

    async def test_log_task_failure__success(self, sip, caplog):
        """Do not log a successful background task."""
        task = asyncio.create_task(asyncio.sleep(0))
        task.add_done_callback(sip.log_task_failure)
        await asyncio.gather(task, return_exceptions=True)
        assert TASK_FAILURE_MESSAGE not in caplog.text

    async def test_log_task_failure__cancellation(self, sip, caplog):
        """Do not log a cancelled background task."""
        task = asyncio.create_task(asyncio.Event().wait())
        task.add_done_callback(sip.log_task_failure)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert TASK_FAILURE_MESSAGE not in caplog.text

    async def test_log_task_failure__raise_value_error(self, sip, caplog):
        """Log the exception when the background task fails."""
        task = asyncio.create_task(sip.dispatch_frame(INVITE_INVALID_BRANCH_BYTES))
        task.add_done_callback(sip.log_task_failure)
        await asyncio.gather(task, return_exceptions=True)
        assert TASK_FAILURE_MESSAGE in caplog.text
        assert INVALID_BRANCH_MESSAGE in caplog.text

    async def test_data_received__logs_invite_failure(self, sip, caplog):
        """Log the exception when the inbound INVITE transaction fails."""
        before = set(asyncio.all_tasks())
        sip.data_received(INVITE_INVALID_BRANCH_BYTES)
        await asyncio.gather(*(asyncio.all_tasks() - before), return_exceptions=True)
        assert INVALID_BRANCH_MESSAGE in caplog.text

    async def test_data_received__logs_bye_failure(self, sip, caplog):
        """Log the exception when the inbound BYE transaction fails."""
        before = set(asyncio.all_tasks())
        sip.data_received(BYE_INVALID_BRANCH_BYTES)
        await asyncio.gather(*(asyncio.all_tasks() - before), return_exceptions=True)
        assert INVALID_BRANCH_MESSAGE in caplog.text

    async def test_datagram_received__logs_invite_failure(self, sip, caplog):
        """Log the exception when the inbound INVITE transaction fails."""
        before = set(asyncio.all_tasks())
        sip.datagram_received(INVITE_INVALID_BRANCH_BYTES, ("192.0.2.1", 5061))
        await asyncio.gather(*(asyncio.all_tasks() - before), return_exceptions=True)
        assert INVALID_BRANCH_MESSAGE in caplog.text

    async def test_datagram_received__logs_bye_failure(self, sip, caplog):
        """Log the exception when the inbound BYE transaction fails."""
        before = set(asyncio.all_tasks())
        sip.datagram_received(BYE_INVALID_BRANCH_BYTES, ("192.0.2.1", 5061))
        await asyncio.gather(*(asyncio.all_tasks() - before), return_exceptions=True)
        assert INVALID_BRANCH_MESSAGE in caplog.text
