"""Tests for the SRTP SDES key parser (voip.srtp, RFC 4568)."""

import base64

import pytest
from voip.srtp import CIPHER_SUITE, SRTPSession


class TestFromSdes:
    """`SRTPSession.from_sdes` parses an `a=crypto:` value into a session."""

    def test_round_trips_generated_attribute(self):
        """Parsing our own `sdes_attribute` reproduces the master key and salt."""
        session = SRTPSession.generate()
        parsed = SRTPSession.from_sdes(session.sdes_attribute)
        assert parsed.master_key == session.master_key
        assert parsed.master_salt == session.master_salt

    def test_decrypts_media_encrypted_with_parsed_session(self):
        """A session built from a remote `a=crypto:` decrypts that remote's SRTP."""
        from voip.rtp import RTPPacket  # noqa: PLC0415

        remote = SRTPSession.generate()
        recv = SRTPSession.from_sdes(remote.sdes_attribute)
        packet = RTPPacket(
            payload_type=0, sequence_number=7, timestamp=160, ssrc=99, payload=b"hello"
        )
        assert recv.decrypt(remote.encrypt(bytes(packet))) is not None

    def test_parses_tag_suite_and_inline_key(self):
        """The tag and cipher suite are accepted; the inline key is base64."""
        session = SRTPSession.generate()
        key_salt = base64.b64encode(session.master_key + session.master_salt).decode()
        value = f"7 {CIPHER_SUITE} inline:{key_salt}"
        parsed = SRTPSession.from_sdes(value)
        assert parsed.master_key == session.master_key
        assert parsed.master_salt == session.master_salt

    def test_ignores_lifetime_and_mki_trailers(self):
        """`|lifetime` and `~MKI` suffixes after the inline key are ignored."""
        session = SRTPSession.generate()
        key_salt = base64.b64encode(session.master_key + session.master_salt).decode()
        value = f"1 {CIPHER_SUITE} inline:{key_salt}|2^32~12345"
        parsed = SRTPSession.from_sdes(value)
        assert parsed.master_key == session.master_key
        assert parsed.master_salt == session.master_salt

    def test_from_sdes__uses_first_of_multiple_inline_keys(self):
        """RFC 4568 §5.1.1: several inline key-params joined by ';' may be present."""
        session = SRTPSession.generate()
        backup = SRTPSession.generate()
        active_key_salt = base64.b64encode(
            session.master_key + session.master_salt
        ).decode()
        backup_key_salt = base64.b64encode(
            backup.master_key + backup.master_salt
        ).decode()
        value = f"1 {CIPHER_SUITE} inline:{active_key_salt};inline:{backup_key_salt}"
        parsed = SRTPSession.from_sdes(value)
        assert parsed.master_key == session.master_key
        assert parsed.master_salt == session.master_salt

    def test_rejects_unsupported_cipher_suite(self):
        """An unknown cipher suite raises `ValueError` (only one suite is implemented)."""
        session = SRTPSession.generate()
        key_salt = base64.b64encode(session.master_key + session.master_salt).decode()
        value = f"1 AES_CM_128_HMAC_SHA1_32 inline:{key_salt}"
        with pytest.raises(ValueError, match="Unsupported SRTP cipher suite"):
            SRTPSession.from_sdes(value)

    def test_rejects_malformed_value(self):
        """A value missing the `inline:` parameter is rejected."""
        with pytest.raises(ValueError, match="Malformed SDES"):
            SRTPSession.from_sdes("1 AES_CM_128_HMAC_SHA1_80 something:else")

    def test_rejects_short_inline_key(self):
        """An inline key shorter than key+salt is rejected."""
        short = base64.b64encode(b"\x00" * 10).decode()
        value = f"1 {CIPHER_SUITE} inline:{short}"
        with pytest.raises(ValueError, match="too short"):
            SRTPSession.from_sdes(value)
