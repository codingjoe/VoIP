"""Secure Real-time Transport Protocol (SRTP) implementation of [RFC 3711].

Provides symmetric key encryption and authentication for RTP media streams
using the AES_CM_128_HMAC_SHA1_80 cipher suite.  Keys are negotiated via SDP
Security Descriptions (SDES, [RFC 4568]) carried in the SIP 200 OK SDP body,
which is itself protected by TLS.

Requires the `cryptography` package (included in any installation).

[RFC 3711]: https://datatracker.ietf.org/doc/html/rfc3711
[RFC 4568]: https://datatracker.ietf.org/doc/html/rfc456
"""

from __future__ import annotations

import base64
import dataclasses
import hmac as _hmac_stdlib
import os
import struct

from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

__all__ = ["CIPHER_SUITE", "SRTPSession"]

#: SRTP cipher suite identifier used in SDP `a=crypto:` attributes.
CIPHER_SUITE = "AES_CM_128_HMAC_SHA1_80"

#: AES key size in bytes (128-bit).
_KEY_SIZE = 16
#: Master salt size in bytes (112-bit).
_SALT_SIZE = 14
#: HMAC-SHA1-80 authentication tag size in bytes.
_AUTH_TAG_SIZE = 10


def _prf(master_key: bytes, label: int, master_salt: bytes, length: int) -> bytes:
    """Compute the SRTP pseudo-random function (RFC 3711 §4.3.1).

    Uses AES Counter Mode with IV derived from the label and master salt.

    Args:
        master_key: 16-byte AES master key.
        label: PRF label (0x00=cipher, 0x01=auth, 0x02=salt).
        master_salt: 14-byte master salt.
        length: Number of output bytes required.

    Returns:
        `length` bytes of pseudo-random keying material.
    """
    # x = (label * 2^48) XOR master_salt  (both treated as 112-bit integers)
    x = ((label << 48) ^ int.from_bytes(master_salt, "big")) & ((1 << 112) - 1)
    # Pad x to 128 bits (AES block size): x occupies bits 127–16, bits 15–0 = 0
    iv = x.to_bytes(14, "big") + b"\x00\x00"
    # AES-CM: encrypt all-zeros to obtain keystream
    cipher = Cipher(algorithms.AES(master_key), modes.CTR(iv))
    enc = cipher.encryptor()
    return enc.update(b"\x00" * length) + enc.finalize()


@dataclasses.dataclass(slots=True)
class SRTPSession:
    """SRTP session for one call leg using AES_CM_128_HMAC_SHA1_80.

    Handles symmetric encryption and authentication of RTP packets.
    Key material is derived from `master_key` and `master_salt` via the
    SRTP pseudo-random function (RFC 3711 §4.3.1).

    Create a fresh session for each answered call via `generate` and pass
    it to the `RTPCall` instance.  The SDP `a=crypto:` attribute is produced
    by `sdes_attribute`.

    Attributes:
        master_key: 16-byte AES master key (randomly generated).
        master_salt: 14-byte master salt (randomly generated).
    """

    master_key: bytes
    master_salt: bytes
    _session_key: bytes = dataclasses.field(init=False)
    _session_auth_key: bytes = dataclasses.field(init=False)
    _session_salt: bytes = dataclasses.field(init=False)
    #: Rollover counter and highest sent sequence number for encryption.
    _send_roc: int = dataclasses.field(init=False, default=0)
    _last_send_seq: int = dataclasses.field(init=False, default=-1)
    #: Rollover counter and highest received sequence number for decryption.
    _recv_roc: int = dataclasses.field(init=False, default=0)
    _last_recv_seq: int = dataclasses.field(init=False, default=-1)

    def __post_init__(self) -> None:
        self._session_key = _prf(self.master_key, 0x00, self.master_salt, _KEY_SIZE)
        self._session_auth_key = _prf(self.master_key, 0x01, self.master_salt, 20)
        self._session_salt = _prf(self.master_key, 0x02, self.master_salt, _SALT_SIZE)

    @classmethod
    def generate(cls) -> SRTPSession:
        """Generate a new SRTP session with a cryptographically random key and salt."""
        return cls(
            master_key=os.urandom(_KEY_SIZE),
            master_salt=os.urandom(_SALT_SIZE),
        )

    @property
    def sdes_attribute(self) -> str:
        """SDP `a=crypto:` attribute value for SDES key exchange (RFC 4568).

        The key material (master key followed by master salt) is base64-encoded
        and wrapped in the standard SDES inline format.  Include this value in
        the SDP `a=crypto:` attribute of the answered media description:

        ```
        a=crypto:1 AES_CM_128_HMAC_SHA1_80 inline:<value>
        ```
        """
        key_salt = base64.b64encode(self.master_key + self.master_salt).decode()
        return f"1 {CIPHER_SUITE} inline:{key_salt}"

    def _compute_iv(self, ssrc: int, index: int) -> bytes:
        """Compute the 128-bit AES-CM IV for a given SSRC and packet index.

        IV = (session_salt * 2^16) XOR (SSRC * 2^64) XOR (index * 2^16)
        per RFC 3711 §4.1.1.
        """
        iv_int = (
            (int.from_bytes(self._session_salt, "big") << 16)
            ^ (ssrc << 64)
            ^ (index << 16)
        )
        return iv_int.to_bytes(16, "big")

    def _auth_tag(self, packet_no_tag: bytes, roc: int) -> bytes:
        """Compute the 10-byte HMAC-SHA1 authentication tag (RFC 3711 §4.2)."""
        roc_bytes = struct.pack(">I", roc)
        mac = hmac.HMAC(self._session_auth_key, hashes.SHA1())  # noqa: S303
        mac.update(packet_no_tag + roc_bytes)
        return mac.finalize()[:_AUTH_TAG_SIZE]

    def _estimate_recv_index(self, seq: int) -> tuple[int, int]:
        """Estimate the packet index and new ROC for a received sequence number.

        Implements the index estimation algorithm from RFC 3711 §3.3.1.

        Args:
            seq: The 16-bit sequence number from the received RTP header.

        Returns:
            A `(index, roc_guess)` tuple where `index` is the estimated
            48-bit packet index and `roc_guess` is the ROC value used.
        """
        s_l = self._last_recv_seq
        roc = self._recv_roc
        if s_l < 0:
            # No packets received yet; use the current ROC.
            return (roc << 16) | seq, roc
        if s_l < 0x8000:  # s_l < 2^15
            if seq - s_l > 0x8000:
                roc_guess = (roc - 1) % (1 << 32)
            else:
                roc_guess = roc
        else:  # s_l >= 2^15
            if s_l - seq > 0x8000:
                roc_guess = (roc + 1) % (1 << 32)
            else:
                roc_guess = roc
        return (roc_guess << 16) | seq, roc_guess

    def encrypt(self, packet: bytes) -> bytes:
        """Encrypt an RTP packet to produce an SRTP packet.

        Encrypts the RTP payload with AES-CM and appends an 80-bit
        HMAC-SHA1 authentication tag.  Tracks sequence number rollover per
        RFC 3711 §3.3.1 to ensure the packet index remains unique.

        Args:
            packet: Raw RTP packet bytes (at least 12 bytes).

        Returns:
            SRTP packet with encrypted payload and appended auth tag.
        """
        if len(packet) < 12:
            return packet
        header = packet[:12]
        payload = packet[12:]
        ssrc = struct.unpack(">I", header[8:12])[0]
        seq = struct.unpack(">H", header[2:4])[0]

        # Detect rollover: the sequence number wrapped from ~65535 back to ~0.
        # For the send side, sequence numbers always increase monotonically so
        # any decrease indicates a rollover.
        if self._last_send_seq >= 0 and seq < self._last_send_seq:
            self._send_roc = (self._send_roc + 1) % (1 << 32)
        self._last_send_seq = seq

        index = (self._send_roc << 16) | seq
        iv = self._compute_iv(ssrc, index)
        cipher = Cipher(algorithms.AES(self._session_key), modes.CTR(iv))
        enc = cipher.encryptor()
        encrypted_payload = enc.update(payload) + enc.finalize()

        srtp_no_tag = header + encrypted_payload
        return srtp_no_tag + self._auth_tag(srtp_no_tag, self._send_roc)

    def decrypt(self, packet: bytes) -> bytes | None:
        """Decrypt and authenticate an SRTP packet.

        Verifies the HMAC-SHA1-80 authentication tag and, if valid, decrypts
        the payload with AES-CM.  The packet index is estimated per
        RFC 3711 §3.3.1, tracking rollovers across the 16-bit sequence space.

        Args:
            packet: Raw SRTP packet bytes (at least 12 + 10 bytes).

        Returns:
            Decrypted RTP packet bytes, or `None` when authentication fails
            or the packet is too short.
        """
        if len(packet) < 12 + _AUTH_TAG_SIZE:
            return None
        srtp_no_tag = packet[:-_AUTH_TAG_SIZE]
        received_tag = packet[-_AUTH_TAG_SIZE:]

        header = packet[:12]
        ssrc = struct.unpack(">I", header[8:12])[0]
        seq = struct.unpack(">H", header[2:4])[0]

        index, roc_guess = self._estimate_recv_index(seq)

        expected_tag = self._auth_tag(srtp_no_tag, roc_guess)
        if not _hmac_stdlib.compare_digest(received_tag, expected_tag):
            return None

        # Authentication passed — update the highest received sequence number
        # and ROC per RFC 3711 §3.3.1.
        if roc_guess == self._recv_roc:
            if self._last_recv_seq < 0 or seq > self._last_recv_seq:
                self._last_recv_seq = seq
        elif roc_guess == (self._recv_roc + 1) % (1 << 32):
            self._recv_roc = roc_guess
            self._last_recv_seq = seq

        encrypted_payload = srtp_no_tag[12:]
        iv = self._compute_iv(ssrc, index)
        cipher = Cipher(algorithms.AES(self._session_key), modes.CTR(iv))
        dec = cipher.decryptor()
        payload = dec.update(encrypted_payload) + dec.finalize()
        return header + payload
