"""End-to-end-encrypted SIP messaging (RFC 3428).

Provides X25519 Diffie-Hellman key agreement with HKDF-SHA-256 key
derivation and AES-256-GCM authenticated encryption for SIP MESSAGE
bodies.

Wire format for an encrypted body::

    eph_pub (32 B) | nonce (12 B) | ciphertext+tag (N+16 B)

The resulting bytes are sent as the raw SIP MESSAGE body with
``Content-Type: application/x-voip-encrypted``.

The sender's long-term public key is advertised via the
``X-Public-Key`` SIP header (URL-safe base64, no padding) so that
contacts can discover it for future sessions.

Example::

    from voip.messaging import MessageCipher

    alice = MessageCipher.generate()
    bob = MessageCipher.generate()

    blob = alice.encrypt("Hello Bob!", bob.public_key_bytes)
    assert bob.decrypt(blob) == b"Hello Bob!"
"""

from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

__all__ = ["CONTENT_TYPE", "MessageCipher"]

#: Content-type for encrypted SIP MESSAGE bodies.
CONTENT_TYPE = "application/x-voip-encrypted"

#: HKDF info label identifying this protocol version.
_HKDF_INFO = b"voip-message-v1"

#: AES-256-GCM nonce size in bytes.
_NONCE_SIZE = 12

#: X25519 raw public key size in bytes.
_PUB_KEY_SIZE = 32


class MessageCipher:
    """X25519 + AES-256-GCM end-to-end encryption for SIP messages.

    Each instance holds a long-term X25519 identity key.  Outbound
    messages use a *fresh ephemeral* key per :meth:`encrypt` call,
    providing forward secrecy even if the sender's long-term key is
    later compromised.

    Args:
        private_key_bytes: 32-byte raw X25519 private key.  A new
            random key is generated when *None* (default).
    """

    def __init__(self, private_key_bytes: bytes | None = None) -> None:
        self._private_key: X25519PrivateKey = (
            X25519PrivateKey.from_private_bytes(private_key_bytes)
            if private_key_bytes is not None
            else X25519PrivateKey.generate()
        )

    @classmethod
    def generate(cls) -> MessageCipher:
        """Return a new :class:`MessageCipher` with a random identity key."""
        return cls()

    @property
    def public_key_bytes(self) -> bytes:
        """32-byte raw X25519 public key."""
        return self._private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )

    @property
    def public_key_b64(self) -> str:
        """URL-safe base64 public key (no padding) for ``X-Public-Key`` headers."""
        return base64.urlsafe_b64encode(self.public_key_bytes).rstrip(b"=").decode()

    def encrypt(self, plaintext: str | bytes, recipient_public_key: bytes) -> bytes:
        """Encrypt *plaintext* for *recipient_public_key*.

        A fresh ephemeral X25519 key pair is generated for each call.
        ECDH shared secret + HKDF derives a 256-bit AES-GCM key; the
        result is the concatenation of the ephemeral public key, a
        random nonce, and the GCM ciphertext (which includes the 16-byte
        authentication tag).

        Args:
            plaintext: Message text or raw bytes to encrypt.
            recipient_public_key: 32-byte raw X25519 public key of the
                intended recipient.

        Returns:
            ``eph_pub (32 B) | nonce (12 B) | ciphertext+tag (N+16 B)``
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode()

        eph_key = X25519PrivateKey.generate()
        recipient_pub = X25519PublicKey.from_public_bytes(recipient_public_key)
        shared = eph_key.exchange(recipient_pub)

        enc_key = HKDF(
            algorithm=SHA256(),
            length=32,
            salt=None,
            info=_HKDF_INFO,
        ).derive(shared)

        nonce = os.urandom(_NONCE_SIZE)
        ciphertext = AESGCM(enc_key).encrypt(nonce, plaintext, None)

        eph_pub_bytes = eph_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return eph_pub_bytes + nonce + ciphertext

    def decrypt(self, blob: bytes) -> bytes:
        """Decrypt and authenticate an encrypted blob.

        Args:
            blob: Encrypted blob as returned by :meth:`encrypt`.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            ValueError: If the blob is too short or authentication fails.
        """
        min_len = _PUB_KEY_SIZE + _NONCE_SIZE + 16  # 16 = GCM tag only, no plaintext
        if len(blob) < min_len:
            raise ValueError("Encrypted blob is too short")

        eph_pub_bytes = blob[:_PUB_KEY_SIZE]
        nonce = blob[_PUB_KEY_SIZE : _PUB_KEY_SIZE + _NONCE_SIZE]
        ciphertext = blob[_PUB_KEY_SIZE + _NONCE_SIZE :]

        eph_pub = X25519PublicKey.from_public_bytes(eph_pub_bytes)
        shared = self._private_key.exchange(eph_pub)

        enc_key = HKDF(
            algorithm=SHA256(),
            length=32,
            salt=None,
            info=_HKDF_INFO,
        ).derive(shared)

        try:
            return AESGCM(enc_key).decrypt(nonce, ciphertext, None)
        except InvalidTag as exc:
            raise ValueError("Message authentication failed") from exc
