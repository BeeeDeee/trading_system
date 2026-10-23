"""Podpis požadavků: Ed25519 (výchozí, D9) a HMAC-SHA256 (fallback, např. testnet)."""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key


class Signer(Protocol):
    api_key: str

    def sign(self, payload: str) -> str: ...


@dataclass(frozen=True, repr=False)
class Ed25519Signer:
    """`signature` = base64(Ed25519(payload)). Payload je percent-encoded query string."""

    api_key: str
    private_key: Ed25519PrivateKey

    @classmethod
    def from_pem(cls, api_key: str, pem: bytes, password: bytes | None = None) -> Ed25519Signer:
        key = load_pem_private_key(pem, password=password)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("privátní klíč není Ed25519")
        return cls(api_key, key)

    @classmethod
    def from_file(cls, api_key: str, path: Path, password: bytes | None = None) -> Ed25519Signer:
        return cls.from_pem(api_key, path.read_bytes(), password)

    def sign(self, payload: str) -> str:
        return base64.b64encode(self.private_key.sign(payload.encode("ascii"))).decode("ascii")

    def __repr__(self) -> str:  # klíče se nikdy nevypisují
        return "Ed25519Signer(<redacted>)"


@dataclass(frozen=True, repr=False)
class HmacSigner:
    """`signature` = hex(HMAC-SHA256(secret, payload))."""

    api_key: str
    secret: str

    def sign(self, payload: str) -> str:
        return hmac.new(self.secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def __repr__(self) -> str:
        return "HmacSigner(<redacted>)"
