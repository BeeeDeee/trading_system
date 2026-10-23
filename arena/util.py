"""Drobné pomocné funkce bez stavu."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def wall_ms() -> int:
    """Lokální čas zdi v ms UTC. Pro čas burzy používej `ServerClock.now_ms()`."""
    return time.time_ns() // 1_000_000


def canonical_json(obj: Any) -> str:
    """Deterministická JSON serializace (seřazené klíče, bez mezer) pro hashování."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, n: int = 6) -> str:
    """Stabilní krátký hash v base32 (A-Z2-7), např. pro `slot_short` v client ID."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return base64.b32encode(digest).decode("ascii")[:n]


def atomic_write_text(path: Path, text: str) -> None:
    """Zápis přes dočasný soubor a `os.replace`, aby čtenář nikdy neviděl půlku souboru."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
