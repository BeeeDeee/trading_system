"""Deterministická ID (SPEC §6): `a5-{slot_short}-{trade_seq}-{kind}`, ≤ 36 znaků, regex Binance."""

from __future__ import annotations

import re

from arena.util import short_hash

CLIENT_ID_RE = re.compile(r"^[\.A-Z\:/a-z0-9_-]{1,36}$")
KINDS = ("E", "TS", "HS", "TP", "TX", "EM")   # vstup, trailing, tvrdý stop, TP, časový výstup, emergency close
PREFIX = "a5-"


def slot_short(slot_id: str) -> str:
    return short_hash(slot_id, 6)


def client_id(slot_id: str, trade_seq: int, kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"neznámý kind {kind}")
    cid = f"{PREFIX}{slot_short(slot_id)}-{trade_seq}-{kind}"
    if not CLIENT_ID_RE.match(cid):
        raise ValueError(f"neplatné client ID {cid}")
    return cid


def paper_trade_id(slot_id: str, trade_seq: int) -> str:
    return f"p-{slot_id}-{trade_seq}"


def is_ours(client_order_id: str | None) -> bool:
    """Příkazy bez prefixu `a5-` patří vlastníkovi a nikdy se na ně nesahá (SPEC §9)."""
    return bool(client_order_id) and client_order_id.startswith(PREFIX)  # type: ignore[union-attr]
