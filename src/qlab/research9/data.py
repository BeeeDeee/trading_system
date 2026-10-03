"""Spot symbol filter of research 9 (prereg §3)."""

import re

from qlab.research8.data import STABLE_BASES

SPOT_RE = re.compile(r"^([A-Z0-9]+)USDT$")
LEVERAGED_RE = re.compile(r"^(.+)(UP|DOWN|BULL|BEAR)$")


def tradable_spot(symbols: set[str]) -> list[str]:
    """USDT spot pairs without stablecoins, wrapped tokens, fiat and leveraged tokens (BTCUP, ETHBEAR...).

    A base ending in UP/DOWN/BULL/BEAR counts as leveraged only when the stem is itself a spot base,
    so JUPUSDT stays and BTCUPUSDT goes.
    """
    bases = {m.group(1) for s in symbols if (m := SPOT_RE.match(s))}
    out = []
    for s in sorted(symbols):
        m = SPOT_RE.match(s)
        if not m or m.group(1) in STABLE_BASES:
            continue
        lev = LEVERAGED_RE.match(m.group(1))
        if lev and lev.group(1) in bases:
            continue
        out.append(s)
    return out
