"""Point-in-time secondary basket of research 15 (prereg §3): monthly top USDT-M perps by trailing quote volume.

Membership for month m uses only daily perp candles that closed before the first day of m, so the basket is
free of survivorship and look-ahead (volume only, no returns). Crypto underlyings only: a perp must map to a
Binance spot pair (`research8.data.spot_pair`, which also drops stablecoins), which removes index and TradFi
perps (BTCDOM, DEFI, stocks, metals); tokenized gold is dropped explicitly.
"""

import io
import zipfile
from datetime import date
from pathlib import Path

import numpy as np

from qlab.research8.data import spot_pair

EXCLUDED = {"BTCUSDT", "ETHUSDT", "PAXGUSDT", "XAUTUSDT"}   # primary pair is separate; gold tokens
TOP_N = 10
WINDOW_DAYS = 30
MIN_HISTORY_DAYS = 60


def daily_quote_volume(perp_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(epoch days, quote volume) of one perp from its monthly 1d zips."""
    days, qv = [], []
    for f in sorted(perp_dir.glob("*-1d-*.zip")):
        with zipfile.ZipFile(f) as z:
            txt = z.read(z.namelist()[0]).decode()
        lines = [ln for ln in txt.splitlines() if ln and ln[0].isdigit()]
        if lines:
            a = np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",", ndmin=2, usecols=(0, 7))
            days.append((a[:, 0] // 86_400_000).astype(np.int64))
            qv.append(a[:, 1])
    if not days:
        return np.zeros(0, np.int64), np.zeros(0)
    d, q = np.concatenate(days), np.concatenate(qv)
    d, idx = np.unique(d, return_index=True)
    return d, q[idx]


def eligible(perps: list[str], spot_symbols: set[str]) -> list[str]:
    return [p for p in perps if p not in EXCLUDED and spot_pair(p, spot_symbols) is not None]


def monthly_members(volumes: dict[str, tuple[np.ndarray, np.ndarray]], months: list[date],
                    top_n: int = TOP_N) -> dict[str, list[str]]:
    """{'YYYY-MM': top_n symbols} ranked by quote volume of the WINDOW_DAYS days before the month starts.

    A symbol qualifies when its first candle is at least MIN_HISTORY_DAYS before the month and it has a candle
    on every day of the window. Ties break by symbol name.
    """
    out = {}
    for m in months:
        start = (np.datetime64(m, "D") - np.datetime64("1970-01-01", "D")).astype(np.int64)
        lo = start - WINDOW_DAYS
        ranked = []
        for sym, (d, q) in volumes.items():
            if len(d) == 0 or d[0] > start - MIN_HISTORY_DAYS:
                continue
            w = (d >= lo) & (d < start)
            if w.sum() < WINDOW_DAYS:
                continue
            ranked.append((-float(q[w].sum()), sym))
        out[m.strftime("%Y-%m")] = [s for _, s in sorted(ranked)[:top_n]]
    return out
