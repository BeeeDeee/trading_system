"""Parity check of a Research Lab strategy of the BTC-lead card with the research 17 engine (dev rows only).

    PYTHONPATH=/home/kapo/ccode/research_lab uv run python scripts/r17_lab_parity.py <strategy.py> [k_sigma hold_candles]

Both sides use the lab's 15m panel, BTC perpetual as leader and the lab's point-in-time universe (top 12) as the
allowed mask; the research 17 engine (`qlab.research17.engine`) gives the reference entries (BTC and ETH are not
followers). Compares the entry decision candles per follower. Prints counts only, no returns. Needs
docs/research17/lab_card_btc_lead_altcoins.yaml semantics: entries at the signal candle, signals ignored while
holding and on the exit row.
"""

import sys
from pathlib import Path

import numpy as np

from lab.framework import intraday_data
from qlab.research15.setup import _load
from qlab.research16 import engine as G
from qlab.research17 import engine as R

path = Path(sys.argv[1])
k_sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
hold = int(sys.argv[3]) if len(sys.argv) > 3 else 16
snap = "binance_2026-10-03"
view = intraday_data.load(Path("data/derived") / snap / "lab_intraday_15m", top_n=12, stage="dev")
spec = __import__("importlib.util").util.spec_from_file_location("lab_strategy_under_test", path)
mod = __import__("importlib.util").util.module_from_spec(spec)
spec.loader.exec_module(mod)
w = np.asarray(mod.target_positions(view, {"k_sigma": k_sigma, "hold_candles": hold}), dtype=float)
absT = view.times.astype("datetime64[s]").astype(np.int64) // 900
ld = R.leader(_load(snap, "perp", "BTCUSDT", None))
n_ref = n_got = n_eq = 0
bad = []
for j, sym in enumerate(view.instruments):
    if sym in ("BTCUSDT", "ETHUSDT") or not view.universe[:, j].any():
        continue
    a = G.Asset.build(_load(snap, "perp", sym, None))
    allowed = np.zeros(len(a.c), bool)
    idx = absT - a.t[0]
    ok = (idx >= 0) & (idx < len(a.c))
    allowed[idx[ok]] = view.universe[ok, j]
    cs = R.signals(a, ld, k_sigma, "LONG", True)
    cs.deadline = cs.k + hold                                   # the engine's fixed 16 candles -> the card's hold_candles
    b = G.trade_book(a, cs, max(int(absT[0] - a.t[0]), 0),
                     int(absT[-1] - a.t[0]) + 1, allowed=allowed, use_funding=False)
    ref = set(a.t[b.entries].tolist())
    prev = np.concatenate([[0.0], w[:-1, j]])
    got = set(absT[(w[:, j] > 0) & (prev == 0)].tolist())
    n_ref, n_got, n_eq = n_ref + len(ref), n_got + len(got), n_eq + len(ref & got)
    if ref != got:
        bad.append((sym, len(ref), len(got), sorted(ref ^ got)[:3]))
print(f"entries: reference {n_ref}, strategy {n_got}, equal {n_eq}; followers with differences: {len(bad)}")
for r in bad[:10]:
    print(r)
sys.exit(0 if not bad else 1)
