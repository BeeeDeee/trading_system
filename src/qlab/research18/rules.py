"""The two rules of research 18 (PREREGISTRATION.md §4-5) on top of the research 17 variants engine.

H1 = `V05_lag0` (altcoin fell while BTC jumped), H2 = `V01_base` entered only while BTC spot closes above its 200-day
mean. Both are spelled out in full, so a later edit of `research17.variants.SPECS` cannot change them (a test pins
the equality with V05 / V01).
"""

from dataclasses import replace

import numpy as np

from qlab.research15 import strategy as S
from qlab.research16 import engine as G
from qlab.research17 import engine as R, variants as V
from qlab.research17.setup import n_days
from qlab.research18.setup import MA_CANDLES

H1 = V.Spec("H1_lag0", "base", K=3.0, ratio=0.0, window=4, hold=16, cost=1e-3, weight=0.1, top=10,
            desc="long after a BTC jump >= 3 sigma when the altcoin fell (rA < 0)")
H2 = V.Spec("H2_btc200", "base", K=3.0, ratio=0.5, window=4, hold=16, cost=1e-3, weight=0.1, top=10,
            desc="primary rule (rA / rB < 0.5), only while BTC spot > its 200-day mean")
HYPOTHESES = {"H1": (H1, False), "H2": (H2, True)}        # name -> (spec, uses the BTC > MA200 gate)


def ma_gate(close: np.ndarray, window: int = MA_CANDLES) -> np.ndarray:
    """Per BTC candle k: close[k] > mean(close[k-window+1 .. k]); False until `window` candles exist (no look ahead)."""
    c = np.asarray(close, float)
    out = np.zeros(len(c), bool)
    if len(c) >= window:
        cs = np.concatenate([[0.0], np.cumsum(c)])
        ma = (cs[window:] - cs[:-window]) / window
        out[window - 1:] = c[window - 1:] > ma
    return out


def run(sp: V.Spec, assets: dict[str, G.Asset], btc: V.LeaderMoves, members: dict[str, list[str]], s0: int,
        gate: np.ndarray | None = None) -> dict:
    """Portfolio of one rule over the followers on the fixed daily grid (call inside the window of `evaluate`).
    `gate` (per leader candle) restricts entries to candles where it is True. Returns the daily grid, the net returns
    of all trades, their entry candles (absolute 15m index), symbols and the number of signal candles."""
    top = {m: v[:sp.top] for m, v in members.items()}
    daily, rets, ent, syms = np.zeros(n_days()), [], [], []
    for sym, a in assets.items():
        n, start = len(a.c), max(s0 - int(a.t[0]), 0)
        allowed = R.member_mask(a, sym, top)
        if gate is not None:
            li = a.t - btc.t0
            ok = (li >= 0) & (li < len(gate))
            allowed = allowed & np.where(ok, gate[np.where(ok, li, 0)], False)
        b = G.trade_book(a, V.candidates(a, btc, sp, {}), start, n, delay=sp.delay, cost=sp.cost, allowed=allowed)
        R._to_grid(a.t, *S.daily(a.t, b.net, start, n), sp.weight, daily)
        rets.append(b.ret)
        ent.append(a.t[b.entries] - sp.delay if len(b.entries) else np.zeros(0, np.int64))
        syms += [sym] * len(b.ret)
    return {"daily": daily, "rets": np.concatenate(rets) if rets else np.zeros(0),
            "events": np.concatenate(ent) if ent else np.zeros(0, np.int64), "syms": syms}


def variant(sp: V.Spec, **kw) -> V.Spec:
    return replace(sp, **kw)
