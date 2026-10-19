"""Predeclared descriptive exit variants for H2_btc200.

This module is deliberately separate from rules.py / evaluate.py: none of these variants
changes the locked H2 forward rule or its decision. Stops are checked conservatively before
targets when both are touched in one 15m candle. Trailing levels use only highs through the
previous candle, avoiding intrabar look-ahead.
"""

from dataclasses import asdict, dataclass

import numpy as np

from qlab.research15 import strategy as S
from qlab.research15.evaluate import stats
from qlab.research16 import engine as G
from qlab.research17 import engine as R, variants as V
from qlab.research17.setup import day0, n_days
from qlab.research18 import rules as H


@dataclass(frozen=True)
class ExitSpec:
    name: str
    kind: str                         # time, target, stop, bracket, trailing
    target: float | None = None       # fraction above entry
    stop: float | None = None         # fraction below entry
    activation: float | None = None   # favorable move before trailing stop activates
    trail: float | None = None        # distance below highest high since entry
    hold: int = 16                    # hard time exit, 15m candles


SPECS = [ExitSpec("X00_time4h", "time")]
SPECS += [ExitSpec(f"TP_{int(x*10000):03d}bp", "target", target=x)
          for x in (0.005, 0.01, 0.015, 0.02, 0.03, 0.04)]
SPECS += [ExitSpec(f"SL_{int(x*10000):03d}bp", "stop", stop=x)
          for x in (0.005, 0.01, 0.015, 0.02, 0.03, 0.04)]
SPECS += [ExitSpec(f"BRK_SL{int(s*10000):03d}_TP{int(t*10000):03d}", "bracket", stop=s, target=t)
          for s, t in ((0.005, 0.01), (0.01, 0.02), (0.015, 0.03), (0.02, 0.04), (0.03, 0.06))]
SPECS += [ExitSpec(f"TR_A{int(a*10000):03d}_D{int(d*10000):03d}", "trailing", activation=a, trail=d)
          for a, d in ((0.0, 0.005), (0.0, 0.01), (0.0, 0.02),
                       (0.005, 0.01), (0.005, 0.02), (0.005, 0.03),
                       (0.01, 0.01), (0.01, 0.02), (0.01, 0.03),
                       (0.02, 0.02), (0.02, 0.03), (0.02, 0.04))]
assert len(SPECS) == 30 and len({s.name for s in SPECS}) == 30


def _exit(a: G.Asset, e: int, last: int, sp: ExitSpec) -> tuple[int, float]:
    entry = float(a.c[e])
    hard_last = min(e + sp.hold, last)
    peak = entry
    stop_price = entry * (1 - sp.stop) if sp.stop is not None else np.nan
    target_price = entry * (1 + sp.target) if sp.target is not None else np.nan
    for j in range(e + 1, hard_last + 1):
        if not a.real[j]:
            continue
        # A trailing stop active before candle j, based on highs no later than j-1.
        if sp.kind == "trailing" and peak >= entry * (1 + float(sp.activation)):
            tr = peak * (1 - float(sp.trail))
            stop_price = max(stop_price, tr) if np.isfinite(stop_price) else tr
        stop_hit = np.isfinite(stop_price) and a.l[j] <= stop_price
        target_hit = np.isfinite(target_price) and a.h[j] >= target_price
        if stop_hit:                       # conservative if both levels touched in one bar
            return j, min(float(a.o[j]), float(stop_price))
        if target_hit:
            return j, max(float(a.o[j]), float(target_price))
        peak = max(peak, float(a.h[j]))
    return hard_last, float(a.c[hard_last])


def run(sp: ExitSpec, assets: dict[str, G.Asset], btc: V.LeaderMoves,
        members: dict[str, list[str]], s0: int, gate: np.ndarray, cost: float = 1e-3) -> dict:
    """Run one H2 exit rule. Uses H2 signals/MA200 entries and the existing costs/funding model."""
    top = {m: v[:10] for m, v in members.items()}
    daily, rets, events = np.zeros(n_days()), [], []
    for sym, a in assets.items():
        n = len(a.c)
        start = max(s0 - int(a.t[0]), 0)
        cand = V.candidates(a, btc, V.SPECS[0], {})
        member = R.member_mask(a, sym, top)
        li = a.t - btc.t0
        valid = (li >= 0) & (li < len(gate))
        idx = np.where(valid, li, 0)
        allowed = member & np.where(valid, gate[idx], False)
        net, blocked = np.zeros(n), -1
        for k in cand.k:
            k = int(k)
            if k < start or k >= n - 1 or k <= blocked or not allowed[k]:
                continue
            e, last = k, min(k + sp.hold, n - 1)
            if e >= last:
                continue
            x, price = _exit(a, e, last, sp)
            hr = S._bar_returns(a, e, x, price, 1, cost, True)
            net[e + 1:x + 1] += hr
            rets.append(float(np.prod(1 + hr) - 1))
            events.append(int(a.t[k]))
            blocked = x
        dates, vals = S.daily(a.t, net, start, n)
        R._to_grid(a.t, dates, vals, 0.1, daily)
    r = np.asarray(rets, float)
    st = stats(daily, r if len(r) else None)
    return {"spec": asdict(sp), "daily": daily, "rets": r, "events": np.asarray(events, np.int64),
            "sharpe": st["sharpe"], "cagr": st["cagr"], "max_dd": st["max_dd"],
            "n_trades": int(len(r)), "mean_trade": st.get("mean_trade")}


def run_all(assets: dict[str, G.Asset], btc: V.LeaderMoves, members: dict[str, list[str]],
            s0: int, gate: np.ndarray) -> dict[str, dict]:
    """Descriptive H2 exit grid; caller supplies already bounded panels and frozen MA200 gate."""
    return {sp.name: run(sp, assets, btc, members, s0, gate) for sp in SPECS}


def run_dynamic(sp: ExitSpec, assets: dict[str, G.Asset], btc: V.LeaderMoves,
                members: dict[str, list[str]], s0: int, gate: np.ndarray,
                target_gross: float = 0.8, max_coin_weight: float = 0.4,
                cost: float = 1e-3) -> dict:
    """Run H2 with unused capital reallocated across open trades.

    Equal weights target 80% gross exposure, capped at 40% per coin. A lone trade therefore
    gets 40%; two trades get 40% each; three or more share 80%. Reweights pay turnover costs.
    """
    if not (0 < max_coin_weight <= target_gross <= 1):
        raise ValueError("require 0 < max_coin_weight <= target_gross <= 1")
    grid_start = int(s0)
    grid_end = max(int(a.t[-1]) for a in assets.values())
    size = grid_end - grid_start + 1
    active = np.zeros(size, np.int16)
    entries = np.zeros(size, np.int16)
    exits = np.zeros(size, np.int16)
    unit_sum = np.zeros(size, float)
    top = {m: v[:10] for m, v in members.items()}
    n_trades = 0
    event_times = []

    for sym, a in assets.items():
        n = len(a.c)
        start = max(s0 - int(a.t[0]), 0)
        cand = V.candidates(a, btc, V.SPECS[0], {})
        member = R.member_mask(a, sym, top)
        li = a.t - btc.t0
        valid = (li >= 0) & (li < len(gate))
        idx = np.where(valid, li, 0)
        allowed = member & np.where(valid, gate[idx], False)
        blocked = -1
        for k in cand.k:
            k = int(k)
            if k < start or k >= n - 1 or k <= blocked or not allowed[k]:
                continue
            e, last = k, min(k + sp.hold, n - 1)
            if e >= last:
                continue
            x, price = _exit(a, e, last, sp)
            hr = S._bar_returns(a, e, x, price, 1, 0.0, True)
            g0, g1 = int(a.t[e + 1] - grid_start), int(a.t[x] - grid_start)
            if g0 < 0 or g1 >= size:
                continue
            active[g0:g1 + 1] += 1
            unit_sum[g0:g1 + 1] += hr
            entries[int(a.t[e] - grid_start)] += 1
            exits[g1] += 1
            n_trades += 1
            event_times.append(int(a.t[k]))
            blocked = x

    weights = np.zeros(size, float)
    has = active > 0
    weights[has] = np.minimum(max_coin_weight, target_gross / active[has])
    gross = weights * unit_sum
    turnover_cost = np.zeros(size, float)
    for t in range(1, size):
        continuing = max(int(active[t - 1]) - int(exits[t - 1]), 0)
        entry_turn = int(entries[t - 1]) * weights[t]
        exit_turn = int(exits[t]) * weights[t]
        rebalance_turn = continuing * abs(weights[t] - weights[t - 1])
        turnover_cost[t] = cost * (entry_turn + exit_turn + rebalance_turn)
    portfolio = gross - turnover_cost

    daily = np.zeros(n_days())
    gtime = np.arange(size, dtype=np.int64) + grid_start
    di = gtime // 96 - day0()
    ok = (di >= 0) & (di < len(daily))
    np.add.at(daily, di[ok], portfolio[ok])
    st = stats(daily)
    return {"spec": asdict(sp), "daily": daily, "sharpe": st["sharpe"], "cagr": st["cagr"],
            "max_dd": st["max_dd"], "total_return": float(np.prod(1 + daily) - 1),
            "n_trades": n_trades, "n_events": len(set(event_times)),
            "avg_gross_exposure": float((weights * active).mean()),
            "max_gross_exposure": float((weights * active).max(initial=0)),
            "total_turnover": float(turnover_cost.sum() / cost) if cost > 0 else 0.0,
            "cost_paid": float(turnover_cost.sum())}


def run_all_dynamic(assets: dict[str, G.Asset], btc: V.LeaderMoves, members: dict[str, list[str]],
                    s0: int, gate: np.ndarray, target_gross: float = 0.8,
                    max_coin_weight: float = 0.4, cost: float = 1e-3) -> dict[str, dict]:
    """All 30 exits with the fixed dynamic allocator; descriptive and non-decision-making."""
    return {sp.name: run_dynamic(sp, assets, btc, members, s0, gate, target_gross, max_coin_weight, cost)
            for sp in SPECS}
