"""Descriptive historical backtest of the 30 LEAD forward variants (FORWARD.md §10) on 2020-05..2026-09.

    uv run python scripts/r17_variants_backtest.py binance_2026-10-03 runs/r17_variants_history.json

Same panel, members, engine and daily grid as research 17 (`V01_base` must reproduce its Sharpe 0.802). Reads the
panels directly (`research15.setup._load`), NOT through the research 17 vault: that vault was spent by the one-shot
final run, so this is a post-hoc, contaminated look. Descriptive only, nothing is selected from it and nothing is
written to the trial registry.

Per variant: daily grid, every trade (signal strength, lag, BTC regime, exit reason, hold), cost and delay
sweeps; across variants: correlation, PBO (CSCV), SPA p-value and stationary-bootstrap Sharpe intervals.
"""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from qlab.research15 import strategy as S
from qlab.research15.evaluate import by_year, stats, window
from qlab.research15.setup import _load, epoch_15m
from qlab.research16 import engine as G
from qlab.research17 import engine as R, variants as V
from qlab.research17.evaluate import N_CUM, SR_VAR
from qlab.research17.setup import START, SUBPERIODS, day0, n_days
from qlab.research18 import rules as H18
from qlab.validation.stats import (deflated_sharpe, pbo_cscv, probabilistic_sharpe, spa_pvalue,
                                   stationary_bootstrap_indices)

COSTS = [0.0, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3]     # per side, sweep for every variant
DELAYS = [0, 1, 2, 3, 4]                                          # candles, absolute (replaces the variant's delay)
REGIME = 30 * 96                                                  # BTC trend lookback for the regime tag (30 days)
N_BOOT = 1000
ALL = [*V.SPECS, H18.H2]                                          # the 30 variants + H2 of research 18 (V05_lag0 is H1)
GATED = {H18.H2.name}                                             # entries only while BTC spot > its 200-day mean
ALIAS = {"V05_lag0": "H1", "H2_btc200": "H2", "V01_base": "základ H2"}
GROUP = {**{k: v for k, v in V.GROUPS.items()}, "r18": "Research 18"}

snapshot, out_path = sys.argv[1], Path(sys.argv[2])
members = json.loads((Path("data/raw") / snapshot / "r15_manifest.json").read_text())["members"]
first = f"{START.year}-{START.month:02d}"
members = {m: v for m, v in members.items() if m >= first}
syms = sorted({s for v in members.values() for s in v})
sym_id = {s: i for i, s in enumerate(syms)}

btc_raw = _load(snapshot, "spot", "BTCUSDT", None)
btc = V.leader_moves(btc_raw)
dates = (day0() + np.arange(n_days())).astype("datetime64[D]")
s0 = int(epoch_15m(START))
nd = n_days()


gate = H18.ma_gate(btc.c)                                         # per BTC spot candle, no look ahead


def on_leader(a, flag):
    """Per follower candle: `flag` (per leader candle) at the same time, False outside the leader's data."""
    li = a.t - btc.t0
    ok = (li >= 0) & (li < len(flag))
    return np.where(ok, flag[np.where(ok, li, 0)], False)


def book(a, cand, sym, sp):
    top = {m: v[:sp.top] for m, v in members.items()}
    allowed = R.member_mask(a, sym, top)
    if sp.name in GATED:
        allowed = allowed & on_leader(a, gate)
    return G.trade_book(a, cand, max(s0 - int(a.t[0]), 0), len(a.c), delay=sp.delay, cost=sp.cost, allowed=allowed)


def to_grid(a, b, w):
    out = np.zeros(nd)
    R._to_grid(a.t, *S.daily(a.t, b.net, max(s0 - int(a.t[0]), 0), len(a.c)), w, out)
    return out


# benchmarks: BTC spot buy and hold, equal-weight buy and hold of the month's members (B_HOLD of research 17)
bt = G.Asset.build(btc_raw)
bench_btc = R.hold_daily(bt, max(s0 - int(bt.t[0]), 0))
hold_sum, hold_cnt = np.zeros(nd), np.zeros(nd)
btc_trend = np.full(len(btc.c), np.nan)
btc_trend[REGIME:] = np.log(btc.c[REGIME:] / btc.c[:-REGIME])

res = {sp.name: {"daily": np.zeros(nd), "cost": np.zeros((len(COSTS), nd)), "delay": np.zeros((len(DELAYS), nd)),
                 "tr": [], "cost_tr": [[] for _ in COSTS], "delay_tr": [[] for _ in DELAYS]} for sp in ALL}
for i, sym in enumerate(syms):
    a = G.Asset.build(_load(snapshot, "perp", sym, None))
    md = R.member_days(sym, members)
    hold_sum += np.where(md, R.hold_daily(a, max(s0 - int(a.t[0]), 0)), 0.0)
    hold_cnt += md
    own: dict = {}
    for sp in ALL:
        cand = V.candidates(a, btc, sp, own)
        o = res[sp.name]
        b = book(a, cand, sym, sp)
        o["daily"] += to_grid(a, b, sp.weight)
        if len(b.entries):
            k = b.entries - sp.delay                            # signal candle on the follower grid
            li = a.t[k] - btc.t0                                # same candle on the leader grid
            rb, sg = btc.cache[sp.window][li], btc.sig[li]
            ra = own[sp.window][k]
            o["tr"].append(np.column_stack([a.t[b.entries], np.full(len(k), sym_id[sym]), b.ret, b.reason,
                                            b.exits - b.entries, rb / (np.sqrt(sp.window) * sg), ra / rb,
                                            btc_trend[li], np.full(len(k), np.nan), gate[li]]))
        for j, c in enumerate(COSTS):
            bc = book(a, cand, sym, replace(sp, cost=c))
            o["cost"][j] += to_grid(a, bc, sp.weight)
            o["cost_tr"][j].append(bc.ret)
        for j, d in enumerate(DELAYS):
            bd = book(a, cand, sym, replace(sp, delay=d))
            o["delay"][j] += to_grid(a, bd, sp.weight)
            o["delay_tr"][j].append(bd.ret)
    del a
    print(f"{i + 1}/{len(syms)} {sym}", flush=True)

bench_ew = np.where(hold_cnt > 0, hold_sum / np.maximum(hold_cnt, 1), 0.0)
mat = np.column_stack([res[sp.name]["daily"] for sp in ALL])

# stationary bootstrap of the daily grid, same resamples for every variant (mean block 10 days)
rng = np.random.default_rng(0)
boot = np.empty((N_BOOT, mat.shape[1]))
for bi in range(N_BOOT):
    x = mat[stationary_bootstrap_indices(nd, 10.0, rng)]
    sd = x.std(0, ddof=1)
    boot[bi] = np.where(sd > 0, x.mean(0) / np.where(sd > 0, sd, 1), 0) * np.sqrt(365)
print("bootstrap done", flush=True)
pbo, logits = pbo_cscv(mat[:, :30], 16)          # PBO and SPA stay on the 30 variants of FORWARD.md §10
spa = spa_pvalue(mat[:, :30], n_boot=2000, mean_block=10.0)
print(f"PBO {pbo:.3f} SPA {spa:.3f}", flush=True)
corr = np.corrcoef(mat.T)


def cat(xs):
    x = np.concatenate(xs) if xs else np.zeros(0)
    return x if len(x) else None


def r4(x):
    return [None if not np.isfinite(v) else round(float(v), 6) for v in x]


def trade_stats(r):
    if not len(r):
        return {}
    w, l = r[r > 0], r[r <= 0]
    signs = np.sign(r)
    run, worst = 0, 0
    for s in signs:
        run = run + 1 if s <= 0 else 0
        worst = max(worst, run)
    return {"n": int(len(r)), "mean": float(r.mean()), "median": float(np.median(r)), "std": float(r.std()),
            "hit": float((r > 0).mean()), "avg_win": float(w.mean()) if len(w) else None,
            "avg_loss": float(l.mean()) if len(l) else None,
            "profit_factor": float(w.sum() / -l.sum()) if len(l) and l.sum() < 0 else None,
            "max_losing_streak": int(worst), "skew": float(((r - r.mean()) ** 3).mean() / r.std() ** 3),
            "q": {str(q): float(np.quantile(r, q / 100)) for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
            "best": float(r.max()), "worst": float(r.min())}


out = {"snapshot": snapshot, "period": [str(START), "2026-09-30"], "n_followers": len(syms), "symbols": syms,
       "dates": [str(d) for d in dates], "costs": COSTS, "delays": DELAYS, "N_CUM_before": N_CUM, "n_boot": N_BOOT,
       "bench": {"BTC buy & hold": r4(bench_btc), "EW členové buy & hold": r4(bench_ew)},
       "bench_stats": {"BTC buy & hold": stats(bench_btc), "EW členové buy & hold": stats(bench_ew)},
       "pbo": pbo, "pbo_logits": [float(x) for x in logits[::10]], "spa_p": spa,
       "corr": [[round(float(x), 3) for x in row] for row in corr], "variants": {}}
for vi, sp in enumerate(ALL):
    o = res[sp.name]
    daily = o["daily"]
    tr = np.concatenate(o["tr"]) if o["tr"] else np.zeros((0, 10))
    tr = tr[np.argsort(tr[:, 0], kind="stable")]
    r = tr[:, 2]
    st = stats(daily, r if len(r) else None)
    ev_t = tr[:, 0].astype(np.int64) - sp.delay
    mean, lo, hi, n_ev = R.event_ci([ev_t], [r], n_boot=2000)
    u, cnt = np.unique(ev_t, return_counts=True)
    size = dict(zip(u.tolist(), cnt.tolist()))
    out["variants"][sp.name] = {
        "spec": asdict(sp), "group": GROUP["r18" if sp.name in GATED else sp.group], "alias": ALIAS.get(sp.name), "stats": st, "n_events": int(n_ev), "n_trades": int(len(r)),
        "trade": trade_stats(r), "event_ci90": [mean, lo, hi],
        "by_year": by_year(dates, daily),
        "subperiods": {k: stats(window(dates, daily, *w))["sharpe"] for k, w in SUBPERIODS.items()},
        "psr": probabilistic_sharpe(daily), "dsr_N1": deflated_sharpe(daily, 1, SR_VAR),
        "dsr_N77": deflated_sharpe(daily, N_CUM + 30, SR_VAR),
        "boot_sharpe": [float(np.quantile(boot[:, vi], q)) for q in (0.05, 0.5, 0.95)],
        "boot_p_le0": float((boot[:, vi] <= 0).mean()),
        "exposure": float((daily != 0).mean()),
        "gate_split": {k: ({"n": int(m.sum()), "mean": float(r[m].mean()), "hit": float((r[m] > 0).mean())} if m.any() else
                          {"n": 0, "mean": None, "hit": None})
                       for k, m in (("above", tr[:, 9] > 0), ("below", tr[:, 9] == 0))},
        "daily": r4(daily),
        "cost_sweep": [{"cost": c, **stats(o["cost"][j], cat(o["cost_tr"][j]))}
                       for j, c in enumerate(COSTS)],
        "delay_sweep": [{"delay": d, **stats(o["delay"][j], cat(o["delay_tr"][j]))}
                        for j, d in enumerate(DELAYS)],
        # trades: [entry 15m index, symbol id, net return, exit reason, hold candles, BTC move in sigma units,
        #          follower move / BTC move, BTC 30-day log return, trades in the same event, BTC > MA200 at the signal]
        "trades": [[int(t[0]), int(t[1]), round(float(t[2]), 5), int(t[3]), int(t[4]), round(float(t[5]), 3),
                    None if not np.isfinite(t[6]) else round(float(t[6]), 3),
                    None if not np.isfinite(t[7]) else round(float(t[7]), 4), size[int(t[0]) - sp.delay], int(t[9])]
                   for t in tr]}
    print(sp.name, round(st["sharpe"], 3), st.get("n_trades"), flush=True)
tt = btc.t0 + np.arange(len(btc.c))
inside = tt >= s0
yrs = (tt[inside] * 900).astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
out["gate"] = {"definition": "BTC spot close > mean of the last 19200 closes (200 days), at the signal candle",
               "share_above": float(gate[inside].mean()),
               "by_year": {int(y): float(gate[inside][yrs == y].mean()) for y in np.unique(yrs)}}
out_path.write_text(json.dumps(out, default=float, separators=(",", ":")))
