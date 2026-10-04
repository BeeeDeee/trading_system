"""Research 13, final holdout evaluation (prereg §8-9). Runs ONCE through the vault.

    uv run python scripts/r13_final.py binance_2026-10-03 freeze
    QLAB_R13_METHODOLOGY=<hash> uv run python scripts/r13_final.py binance_2026-10-03 run

Every run starts with an empty book after the close before the holdout start (prereg §7); B_EW uses the
same universe mask as the strategy (identical to research 9 in the top 20).
"""

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from qlab.engine.vector import simulate as simulate_vector
from qlab.research9 import panel as P9, strategy as S9
from qlab.research13 import strategy as S
from qlab.research13.setup import FINAL_ENV, HOLDOUT, SUBPERIODS, index_of, load_final, symbols, vault
from qlab.validation.metrics import max_drawdown
from qlab.validation.registry import current_commit
from qlab.validation.stats import deflated_sharpe, sharpe, stationary_bootstrap_indices

ANN = np.sqrt(365)
N_RND = 500
snapshot, action = sys.argv[1], sys.argv[2]
FILES = ["docs/research13/PREREGISTRATION.md", "docs/research13/results/dev.json", "scripts/r13_final.py",
         "src/qlab/research5/sim.py", *sorted(str(f) for f in Path("src/qlab/research13").glob("*.py"))]


def methodology_hash() -> str:
    h = hashlib.sha256()
    for f in FILES:
        h.update(f.encode() + b"\0" + Path(f).read_bytes() + b"\0")
    return h.hexdigest()


if action == "freeze":
    print(f"frozen: {(mh := methodology_hash())}")
    vault().freeze(mh)
    sys.exit(0)
if action != "run":
    raise SystemExit(__doc__)
mh = os.environ.get(FINAL_ENV, "")
if mh != methodology_hash():
    raise SystemExit("methodology changed since freeze (or env var missing)")
vault().open_final(mh, snapshot, current_commit())
p, extra = load_final(snapshot)
syms = symbols(snapshot)
btc = syms.index("BTCUSDT")
qv = np.asarray(extra["qv"])
dev = json.loads(Path("docs/research13/results/dev.json").read_text())
cfg_p = {c.id: c for c in S.grid()}[dev["P"]]
h0, T = index_of(p, HOLDOUT[0]), p.shape[0]
start = h0 - 1
cost, cost2 = P9.cost_rate(qv), P9.cost_rate(qv, 2.0)
excl = S.excluded(syms)
univ = {n: S.universe_mask(p, qv, n, excl) for n in (10, S9.UNIVERSE_N, 40)}


def stats(r: np.ndarray) -> dict:
    nav = np.cumprod(1 + r)
    return {"cagr": float(nav[-1] ** (365 / len(r)) - 1), "sharpe": float(sharpe(r) * ANN),
            "vol": float(r.std(ddof=1) * ANN), "max_dd": float(max_drawdown(nav)[0])}


def sr(r: np.ndarray) -> float:
    return float(sharpe(r) * ANN)


def run(cfg, c=cost, **kw):
    """Holdout days h0..T-1 (day `start` is the decision close before the holdout)."""
    return S.run(p, extra, cfg, c, start, T, univ[cfg.universe_n], **kw)


def ew(c=cost, n=S9.UNIVERSE_N) -> np.ndarray:
    return simulate_vector(p, S.ew_decisions(p, univ[n], start), c).returns[h0:]


def trade_stats(tr) -> dict:
    reasons = tr["reason"].value_counts()
    return {"n_trades": tr.height, "win_rate": float((tr["net_ret"] > 0).mean()),
            "mean_net": float(tr["net_ret"].mean()), "median_net": float(tr["net_ret"].median()),
            "median_held": float(tr["held"].median()),
            "exits": {r: int(n) for r, n in zip(reasons["reason"], reasons["count"])}}


o = run(cfg_p)
r, b = o.returns[1:], ew()
r2, b2 = run(cfg_p, cost2).returns[1:], ew(cost2)
btc_hold = simulate_vector(p, S9.hold(p, btc, start), cost).returns[h0:]
assert len(r) == len(b) == len(btc_hold) == T - h0

rng = np.random.default_rng(0)
x = np.column_stack([r, b])
boots = np.array([sr(y[:, 0]) - sr(y[:, 1]) for y in (x[stationary_bootstrap_indices(len(x), 21.0, rng)]
                                                       for _ in range(1000))])
lo, hi = np.quantile(boots, [0.05, 0.95])
d = sr(r) - sr(b)
subs = {}
for k, (a_, e_) in SUBPERIODS.items():
    s_ = slice(index_of(p, a_) - h0, index_of(p, e_) + 1 - h0)
    subs[k] = sr(r[s_]) - sr(b[s_])
rnd = np.array([sr(S.random_benchmark(p, univ[S9.UNIVERSE_N], o, cost, start, T, seed).returns[1:])
                for seed in range(N_RND)])
rnd_p95 = float(np.percentile(rnd, 95))
dsr = deflated_sharpe(r - b, dev["n_trials"], dev["sr_variance_active_daily"])

crit = {"1_sharpe_diff_vs_ew_ci": bool(d > 0 and lo > 0),
        "2_sharpe_above_btc": bool(sr(r) > sr(btc_hold)),
        "3_sharpe_above_rnd_p95": bool(sr(r) > rnd_p95),
        "4_both_subperiods": all(v > 0 for v in subs.values()),
        "5_stress": bool(sr(r2) - sr(b2) > 0),
        "6_dsr": bool(dsr >= 0.95)}

dates = p.dates[h0:]
years = {}
for y in range(HOLDOUT[0].year, HOLDOUT[1].year + 1):
    m = dates.astype("datetime64[Y]").astype(int) + 1970 == y
    years[str(y)] = {"strategy": stats(r[m]), "B_EW": stats(b[m]), "B_BTC": stats(btc_hold[m])}

rob = {"fill_one_day_later": stats(run(cfg_p, entry_delay=2).returns[1:]),
       "pivots_from_close": stats(run(replace(cfg_p, pivot_src="close")).returns[1:]),
       "stop_at_stop_price": stats(run(cfg_p, stop_at_level=True).returns[1:])}
for n in (10, 40):
    rob[f"universe_top{n}"] = {"strategy": stats(run(replace(cfg_p, universe_n=n)).returns[1:]),
                               "B_EW": stats(ew(n=n))}

res = {"snapshot": snapshot, "commit": current_commit(), "methodology_hash": mh,
       "period": [str(HOLDOUT[0]), str(HOLDOUT[1])], "P": cfg_p.id,
       "strategy": {**stats(r), **trade_stats(o.trades), "exposure": float(o.exposure[1:].mean()),
                    "costs_ann": float(o.costs[1:].mean() * 365), "turnover_ann": float(o.turnover[1:].mean() * 365)},
       "B_EW": stats(b), "B_BTC": stats(btc_hold),
       "B_RND": {"n": N_RND, "median": float(np.median(rnd)), "p95": rnd_p95,
                 "share_below_P": float((rnd < sr(r)).mean())},
       "sharpe_diff_vs_ew": d, "ci90": [float(lo), float(hi)], "p_one_sided": float((boots <= 0).mean()),
       "subperiods_sharpe_diff": subs, "stress_sharpe_diff": sr(r2) - sr(b2), "dsr": float(dsr),
       "criteria": crit, "pass": all(crit.values()), "years": years, "robustness": rob,
       # descriptive: every candidate on the holdout (no selection from this)
       "all_candidates_holdout": {c.id: stats(run(c).returns[1:]) for c in S.grid()}}
Path("docs/research13/results/final.json").write_text(json.dumps(res, indent=1))

s_ = res["strategy"]
print(f"P {cfg_p.id}: SR {s_['sharpe']:.2f} (CAGR {s_['cagr'] * 100:+.1f} %, MDD {s_['max_dd'] * 100:.1f} %)  "
      f"B_EW {res['B_EW']['sharpe']:.2f}  B_BTC {res['B_BTC']['sharpe']:.2f}  B_RND median {res['B_RND']['median']:.2f} "
      f"p95 {rnd_p95:.2f}")
print(f"ΔSR vs EW {d:+.2f} CI90 [{lo:+.2f}, {hi:+.2f}]  subperiods {subs}  stress {res['stress_sharpe_diff']:+.2f}  "
      f"DSR {dsr:.3f}")
print(f"-> {'PASS' if res['pass'] else 'FAIL'} {crit}")
