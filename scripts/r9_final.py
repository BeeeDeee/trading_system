"""Research 9, final holdout evaluation (prereg §6-7). Runs ONCE through the vault.

    uv run python scripts/r9_final.py binance_2026-10-03 freeze
    QLAB_R9_METHODOLOGY=<hash> uv run python scripts/r9_final.py binance_2026-10-03 run
"""

import hashlib
import io
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from qlab.engine.vector import Decisions, simulate
from qlab.research8.panel import tbill_daily
from qlab.research9 import panel as P, strategy as S
from qlab.research9.setup import DEV, FINAL_ENV, HOLDOUT, SUBPERIODS, index_of, load_final, symbols, vault
from qlab.validation.metrics import max_drawdown
from qlab.validation.registry import current_commit
from qlab.validation.stats import deflated_sharpe, sharpe, stationary_bootstrap_indices

ANN = np.sqrt(365)
snapshot, action = sys.argv[1], sys.argv[2]
FILES = ["docs/research9/PREREGISTRATION.md", "docs/research9/results/dev.json", "scripts/r9_final.py",
         *sorted(str(f) for f in Path("src/qlab/research9").glob("*.py"))]


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
btc, eth = syms.index("BTCUSDT"), syms.index("ETHUSDT")
qv = np.asarray(extra["qv"])
dev = json.loads(Path("docs/research9/results/dev.json").read_text())
by_id = {c.id: c for c in S.grid()}
first = index_of(p, DEV[0]) - 1
h0 = index_of(p, HOLDOUT[0])
H = slice(h0, None)
cost, cost2 = P.cost_rate(qv), P.cost_rate(qv, 2.0)
fred = pl.read_csv(io.BytesIO((Path("data/raw") / snapshot / "fred_DTB3.csv").read_bytes()), infer_schema_length=0)
tbill = tbill_daily(fred, p.dates)


def stats(r):
    nav = np.cumprod(1 + r)
    return {"cagr": float(nav[-1] ** (365 / len(r)) - 1), "sharpe": float(sharpe(r) * ANN),
            "vol": float(r.std(ddof=1) * ANN), "max_dd": float(max_drawdown(nav)[0])}


def bench(fam, c=cost, n=S.UNIVERSE_N):
    d = S.monthly_5050(p, btc, eth, first) if fam == "T" else S.ew_universe_decisions(p, qv, n, first)
    return simulate(p, d, c).returns


def run(cfg, c=cost, shift=0, cash_ret=None, only=None):
    if cfg.family == "T":
        coins = only or (btc, eth)
        sig = {j: np.r_[np.zeros(shift), S.trend_signal(p.close_u[:, j], cfg)[:len(p.dates) - shift]] for j in coins}
        return S.simulate_trend(p, sig, c, first, cash_ret).returns
    d = S.x_decisions(p, qv, cfg, btc, first)
    if shift:
        k = d.days + shift < len(p.dates)
        d = Decisions(d.days[k] + shift, d.weights[k])
    return simulate(p, d, c, cash_ret).returns


def sr_diff_boot(a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.column_stack([a, b])
    out = []
    for _ in range(n):
        y = x[stationary_bootstrap_indices(len(x), 21.0, rng)]
        out.append((sharpe(y[:, 0]) - sharpe(y[:, 1])) * ANN)
    return np.array(out)


res, pv = {"snapshot": snapshot, "commit": current_commit(), "methodology_hash": mh, "candidates": {}}, {}
for name, cid in (("P_T", dev["P_T"]), ("P_X", dev["P_X"])):
    cfg = by_id[cid]
    r, b = run(cfg), bench(cfg.family)
    r2, b2 = run(cfg, cost2), bench(cfg.family, cost2)
    boots = sr_diff_boot(r[H], b[H])
    d = (sharpe(r[H]) - sharpe(b[H])) * ANN
    lo, hi = np.quantile(boots, [0.05, 0.95])
    pv[name] = float((boots <= 0).mean())
    subs = {}
    for k, (a_, e_) in SUBPERIODS.items():
        s_ = slice(index_of(p, a_), index_of(p, e_) + 1)
        subs[k] = (sharpe(r[s_]) - sharpe(b[s_])) * ANN
    st, bt = stats(r[H]), stats(b[H])
    dsr = deflated_sharpe(r[H] - b[H], dev["n_trials"], dev["sr_variance_active_daily"])
    years = {}
    for y in range(2022, 2027):
        yi = np.array([str(x)[:4] == str(y) for x in p.dates])
        yi[:h0] = False
        if yi.any():
            years[str(y)] = {"strategy": stats(r[yi]), "benchmark": stats(b[yi])}
    rob = {"cash_tbill": stats(run(cfg, cash_ret=tbill)[H]), "fill_t+1": stats(run(cfg, shift=1)[H])}
    if cfg.family == "T":
        rob["btc_only"] = stats(run(cfg, only=(btc,))[H])
        rob["eth_only"] = stats(run(cfg, only=(eth,))[H])
    else:
        for n_ in (10, 40):
            rob[f"universe_top{n_}"] = {"strategy": stats(run(replace(cfg, universe_n=n_))[H]),
                                        "benchmark": stats(bench("X", n=n_)[H])}
    res["candidates"][name] = {"id": cid, "strategy": st, "benchmark": bt, "sharpe_diff": d, "ci90": [lo, hi],
                               "p_one_sided": pv[name], "subperiods_sharpe_diff": subs,
                               "stress_sharpe_diff": (sharpe(r2[H]) - sharpe(b2[H])) * ANN, "dsr": dsr,
                               "years": years, "robustness": rob}
order = sorted(pv, key=pv.get)
holm = {k: min(1.0, max(pv[order[i]] * (len(order) - i) for i in range(order.index(k) + 1))) for k in pv}
for name, c in res["candidates"].items():
    crit4 = (c["strategy"]["max_dd"] > c["benchmark"]["max_dd"]) if name == "P_T" else (c["strategy"]["cagr"] > c["benchmark"]["cagr"])
    c["criteria"] = {"1_sharpe_diff_ci_holm": holm[name] < 0.05 and c["ci90"][0] > 0,
                     "2_both_subperiods": all(v > 0 for v in c["subperiods_sharpe_diff"].values()),
                     "3_stress": c["stress_sharpe_diff"] > 0, "4_dd_or_cagr": bool(crit4), "5_dsr": c["dsr"] >= 0.95}
    c["p_holm"], c["pass"] = holm[name], all(c["criteria"].values())
# descriptive: every candidate on the holdout (no selection from this)
res["all_candidates_holdout"] = {c.id: stats(run(c)[H]) for c in S.grid()}
res["benchmarks_holdout"] = {"T_5050": stats(bench("T")[H]), "X_ew20": stats(bench("X")[H]),
                             "BTC_hold": stats(simulate(p, S.hold(p, btc, first), cost).returns[H])}
Path("docs/research9/results/final.json").write_text(json.dumps(res, indent=1))
for name, c in res["candidates"].items():
    s, b = c["strategy"], c["benchmark"]
    print(f"{name} {c['id']}: SR {s['sharpe']:.2f} vs {b['sharpe']:.2f} (Δ {c['sharpe_diff']:+.2f}, CI90 [{c['ci90'][0]:+.2f}, "
          f"{c['ci90'][1]:+.2f}])  CAGR {s['cagr'] * 100:+.1f} % vs {b['cagr'] * 100:+.1f} %  MDD {s['max_dd'] * 100:.1f} % vs "
          f"{b['max_dd'] * 100:.1f} %  -> {'PASS' if c['pass'] else 'FAIL'} {c['criteria']}")
