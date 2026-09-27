"""Research 7: SPY core + VIX-gated STR-TF sleeve vs. SPY (docs/research7/PREREGISTRATION.md).

Usage: python scripts/r7_evaluate.py sharadar_YYYY-MM-DD
Output: docs/research7/results/summary.json
All history is declared as seen (pre-registration §2); the panel is loaded without the vault.
"""

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from qlab.research5.report import yearly
from qlab.research5.setup import load_full, registry as registry_r5
from qlab.research5.strategy import DEFAULT, Context, run
from qlab.validation.metrics import max_drawdown
from qlab.validation.registry import TrialRegistry
from qlab.validation.stats import bootstrap_ci, deflated_sharpe

snapshot = sys.argv[1]
t0 = time.time()
res_dir = Path("docs/research7/results")
res_dir.mkdir(parents=True, exist_ok=True)
EVAL_FROM = "2003-01-02"
SUBPERIODS = {"2003-2014": ("2003-01-01", "2014-12-31"), "2015-2019": ("2015-01-01", "2019-12-31"),
              "2020-2026": ("2020-01-01", "2026-12-31")}

primary = replace(DEFAULT, vix_gate="abs25", core="spy")
runs = {
    "primary": primary,
    "cost_x2": replace(primary, cost_mult=2.0),
    "vix20": replace(primary, vix_gate="abs20"),
    "vix30": replace(primary, vix_gate="abs30"),
    "no_vix_gate": replace(primary, vix_gate=None),
    "entry_t_plus_2": replace(primary, entry_delay=2),
}
reg = TrialRegistry("runs/registry/research7.sqlite")
reg.record("methodology_eval", {n: asdict(c) for n, c in runs.items()}, len(runs),
           "research7 all runs, recorded before computing")

full = load_full(snapshot)


def ctx_from(day: str) -> Context:
    return Context(full.panel, full.extra, int(np.searchsorted(full.panel.dates, np.datetime64(day))),
                   full.cache_dir, full.spy)


ctx = ctx_from(EVAL_FROM)
spy_all = ((1 + np.asarray(full.panel.ret_co[:, full.spy]))
           * (1 + np.asarray(full.panel.ret_oc[:, full.spy])) - 1)


def sharpe(r: np.ndarray) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std() > 0 else 0.0


def cagr(r: np.ndarray) -> float:
    return float(np.prod(1 + r) ** (252 / len(r)) - 1)


def compare(r: np.ndarray, s: np.ndarray) -> dict:
    x = np.cov(r, s)
    return {"sharpe": sharpe(r), "sharpe_spy": sharpe(s), "cagr": cagr(r), "cagr_spy": cagr(s),
            "cagr_diff": cagr(r) - cagr(s), "sharpe_diff": sharpe(r) - sharpe(s),
            "max_dd": max_drawdown(np.cumprod(1 + r))[0], "max_dd_spy": max_drawdown(np.cumprod(1 + s))[0],
            "vol": float(r.std(ddof=1) * np.sqrt(252)), "vol_spy": float(s.std(ddof=1) * np.sqrt(252)),
            "beta": float(x[0, 1] / x[1, 1])}


def windows(out, s: np.ndarray) -> dict:
    d = out.dates.astype("datetime64[D]")
    res = {"all": compare(out.returns, s)}
    for k, (lo, hi) in SUBPERIODS.items():
        m = (d >= np.datetime64(lo)) & (d <= np.datetime64(hi))
        if m.sum() > 20:
            res[k] = compare(out.returns[m], s[m])
    tr = out.trades.filter(out.trades["reason"] != "end")
    res["n_trades"] = tr.height
    res["avg_sleeve_exposure"] = float(out.exposure.mean())
    return res


results = {"primary": asdict(primary), "runs": {}}
outs = {}
spy = spy_all[ctx.start:ctx.end]
for name, cfg in runs.items():
    outs[name] = run(ctx, cfg)
    results["runs"][name] = windows(outs[name], spy)
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)

early = ctx_from("1999-01-04")
e_out = run(early, primary)
d = e_out.dates.astype("datetime64[D]")
m = d <= np.datetime64("2002-12-31")
results["primary_1999_2002"] = compare(e_out.returns[m], spy_all[early.start:early.end][m])

r = outs["primary"].returns
active = r - spy
_, lo, hi = bootstrap_ci(np.column_stack([r, spy]), lambda x: sharpe(x[:, 0]) - sharpe(x[:, 1]),
                         n_boot=1000, mean_block=21.0, alpha=0.10, seed=0)
grid_ret = np.load("runs/research5/grid_returns.npy", mmap_mode="r")
daily_sr = np.array([np.nanmean(grid_ret[:, j]) / np.nanstd(grid_ret[:, j], ddof=1)
                     for j in range(grid_ret.shape[1])])
n_trials = (registry_r5().total_configs() + TrialRegistry("runs/registry/research6.sqlite").total_configs()
            + reg.total_configs())
dsr = deflated_sharpe(active, n_trials, float(np.nanvar(daily_sr, ddof=1)))
p = results["runs"]["primary"]
criteria = {
    "C1_sharpe_diff_ci_lower_gt_0": bool(p["all"]["sharpe_diff"] > 0 and lo > 0),
    "C2_cagr_diff_gt_0_each_subperiod": bool(all(p[k]["cagr_diff"] > 0 for k in SUBPERIODS)),
    "C3_cagr_diff_gt_0_cost_x2": bool(results["runs"]["cost_x2"]["all"]["cagr_diff"] > 0),
    "C4_dsr_active_ge_0_95": bool(dsr >= 0.95),
}
results["criteria"] = criteria
results["passed_all"] = all(criteria.values())
results["criteria_inputs"] = {
    "sharpe_diff": p["all"]["sharpe_diff"], "sharpe_diff_ci90": [lo, hi],
    "cagr_diff_by_subperiod": {k: p[k]["cagr_diff"] for k in SUBPERIODS},
    "cagr_diff_cost_x2": results["runs"]["cost_x2"]["all"]["cagr_diff"],
    "active_ann_mean": float(active.mean() * 252), "active_ann_vol": float(active.std(ddof=1) * np.sqrt(252)),
    "dsr_active": dsr, "n_trials": n_trials}
yp = yearly(outs["primary"]).to_dicts()
dd = outs["primary"].dates.astype("datetime64[D]")
for row in yp:
    mm = (dd >= np.datetime64(f"{row['year']}-01-01")) & (dd <= np.datetime64(f"{row['year']}-12-31"))
    row["spy_ret"] = float(np.prod(1 + spy[mm]) - 1)
results["yearly"] = yp
(res_dir / "summary.json").write_text(json.dumps(results, indent=2, default=float))
print(json.dumps({"criteria": criteria, **results["criteria_inputs"]}, indent=2, default=float))
print(f"done in {time.time() - t0:.0f}s")
