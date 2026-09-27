"""Research 6: STR-TF with new entries only on high-VIX days (docs/research6/PREREGISTRATION.md).

Usage: python scripts/r6_evaluate.py sharadar_YYYY-MM-DD [n_random_sims]
Output: docs/research6/results/summary.json
All history is declared as seen (pre-registration §2); the panel is loaded without the vault.
"""

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import polars as pl

from qlab.research5.report import portfolio_stats, trade_stats, yearly
from qlab.research5.setup import load_full, registry as registry_r5
from qlab.research5.strategy import DEFAULT, Config, Context, random_benchmark, run
from qlab.validation.registry import TrialRegistry
from qlab.validation.stats import deflated_sharpe

snapshot = sys.argv[1]
n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
t0 = time.time()
res_dir = Path("docs/research6/results")
res_dir.mkdir(parents=True, exist_ok=True)
EVAL_FROM = "2003-01-02"
SUBPERIODS = {"2003-2014": ("2003-01-01", "2014-12-31"), "2015-2019": ("2015-01-01", "2019-12-31"),
              "2020-2026": ("2020-01-01", "2026-12-31")}

primary = replace(DEFAULT, vix_gate="abs25")
r5_selected = Config(**json.loads(Path("docs/research5/results/dev_summary.json").read_text())
                     ["selected"])
runs = {
    "primary_vix25": primary,
    "ungated": DEFAULT,
    "primary_cost_x2": replace(primary, cost_mult=2.0),
    "primary_cost_x3": replace(primary, cost_mult=3.0),
    "primary_entry_t_plus_2": replace(primary, entry_delay=2),
    "vix20": replace(primary, vix_gate="abs20"),
    "vix30": replace(primary, vix_gate="abs30"),
    "vix_rel80": replace(primary, vix_gate="rel80"),
    "r5_selected_vix25": replace(r5_selected, vix_gate="abs25"),
    "r5_selected_ungated": r5_selected,
}
reg = TrialRegistry("runs/registry/research6.sqlite")
reg.record("methodology_eval", {name: asdict(c) for name, c in runs.items()}, len(runs),
           "research6 all runs, recorded before computing")
reg.record("other", {"random_benchmark": asdict(primary), "n": n_sims}, 1, "research6 random")

full = load_full(snapshot)
ctx = Context(full.panel, full.extra, int(np.searchsorted(full.panel.dates,
                                                          np.datetime64(EVAL_FROM))),
              full.cache_dir, full.spy)


def window_stats(out, lo: str, hi: str) -> dict:
    d = out.dates.astype("datetime64[D]")
    m = (d >= np.datetime64(lo)) & (d <= np.datetime64(hi))
    r = out.returns[m]
    ed = full.panel.dates[out.trades["entry_day"].to_numpy()].astype("datetime64[D]")
    tm = (ed >= np.datetime64(lo)) & (ed <= np.datetime64(hi))
    ts = trade_stats(out.trades.filter(pl.Series(tm)))
    return {"sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std() > 0 else 0.0,
            "cagr": float(np.prod(1 + r) ** (252 / len(r)) - 1), **ts}


results = {"primary": asdict(primary), "runs": {}}
outs = {}
for name, cfg in runs.items():
    out = run(ctx, cfg)
    outs[name] = out
    results["runs"][name] = {"all_2003_2026": {**portfolio_stats(out), **trade_stats(out.trades)},
                             **{k: window_stats(out, *v) for k, v in SUBPERIODS.items()}}
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)

# 1999-2002 reported separately (not used by the criteria).
early = Context(full.panel, full.extra, full.start, full.cache_dir, full.spy)
early_out = run(early, primary)
results["primary_1999_2002"] = window_stats(early_out, "1999-01-01", "2002-12-31")
results["ungated_1999_2002"] = window_stats(run(early, DEFAULT), "1999-01-01", "2002-12-31")

p = results["runs"]["primary_vix25"]
u = results["runs"]["ungated"]["all_2003_2026"]
rand = random_benchmark(ctx, primary, outs["primary_vix25"], n_sims=n_sims, seed=0)
grid_ret = np.load("runs/research5/grid_returns.npy", mmap_mode="r")
daily_sr = np.array([np.nanmean(grid_ret[:, j]) / np.nanstd(grid_ret[:, j], ddof=1)
                     for j in range(grid_ret.shape[1])])
n_trials = registry_r5().total_configs() + reg.total_configs()
dsr = deflated_sharpe(outs["primary_vix25"].returns, n_trials, float(np.nanvar(daily_sr, ddof=1)))
a = p["all_2003_2026"]
criteria = {
    "C1_gross_ge_2x_cost": bool(a["gross_to_cost"] >= 2.0),
    "C2_cagr_cost_x2_gt_0": bool(results["runs"]["primary_cost_x2"]["all_2003_2026"]["cagr"] > 0),
    "C3_net_trade_positive_each_subperiod": bool(all(p[k].get("net_bps", -1) > 0 for k in SUBPERIODS)),
    "C4_gate_beats_ungated_net_per_trade": bool(a["net_bps"] > u["net_bps"]),
    "C5_sharpe_above_random_p95": bool(a["sharpe"] > np.quantile(rand, 0.95)),
    "C6_dsr_ge_0_95": bool(dsr >= 0.95),
}
results["criteria"] = criteria
results["passed_all"] = all(criteria.values())
results["criteria_inputs"] = {"gross_to_cost": a["gross_to_cost"],
                              "cagr_cost_x2": results["runs"]["primary_cost_x2"]["all_2003_2026"]["cagr"],
                              "net_bps_by_subperiod": {k: p[k].get("net_bps") for k in SUBPERIODS},
                              "net_bps_gated": a["net_bps"], "net_bps_ungated": u["net_bps"],
                              "sharpe": a["sharpe"], "random_p50": float(np.median(rand)),
                              "random_p95": float(np.quantile(rand, 0.95)), "dsr": dsr,
                              "n_trials": n_trials}
results["yearly_primary"] = yearly(outs["primary_vix25"]).to_dicts()
results["yearly_ungated"] = yearly(outs["ungated"]).to_dicts()
(res_dir / "summary.json").write_text(json.dumps(results, indent=2, default=float))
print(json.dumps({"criteria": criteria, **results["criteria_inputs"]}, indent=2, default=float))
print(f"done in {time.time() - t0:.0f}s")
