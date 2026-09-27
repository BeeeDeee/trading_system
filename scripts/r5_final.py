"""Research 5 / R5.4: one validation run (2015–2019) and one late-period run (2020–) of the
configuration selected on the development period (pre-registration §7, §10).

Usage: python scripts/r5_final.py sharadar_YYYY-MM-DD validation|late
Needs docs/research5/results/dev_summary.json. Each stage refuses to run twice. The late stage
freezes the methodology in runs/vault_r5 and opens the vault once; it must run from a clean,
committed tree.
"""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import polars as pl

from qlab.research5.report import portfolio_stats, spy_returns, trade_stats, yearly
from qlab.research5.setup import FINAL_ENV, load_stage, log_experiment, once, vault
from qlab.research5.strategy import Config, run
from qlab.validation.registry import config_hash, current_commit

snapshot, stage = sys.argv[1], sys.argv[2]
res_dir = Path("docs/research5/results")
dev = json.loads((res_dir / "dev_summary.json").read_text())
selected = Config(**dev["selected"])
prereg = hashlib.sha256(Path("docs/research5/PREREGISTRATION.md").read_bytes()).hexdigest()[:16]
methodology = config_hash({"selected": asdict(selected), "prereg": prereg})

if stage == "late":
    if subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "docs/research5"],
                      capture_output=True, text=True).stdout.strip():
        sys.exit("commit the code first: the vault session is bound to the commit")
    v = vault()
    if not v.lock_path.exists():
        v.freeze(methodology)
    v.open_final(methodology, snapshot, current_commit())
    os.environ[FINAL_ENV] = methodology
elif stage != "validation":
    sys.exit(f"unknown stage {stage!r}")

once(stage, asdict(selected))
ctx = load_stage(snapshot, stage)
spy = spy_returns(ctx)
out = run(ctx, selected)
stats = {**portfolio_stats(out, spy), **trade_stats(out.trades)}
x2 = run(ctx, replace(selected, cost_mult=2.0))
yr = yearly(out)
result = {"stage": stage, "selected": asdict(selected), "methodology_hash": methodology,
          "period": [str(out.dates[0]), str(out.dates[-1])], "stats": stats,
          "cost_x2": portfolio_stats(x2), "yearly": yr.to_dicts()}

if stage == "validation":
    dev_years = pl.DataFrame(dev["yearly"]).select("year", "log_ret")
    all_years = pl.concat([dev_years, yr.select("year", "log_ret")])
    total = all_years["log_ret"].sum()
    early = all_years.filter(pl.col("year") <= 2002)["log_ret"].sum()
    share = float(early / total) if total > 0 else None
    result["kill"] = {"K2_validation_sharpe_lt_0_7": bool(stats["sharpe"] < 0.7),
                      "K4_gt_50pct_from_1999_2002": bool(share is None or share > 0.5)}
    result["share_1999_2002"] = share
else:
    d = out.dates.astype("datetime64[D]")
    for y in (2020, 2022):
        m = (d >= np.datetime64(f"{y}-01-01")) & (d <= np.datetime64(f"{y}-12-31"))
        r = out.returns[m]
        result[f"year_{y}"] = {"ret": float(np.prod(1 + r) - 1),
                               "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252))}
    r_spy = spy
    result["spy"] = {"cagr": float(np.prod(1 + r_spy) ** (252 / len(r_spy)) - 1),
                     "sharpe": float(r_spy.mean() / r_spy.std(ddof=1) * np.sqrt(252))}

log_experiment(stage, asdict(selected), {k: stats.get(k) for k in
                                         ("sharpe", "cagr", "max_drawdown", "n_trades")}, "final")
(res_dir / f"{stage}_summary.json").write_text(json.dumps(result, indent=2, default=float))
print(json.dumps({k: v for k, v in result.items() if k != "yearly"}, indent=2, default=float))
