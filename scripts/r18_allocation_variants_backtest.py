"""Rerun all H2 TP/SL/trailing exits with the predeclared dynamic allocator.

    uv run python scripts/r18_allocation_variants_backtest.py binance_2026-10-03 \
        runs/r18_allocation_variants_history.json

Historical results are descriptive and post-hoc. The trial is recorded before panels are loaded.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

from qlab.research15.setup import _load, epoch_15m
from qlab.research16 import engine as G
from qlab.research17 import variants as V
from qlab.research17.setup import START, END
from qlab.research18 import exits as X, rules as H
from qlab.validation.registry import TrialRegistry, current_commit


snapshot, out_path = sys.argv[1], Path(sys.argv[2])
target_gross, max_coin_weight, cost = 0.8, 0.4, 1e-3
config = {"study": "research18_h2_dynamic_allocation", "prereg": "docs/research18/ALLOCATION_VARIANTS.md v0.1",
          "snapshot": snapshot, "period": [str(START), str(END)], "target_gross": target_gross,
          "max_coin_weight": max_coin_weight, "cost": cost, "exit_variants": [asdict(s) for s in X.SPECS]}
TrialRegistry("runs/registry/research18.sqlite").record(
    "candidates", config, n_configs=len(X.SPECS),
    note="H2 dynamic exposure allocation x 30 exit variants, post-hoc historical descriptive run")

manifest = json.loads((Path("data/raw") / snapshot / "r15_manifest.json").read_text())
members = {m: v for m, v in manifest["members"].items() if m >= f"{START.year}-{START.month:02d}"}
symbols = sorted({s for vals in members.values() for s in vals})
btc_raw = _load(snapshot, "spot", "BTCUSDT", None)
btc = V.leader_moves(btc_raw)
gate = H.ma_gate(btc.c)
assets = {s: G.Asset.build(_load(snapshot, "perp", s, None)) for s in symbols}
s0 = int(epoch_15m(START))

results = {}
for i, spec in enumerate(X.SPECS, 1):
    r = X.run_dynamic(spec, assets, btc, members, s0, gate, target_gross, max_coin_weight, cost)
    results[spec.name] = {k: v for k, v in r.items() if k != "daily"}
    print(f"{i}/{len(X.SPECS)} {spec.name}: Sharpe={r['sharpe']:.3f}, CAGR={r['cagr']:.3%}, "
          f"DD={r['max_dd']:.3%}, avg gross={r['avg_gross_exposure']:.1%}", flush=True)

fixed_path = Path("runs/r18_exit_variants_history.json")
fixed = json.loads(fixed_path.read_text())["results"] if fixed_path.exists() else {}
comparisons = {}
for name, dyn in results.items():
    if name in fixed:
        comparisons[name] = {metric: {"fixed_10pct": fixed[name].get(metric), "dynamic": dyn.get(metric)}
                             for metric in ("sharpe", "cagr", "max_dd", "total_return")}

out = {"study": "research18_h2_dynamic_allocation", "snapshot": snapshot, "commit": current_commit(),
       "period": [str(START), str(END)], "n_symbols": len(symbols), "n_variants": len(X.SPECS),
       "allocation": {"target_gross": target_gross, "max_coin_weight": max_coin_weight,
                      "formula": "min(0.40, 0.80 / open_positions)", "cost_per_side": cost,
                      "cost_on": "entry, exit, and rebalancing turnover"},
       "note": "Post-hoc historical descriptive results; do not alter or decide the H2 forward test.",
       "results": results, "vs_fixed_10pct": comparisons}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, separators=(",", ":"), default=float))
print(f"saved {out_path}")
