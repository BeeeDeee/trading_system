"""Run the predeclared descriptive H2 exit grid on historical data.

    uv run python scripts/r18_exit_variants_backtest.py binance_2026-10-03 runs/r18_exit_variants_history.json

This is a contaminated, post-hoc historical look. It never changes the H2 forward decision.
The full grid is recorded before loading/processing price panels.
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
config = {"study": "research18_h2_exits", "prereg": "docs/research18/EXIT_VARIANTS.md v0.1",
          "snapshot": snapshot, "period": [str(START), str(END)], "variants": [asdict(s) for s in X.SPECS]}
TrialRegistry("runs/registry/research18.sqlite").record(
    "candidates", config, n_configs=len(X.SPECS),
    note="H2 TP/SL/trailing exit grid, post-hoc historical descriptive run")

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
    r = X.run(spec, assets, btc, members, s0, gate)
    results[spec.name] = {k: v for k, v in r.items() if k not in ("daily", "rets", "events")}
    results[spec.name]["n_events"] = int(len(set(r["events"].tolist())))
    results[spec.name]["total_return"] = float((1 + r["daily"]).prod() - 1)
    print(f"{i}/{len(X.SPECS)} {spec.name}: Sharpe={r['sharpe']:.3f}, trades={r['n_trades']}", flush=True)

out = {"study": "research18_h2_exits", "snapshot": snapshot, "commit": current_commit(),
       "period": [str(START), str(END)], "n_symbols": len(symbols), "n_variants": len(X.SPECS),
       "note": "Post-hoc historical descriptive results; do not alter or decide the H2 forward test.",
       "results": results}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, separators=(",", ":"), default=float))
print(f"saved {out_path}")
