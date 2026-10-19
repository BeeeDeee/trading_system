"""Post-hoc descriptive BTC spot buy-and-hold / H2 portfolio mix backtest.

    uv run python scripts/r18_btc_h2_mixes_backtest.py binance_2026-10-03 runs/r18_btc_h2_mixes_history.json

The complete mix grid is recorded before any price panels are loaded.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from qlab.research15.setup import _load, epoch_15m, to_dates
from qlab.research15.evaluate import by_year, stats
from qlab.research16 import engine as G
from qlab.research17 import variants as V
from qlab.research17.setup import END, START
from qlab.research18 import exits as X, rules as H
from qlab.validation.registry import TrialRegistry, current_commit


snapshot, out_path = sys.argv[1], Path(sys.argv[2])
weights = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4),
           (0.5, 0.5), (0.4, 0.6), (0.25, 0.75), (0.2, 0.8)]
config = {"study": "research18_btc_h2_mixes", "prereg": "docs/research18/BTC_H2_MIXES.md v0.1",
          "snapshot": snapshot, "period": [str(START), str(END)],
          "base_h2": "H2_btc200 / X00_time4h", "weights_btc_h2": weights,
          "rebalancing": "daily target weights"}
TrialRegistry("runs/registry/research18.sqlite").record(
    "candidates", config, n_configs=len(weights),
    note="Post-hoc descriptive BTC spot B&H + H2 target weight mix grid")

manifest = json.loads((Path("data/raw") / snapshot / "r15_manifest.json").read_text())
members = {m: v for m, v in manifest["members"].items() if m >= f"{START.year}-{START.month:02d}"}
symbols = sorted({s for vals in members.values() for s in vals})
btc_raw = _load(snapshot, "spot", "BTCUSDT", None)
btc = V.leader_moves(btc_raw)
gate = H.ma_gate(btc.c)
assets = {s: G.Asset.build(_load(snapshot, "perp", s, None)) for s in symbols}
s0 = int(epoch_15m(START))

# H2 and BTC series are aligned on the same fixed UTC-day grid.
h2_result = X.run(X.SPECS[0], assets, btc, members, s0, gate)
btc_t = btc_raw["t"]
btc_days = (btc_t // 96)
# Last candle close of each day; spot BTC is continuously traded in this panel.
_, reverse_idx = np.unique(btc_days[::-1], return_index=True)
last_idx = len(btc_days) - 1 - reverse_idx
last_idx.sort()
day_ids = btc_days[last_idx]
close_by_day = btc.c[last_idx]
grid_start_day = int(s0 // 96)
grid_days = np.arange(grid_start_day, grid_start_day + len(h2_result["daily"]))
btc_daily = np.zeros(len(grid_days), dtype=float)
lookup = {int(d): float(c) for d, c in zip(day_ids, close_by_day)}
for i, d in enumerate(grid_days):
    prev, cur = lookup.get(int(d - 1)), lookup.get(int(d))
    if prev is not None and cur is not None and prev > 0:
        btc_daily[i] = cur / prev - 1

dates = (grid_days * 86400).astype("datetime64[s]").astype("datetime64[D]")
btc_stats = stats(btc_daily)
h2_stats = stats(h2_result["daily"], h2_result["rets"])
rows = {
    "BTC_BH": {**btc_stats, "total_return": float(np.prod(1 + btc_daily) - 1),
               "worst_year_return": min(by_year(dates, btc_daily).values(), key=lambda x: x["ret"])["ret"],
               "by_year": by_year(dates, btc_daily)},
    "H2_btc200": {**h2_stats, "total_return": float(np.prod(1 + h2_result["daily"]) - 1),
                  "worst_year_return": min(by_year(dates, h2_result["daily"]).values(), key=lambda x: x["ret"])["ret"],
                  "by_year": by_year(dates, h2_result["daily"])},
}
for wb, wh in weights:
    daily = wb * btc_daily + wh * h2_result["daily"]
    annual = by_year(dates, daily)
    rows[f"BTC{int(wb*100):02d}_H2{int(wh*100):02d}"] = {
        **stats(daily), "total_return": float(np.prod(1 + daily) - 1),
        "worst_year_return": min(annual.values(), key=lambda x: x["ret"])["ret"],
        "by_year": annual,
    }

out = {"study": "research18_btc_h2_mixes", "snapshot": snapshot, "commit": current_commit(),
       "period": [str(START), str(END)], "n_symbols": len(symbols), "n_mixes": len(weights),
       "note": "Post-hoc descriptive, selection-biased historical results; not OOS evidence and do not alter H2 forward.",
       "results": rows}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, separators=(",", ":"), default=float))
for name, result in rows.items():
    print(f"{name}: CAGR={result['cagr']:.2%}, maxDD={result['max_dd']:.2%}, Sharpe={result['sharpe']:.3f}, total={result['total_return']:.2%}")
print(f"saved {out_path}")
