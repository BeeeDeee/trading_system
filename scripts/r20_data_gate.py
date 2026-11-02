"""Research 20 data quality gate (prereg §3.3): Yahoo-built TR vs Sharadar over 2025-09-26 .. 2026-09-25.

Usage: python scripts/r20_data_gate.py   -> docs/research20/results/data_gate.json (no strategy is computed)
"""

import json
import time
from datetime import datetime, timezone

import numpy as np

from qlab.research19 import setup as S19
from qlab.research20 import data as D
from qlab.research20 import setup as S

p, _, cols = S19.load_etf()
lo, hi = (np.datetime64(d, "D") for d in S.GATE_WINDOW)
win = (p.dates >= lo) & (p.dates <= hi)
out, raw = {}, {}
for tk in S19.ALL_ETF:
    j = D.fetch_yahoo(tk)
    raw[tk] = j
    y = D.tr_bars(D.parse_yahoo(j))
    k = cols.index(tk)
    tr_s = (1 + np.asarray(p.ret_co[win, k])) * (1 + np.asarray(p.ret_oc[win, k])) - 1
    live = np.asarray(p.listed[win, k])
    m = (y["dates"] >= lo) & (y["dates"] <= hi)
    st = D.gate_stats(y["dates"][m], y["tr"][m], p.dates[win][live], tr_s[live])
    st["pass"] = bool(st["n_common"] > 0 and st["n_only_yahoo"] == 0 and st["n_only_sharadar"] == 0
                      and st["median_abs_bps"] <= S.GATE_MEDIAN_BPS and st["p99_abs_bps"] <= S.GATE_P99_BPS
                      and st["max_abs_bps"] <= S.GATE_MAX_BPS and abs(st["cum_diff_pct"]) <= S.GATE_CUM_PCT)
    out[tk] = st
    time.sleep(0.5)
res = {"window": [str(lo), str(hi)], "thresholds": {"median_bps": S.GATE_MEDIAN_BPS, "p99_bps": S.GATE_P99_BPS,
       "max_bps": S.GATE_MAX_BPS, "cum_pct": S.GATE_CUM_PCT}, "tickers": out,
       "passed": all(v["pass"] for v in out.values()), "created_at": datetime.now(timezone.utc).isoformat()}
S.RESULTS.mkdir(parents=True, exist_ok=True)
(S.RESULTS / "data_gate.json").write_text(json.dumps(res, indent=1))
print(f"{'tk':5} {'n':>4} {'onlyY':>5} {'onlyS':>5} {'med':>6} {'p99':>7} {'max':>7} {'cum%':>7}  pass  worst")
for tk, v in out.items():
    print(f"{tk:5} {v['n_common']:4d} {v['n_only_yahoo']:5d} {v['n_only_sharadar']:5d} {v['median_abs_bps']:6.2f} "
          f"{v['p99_abs_bps']:7.2f} {v['max_abs_bps']:7.2f} {v['cum_diff_pct']:7.3f}  {v['pass']!s:5} {v['worst_dates']}")
print("GATE PASSED" if res["passed"] else "GATE FAILED")
