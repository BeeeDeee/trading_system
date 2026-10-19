"""Interactive HTML report of the historical backtest of the 30 LEAD variants (one self-contained file).

    uv run python scripts/r17_variants_report.py runs/r17_variants_history.json \
        docs/research17/results/forward_<asof>.json docs/research17/results/variants_backtest.html

Served by the portal at /lead/backtest.html (linked from the LEAD forward tab). Descriptive, post hoc.
Commentary lives in `docs/research17/variants_backtest_comments.json` (section key -> HTML).
"""

import json
import sys
from pathlib import Path

import numpy as np

from qlab.research17.evaluate import SR_VAR
from qlab.validation.stats import deflated_sharpe, expected_max_sharpe, sharpe

hist = json.loads(Path(sys.argv[1]).read_text())
fwd = json.loads(Path(sys.argv[2]).read_text())
final = json.loads(Path("docs/research17/results/final.json").read_text())
cpath = Path("docs/research17/variants_backtest_comments.json")
comments = json.loads(cpath.read_text()) if cpath.exists() else {}
# second DSR: variance of the daily Sharpe across these 30 variants instead of the research 16 grid proxy
daily = {n: np.array(v["daily"]) for n, v in hist["variants"].items()}
var30 = float(np.var([sharpe(x) for x in daily.values()], ddof=1))
for n, v in hist["variants"].items():
    v["dsr_N77_var30"] = deflated_sharpe(daily[n], hist["N_CUM_before"] + 30, var30)
hist["dsr_alt"] = {"var30": var30, "emax77_var30": expected_max_sharpe(77, var30) * 365 ** 0.5,
                   "emax77_proxy": expected_max_sharpe(77, SR_VAR) * 365 ** 0.5}
data = {"hist": hist, "fwd": {"asof": fwd["asof"], "days": fwd["days"], "variants": fwd["variants"]},
        "final": {"sharpe": final["P_stats"]["sharpe"], "n_trades": final["P_stats"]["n_trades"],
                  "n_events": final["n_events"]}}

PAGE = Path(__file__).with_name("r17_variants_report.html").read_text()
html = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace(
    "__COMMENTS__", json.dumps(comments, ensure_ascii=False))
Path(sys.argv[3]).write_text(html)
print(f"{sys.argv[3]}: {len(html) / 1e6:.1f} MB")
