"""Research 12 §8: descriptive historical run (NOT a criterion; all history is consumed by research 1-11).
Walk-forward predictions of research 11 for 2010-01..2026-08; plain top 50 vs hysteresis 50/200.

    uv run python scripts/r12_history.py sharadar_2026-09-25     -> docs/research12/results/history.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from r11_common import stats  # noqa: E402

from qlab.engine.vector import Decisions, simulate  # noqa: E402
from qlab.pipeline import load_config  # noqa: E402
from qlab.research4.sleeves import cost_rates, pct_rank, stock_decisions  # noqa: E402
from qlab.research11.model import MODELS, design, forward_returns, walk_forward  # noqa: E402
from qlab.research11.setup import _load, scores  # noqa: E402
from qlab.research12.portfolio import KEEP_RANK, N_HOLD, decisions  # noqa: E402

snapshot = sys.argv[1]
p, extra, meta, days, _ = _load(snapshot, None)                  # full history: declared consumed (prereg §8)
days = days[p.dates[days] <= np.datetime64("2026-08-31")]
sc = scores(snapshot, "final", p, meta, days)
universe = np.asarray(extra["in_liq1000"][days])
cash = np.asarray(extra["cash_ret"])
rate = cost_rates(extra["liq_rank"], p.dates, load_config()["costs"], list(meta["etf"].values()))
years = list(range(2010, 2027))
fwd = forward_returns(p, days)
y = pct_rank(fwd, universe)
dd = np.asarray(p.dates[days], dtype="datetime64[D]")
test = np.flatnonzero(dd >= np.datetime64("2010-01-01"))
tdays = days[test]
s0 = tdays[0] + 1
end = min(int(tdays[-1]) + 22, p.shape[0])
cut = int(np.searchsorted(p.dates, np.datetime64("2020-01-01")))
periods = {"2010-2019": slice(s0, cut), "2020-2026": slice(cut, end), "2010-2026": slice(s0, end)}


def report(res) -> dict:
    out = {k: stats(res.returns[s]) for k, s in periods.items()}
    out["turnover_ann"] = float(res.turnover[s0:end].mean() * 252)
    out["costs_ann"] = float(res.costs[s0:end].mean() * 252)
    return out


res = {"note": "descriptive, post hoc; not a criterion (prereg §8)", "n_hold": N_HOLD, "keep_rank": KEEP_RANK,
       "runs": {}}
for name in ("M_P", "M_ALL"):
    X = design(sc, universe, MODELS[name])
    pred, _ = walk_forward(X, y, universe, p.dates, days, years)
    del X
    top = stock_decisions(pred[test], universe[test], tdays, N_HOLD)
    hyst, masks = decisions(pred[test], universe[test], tdays, N_HOLD, KEEP_RANK)
    res["runs"][f"{name}_top50"] = report(simulate(p, top, rate, cash))
    res["runs"][f"{name}_H"] = report(simulate(p, hyst, rate, cash))
    res["runs"][f"{name}_H"]["names_changed_per_month"] = float(np.abs(np.diff(masks.astype(int), axis=0)).sum() / 2
                                                                / (len(masks) - 1))
    print(name, "done", flush=True)
w = np.zeros((len(tdays), p.shape[1])); w[:, meta["etf"]["SPY"]] = 1.0
res["runs"]["SPY"] = report(simulate(p, Decisions(tdays, w), rate, cash))
ew = stock_decisions(np.where(universe[test], 1.0, np.nan), universe[test], tdays, 1000)
res["runs"]["EW_LIQ1000"] = report(simulate(p, ew, rate, cash))
Path("docs/research12/results").mkdir(parents=True, exist_ok=True)
Path("docs/research12/results/history.json").write_text(json.dumps(res, indent=1))
for k, v in res["runs"].items():
    print(f"{k:<12} turnover {v['turnover_ann']:5.1f}x  costs {v['costs_ann'] * 100:4.2f} %  " + "  ".join(
        f"{per}: CAGR {v[per]['cagr'] * 100:5.1f} % SR {v[per]['sharpe']:.2f} MDD {v[per]['max_dd'] * 100:4.1f} %"
        for per in periods) + (f"  changes/m {v['names_changed_per_month']:.1f}" if "names_changed_per_month" in v else ""))
