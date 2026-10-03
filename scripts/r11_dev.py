"""Research 11, development stage (prereg §6): walk-forward 2010-2019 on data <= 2019-12-31.

    uv run python scripts/r11_dev.py sharadar_2026-09-25     -> docs/research11/results/dev.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from r11_common import evaluate, stats, yearly  # noqa: E402

from qlab.research11.model import MODELS  # noqa: E402
from qlab.research11.setup import DEV_YEARS, load_dev, registry, scores  # noqa: E402
from qlab.validation.stats import sharpe  # noqa: E402

snapshot = sys.argv[1]
p, extra, meta, days, _ = load_dev(snapshot)
sc = scores(snapshot, "dev", p, meta, days)
print(f"dev panel to {p.dates[-1]}, {len(days)} decisions; features {len(sc)}", flush=True)
reg = registry()
for name, feats in MODELS.items():
    cfg = {"study": "research11", "model": name, "features": feats}
    if not reg.has(cfg):
        reg.record("candidates", cfg, note=f"r11 {name}")
ev = evaluate(p, extra, meta, days, sc, DEV_YEARS)
s0 = ev["start"]
s14 = int(np.searchsorted(p.dates, np.datetime64("2014-01-02")))
res = {"stage": "dev", "years": DEV_YEARS, "models": ev["models"], "perf": {}, "perf_2014_2019": {}, "yearly": {}}
for k, (r, r2) in ev["returns"].items():
    res["perf"][k] = {**stats(r[s0:]), "stress_sharpe": stats(r2[s0:])["sharpe"]}
    res["perf_2014_2019"][k] = stats(r[s14:])
    res["yearly"][k] = yearly(r, p.dates, s0)
spy = ev["returns"]["SPY"][0]
act = [sharpe(ev["returns"][m][0][s0:] - spy[s0:]) for m in MODELS]
res["sr_variance_active_daily"] = float(np.var(act, ddof=1))
res["n_trials"] = registry().total_configs("candidates")
Path("docs/research11/results").mkdir(parents=True, exist_ok=True)
Path("docs/research11/results/dev.json").write_text(json.dumps(res, indent=1))
for k in res["perf"]:
    a, b = res["perf"][k], res["perf_2014_2019"][k]
    print(f"{k:<11} 2010-19 CAGR {a['cagr'] * 100:5.1f} %  SR {a['sharpe']:.2f}  MDD {a['max_dd'] * 100:5.1f} %  "
          f"stress SR {a['stress_sharpe']:.2f} | 2014-19 CAGR {b['cagr'] * 100:5.1f} %  SR {b['sharpe']:.2f}")
for k, v in res["models"].items():
    print(f"{k}: IC {v['ic']['mean']:+.4f} (t {v['ic']['t']:.2f}, {v['ic']['share_pos'] * 100:.0f} % pos)  "
          f"turnover {v['turnover_ann']:.1f}x  costs {v['costs_ann'] * 100:.2f} %  IC by year "
          + " ".join(f"{y}:{x:+.3f}" for y, x in v["ic_by_year"].items()))
    last = max(v["importance_by_group"])
    print(f"   importance {last}: {v['importance_by_group'][last]}")
print("yearly:")
for y in res["yearly"]["SPY"]:
    print(f"  {y} " + "  ".join(f"{k} {res['yearly'][k][y] * 100:+6.1f} %" for k in res["yearly"]))
