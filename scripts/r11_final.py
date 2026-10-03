"""Research 11, final stage 2020-01..2026-08 (prereg §6, §8). Runs ONCE through the vault.

    uv run python scripts/r11_final.py sharadar_2026-09-25 freeze
    QLAB_R11_METHODOLOGY=<hash> uv run python scripts/r11_final.py sharadar_2026-09-25 run
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from r11_common import ANN, evaluate, stats, yearly  # noqa: E402

from qlab.research11.setup import FINAL_ENV, FINAL_YEARS, SUBPERIODS, load_final, scores, vault  # noqa: E402
from qlab.validation.registry import current_commit  # noqa: E402
from qlab.validation.stats import deflated_sharpe, sharpe, stationary_bootstrap_indices  # noqa: E402

snapshot, action = sys.argv[1], sys.argv[2]
FILES = ["docs/research11/PREREGISTRATION.md", "docs/research11/results/dev.json", "scripts/r11_final.py",
         "scripts/r11_common.py", *sorted(str(f) for f in Path("src/qlab/research11").glob("*.py"))]


def methodology_hash() -> str:
    h = hashlib.sha256()
    for f in FILES:
        h.update(f.encode() + b"\0" + Path(f).read_bytes() + b"\0")
    return h.hexdigest()


if action == "freeze":
    print(f"frozen: {(mh := methodology_hash())}")
    vault().freeze(mh)
    sys.exit(0)
if action != "run":
    raise SystemExit(__doc__)
mh = os.environ.get(FINAL_ENV, "")
if mh != methodology_hash():
    raise SystemExit("methodology changed since freeze (or env var missing)")
vault().open_final(mh, snapshot, current_commit())
p, extra, meta, days, _ = load_final(snapshot)
days = days[p.dates[days] <= np.datetime64("2026-08-31")]          # last decision 2026-08 (prereg §6)
dev = json.loads(Path("docs/research11/results/dev.json").read_text())
sc = scores(snapshot, "final", p, meta, days)
ev = evaluate(p, extra, meta, days, sc, FINAL_YEARS)
s0 = ev["start"]
end = int(days[-1]) + 1 + 21                                          # hold the last decision ~1 month
end = min(end, p.shape[0])
H = slice(s0, end)
R = {k: (r[H], r2[H]) for k, (r, r2) in ev["returns"].items()}
a, spy, ew, mp = R["M_ALL"][0], R["SPY"][0], R["EW_LIQ1000"][0], R["M_P"][0]
rng = np.random.default_rng(0)
x = np.column_stack([a, spy])
boots = [(sharpe(y[:, 0]) - sharpe(y[:, 1])) * ANN
         for y in (x[stationary_bootstrap_indices(len(x), 21.0, rng)] for _ in range(1000))]
lo, hi = np.quantile(boots, [0.05, 0.95])
dsr = lambda n: float(deflated_sharpe(a - spy, n, dev["sr_variance_active_daily"]))  # noqa: E731
d = lambda u, v: float((sharpe(u) - sharpe(v)) * ANN)                                # noqa: E731
dates = p.dates[H]
subs = {}
for k, (s_, e_) in SUBPERIODS.items():
    m = (dates >= np.datetime64(s_)) & (dates <= np.datetime64(e_))
    subs[k] = d(a[m], spy[m])
ic_all, ic_p = ev["models"]["M_ALL"]["ic"]["mean"], ev["models"]["M_P"]["ic"]["mean"]
crit = {"1_vs_spy_ci": d(a, spy) > 0 and lo > 0, "2_vs_ew": d(a, ew) > 0,
        "3_vs_price_only": d(a, mp) > 0 and ic_all > ic_p, "4_both_subperiods": all(v > 0 for v in subs.values()),
        "5_stress": d(R["M_ALL"][1], R["SPY"][1]) > 0, "6_dsr": dsr(3) >= 0.95}
res = {"snapshot": snapshot, "commit": current_commit(), "methodology_hash": mh, "period": [str(dates[0]), str(dates[-1])],
       "perf": {k: {**stats(r), "stress_sharpe": stats(r2)["sharpe"]} for k, (r, r2) in R.items()},
       "models": ev["models"], "sharpe_diff_spy": d(a, spy), "ci90": [float(lo), float(hi)],
       "sharpe_diff_ew": d(a, ew), "sharpe_diff_price_only": d(a, mp), "subperiods_sharpe_diff_spy": subs,
       "dsr_n3": dsr(3), "dsr_n3027": dsr(3027), "criteria": crit, "pass": all(crit.values()),
       "yearly": {k: yearly(np.r_[np.zeros(s0), r], p.dates[:end], s0) for k, (r, _) in R.items()}}
Path("docs/research11/results/final.json").write_text(json.dumps(res, indent=1, default=float))
for k, v in res["perf"].items():
    print(f"{k:<11} CAGR {v['cagr'] * 100:5.1f} %  SR {v['sharpe']:.2f}  MDD {v['max_dd'] * 100:5.1f} %  stress SR {v['stress_sharpe']:.2f}")
for k, v in res["models"].items():
    print(f"{k}: IC {v['ic']['mean']:+.4f} (t {v['ic']['t']:.2f})  turnover {v['turnover_ann']:.1f}x  costs {v['costs_ann'] * 100:.2f} %")
print(f"ΔSR vs SPY {res['sharpe_diff_spy']:+.2f} CI90 [{lo:+.2f}, {hi:+.2f}]  vs EW {res['sharpe_diff_ew']:+.2f}  "
      f"vs M_P {res['sharpe_diff_price_only']:+.2f}  subperiods {subs}  DSR {res['dsr_n3']:.3f}")
for y in res["yearly"]["SPY"]:
    print(f"  {y} " + "  ".join(f"{k} {res['yearly'][k][y] * 100:+6.1f} %" for k in res["yearly"]))
print("criteria:", crit, "PASS" if res["pass"] else "FAIL")
