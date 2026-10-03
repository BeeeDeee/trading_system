"""Research 10, part C holdout (prereg §5.2, §7). Runs ONCE through the vault.

    uv run python scripts/r10_crypto_final.py binance_2026-10-03 freeze
    QLAB_R10_METHODOLOGY=<hash> uv run python scripts/r10_crypto_final.py binance_2026-10-03 run
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from r10_crypto_common import ANN, Runner, stats  # noqa: E402

from qlab.research10 import crypto as C  # noqa: E402
from qlab.research10.setup import (C_DEV, C_HOLDOUT, C_SUBPERIODS, FINAL_ENV, crypto_vault, index_of,  # noqa: E402
                                   load_crypto_final)
from qlab.validation.registry import current_commit  # noqa: E402
from qlab.validation.stats import deflated_sharpe, sharpe, stationary_bootstrap_indices  # noqa: E402

snapshot, action = sys.argv[1], sys.argv[2]
FILES = ["docs/research10/PREREGISTRATION.md", "docs/research10/results/crypto_dev.json", "scripts/r10_crypto_final.py",
         "scripts/r10_crypto_common.py", *sorted(str(f) for f in Path("src/qlab/research10").glob("*.py"))]


def methodology_hash() -> str:
    h = hashlib.sha256()
    for f in FILES:
        h.update(f.encode() + b"\0" + Path(f).read_bytes() + b"\0")
    return h.hexdigest()


if action == "freeze":
    print(f"frozen: {(mh := methodology_hash())}")
    crypto_vault().freeze(mh)
    sys.exit(0)
if action != "run":
    raise SystemExit(__doc__)
mh = os.environ.get(FINAL_ENV, "")
if mh != methodology_hash():
    raise SystemExit("methodology changed since freeze (or env var missing)")
crypto_vault().open_final(mh, snapshot, current_commit())
p, extra, syms, fund = load_crypto_final(snapshot)
dev = json.loads(Path("docs/research10/results/crypto_dev.json").read_text())
R = Runner(p, extra, syms, fund, first=index_of(p.dates, C_DEV[0]) - 1)
H = slice(index_of(p.dates, C_HOLDOUT[0]), None)
cfg = {c.id: c for c in C.grid()}[dev["P_C"]]

r, hold, lc = R.run(cfg), R.hold(), R.lc(cfg.n)
rng = np.random.default_rng(0)
x = np.column_stack([r[H], hold[H]])
boots = []
for _ in range(1000):
    y = x[stationary_bootstrap_indices(len(x), 21.0, rng)]
    boots.append((sharpe(y[:, 0]) - sharpe(y[:, 1])) * ANN)
lo, hi = np.quantile(boots, [0.05, 0.95])
d_hold = (sharpe(r[H]) - sharpe(hold[H])) * ANN
d_lc = (sharpe(r[H]) - sharpe(lc[H])) * ANN
subs = {}
for k, (a, b) in C_SUBPERIODS.items():
    s = slice(index_of(p.dates, a), index_of(p.dates, b) + 1)
    subs[k] = float((sharpe(r[s]) - sharpe(hold[s])) * ANN)
stress_cost = (sharpe(R.run(cfg, mult=2.0)[H]) - sharpe(R.hold(2.0)[H])) * ANN
stress_fund = (sharpe(R.run(cfg, stress_funding=True)[H]) - sharpe(hold[H])) * ANN
dsr = deflated_sharpe(r[H] - hold[H], dev["n_trials"], dev["sr_variance_active_daily"])
crit = {"1_sharpe_diff_ci": bool(d_hold > 0 and lo > 0), "2_vs_long_cash": bool(d_lc > 0),
        "3_both_subperiods": all(v > 0 for v in subs.values()),
        "4_stress": bool(stress_cost > 0 and stress_fund > 0), "5_dsr": bool(dsr >= 0.95)}
years = {}
for y in range(2022, 2027):
    yi = np.array([str(d)[:4] == str(y) for d in p.dates])
    yi[:H.start] = False
    years[str(y)] = {"P_C": stats(r[yi]), "HOLD": stats(hold[yi]), "LC": stats(lc[yi])}
res = {"snapshot": snapshot, "commit": current_commit(), "methodology_hash": mh, "P_C": cfg.id,
       "strategy": stats(r[H]), "HOLD": stats(hold[H]), f"LC_{cfg.n}": stats(lc[H]),
       "sharpe_diff_hold": float(d_hold), "ci90": [float(lo), float(hi)], "sharpe_diff_lc": float(d_lc),
       "subperiods_sharpe_diff": subs, "stress_cost_sharpe_diff": float(stress_cost),
       "stress_funding_sharpe_diff": float(stress_fund), "dsr": float(dsr), "criteria": crit,
       "pass": all(crit.values()), "years": years,
       "all_candidates_holdout": {c.id: stats(R.run(c)[H]) for c in C.grid()},      # descriptive only
       "all_lc_holdout": {f"LC_{n}": stats(R.lc(n)[H]) for n in (20, 50, 100, 200)}}
Path("docs/research10/results/crypto_final.json").write_text(json.dumps(res, indent=1))
print(f"P_C {cfg.id}: SR {res['strategy']['sharpe']:.2f} vs HOLD {res['HOLD']['sharpe']:.2f} (Δ {d_hold:+.2f}, CI90 "
      f"[{lo:+.2f}, {hi:+.2f}]), vs LC_{cfg.n} {res[f'LC_{cfg.n}']['sharpe']:.2f} (Δ {d_lc:+.2f})")
print(f"CAGR {res['strategy']['cagr'] * 100:+.1f} % vs {res['HOLD']['cagr'] * 100:+.1f} %  MDD {res['strategy']['max_dd'] * 100:.1f} % "
      f"vs {res['HOLD']['max_dd'] * 100:.1f} %  subperiods {subs}  stress {stress_cost:+.2f}/{stress_fund:+.2f}  DSR {dsr:.3f}")
for k, v in res["all_candidates_holdout"].items():
    print(f"  {k:<7} SR {v['sharpe']:5.2f}  CAGR {v['cagr'] * 100:+6.1f} %  MDD {v['max_dd'] * 100:5.1f} %")
for k, v in res["all_lc_holdout"].items():
    print(f"  {k:<7} SR {v['sharpe']:5.2f}  CAGR {v['cagr'] * 100:+6.1f} %  MDD {v['max_dd'] * 100:5.1f} %")
for y, v in years.items():
    print(f"  {y}: P_C {v['P_C']['cagr'] * 100:+6.1f} %  HOLD {v['HOLD']['cagr'] * 100:+6.1f} %  LC {v['LC']['cagr'] * 100:+6.1f} %")
print("criteria:", crit, "PASS" if res["pass"] else "FAIL")
