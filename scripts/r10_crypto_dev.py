"""Research 10, part C development (prereg §5.2): 8 long/short candidates on 2018-04-01..2021-12-31, choose P_C.

    uv run python scripts/r10_crypto_dev.py binance_2026-10-03     -> docs/research10/results/crypto_dev.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from r10_crypto_common import Runner, stats  # noqa: E402

from qlab.research10 import crypto as C  # noqa: E402
from qlab.research10.setup import C_DEV, index_of, load_crypto_dev, registry  # noqa: E402
from qlab.validation.stats import sharpe  # noqa: E402

snapshot = sys.argv[1]
p, extra, syms, fund = load_crypto_dev(snapshot)
i0 = index_of(p.dates, C_DEV[0])
R = Runner(p, extra, syms, fund, first=i0 - 1)
sl = slice(i0, None)
hold = R.hold()
out = {"stage": "dev", "period": [str(x) for x in C_DEV], "HOLD": stats(hold[sl]),
       "funding_days_in_dev": int((fund[sl, [R.btc, R.eth]] != 0).any(axis=1).sum()), "LC": {}, "candidates": {}}
for n in (20, 50, 100, 200):
    out["LC"][f"LC_{n}"] = stats(R.lc(n)[sl])
reg = registry()
act = {}
for cfg in C.grid():
    if not reg.has(cfg.as_dict()):
        reg.record("candidates", cfg.as_dict(), note=f"r10 C dev {cfg.id}")
    r = R.run(cfg)
    s = stats(r[sl])
    s.update({"sharpe_diff_hold": s["sharpe"] - out["HOLD"]["sharpe"],
              "sharpe_diff_lc": s["sharpe"] - out["LC"][f"LC_{cfg.n}"]["sharpe"],
              "stress_cost_sharpe": stats(R.run(cfg, mult=2.0)[sl])["sharpe"],
              "stress_funding_sharpe": stats(R.run(cfg, stress_funding=True)[sl])["sharpe"]})
    act[cfg.id] = sharpe(r[sl] - hold[sl])
    out["candidates"][cfg.id] = {"config": cfg.as_dict(), **s}
    print(f"{cfg.id:<7} CAGR {s['cagr'] * 100:+7.1f} %  SR {s['sharpe']:5.2f}  ΔSR hold {s['sharpe_diff_hold']:+5.2f}  "
          f"ΔSR LC {s['sharpe_diff_lc']:+5.2f}  MDD {s['max_dd'] * 100:5.1f} %  stress cost {s['stress_cost_sharpe']:.2f}  "
          f"stress funding {s['stress_funding_sharpe']:.2f}")
print(f"HOLD    CAGR {out['HOLD']['cagr'] * 100:+7.1f} %  SR {out['HOLD']['sharpe']:5.2f}  MDD {out['HOLD']['max_dd'] * 100:5.1f} %")
for k, v in out["LC"].items():
    print(f"{k:<7} CAGR {v['cagr'] * 100:+7.1f} %  SR {v['sharpe']:5.2f}  MDD {v['max_dd'] * 100:5.1f} %")
out["P_C"] = max(out["candidates"], key=lambda k: out["candidates"][k]["sharpe"])
out["sr_variance_active_daily"] = float(np.var(list(act.values()), ddof=1))
out["n_trials"] = reg.total_configs("candidates") - 12 + 26     # part C (8) + research 9 (26); part B excluded
Path("docs/research10/results/crypto_dev.json").write_text(json.dumps(out, indent=1))
print(f"\nP_C = {out['P_C']}, N for DSR = {out['n_trials']}, funding days in dev {out['funding_days_in_dev']}")
