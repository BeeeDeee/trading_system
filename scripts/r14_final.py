"""Research 14, final holdout evaluation (prereg §8-9). Runs ONCE through the vault.

    uv run python scripts/r14_final.py binance_2026-10-03 freeze
    QLAB_R14_METHODOLOGY=<hash> uv run python scripts/r14_final.py binance_2026-10-03 run

Every run starts with an empty book at 2022-01-01 00:00 UTC (prereg §7); indicators use earlier data.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from qlab.research14 import evaluate as E, strategy as S
from qlab.research14.setup import (FINAL_ENV, HOLDOUT, SECONDARY, SUBPERIODS, epoch_hour, load_final,
                                   registry, vault)
from qlab.validation.registry import current_commit
from qlab.validation.stats import deflated_sharpe

N_RND = 500
snapshot, action = sys.argv[1], sys.argv[2]
FILES = ["docs/research14/PREREGISTRATION.md", "docs/research14/results/dev.json", "scripts/r14_final.py",
         *sorted(str(f) for f in Path("src/qlab/research14").glob("*.py"))]


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
registry().record("methodology_eval", {"study": "research14", "stage": "final", "hash": mh},
                  note="r14 one-time holdout")
dev = json.loads(Path("docs/research14/results/dev.json").read_text())
assets = {s: E.Asset.build(d) for s, d in load_final(snapshot).items()}
hours = next(iter(assets.values())).hours
start, end = int(epoch_hour(HOLDOUT[0]) - hours[0]), len(hours)
P = {c.id: c for c in S.grid()}[dev["P"]]

rp = E.run(assets, P, start, end)
sp = E.stats(rp.daily, rp.all_trades())
mean, lo, hi = E.trade_ci(rp.all_trades())
rnd = np.array([E.stats(E.run_random(assets, P, rp, start, end, seed).daily)["sharpe"] for seed in range(N_RND)])
nohtf = E.run(assets, P, start, end, nohtf=True)
s_nohtf = E.stats(nohtf.daily, nohtf.all_trades())
s_x2 = E.stats(E.run(assets, P, start, end, cost=2 * S.COST).daily)
sub = {k: E.stats(E.window(rp.dates, rp.daily, *v))["sharpe"] for k, v in SUBPERIODS.items()}
per_asset = {s: float(t[2].mean()) if len(t[2]) else float("nan") for s, t in rp.trades.items()}
dsr = deflated_sharpe(rp.daily, len(S.grid()), dev["sr_daily_var"])

criteria = {
    "1_sharpe_and_trade_ci": {"sharpe": sp["sharpe"], "mean_trade_ci90": [mean, lo, hi],
                              "pass": bool(sp["sharpe"] > 0 and mean > 0 and lo > 0)},
    "2_above_rnd_p95": {"sharpe": sp["sharpe"], "rnd_p95": float(np.quantile(rnd, 0.95)),
                        "pass": bool(sp["sharpe"] > np.quantile(rnd, 0.95))},
    "3_above_nohtf_trade": {"P": sp.get("mean_trade"), "B_NOHTF": s_nohtf.get("mean_trade"),
                            "pass": bool(sp.get("mean_trade", -1) > s_nohtf.get("mean_trade", 1))},
    "4_cost_x2": {"sharpe": s_x2["sharpe"], "pass": bool(s_x2["sharpe"] > 0)},
    "5_subperiods": {"sharpe": sub, "pass": bool(all(v > 0 for v in sub.values()))},
    "6_btc_and_eth_trade": {"mean_trade": per_asset, "pass": bool(all(v > 0 for v in per_asset.values()))},
}
htf_d, htf = E.run_htf(assets, P.side, start, end)
_, htf_ls = E.run_htf(assets, "LS", start, end)
_, hold = E.run_hold(assets, start, end)
sides = {"long": [], "short": []}
for a in assets.values():
    e, sd = S.pick_entries(S.signals(a.hours, a.close, a.real, P, a.reg, a.sig), P.H, start, end)
    _, tr = S.simulate(a.close, a.fund, e, sd, P.H, end)
    ok = np.isfinite(tr)
    sides["long"] += tr[ok & (sd == 1)].tolist()
    sides["short"] += tr[ok & (sd == -1)].tolist()
sides = {k: {"n": len(v), "mean": float(np.mean(v)) if v else None} for k, v in sides.items()}
sec_raw = load_final(snapshot, SECONDARY)
sec = {s: E.Asset.build(d) for s, d in sec_raw.items()}
rs = E.run(sec, P, start, end)
out = {
    "snapshot": snapshot, "commit": current_commit(), "methodology_hash": mh, "P": P.id,
    "criteria": criteria, "n_pass": int(sum(c["pass"] for c in criteria.values())),
    "P_stats": sp, "dsr": dsr,
    "B_RND": {"median": float(np.median(rnd)), "p95": float(np.quantile(rnd, 0.95)),
              "share_below_P": float((rnd < sp["sharpe"]).mean())},
    "B_NOHTF": s_nohtf, "B_HTF_same_side": E.stats(htf), "B_HTF_LS": E.stats(htf_ls), "B_HOLD": E.stats(hold),
    "robust": {
        "all_candidates": {c.id: E.stats((r := E.run(assets, c, start, end)).daily, r.all_trades())
                           for c in S.grid()},
        "by_year": E.by_year(rp.dates, rp.daily),
        "delay_1h": E.stats((r := E.run(assets, P, start, end, delay=1)).daily, r.all_trades()),
        "no_funding": E.stats((r := E.run(assets, P, start, end, use_funding=False)).daily, r.all_trades()),
        "sides": sides,
        "secondary_basket": E.stats(rs.daily, rs.all_trades()),
        "secondary_hold": E.stats(E.run_hold(sec, start, end)[1]),
    },
}
Path("docs/research14/results/final.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k != "robust"}, indent=1, default=float))
