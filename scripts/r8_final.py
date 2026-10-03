"""Research 8, final holdout evaluation (prereg §6-7). Runs ONCE through the vault.

    uv run python scripts/r8_final.py binance_2026-10-03 freeze     # write the methodology hash (no data)
    QLAB_R8_METHODOLOGY=<hash> uv run python scripts/r8_final.py binance_2026-10-03 run

The methodology hash covers the pre-registration, every research-8 module, this script and the dev
choice of P2. The simulation runs continuously from the dev start; holdout metrics use the returns of
2023-01-01..2026-09-30 only.
"""

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from qlab.research8 import report as R, strategy as S
from qlab.research8.setup import DEV, FINAL_ENV, HOLDOUT, SUBPERIODS, index_of, load_final, vault
from qlab.research8.sim import Costs, simulate
from qlab.validation.registry import current_commit
from qlab.validation.stats import deflated_sharpe, probabilistic_sharpe, stationary_bootstrap_indices

snapshot, action = sys.argv[1], sys.argv[2]
FILES = ["docs/research8/PREREGISTRATION.md", "docs/research8/results/dev.json", "scripts/r8_final.py",
         *sorted(str(f) for f in Path("src/qlab/research8").glob("*.py"))]


def methodology_hash() -> str:
    h = hashlib.sha256()
    for f in FILES:
        h.update(f.encode() + b"\0" + Path(f).read_bytes() + b"\0")
    return h.hexdigest()


if action == "freeze":
    mh = methodology_hash()
    vault().freeze(mh)
    print(f"frozen: {mh}")
    sys.exit(0)
if action != "run":
    raise SystemExit(__doc__)

mh = os.environ.get(FINAL_ENV, "")
if mh != methodology_hash():
    raise SystemExit("methodology changed since freeze (or env var missing)")
vault().open_final(mh, snapshot, current_commit())
p = load_final(snapshot)
dev = json.loads(Path("docs/research8/results/dev.json").read_text())
by_id = {c.id: c for c in S.grid()}
d0, h0 = index_of(p, DEV[0]), index_of(p, HOLDOUT[0])
H = slice(h0, None)


def run(cfg, costs=Costs(), shift=0, universe_n=None):
    if universe_n:
        cfg = replace(cfg, universe_n=universe_n)
    tg = {t + shift: w for t, w in S.targets(p, cfg).items() if t >= d0 and t + shift < len(p.dates)}
    return simulate(p, tg, s=cfg.s, costs=costs, liq_price=cfg.liq_price)


def boot_excess(ret, tb, n=1000, seed=0):
    ex = (ret - tb) * R.ANN
    rng = np.random.default_rng(seed)
    return np.array([ex[stationary_bootstrap_indices(len(ex), 21.0, rng)].mean() for _ in range(n)])


out = {"snapshot": snapshot, "commit": current_commit(), "methodology_hash": mh, "candidates": {}}
pvals = {}
for name, cid in (("P1", "S0"), ("P2", dev["P2"])):
    cfg = by_id[cid]
    base, stress = run(cfg), run(cfg, Costs(mult=2.0))
    tb = p.tbill[H]
    est, lo, hi = R.excess_ci(base.ret[H], tb)
    bs = boot_excess(base.ret[H], tb)
    pvals[name] = float((bs <= 0).mean())
    subs = {}
    for k, (a, b) in SUBPERIODS.items():
        sl = slice(index_of(p, a), index_of(p, b) + 1)
        subs[k] = R.summary(base.ret[sl], p.tbill[sl])
    ex = base.ret[H] - tb
    stat = (probabilistic_sharpe(ex) if name == "P1"
            else deflated_sharpe(ex, dev["n_trials"], dev["sr_variance_daily"]))
    rob = {}
    for label, kw in (("s_1/2", {"cfg": replace(cfg, s=0.5)}), ("s_3/4", {"cfg": replace(cfg, s=0.75)}),
                      ("fill_t+1", {"shift": 1}),
                      *((("universe_top10", {"universe_n": 10}), ("universe_top40", {"universe_n": 40}))
                        if cfg.family == "S2" else ())):
        r = run(kw.pop("cfg", cfg), **kw)
        rob[label] = R.summary(r.ret[H], tb)
    years = {}
    for y in range(2023, 2027):
        yi = np.array([str(d)[:4] == str(y) for d in p.dates])
        yi[:h0] = False
        if yi.any():
            years[str(y)] = R.summary(base.ret[yi], p.tbill[yi], base.funding[yi], base.costs[yi])
    hold_liq = [(str(p.dates[t]), p.symbols[j]) for t, j in base.liquidations if t >= h0]
    out["candidates"][name] = {
        "id": cid, "base": R.summary(base.ret[H], tb, base.funding[H], base.costs[H], base.turnover[H]),
        "stress": R.summary(stress.ret[H], tb), "excess_ci90": [est, lo, hi], "p_one_sided": pvals[name],
        "subperiods": subs, "psr_or_dsr": stat, "liquidations": hold_liq,
        "delistings": [(str(p.dates[t]), p.symbols[j]) for t, j in base.delistings if t >= h0],
        "years": years, "robustness": rob,
    }

# Holm over the two candidates for criterion 1 (one-sided bootstrap p < 0.05 <=> 90 % CI lower bound > 0)
order = sorted(pvals, key=pvals.get)
holm = {k: min(1.0, max(pvals[order[i]] * (len(order) - i) for i in range(order.index(k) + 1))) for k in pvals}
for name, c in out["candidates"].items():
    crit = {
        "1_excess_ci_holm": holm[name] < 0.05 and c["excess_ci90"][1] > 0,
        "2_both_subperiods": all(v["excess_ann"] > 0 for v in c["subperiods"].values()),
        "3_stress": c["stress"]["excess_ann"] > 0,
        "4_psr_dsr": c["psr_or_dsr"] >= 0.95,
    }
    c["criteria"], c["p_holm"], c["pass"] = crit, holm[name], all(crit.values())

# descriptive: trailing 7-day funding of BTC and ETH by quarter (premium compression)
sym = {s: j for j, s in enumerate(p.symbols)}
q = {}
for t in range(h0, len(p.dates)):
    key = f"{str(p.dates[t])[:4]}Q{(int(str(p.dates[t])[5:7]) - 1) // 3 + 1}"
    for c in ("BTCUSDT", "ETHUSDT"):
        q.setdefault(c, {}).setdefault(key, []).append(p.m["fund_sig"][t, sym[c]] * 365)
out["funding_by_quarter"] = {c: {k: float(np.nanmean(v)) for k, v in d.items()} for c, d in q.items()}
Path("docs/research8/results/final.json").write_text(json.dumps(out, indent=1))
for name, c in out["candidates"].items():
    b = c["base"]
    print(f"{name} {c['id']}: excess {b['excess_ann'] * 100:+.2f} % p.a. CI90 [{c['excess_ci90'][1] * 100:+.2f}, "
          f"{c['excess_ci90'][2] * 100:+.2f}]  SR {b['sharpe_excess']:.2f}  MDD {b['max_dd'] * 100:.1f} %  "
          f"liq {len(c['liquidations'])}  -> {'PASS' if c['pass'] else 'FAIL'} {c['criteria']}")
