"""Research 22 gate G0 (integrity) on the DEV stage of a run; reads no validation / late data.

    uv run python scripts/r22_g0.py --run A

1. Parity of the weekly portfolio with `engine.vector.simulate` for the three canonical strategies at 1x costs: NAV
   after the open trade, turnover and number of positions (intraday returns are zeroed in a copy so that the engine NAV
   at a close equals the NAV after the open trade);
2. the holding-week return equals the exact product of the daily returns on 2 000 random real (week, name) cells;
3. delisted names occur in the universe and are flagged as paid out.
Writes docs/research22/results/g0_<run>.json. G0 passes if all three hold; a failure means fix, not result.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from qlab.data.panel import load_panel
from qlab.engine.vector import Decisions, simulate
from qlab.research22 import setup
from qlab.research22.compact import build_compact
from qlab.research22.grammar import Evaluator
from qlab.research22.late import CANONICAL

PARITY_TOL = 1e-9
EXACT_TOL = 1e-9


def select_weights(c, score, topk):
    w = np.zeros((c.shape[0], c.n_assets))
    for i in range(c.shape[0]):
        s = np.where(c.valid[i] & np.isfinite(score[i]), score[i], -np.inf)
        k = min(topk, int(np.isfinite(s).sum()))
        w[i, c.idx[i, np.argsort(-s, kind="stable")[:k]]] = 1.0 / k
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["A", "B"], default="A")
    args = ap.parse_args()
    st = setup.RUNS[args.run]["dev"]
    panel, extra = load_panel(setup.panel_dir(), mmap=True)
    first, last = setup.stage_rows(panel.dates, st)
    out: dict = {"run": args.run, "stage": "dev", "parity_tol": PARITY_TOL, "exact_tol": EXACT_TOL}

    # 1. parity with the vector engine (intraday returns zeroed in a copy)
    sliced = panel.slice(last + 1)
    zero_oc = replace(sliced, ret_oc=np.zeros(sliced.shape), ret_co=np.nan_to_num(np.asarray(sliced.ret_co)))
    c0 = build_compact(zero_oc, extra, first, last, st.embargo, setup.COST_MODEL)
    ev0 = Evaluator(c0)
    parity = {}
    for name, tree in CANONICAL.items():
        score = ev0.score(tree)
        weekly = c0.weekly(score, setup.TOPK, 1.0)
        cost_rate = np.zeros(sliced.shape)
        cost_rate[c0.dec_rows + 1] = c0.rate
        res = simulate(zero_oc, Decisions(c0.dec_rows, select_weights(c0, score, setup.TOPK)), cost_rate)
        rows = c0.dec_rows + 1
        parity[name] = {"max_rel_nav_diff": float(np.abs(res.nav[rows] / weekly.nav_trade - 1.0).max()),
                        "max_turnover_diff": float(np.abs(res.turnover[rows] - weekly.turnover).max()),
                        "positions_equal": bool((res.n_positions[rows] == weekly.n_positions).all()),
                        "weeks_with_stuck_positions": int((weekly.n_positions > setup.TOPK).sum())}
    out["parity"] = parity
    out["parity_ok"] = all(v["max_rel_nav_diff"] <= PARITY_TOL and v["max_turnover_diff"] <= PARITY_TOL
                           and v["positions_equal"] for v in parity.values())

    # 2. exact holding return on real cells (with the real intraday returns)
    c = setup.load_compact(args.run, "dev")
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(2000):
        i = int(rng.integers(c.shape[0]))
        j = int(rng.integers(int(c.valid[i].sum())))
        a, f1, g = int(c.idx[i, j]), int(c.dec_rows[i]) + 1, int(c.dec_rows[i + 1]) if i + 1 < c.shape[0] else None
        if g is None:
            continue
        ro, rc = np.nan_to_num(np.asarray(panel.ret_oc[f1: g + 1, a])), np.nan_to_num(np.asarray(panel.ret_co[f1: g + 2, a]))
        p = 1.0 + ro[0]
        for d in range(1, g - f1 + 1):
            p *= (1.0 + rc[d]) * (1.0 + ro[d])
        p *= 1.0 + rc[g - f1 + 1]
        errs.append(abs(c.wk[i, j] - (p - 1.0)))
    out["exact_holding_return"] = {"n_cells": len(errs), "max_abs_err": float(max(errs))}
    out["exact_ok"] = bool(max(errs) <= EXACT_TOL)

    # 3. delisted names are in the universe and flagged
    dead_assets = np.unique(c.idx[c.dead & c.valid])
    out["delisted_in_universe"] = {"n_assets": int(len(dead_assets)), "n_flags": int((c.dead & c.valid).sum()),
                                   "worst_terminal_week_return": float(c.wk[c.dead & c.valid].min())}
    out["delisted_ok"] = bool(len(dead_assets) > 0)

    out["passed"] = bool(out["parity_ok"] and out["exact_ok"] and out["delisted_ok"])
    path = setup.RESULTS / f"g0_{args.run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
