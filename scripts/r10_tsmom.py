"""Research 10, part B (prereg §5.1): 50 % SPY + 50 % TSMOM long/short on 9 ETFs. Data <= 2019-12-31;
selection 2004-2012 (P_B = best LS by 50/50 Sharpe), check 2013-2019.

    uv run python scripts/r10_tsmom.py sharadar_2026-09-25     -> docs/research10/results/tsmom.json
"""

import json
import sys
from pathlib import Path

import numpy as np

from qlab.engine.vector import Decisions, simulate
from qlab.research10 import tsmom as B
from qlab.research10.setup import B_SELECT, A_CHECK, CHECK_SUBPERIODS, load_etf, registry, window
from qlab.research10.sim import monthly_mix, simulate_signed
from qlab.schedule import period_starts
from qlab.validation.metrics import max_drawdown
from qlab.validation.stats import sharpe

ANN = np.sqrt(252)
snapshot = sys.argv[1]
p, extra, tickers = load_etf(snapshot)
assert not p.delisting.any()
cash = np.asarray(extra["cash_ret"])
cols = [tickers.index(t) for t in B.UNIVERSE]
spy, ief = tickers.index("SPY"), tickers.index("IEF")
T, N = p.shape
first = int(np.flatnonzero(p.listed[:, spy])[0])


def stats(r: np.ndarray) -> dict:
    nav = np.cumprod(1 + r)
    return {"cagr": float(nav[-1] ** (252 / len(r)) - 1), "sharpe": float(sharpe(r) * ANN),
            "vol": float(r.std(ddof=1) * ANN), "max_dd": float(max_drawdown(nav)[0])}


def sub_sr(r: np.ndarray, d: np.ndarray) -> dict:
    return {k: float(sharpe(r[window(d, a, b)]) * ANN) for k, (a, b) in CHECK_SUBPERIODS.items()}


def monthly_targets(w: dict[int, float]) -> Decisions:
    days = np.flatnonzero(period_starts(p.dates, "M"))
    days = days[(days >= first) & (days < T - 1)]
    rows = np.zeros((len(days), N))
    for j, x in w.items():
        rows[:, j] = x
    return Decisions(days, rows)


spy_r = simulate(p, monthly_targets({spy: 1.0}), B.COST, cash).returns
b6040 = simulate(p, monthly_targets({spy: 0.6, ief: 0.4}), B.COST, cash).returns
spy_r2 = simulate(p, monthly_targets({spy: 1.0}), B.COST_STRESS, cash).returns
b6040_2 = simulate(p, monthly_targets({spy: 0.6, ief: 0.4}), B.COST_STRESS, cash).returns
borrow = np.full((T, N), -B.BORROW)


def run(cfg: B.Config, cost: float = B.COST):
    d, w = B.decisions(p, cash, cols, cfg)
    return simulate_signed(p.ret_co, p.ret_oc, d, w, cost, cash, borrow, p.tradable)


def combo(r_tsmom: np.ndarray, r_core: np.ndarray) -> np.ndarray:
    return monthly_mix(np.column_stack([r_core, r_tsmom]), np.array([0.5, 0.5]), p.dates)


reg = registry()
sel, chk = window(p.dates, *B_SELECT), window(p.dates, *A_CHECK)
res = {"selection": {"period": [str(x) for x in B_SELECT], "SPY": stats(spy_r[sel]), "60/40": stats(b6040[sel])},
       "check": {"period": [str(x) for x in A_CHECK]}, "candidates": {}}
runs = {}
for cfg in B.grid():
    if not reg.has(cfg.as_dict()):
        reg.record("candidates", cfg.as_dict(), note=f"r10 B {cfg.id}")
    s = run(cfg)
    c = combo(s.returns, spy_r)
    runs[cfg.id] = (s, c)
    res["candidates"][cfg.id] = {
        "config": cfg.as_dict(), "tsmom_selection": stats(s.returns[sel]), "combo_selection": stats(c[sel]),
        "tsmom_check": stats(s.returns[chk]), "combo_check": stats(c[chk]),
        "corr_spy_selection": float(np.corrcoef(s.returns[sel], spy_r[sel])[0, 1]),
        "mean_net": float(s.net[sel].mean()), "mean_gross": float(s.gross[sel].mean()),
        "turnover_ann": float(s.turnover[sel].mean() * 252), "costs_ann": float(s.costs[sel].mean() * 252)}
    v = res["candidates"][cfg.id]
    print(f"{cfg.id:<10} TSMOM sel SR {v['tsmom_selection']['sharpe']:5.2f} CAGR {v['tsmom_selection']['cagr'] * 100:5.1f} %  "
          f"corr SPY {v['corr_spy_selection']:+.2f}  net {v['mean_net']:+.2f}  | 50/50 sel SR {v['combo_selection']['sharpe']:.2f} "
          f"MDD {v['combo_selection']['max_dd'] * 100:5.1f} %  | check: TSMOM SR {v['tsmom_check']['sharpe']:5.2f}  "
          f"50/50 SR {v['combo_check']['sharpe']:.2f}")

ls = [c for c in B.grid() if c.kind == "LS"]
pb = max(ls, key=lambda c: res["candidates"][c.id]["combo_selection"]["sharpe"])
plo = B.Config("LO", pb.L, pb.weighting)
d_chk = p.dates[chk]
cb, clo = runs[pb.id][1][chk], runs[plo.id][1][chk]
cb2 = combo(run(pb, B.COST_STRESS).returns, spy_r2)[chk]
st, sspy, s6040 = stats(cb), stats(spy_r[chk]), stats(b6040[chk])
crit = {"1_sr_vs_spy": st["sharpe"] > sspy["sharpe"], "2_sr_vs_6040": st["sharpe"] > s6040["sharpe"],
        "3_sr_vs_LO_combo": st["sharpe"] > stats(clo)["sharpe"],
        "4_stress_sr_vs_6040": stats(cb2)["sharpe"] > stats(b6040_2[chk])["sharpe"],
        "5_mdd_vs_spy": st["max_dd"] < sspy["max_dd"]}
years = {}
for y in range(2004, 2020):
    yi = np.array([str(x)[:4] == str(y) for x in p.dates])
    years[str(y)] = {"SPY": float(np.prod(1 + spy_r[yi]) - 1), "60/40": float(np.prod(1 + b6040[yi]) - 1),
                     "P_B_tsmom": float(np.prod(1 + runs[pb.id][0].returns[yi]) - 1),
                     "P_B_combo": float(np.prod(1 + runs[pb.id][1][yi]) - 1),
                     "P_LO_tsmom": float(np.prod(1 + runs[plo.id][0].returns[yi]) - 1)}
res["check"].update({"P_B": pb.id, "P_LO": plo.id, "combo": {**st, "subperiods_sharpe": sub_sr(cb, d_chk)},
                     "combo_LO": stats(clo), "SPY": {**sspy, "subperiods_sharpe": sub_sr(spy_r[chk], d_chk)},
                     "60/40": {**s6040, "subperiods_sharpe": sub_sr(b6040[chk], d_chk)},
                     "stress_combo": stats(cb2), "stress_60/40": stats(b6040_2[chk]),
                     "criteria": crit, "pass": all(crit.values())})
res["years"] = years
res["n_trials"] = reg.total_configs("candidates")
Path("docs/research10/results/tsmom.json").write_text(json.dumps(res, indent=1))
print(f"\nselection SPY SR {res['selection']['SPY']['sharpe']:.2f}, 60/40 SR {res['selection']['60/40']['sharpe']:.2f}")
print(f"P_B = {pb.id} (control {plo.id})")
for k in ("combo", "combo_LO", "SPY", "60/40", "stress_combo", "stress_60/40"):
    v = res["check"][k]
    print(f"  check {k:<13} CAGR {v['cagr'] * 100:5.1f} %  SR {v['sharpe']:.2f}  MDD {v['max_dd'] * 100:5.1f} %")
print("years:")
for y, v in years.items():
    print(f"  {y} " + "  ".join(f"{k} {x * 100:+6.1f} %" for k, x in v.items()))
print("criteria:", crit, "PASS" if res["check"]["pass"] else "FAIL")
