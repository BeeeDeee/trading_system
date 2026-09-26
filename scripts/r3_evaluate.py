"""Research 3: passive portfolio rules (docs/research3/PREREGISTRATION.md).

Usage: python scripts/r3_evaluate.py sharadar_YYYY-MM-DD
Output: data/derived/<snapshot>/research3/results.parquet + summary.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from qlab.data.panel import load_panel
from qlab.engine.vector import Decisions, simulate
from qlab.schedule import period_starts
from qlab.validation.metrics import max_drawdown, summary
from qlab.validation.stats import stationary_bootstrap_indices

COST = 5e-4
N_BOOT = 2000
ALPHA = 0.10

snapshot = sys.argv[1]
t0 = time.time()
der = Path("data/derived") / snapshot
panel, extra = load_panel(der / "panel_etf", mmap=False)
col = {t: i for i, t in enumerate(json.loads((der / "panel_etf" / "tickers.json").read_text())["columns"])}
cash = np.asarray(extra["cash_ret"])
dates = panel.dates
c2c = (1 + panel.ret_co) * (1 + panel.ret_oc) - 1


def schedule(rule: str, start: int) -> np.ndarray:
    months = period_starts(dates, "M")
    m = np.asarray(dates, dtype="datetime64[M]").astype(int) % 12  # 0 = January
    days = {"monthly": months, "quarterly": months & (m % 3 == 0), "annual": months & (m == 0),
            "never": np.zeros(len(dates), bool)}[rule].copy()
    days[:start] = False
    days[start] = True
    return days


def band_days(target: dict[int, float], band: float, start: int) -> np.ndarray:
    """Decide after any close where a weight drifted more than `band` from target (past data only)."""
    days = np.zeros(len(dates), bool)
    days[start] = True
    cols = list(target)
    tw = np.array([target[c] for c in cols])
    w = tw.copy()
    for t in range(start + 1, len(dates)):
        w = w * (1 + c2c[t, cols])
        w = w / w.sum()
        if np.abs(w - tw).max() > band:
            days[t] = True
            w = tw.copy()
    return days


def run(target: dict[str, float], start: int, rule: str = "monthly") -> np.ndarray:
    tgt = {col[k]: v for k, v in target.items()}
    days = (band_days(tgt, float(rule[4:]) / 100, start) if rule.startswith("band")
            else schedule(rule, start))
    idx = np.flatnonzero(days)
    w = np.zeros((len(idx), panel.shape[1]))
    for c, x in tgt.items():
        w[:, c] = x
    return simulate(panel, Decisions(idx, w), COST, cash)


def compare(r: np.ndarray, ref: np.ndarray, sl: slice, seed: int = 0) -> dict:
    """Sharpe / CAGR difference vs reference with stationary-bootstrap CI and two-sided p."""
    x, y, f = r[sl], ref[sl], cash[sl]
    n = len(x)

    def stats(ix):
        ex, ey = x[ix] - f[ix], y[ix] - f[ix]
        sx = ex.mean() / ex.std(ddof=1) * np.sqrt(252) if ex.std() > 0 else 0.0
        sy = ey.mean() / ey.std(ddof=1) * np.sqrt(252)
        gx = np.exp(np.log1p(x[ix]).sum() * 252 / n) - 1
        gy = np.exp(np.log1p(y[ix]).sum() * 252 / n) - 1
        return sx - sy, gx - gy

    point = stats(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = np.array([stats(stationary_bootstrap_indices(n, 21, rng)) for _ in range(N_BOOT)])
    lo, hi = np.quantile(boots, [ALPHA / 2, 1 - ALPHA / 2], axis=0)
    p = 2 * min((boots[:, 0] <= 0).mean(), (boots[:, 0] >= 0).mean())
    return {"d_sharpe": point[0], "d_sharpe_lo": lo[0], "d_sharpe_hi": hi[0],
            "d_cagr": point[1], "d_cagr_lo": lo[1], "d_cagr_hi": hi[1], "p_sharpe": min(p, 1.0)}


REF = {"SPY": 0.6, "IEF": 0.4}
QUESTIONS = {
    "Q1 rebalance": ("2002-08-01", [("monthly (ref)", REF, "monthly"), ("quarterly", REF, "quarterly"),
                                    ("annual", REF, "annual"), ("never", REF, "never"),
                                    ("band 5 pp", REF, "band5"), ("band 10 pp", REF, "band10")]),
    "Q2 equity share": ("2002-08-01", [(f"{e}/{100 - e}" + (" (ref)" if e == 60 else ""),
                                        {"SPY": e / 100, "IEF": 1 - e / 100}, "monthly")
                                       for e in (0, 20, 40, 60, 80, 100)]),
    "Q3 international": ("2003-05-01", [("US only (ref)", REF, "monthly"),
                                        ("global A 60/30/10", {"SPY": .36, "EFA": .18, "EEM": .06, "IEF": .4}, "monthly"),
                                        ("global B 50/40/10", {"SPY": .30, "EFA": .24, "EEM": .06, "IEF": .4}, "monthly")]),
    "Q4 bonds": ("2004-01-02", [("IEF (ref)", REF, "monthly"), ("SHY", {"SPY": .6, "SHY": .4}, "monthly"),
                                ("TLT", {"SPY": .6, "TLT": .4}, "monthly"), ("AGG", {"SPY": .6, "AGG": .4}, "monthly"),
                                ("TIP", {"SPY": .6, "TIP": .4}, "monthly"),
                                ("IEF+TIP", {"SPY": .6, "IEF": .2, "TIP": .2}, "monthly")]),
    "Q5 gold": ("2004-12-01", [("60/40 (ref)", REF, "monthly"),
                               ("5 % gold", {"SPY": .57, "IEF": .38, "GLD": .05}, "monthly"),
                               ("10 % gold", {"SPY": .54, "IEF": .36, "GLD": .10}, "monthly")]),
}

rows = []
for q, (start_date, variants) in QUESTIONS.items():
    start = int(np.searchsorted(dates, np.datetime64(start_date)))
    ev = start + 1                       # first full day after the initial purchase
    mid = ev + (len(dates) - ev) // 2
    sims = {name: run(target, start, rule) for name, target, rule in variants}
    ref_name = next(n for n in sims if "(ref)" in n)
    ref = sims[ref_name].returns
    for k, (name, sim) in enumerate(sims.items()):
        r = sim.returns
        full = slice(ev, len(dates))
        m = summary(r[full], cash[full], sim.turnover[full], sim.costs[full])
        row = {"question": q, "variant": name, "period": f"{dates[ev]}..{dates[-1]}",
               "cagr": m["cagr"], "volatility": m["volatility"], "sharpe": m["sharpe"],
               "max_drawdown": m["max_drawdown"], "max_dd_days": m["max_drawdown_days"],
               "turnover": m["turnover_annual"], "costs_bps": m["costs_bps_annual"],
               "rebalances": int(np.count_nonzero(sim.turnover[full] > 0))}
        for half, sl in (("h1", slice(ev, mid)), ("h2", slice(mid, len(dates)))):
            row[f"sharpe_{half}"] = summary(r[sl], cash[sl])["sharpe"]
            row[f"cagr_{half}"] = float(np.prod(1 + r[sl]) ** (252 / (sl.stop - sl.start)) - 1)
        if name != ref_name:
            row.update(compare(r, ref, full, seed=k))
        rows.append(row)

df = pl.DataFrame(rows)
# Holm-Bonferroni within each question (Q2 is descriptive but reported the same way).
holm = []
for q in QUESTIONS:
    sub = df.filter((pl.col("question") == q) & pl.col("p_sharpe").is_not_null())
    order = np.argsort(sub["p_sharpe"].to_numpy())
    m = len(order)
    passed, still = {}, True
    for rank, i in enumerate(order):
        ok = still and sub["p_sharpe"][int(i)] <= ALPHA / (m - rank)
        still = ok
        passed[sub["variant"][int(i)]] = ok
    holm += [{"question": q, "variant": v, "holm_significant": s} for v, s in passed.items()]
df = df.join(pl.DataFrame(holm), on=["question", "variant"], how="left")
out = der / "research3"
out.mkdir(parents=True, exist_ok=True)
df.write_parquet(out / "results.parquet")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(300)
pl.Config.set_float_precision(3)
pl.Config.set_tbl_cols(30)
print(df.select("question", "variant", "cagr", "volatility", "sharpe", "max_drawdown", "turnover",
                "costs_bps", "rebalances", "d_sharpe", "d_sharpe_lo", "d_sharpe_hi", "d_cagr",
                "d_cagr_lo", "d_cagr_hi", "p_sharpe", "holm_significant"))
print(df.select("question", "variant", "sharpe_h1", "sharpe_h2", "cagr_h1", "cagr_h2"))
print(f"{time.time() - t0:.0f}s")
