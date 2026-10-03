"""Shared evaluation of research 11 for the dev and the final script (prereg §5, §7)."""

import numpy as np

from qlab.engine.vector import Decisions, simulate
from qlab.pipeline import load_config
from qlab.research4.sleeves import cost_rates, pct_rank, stock_decisions
from qlab.research11.model import GROUP, MODELS, TOP_N, design, forward_returns, rank_ic, walk_forward
from qlab.validation.metrics import max_drawdown
from qlab.validation.stats import sharpe

ANN = np.sqrt(252)


def stats(r: np.ndarray) -> dict:
    nav = np.cumprod(1 + r)
    return {"cagr": float(nav[-1] ** (252 / len(r)) - 1), "sharpe": float(sharpe(r) * ANN),
            "vol": float(r.std(ddof=1) * ANN), "max_dd": float(max_drawdown(nav)[0])}


def ic_stats(ic: np.ndarray) -> dict:
    x = ic[np.isfinite(ic)]
    return {"mean": float(x.mean()), "t": float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))),
            "share_pos": float((x > 0).mean()), "n": int(len(x))}


def evaluate(p, extra, meta, days, sc, years):
    universe = np.asarray(extra["in_liq1000"][days])
    cash = np.asarray(extra["cash_ret"])
    cfg = load_config()
    etf_cols = list(meta["etf"].values())
    rate = {m: cost_rates(extra["liq_rank"], p.dates, cfg["costs"], etf_cols, m) for m in (1.0, 2.0)}
    fwd = forward_returns(p, days)
    y = pct_rank(fwd, universe)
    dd = np.asarray(p.dates[days], dtype="datetime64[D]")
    test = np.flatnonzero(np.isin(dd.astype("datetime64[Y]").astype(int) + 1970, years))
    tdays = days[test]
    start = tdays[0] + 1                                       # first fill
    out = {"models": {}, "returns": {}}
    for name, feats in MODELS.items():
        X = design(sc, universe, feats)
        pred, imp = walk_forward(X, y, universe, p.dates, days, years)
        del X
        dec = stock_decisions(pred[test], universe[test], tdays, TOP_N)
        res = {m: simulate(p, dec, rate[m], cash) for m in (1.0, 2.0)}
        ic = rank_ic(pred, fwd, universe)
        g = np.array([GROUP[f] for f in feats])
        imp_g = {Y: {k: float(np.asarray(v)[g == k].sum() / max(np.sum(v), 1e-12)) for k in "PFIH" if (g == k).any()}
                 for Y, v in imp.items()}
        out["returns"][name] = (res[1.0].returns, res[2.0].returns)
        out["models"][name] = {"ic": ic_stats(ic[test]), "ic_by_year": {
            str(Y): float(np.nanmean(ic[test][np.isin(dd[test].astype("datetime64[Y]").astype(int) + 1970, [Y])]))
            for Y in years}, "importance_by_group": imp_g,
            "turnover_ann": float(res[1.0].turnover[start:].mean() * 252),
            "costs_ann": float(res[1.0].costs[start:].mean() * 252)}
        print(f"{name}: IC {out['models'][name]['ic']}", flush=True)
    spy = meta["etf"]["SPY"]
    w = np.zeros((len(tdays), p.shape[1])); w[:, spy] = 1.0
    out["returns"]["SPY"] = tuple(simulate(p, Decisions(tdays, w), rate[m], cash).returns for m in (1.0, 2.0))
    ew = stock_decisions(np.where(universe[test], 1.0, np.nan), universe[test], tdays, 1000)
    out["returns"]["EW_LIQ1000"] = tuple(simulate(p, ew, rate[m], cash).returns for m in (1.0, 2.0))
    out["start"] = int(start)
    return out


def yearly(r: np.ndarray, dates: np.ndarray, start: int) -> dict:
    y = np.asarray(dates, dtype="datetime64[Y]").astype(int) + 1970
    return {str(Y): float(np.prod(1 + r[start:][y[start:] == Y]) - 1) for Y in np.unique(y[start:])}
