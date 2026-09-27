"""Sleeve library (research 4 pre-registration §3).

A sleeve is a fully specified long-only strategy. Stock sleeves hold the top-N LIQ1000 names by a
score, equal weight, rebalanced after the close of the first trading day of each month. ETF
sleeves buy and hold one fund. Scores are computed only on decision rows: row k of a score
matrix belongs to panel row `days[k]` and uses panel rows 0..days[k] only.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from qlab.data.panel import Panel
from qlab.engine.vector import Decisions

ETF_FAMILY = {
    "SPY": "us_equity", "QQQ": "us_equity", "IWM": "us_equity", "MDY": "us_equity",
    "EFA": "intl_equity", "EEM": "intl_equity", "EWJ": "intl_equity",
    "IEF": "bonds", "TLT": "bonds", "SHY": "bonds", "LQD": "bonds", "TIP": "bonds",
    "GLD": "gold", "DBC": "commodities", "VNQ": "real_estate",
    "XLK": "sector", "XLF": "sector", "XLE": "sector", "XLV": "sector", "XLY": "sector",
    "XLP": "sector", "XLI": "sector", "XLB": "sector", "XLU": "sector",
}
ETFS = tuple(ETF_FAMILY)

# score key -> family (prereg §3 table, in order)
STOCK_SCORES = {
    "mom_12_1": "momentum", "mom_6_1": "momentum", "mom_3": "momentum",
    "lowvol_63": "low_vol", "lowvol_252": "low_vol",
    "lowbeta_252": "low_beta", "highbeta_252": "high_beta",
    "strev_21": "st_reversal", "strev_5": "st_reversal",
    "ltrev_36_12": "lt_reversal",
    "high52": "high52",
    "trend_200": "trend", "trend_50": "trend",
    "value_ep": "value", "value_bp": "value", "value_ebitda_ev": "value", "value_sp": "value",
    "value_comp": "value",
    "quality_gpa": "quality", "quality_roe": "quality", "quality_lowlev": "quality",
    "quality_comp": "quality",
    "invest_lowag": "investment",
    "growth_rev": "growth", "growth_eps": "growth",
    "dividend": "dividend",
    "size_small": "size", "size_large": "size",
    "insider": "insider",
    "value_momentum": "value_momentum",
    "quality_value": "quality_value",
}
TOP_NS = (20, 50)
MAX_WEIGHT = 0.10
ETF_MIN_HISTORY = 252


@dataclass(frozen=True)
class Sleeve:
    name: str
    family: str
    kind: str          # stock | etf | cash
    score: str = ""    # stock: key of STOCK_SCORES
    top_n: int = 0
    ticker: str = ""   # etf


def sleeve_library() -> list[Sleeve]:
    out = [Sleeve(f"{s}_top{n}", fam, "stock", score=s, top_n=n)
           for s, fam in STOCK_SCORES.items() for n in TOP_NS]
    out += [Sleeve(f"etf_{t}", fam, "etf", ticker=t) for t, fam in ETF_FAMILY.items()]
    out.append(Sleeve("cash", "cash", "cash"))
    return out


# ---------------------------------------------------------------- price-based scores

def _daily_returns(panel: Panel, a: int, b: int) -> np.ndarray:
    """Close-to-close total returns of rows a..b-1."""
    return (1.0 + np.asarray(panel.ret_co[a:b])) * (1.0 + np.asarray(panel.ret_oc[a:b])) - 1.0


def price_scores(panel: Panel, days: np.ndarray, spy: int) -> dict[str, np.ndarray]:
    """(D, N) price-based scores on decision rows `days`; NaN = not rankable.

    A score needs as many listed days as its longest window and a listing on the decision day.
    """
    D, N = len(days), panel.shape[1]
    listed_days = np.zeros(N, dtype=np.int64)
    keys = ["mom_12_1", "mom_6_1", "mom_3", "lowvol_63", "lowvol_252", "lowbeta_252",
            "highbeta_252", "strev_21", "strev_5", "ltrev_36_12", "high52", "trend_200",
            "trend_50"]
    out = {k: np.full((D, N), np.nan, dtype=np.float32) for k in keys}
    prev = 0
    for k, t in enumerate(days):
        listed_days += np.asarray(panel.listed[prev:t + 1]).sum(axis=0)
        prev = t + 1
        on = np.asarray(panel.listed[t])
        a = max(0, t + 1 - 757)
        r = _daily_returns(panel, a, t + 1)          # rows a..t
        # log growth avoids the 0-index problem after a -100 % day (-> -inf, filtered below)
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.cumsum(np.log1p(np.maximum(r, -1.0)), axis=0)
        n = len(r)

        def ret(lookback, skip=0):
            if n <= lookback:
                return np.full(N, np.nan)
            with np.errstate(invalid="ignore"):
                x = np.expm1(lg[n - 1 - skip] - lg[n - 1 - lookback])
            x[(listed_days <= lookback) | ~on] = np.nan
            return x

        def vol(lookback):
            if n < lookback:
                return np.full(N, np.nan)
            x = r[n - lookback:].std(axis=0, ddof=1)
            x[(listed_days <= lookback) | ~on] = np.nan
            return x

        def idx_ratio(window, fn):
            if n < window:
                return np.full(N, np.nan)
            with np.errstate(invalid="ignore", over="ignore"):
                level = np.exp(lg[n - window:] - lg[n - 1])  # index relative to today's close
                x = 1.0 / fn(level, axis=0)                   # today / fn(window)
            x[(listed_days < window) | ~on] = np.nan
            return x

        out["mom_12_1"][k] = ret(252, 21)
        out["mom_6_1"][k] = ret(126, 21)
        out["mom_3"][k] = ret(63)
        out["lowvol_63"][k] = -vol(63)
        out["lowvol_252"][k] = -vol(252)
        if n >= 252:
            x = r[n - 252:]
            m = x[:, spy]
            cov = ((x - x.mean(axis=0)) * (m - m.mean())[:, None]).sum(axis=0) / 251
            with np.errstate(invalid="ignore", divide="ignore"):
                beta = cov / m.var(ddof=1)
            beta[(listed_days <= 252) | ~on] = np.nan
            out["lowbeta_252"][k], out["highbeta_252"][k] = -beta, beta
        out["strev_21"][k] = -ret(21)
        out["strev_5"][k] = -ret(5)
        lt = ret(756, 252)
        out["ltrev_36_12"][k] = -lt
        out["high52"][k] = idx_ratio(252, np.max)
        out["trend_200"][k] = idx_ratio(200, np.mean) - 1.0
        out["trend_50"][k] = idx_ratio(50, np.mean) - 1.0
    for v in out.values():
        v[~np.isfinite(v)] = np.nan
    return out


def pct_rank(score: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """Per-row percentile rank (0..1] among universe members with a finite score; NaN otherwise."""
    out = np.full(score.shape, np.nan)
    for k in range(score.shape[0]):
        ok = universe[k] & np.isfinite(score[k])
        if ok.sum() == 0:
            continue
        s = score[k, ok]
        order = np.argsort(np.argsort(s, kind="stable"), kind="stable")
        out[k, np.flatnonzero(ok)] = (order + 1) / ok.sum()
    return out


def composite(parts: list[np.ndarray], universe: np.ndarray, min_parts: int) -> np.ndarray:
    """Mean percentile rank over the parts that are available (at least `min_parts`)."""
    ranks = np.stack([pct_rank(p, universe) for p in parts])
    cnt = np.isfinite(ranks).sum(axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(np.where(np.isfinite(ranks), ranks, np.nan), axis=0)
    m[cnt < min_parts] = np.nan
    return m


def add_composites(scores: dict[str, np.ndarray], universe: np.ndarray) -> None:
    scores["value_comp"] = composite([scores[k] for k in ("value_ep", "value_bp",
                                                          "value_ebitda_ev", "value_sp")],
                                     universe, 2)
    scores["quality_comp"] = composite([scores[k] for k in ("quality_gpa", "quality_roe",
                                                            "quality_lowlev")], universe, 2)
    scores["value_momentum"] = composite([scores["value_ep"], scores["mom_12_1"]], universe, 2)
    scores["quality_value"] = composite([scores["quality_comp"], scores["value_comp"]],
                                        universe, 2)


# ---------------------------------------------------------------- decisions

def stock_decisions(score: np.ndarray, universe: np.ndarray, days: np.ndarray,
                    top_n: int) -> Decisions:
    """Top-N by score among universe members, 1/N each (capped), unfilled slots stay in cash."""
    w_each = min(1.0 / top_n, MAX_WEIGHT)
    weights = np.zeros((len(days), score.shape[1]))
    for k in range(len(days)):
        ok = np.flatnonzero(universe[k] & np.isfinite(score[k]))
        best = ok[np.argsort(-score[k, ok], kind="stable")[:top_n]]
        weights[k, best] = w_each
    return Decisions(np.asarray(days), weights)


def etf_decisions(panel: Panel, col: int, days: np.ndarray) -> tuple[Decisions, int]:
    """Buy and hold one ETF from the first decision day with enough history.

    Returns the decisions and the first active decision index (len(days) if never).
    """
    hist = np.cumsum(np.asarray(panel.listed[:, col]), dtype=np.int64)[days]
    active = hist >= ETF_MIN_HISTORY
    first = int(np.argmax(active)) if active.any() else len(days)
    weights = np.zeros((len(days) - first, panel.shape[1]))
    weights[:, col] = 1.0
    return Decisions(np.asarray(days[first:]), weights), first


ETF_COST = 5e-4  # per side, all funds (research 2 convention)


def cost_rates(liq_rank: np.ndarray, dates: np.ndarray, costs_cfg: dict, etf_cols: list[int],
               multiplier: float = 1.0) -> np.ndarray:
    """(T, N) cost per unit traded: research-1 tier model for stocks, ETF_COST for funds."""
    from qlab.pipeline import cost_model
    rate = cost_model(costs_cfg, multiplier).rate(np.asarray(liq_rank), dates)
    rate[:, etf_cols] = ETF_COST * multiplier
    return rate
