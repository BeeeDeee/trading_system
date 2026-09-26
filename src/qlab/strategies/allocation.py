"""Multi-asset allocation with per-asset trend filters (research 2, pre-registration §4-§5, §10b).

On each decision day t (after the close), over the given asset columns:
1. eligible = listed with at least `min_history` bars and finite signals,
2. optional relative momentum: keep the top-k eligible assets by trailing return,
3. base weights among them: equal or inverse trailing volatility,
4. cap each weight at `max_weight`, redistributing the excess (rest stays in cash if infeasible),
5. trend filter: an asset that fails its trend test gets weight 0 - its share goes to cash.
"""

from dataclasses import dataclass

import numpy as np

from qlab.data.panel import Panel
from qlab.engine.vector import Decisions
from qlab.features.basic import sma_ratio, trailing_return, trailing_volatility
from qlab.strategies.portfolio import cap_weights

WEIGHTINGS = ("equal", "iv63", "iv126", "iv252")
TRENDS = ("none", "sma100", "sma200", "tsmom126", "tsmom252")
RELMOMS = ("none", "top3_126", "top3_252", "top5_126", "top5_252")


@dataclass(frozen=True)
class AllocationConfig:
    weighting: str = "iv126"
    trend: str = "sma200"
    relmom: str = "none"
    max_weight: float = 0.40
    family: str = "allocation"  # select() looks at .family

    @property
    def key(self) -> str:
        return f"{self.weighting}|{self.trend}|{self.relmom}"


H1 = AllocationConfig("iv126", "sma200", "none")


def allocation_grid() -> list[AllocationConfig]:
    return [AllocationConfig(w, t, r) for w in WEIGHTINGS for t in TRENDS for r in RELMOMS]


def allocation_neighbours(grid: list[AllocationConfig]) -> list[np.ndarray]:
    """Neighbours differ in exactly one dimension by one step of its pre-registered order."""
    pos = [(WEIGHTINGS.index(c.weighting), TRENDS.index(c.trend), RELMOMS.index(c.relmom))
           for c in grid]
    out = []
    for a in pos:
        near = [j for j, b in enumerate(pos)
                if sum(x != y for x, y in zip(a, b)) <= 1
                and all(abs(x - y) <= 1 for x, y in zip(a, b))]
        out.append(np.array(near))
    return out


def _cash_growth(cash_ret: np.ndarray, lookback: int) -> np.ndarray:
    idx = np.cumprod(1.0 + np.asarray(cash_ret, dtype=float))
    out = np.full(len(idx), np.nan)
    out[lookback:] = idx[lookback:] / idx[:-lookback] - 1.0
    return out


def trend_pass(panel: Panel, trend: str, cash_ret: np.ndarray) -> np.ndarray:
    """(T, N) bool: asset passes its trend test at t (always True for 'none')."""
    if trend == "none":
        return np.ones(panel.shape, bool)
    kind, lookback = trend[:-3], int(trend[-3:])
    if kind == "sma":
        return sma_ratio(panel, lookback) > 0
    if kind == "tsmom":
        return trailing_return(panel, lookback) > _cash_growth(cash_ret, lookback)[:, None]
    raise ValueError(f"unknown trend {trend!r}")


def allocation_decisions(panel: Panel, columns: list[int], cfg: AllocationConfig,
                         cash_ret: np.ndarray, decision_days: np.ndarray,
                         min_history: int = 252) -> Decisions:
    cols = np.asarray(columns)
    history = np.cumsum(panel.listed, axis=0, dtype=np.int32) >= min_history
    passes = trend_pass(panel, cfg.trend, cash_ret)
    vol = (trailing_volatility(panel, int(cfg.weighting[2:])) if cfg.weighting != "equal"
           else None)
    mom = None
    if cfg.relmom != "none":
        k, lookback = int(cfg.relmom[3]), int(cfg.relmom.split("_")[1])
        mom = trailing_return(panel, lookback)

    days = np.flatnonzero(decision_days)
    weights = np.zeros((len(days), panel.shape[1]))
    for i, t in enumerate(days):
        elig = cols[history[t, cols] & panel.listed[t, cols]]
        if vol is not None:
            elig = elig[np.isfinite(vol[t, elig]) & (vol[t, elig] > 0)]
        if mom is not None:
            elig = elig[np.isfinite(mom[t, elig])]
            elig = elig[np.argsort(-mom[t, elig], kind="stable")[:k]]
        if len(elig) == 0:
            continue
        base = (np.full(len(elig), 1.0 / len(elig)) if vol is None
                else (1.0 / vol[t, elig]) / (1.0 / vol[t, elig]).sum())
        w = cap_weights(base, cfg.max_weight)
        w[~passes[t, elig]] = 0.0
        weights[i, elig] = w
    return Decisions(days, weights)
