"""Portfolio construction (spec §6.1): selection with hysteresis, weighting, overlay.

On each decision day t (after the close):
1. eligible = in universe at t and finite score; rank eligible by score (1 = best),
2. keep previous selections that are eligible, not exiting and ranked within N * hysteresis,
3. fill up to N with the best-ranked eligible securities that have an entry and no exit signal,
4. each position gets 1/N of capital (EW) or an inverse-volatility share of the same total,
   capped at `max_weight`; unfilled slots stay in cash,
5. everything is scaled by the overlay exposure of day t.
"""

from dataclasses import dataclass

import numpy as np

from qlab.engine.vector import Decisions
from qlab.strategies.signals import Signal


@dataclass(frozen=True)
class PortfolioSpec:
    top_n: int
    weighting: str = "equal"      # equal | inverse_vol
    hysteresis: float = 1.0
    max_weight: float = 0.10


def cap_weights(w: np.ndarray, cap: float) -> np.ndarray:
    """Cap weights at `cap`, redistributing the excess over uncapped names (total preserved if
    feasible, otherwise the remainder stays in cash)."""
    w = w.copy()
    total = w.sum()
    for _ in range(len(w)):
        over = w > cap + 1e-15
        if not over.any():
            break
        w[over] = cap
        free = (w > 0) & (w < cap)
        missing = total - w.sum()
        if not free.any() or missing <= 0:
            break
        w[free] += missing * w[free] / w[free].sum()
    return w


def build_decisions(signal: Signal, universe: np.ndarray, decision_days: np.ndarray,
                    spec: PortfolioSpec, volatility: np.ndarray | None = None,
                    exposure: np.ndarray | None = None) -> Decisions:
    n_assets = universe.shape[1]
    days = np.flatnonzero(decision_days)
    weights = np.zeros((len(days), n_assets))
    held = np.zeros(n_assets, bool)
    for k, t in enumerate(days):
        score = np.asarray(signal.score[t], dtype=float)
        eligible = np.asarray(universe[t], bool) & np.isfinite(score)
        order = np.flatnonzero(eligible)[np.argsort(-score[eligible], kind="stable")]
        rank = np.full(n_assets, np.inf)
        rank[order] = np.arange(1, len(order) + 1)

        exiting = np.asarray(signal.exit[t], bool)
        keep = held & eligible & ~exiting & (rank <= spec.top_n * spec.hysteresis)
        if keep.sum() > spec.top_n:
            kept = np.flatnonzero(keep)
            keep[:] = False
            keep[kept[np.argsort(rank[kept], kind="stable")[: spec.top_n]]] = True
        entry = np.asarray(signal.entry[t], bool)
        fill = order[entry[order] & ~exiting[order] & ~keep[order]][: spec.top_n - int(keep.sum())]
        selected = keep.copy()
        selected[fill] = True

        n_sel = int(selected.sum())
        if n_sel:
            total = n_sel / spec.top_n
            if spec.weighting == "inverse_vol" and volatility is not None:
                vol = np.asarray(volatility[t], dtype=float)[selected]
                inv = np.where(np.isfinite(vol) & (vol > 0), 1.0 / np.maximum(vol, 1e-12), 0.0)
                w = inv / inv.sum() * total if inv.sum() > 0 else np.full(n_sel, total / n_sel)
            elif spec.weighting in ("equal", "inverse_vol"):
                w = np.full(n_sel, total / n_sel)
            else:
                raise ValueError(f"unknown weighting {spec.weighting!r}")
            weights[k, selected] = cap_weights(w, spec.max_weight)
        if exposure is not None:
            weights[k] *= exposure[t]
        held = selected
    return Decisions(days, weights)
