"""Hysteresis selection of research 12 (prereg §4).

Keep held stocks that are in the universe and ranked <= `keep`; fill up to `n` with the best-ranked stocks not
held; equal weights. Ranking: prediction descending, ties -> lower column. Point in time: uses row k only.
"""

import numpy as np

from qlab.engine.vector import Decisions

N_HOLD = 50
KEEP_RANK = 200


def ranking(pred: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """Columns of eligible stocks (universe and finite prediction), best first."""
    ok = np.flatnonzero(universe & np.isfinite(pred))
    return ok[np.lexsort((ok, -pred[ok]))]


def hysteresis(pred: np.ndarray, universe: np.ndarray, held: np.ndarray, n: int = N_HOLD,
               keep: int = KEEP_RANK) -> np.ndarray:
    """New held mask (bool, N) from one decision row and the currently held mask."""
    order = ranking(pred, universe)
    kept = order[:keep][held[order[:keep]]]
    new = np.zeros_like(held, dtype=bool)
    new[kept[:n]] = True
    for j in order:
        if new.sum() >= n:
            break
        new[j] = True
    return new


def decisions(pred: np.ndarray, universe: np.ndarray, days: np.ndarray, n: int = N_HOLD, keep: int = KEEP_RANK,
              held: np.ndarray | None = None) -> tuple[Decisions, np.ndarray]:
    """Equal-weight decisions over the rows of `pred`; returns (Decisions, held masks per row)."""
    held = np.zeros(pred.shape[1], dtype=bool) if held is None else held.copy()
    masks = np.zeros(pred.shape, dtype=bool)
    for k in range(len(days)):
        held = hysteresis(pred[k], universe[k], held, n, keep) if (universe[k] & np.isfinite(pred[k])).any() else held
        masks[k] = held
    w = masks / np.maximum(masks.sum(axis=1, keepdims=True), 1)
    return Decisions(np.asarray(days), np.minimum(w, 1.0 / n)), masks
