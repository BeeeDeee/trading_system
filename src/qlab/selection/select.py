"""select(): the whole selection methodology as a pure function of a training window (spec §9.2).

It receives only the rows of the candidate matrices inside the training window; nothing after the
window can influence the result by construction.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qlab.selection.metrics import window_summary
from qlab.selection.ranking import (cluster_representatives, correlation_clusters, hard_filter,
                                    robust_score)
from qlab.strategies.grid import StrategyConfig


@dataclass(frozen=True)
class Policy:
    members: np.ndarray        # candidate columns, best first (may be shorter than k)
    weight_each: float         # capital share per member (1/k); missing members = cash
    n_passed: int              # candidates passing the hard filters
    n_clusters: int            # correlation clusters among them
    scores: np.ndarray         # robust scores of the members


def select(R: np.ndarray, rf: np.ndarray, dates: np.ndarray, turnover: np.ndarray,
           n_pos: np.ndarray, grid: Sequence[StrategyConfig], neigh: list[np.ndarray],
           cfg: dict, k: int) -> Policy:
    """Choose an ensemble of k candidates from a training window (all arrays = window rows)."""
    m = window_summary(R, rf, dates, turnover, n_pos)
    m["is_regime"] = np.array([c.family == "regime_market" for c in grid])
    passed = hard_filter(m, cfg["hard_filters"])
    score, _ = robust_score(m["sharpe"], neigh)
    idx = np.flatnonzero(passed)
    if len(idx) == 0:
        return Policy(np.empty(0, int), 1.0 / k, 0, 0, np.empty(0))
    labels = correlation_clusters(np.asarray(R[:, idx]), cfg["ranking"]["dedup_correlation"])
    reps = idx[cluster_representatives(labels, score[idx])]
    members = reps[:k]
    return Policy(members, 1.0 / k, len(idx), len(np.unique(labels)), score[members])
