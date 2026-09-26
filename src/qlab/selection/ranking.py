"""Hard filters, robust score and de-duplication (spec §9.4, §9.5).

All functions take metrics computed on one window (the training window inside select()).
"""

from collections.abc import Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from qlab.strategies.grid import StrategyConfig, neighbours


def hard_filter(m: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    """Boolean mask of candidates passing the frozen hard filters."""
    ok = ((m["max_drawdown"] <= cfg["max_drawdown"])
          & (m["sharpe"] >= cfg["min_sharpe"])
          & (m["share_positive_years"] >= cfg["min_share_positive_years"]))
    if "turnover_annual" in m:
        ok &= m["turnover_annual"] <= cfg["max_turnover_annual"]
    if "avg_positions" in m:
        is_regime = m.get("is_regime", np.zeros_like(ok))
        ok &= (m["avg_positions"] >= cfg["min_avg_positions"]) | is_regime
    return ok


def neighbour_lists(grid: Sequence[StrategyConfig], grid_cfg: dict) -> list[np.ndarray]:
    """For each candidate, the indices of its grid neighbours (including itself)."""
    by_group: dict[tuple, list[int]] = {}
    for i, c in enumerate(grid):
        by_group.setdefault((c.family, c.weighting, c.rebalance, c.overlay), []).append(i)
    out: list[np.ndarray] = [np.empty(0, int)] * len(grid)
    for members in by_group.values():
        for i in members:
            out[i] = np.array([j for j in members if neighbours(grid[i], grid[j], grid_cfg)])
    return out


def robust_score(sharpe: np.ndarray, neigh: list[np.ndarray]) -> np.ndarray:
    """(min(own Sharpe, neighbourhood median), neighbourhood median); isolated peaks are penalized."""
    med = np.array([np.median(sharpe[n]) for n in neigh])
    return np.minimum(sharpe, med), med


def correlation_clusters(R: np.ndarray, threshold: float) -> np.ndarray:
    """Average-linkage clusters of daily returns cut at correlation `threshold`; labels 1..K."""
    if R.shape[1] == 1:
        return np.ones(1, int)
    X = np.asarray(R, dtype=np.float64)
    X = X - X.mean(axis=0)
    sd = X.std(axis=0)
    X = X / np.where(sd > 0, sd, 1.0)
    corr = np.clip(X.T @ X / len(X), -1.0, 1.0)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    return fcluster(Z, t=1.0 - threshold, criterion="distance")


def cluster_representatives(labels: np.ndarray, score: np.ndarray,
                            eligible: np.ndarray | None = None) -> np.ndarray:
    """Index of the best-scoring (eligible) candidate of each cluster, ordered by score."""
    eligible = np.ones(len(labels), bool) if eligible is None else eligible
    reps = []
    for lab in np.unique(labels):
        idx = np.flatnonzero((labels == lab) & eligible)
        if len(idx):
            reps.append(idx[np.argmax(score[idx])])
    reps = np.array(reps, dtype=int)
    return reps[np.argsort(-score[reps], kind="stable")]
