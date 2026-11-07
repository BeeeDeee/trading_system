"""Multiple-testing, clustering and track-record helpers for research 22.

Reuses `qlab.validation.stats` (PSR/DSR, PBO via CSCV, stationary bootstrap). Adds what it lacks:
- `multiple_testing`: Romano-Wolf StepM and Benjamini-Yekutieli FDR on a T x k matrix of weekly excess returns,
  both from one studentised stationary bootstrap (H0 per column: mean excess <= 0);
- `cluster_returns`: correlation clustering of return streams ("ONC-lite": average-linkage hierarchical
  clustering with the number of clusters chosen by the silhouette, instead of the recursive k-means of
  Lopez de Prado & Lewis; the cluster count is used as the effective number of independent trials);
- `min_track_record_length` (Bailey & Lopez de Prado) and `unique_columns` (exact duplicate streams).
"""

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from qlab.validation.stats import sharpe, stationary_bootstrap_indices

_N = NormalDist()


def by_fdr(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Yekutieli step-up: boolean rejection vector at FDR level q (any dependence)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    rej = np.zeros(m, dtype=bool)
    if m == 0:
        return rej
    order = np.argsort(p)
    c_m = (1.0 / np.arange(1, m + 1)).sum()
    ok = p[order] <= q * np.arange(1, m + 1) / (m * c_m)
    if ok.any():
        rej[order[: int(np.flatnonzero(ok).max()) + 1]] = True
    return rej


def by_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Yekutieli adjusted p-values (monotone, capped at 1); NaN inputs stay NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    c_m = (1.0 / np.arange(1, m + 1)).sum()
    order = np.argsort(p[ok])
    adj = p[ok][order] * m * c_m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


@dataclass(frozen=True)
class MultiTest:
    t: np.ndarray          # studentised mean excess per column
    p: np.ndarray          # one-sided bootstrap p-values
    stepm: np.ndarray      # Romano-Wolf rejections at FWER alpha
    by: np.ndarray         # Benjamini-Yekutieli rejections at FDR alpha

    @property
    def passed(self) -> np.ndarray:
        return self.stepm | self.by


def multiple_testing(excess: np.ndarray, alpha: float = 0.10, n_boot: int = 2000, mean_block: float = 4.0,
                     seed: int = 0) -> MultiTest:
    """StepM and BY-FDR for H0_j: E[excess_j] <= 0, with a stationary bootstrap of the weekly rows."""
    x = np.asarray(excess, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    n, k = x.shape
    if k == 0:
        e = np.zeros(0)
        return MultiTest(e, e, e.astype(bool), e.astype(bool))
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, k))
    for b in range(n_boot):
        boot[b] = x[stationary_bootstrap_indices(n, mean_block, rng)].mean(axis=0)
    mean = x.mean(axis=0)
    sd = boot.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.inf)
    t = mean / sd
    t_boot = (boot - mean) / sd
    p = (1.0 + (t_boot >= t).sum(axis=0)) / (n_boot + 1.0)

    reject = np.zeros(k, dtype=bool)
    active = np.ones(k, dtype=bool)
    while active.any():
        crit = np.quantile(t_boot[:, active].max(axis=1), 1.0 - alpha)
        new = active & (t > crit)
        if not new.any():
            break
        reject |= new
        active &= ~new
    return MultiTest(t, p, reject, by_fdr(p, alpha))


def min_track_record_length(returns: np.ndarray, sr_benchmark: float = 0.0, alpha: float = 0.05) -> float:
    """Number of periods after which the observed per-period Sharpe beats `sr_benchmark` at confidence 1-alpha."""
    r = np.asarray(returns, dtype=float)
    sr = sharpe(r)
    if sr <= sr_benchmark:
        return float("inf")
    z = (r - r.mean()) / r.std(ddof=0)
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    factor = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return 1.0 + max(factor, 1e-12) * (_N.inv_cdf(1.0 - alpha) / (sr - sr_benchmark)) ** 2


def unique_columns(x: np.ndarray) -> np.ndarray:
    """Indices of the first occurrence of every distinct column (exact equality)."""
    seen: dict[bytes, int] = {}
    keep = []
    for j in range(x.shape[1]):
        key = np.ascontiguousarray(x[:, j]).tobytes()
        if key not in seen:
            seen[key] = j
            keep.append(j)
    return np.array(keep, dtype=int)


def cluster_returns(x: np.ndarray, max_clusters: int = 200, n_grid: int = 12) -> np.ndarray:
    """Cluster labels (0..K-1) of the columns of the T x m matrix `x` by return correlation.

    Distance sqrt((1 - rho) / 2), average linkage; K is the silhouette-maximising value on a roughly geometric grid
    in [2, min(max_clusters, m - 1)]. Columns without variance form singleton clusters of their own.
    """
    x = np.asarray(x, dtype=float)
    m = x.shape[1]
    if m < 3:
        return np.arange(m)
    flat = x.std(axis=0) < 1e-14
    live = np.flatnonzero(~flat)
    labels = np.full(m, -1)
    if len(live) >= 3:
        corr = np.nan_to_num(np.corrcoef(x[:, live].T))
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        z = linkage(squareform(dist, checks=False), method="average")
        k_max = min(max_clusters, len(live) - 1)
        grid = sorted({int(round(v)) for v in np.geomspace(2, max(k_max, 2), n_grid)})
        best_k, best_s = 2, -2.0
        for k in grid:
            lab = fcluster(z, k, criterion="maxclust")
            if 2 <= len(set(lab)) < len(live):
                s = silhouette_score(dist, lab, metric="precomputed")
                if s > best_s:
                    best_k, best_s = k, s
        labels[live] = fcluster(z, best_k, criterion="maxclust") - 1
    else:
        labels[live] = np.arange(len(live))
    nxt = labels.max() + 1
    for j in np.flatnonzero(labels < 0):
        labels[j] = nxt
        nxt += 1
    return labels
