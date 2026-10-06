"""Random-entry null for cross-sectional books (G2): same exposure, holding period and turnover, random names.

The first version assigned each stock's weight series to a random other column for the whole window, drawn from
the stocks listed in more than half of it. That is survivorship: the pool was chosen with knowledge of who
lasted, so the null beat even the market (calibration 2026-10-06: median Sharpe 0.62 against SPY 0.36) and no
long-only stock book could pass. Now the null is built date by date from information of that date only:

- at every decision row the fake book holds as many names as the real one (longs and shorts separately), drawn
  from the instruments eligible on that date (universe member, tradable),
- each name of the fake book is kept from the previous decision with the probability with which the real book
  kept its names (retention), so turnover and holding periods match in distribution,
- the real weights are dealt out to the chosen names at random (equal-weight books: identical).
"""

import numpy as np


def random_book(w: np.ndarray, eligible: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A fake book with the structure of `w` (T, N; NaN row = no decision) on `eligible` (T, N bool)."""
    out = np.full_like(w, np.nan)
    prev_real = {1: np.empty(0, int), -1: np.empty(0, int)}
    prev_fake = {1: np.empty(0, int), -1: np.empty(0, int)}
    for t in np.flatnonzero(~np.isnan(w).all(axis=1)):
        row = np.nan_to_num(w[t])
        out[t] = 0.0
        pool = np.flatnonzero(eligible[t])
        taken = np.zeros(w.shape[1], dtype=bool)
        for sign in (1, -1):
            real = np.flatnonzero(row * sign > 1e-12)
            if len(real) == 0:
                prev_real[sign], prev_fake[sign] = real, real
                continue
            keep_p = len(np.intersect1d(real, prev_real[sign])) / max(len(prev_real[sign]), 1)
            q = prev_fake[sign][eligible[t, prev_fake[sign]] & ~taken[prev_fake[sign]]]
            kept = q[rng.random(len(q)) < keep_p][: len(real)]
            taken[kept] = True
            free = pool[~taken[pool]]
            need = len(real) - len(kept)
            fill = rng.choice(free, min(need, len(free)), replace=False) if need > 0 and len(free) else np.empty(0, int)
            taken[fill] = True
            chosen = np.concatenate([kept, fill]).astype(int)
            vals = rng.permutation(row[real])[: len(chosen)]
            out[t, chosen] = vals
            prev_real[sign], prev_fake[sign] = real, chosen
    return out


def turnover(w: np.ndarray) -> float:
    """Mean one-way change of the held weights per decision row (decision rows only)."""
    rows = w[~np.isnan(w).all(axis=1)]
    return float(np.abs(np.diff(np.nan_to_num(rows), axis=0)).sum(axis=1).mean() / 2) if len(rows) > 1 else 0.0
