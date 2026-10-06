"""Random-entry null: date-by-date, from eligible names only, same counts and (in distribution) turnover."""

import numpy as np

from lab.framework import nulls


def book(T=300, N=60, k=10, stay=0.7, seed=0):
    """A monthly equal-weight book of k names that keeps each name with probability `stay`."""
    rng = np.random.default_rng(seed)
    w = np.full((T, N), np.nan)
    cur = rng.choice(N, k, replace=False)
    for t in range(0, T, 20):
        keep = cur[rng.random(len(cur)) < stay]
        pool = np.setdiff1d(np.arange(N), keep)
        cur = np.concatenate([keep, rng.choice(pool, k - len(keep), replace=False)])
        w[t] = 0.0
        w[t, cur] = 1.0 / k
    return w


def test_counts_weights_and_eligibility_are_respected():
    w = book()
    rng = np.random.default_rng(1)
    elig = rng.random(w.shape) < 0.5
    elig[:, :12] = True
    f = nulls.random_book(w, elig, rng)
    rows = ~np.isnan(w).all(axis=1)
    assert (np.isnan(f).all(axis=1) == ~rows).all()
    assert np.allclose(np.nansum(f, axis=1)[rows], 1.0) and ((f > 0).sum(axis=1)[rows] == 10).all()
    assert not ((f > 0) & ~elig).any()                      # only names eligible on that date


def test_turnover_matches_in_distribution():
    w = book(stay=0.7)
    elig = np.ones(w.shape, bool)
    real = nulls.turnover(w)
    fake = np.mean([nulls.turnover(nulls.random_book(w, elig, np.random.default_rng(s))) for s in range(40)])
    assert abs(fake - real) < 0.08 * max(real, 1e-9) + 0.03
    fresh = book(stay=0.0)                                   # a book that turns over fully each month
    assert nulls.turnover(nulls.random_book(fresh, elig, np.random.default_rng(0))) > 0.8 * nulls.turnover(fresh)


def test_no_survivorship_names_enter_only_when_eligible():
    w = book()
    elig = np.ones(w.shape, bool)
    elig[:150, 30:] = False                                  # names 30+ do not exist in the first half
    f = nulls.random_book(w, elig, np.random.default_rng(2))
    assert not (f[:150, 30:] > 0).any() and (f[150:, 30:] > 0).any()


def test_signed_books_keep_both_legs():
    w = book()
    s = w.copy()
    s[~np.isnan(s)] = 0.0
    for t in np.flatnonzero(~np.isnan(w).all(axis=1)):
        idx = np.flatnonzero(w[t] > 0)
        s[t, idx[:5]], s[t, idx[5:]] = 0.1, -0.1
    f = nulls.random_book(s, np.ones(s.shape, bool), np.random.default_rng(0))
    rows = ~np.isnan(s).all(axis=1)
    assert ((f > 0).sum(axis=1)[rows] == 5).all() and ((f < 0).sum(axis=1)[rows] == 5).all()
