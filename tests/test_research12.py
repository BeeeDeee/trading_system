"""Research 12: hysteresis rule."""

import numpy as np

from qlab.research12.portfolio import decisions, hysteresis, ranking


def test_ranking_ties_and_eligibility():
    pred = np.array([0.5, np.nan, 0.9, 0.5, 0.1])
    uni = np.array([True, True, True, True, False])
    assert ranking(pred, uni).tolist() == [2, 0, 3]


def test_hysteresis_keeps_inside_band_and_fills():
    N = 300
    pred = -np.arange(N, dtype=float)                # column j has rank j + 1
    uni = np.ones(N, dtype=bool)
    held = np.zeros(N, dtype=bool)
    held[[10, 150, 199, 200, 250]] = True           # ranks 11, 151, 200 kept; 201, 251 sold
    new = hysteresis(pred, uni, held, n=50, keep=200)
    assert new.sum() == 50
    assert new[[10, 150, 199]].all() and not new[[200, 250]].any()
    assert new[:48].all() and new[48:].sum() == 2    # 48 best (10 among them) + 150 + 199


def test_held_stock_leaving_universe_is_sold():
    pred = -np.arange(100, dtype=float)
    uni = np.ones(100, dtype=bool)
    held = np.zeros(100, dtype=bool)
    held[5] = True
    uni[5] = False
    assert not hysteresis(pred, uni, held, n=10, keep=50)[5]


def test_turnover_lower_than_top_n():
    rng = np.random.default_rng(0)
    D, N = 60, 1000
    base = rng.normal(size=N)
    pred = base + rng.normal(scale=1.0, size=(D, N))    # persistent signal + noise
    uni = np.ones((D, N), dtype=bool)
    _, m_h = decisions(pred, uni, np.arange(D), 50, 200)
    _, m_t = decisions(pred, uni, np.arange(D), 50, 50)  # keep = n -> plain top 50
    changes = lambda m: np.abs(np.diff(m.astype(int), axis=0)).sum() / 2  # noqa: E731
    assert (m_h.sum(axis=1) == 50).all() and changes(m_h) < 0.6 * changes(m_t)
