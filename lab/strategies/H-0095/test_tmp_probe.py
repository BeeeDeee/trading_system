"""Edge cases on very short views (G0 cuts can be shorter than every window)."""
import numpy as np

from strategy import build_book, entry_flags, lagged_median, lagged_std, trailing_mean


def test_short_views_do_not_crash():
    x = np.ones((5, 3))
    assert np.all(np.isnan(lagged_std(x, 63)))
    assert np.all(np.isnan(lagged_median(x, np.ones((5, 3), dtype=bool), 63)[0]))
    assert np.allclose(trailing_mean(x, 63), 1.0)
    score = np.full((5, 3), np.nan)
    assert not entry_flags(score, 0.7).any()
    W = build_book(np.zeros((5, 3), dtype=bool), np.ones((5, 3), dtype=bool), np.ones((5, 3)),
                   np.ones((5, 3)), np.array([True, False, False, False, False]))
    assert np.allclose(W, 1.0 / 3.0)


def test_no_members_means_cash():
    W = build_book(np.zeros((3, 2), dtype=bool), np.zeros((3, 2), dtype=bool), np.ones((3, 2)),
                   np.ones((3, 2)), np.array([True, False, False]))
    assert np.all(W == 0.0)
