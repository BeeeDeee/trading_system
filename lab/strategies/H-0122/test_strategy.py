import dataclasses

import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import daily_total_return, grid_features, realized_vol, build_features, gbm_scores, book
from lab.framework.data import synthetic


def test_params():
    assert PARAMS == {"n_estimators": 100, "top": 0.1}


def test_total_return():
    r = daily_total_return(np.array([[0.01]]), np.array([[0.02]]))
    assert r[0, 0] == pytest.approx(1.01 * 1.02 - 1)


def test_cents_and_roundness():
    p = np.array([[9.99], [10.01], [12.49], [12.50], [49.6], [100.2], [55.0], [7.3]])
    r = np.zeros_like(p)
    cents, rnd, logp, _ = grid_features(p, r)
    assert cents[:, 0] == pytest.approx([-0.01, 0.01, 0.49, -0.50, -0.4, 0.2, 0.0, 0.3], abs=1e-9)
    # levels: 10, 10, 12, 13, 50, 100, 55, 7
    assert list(rnd[:, 0]) == [2.0, 2.0, 0.0, 0.0, 3.0, 3.0, 1.0, 0.0]
    assert np.all(cents >= -0.5) and np.all(cents < 0.5)
    assert logp[0, 0] == pytest.approx(np.log(9.99))


def test_crossing_and_split():
    # rows: 9.80 -> 10.05 up through 10; 10.05 -> 9.97 down through 10; 9.97 -> 9.98 nothing;
    # 9.98 -> 20.04 is a reverse split (r = 0) -> crossing set to 0 although 20 crossed
    p = np.array([[9.80], [10.05], [9.97], [9.98], [20.04], [20.00]])
    r = np.full_like(p, np.nan)
    r[1:] = p[1:] / p[:-1] - 1
    r[4] = 0.0
    _, _, _, cr = grid_features(p, r)
    assert np.isnan(cr[0, 0])                       # no previous close
    assert list(cr[1:, 0]) == [1.0, -1.0, 0.0, 0.0, 0.0]   # 20.04 -> 20.00: level 20, prev >= 20 and 20 <= 20 -> 0


def test_crossing_exact_level():
    p = np.array([[9.9], [10.0], [11.2], [10.9]])
    r = np.full_like(p, np.nan)
    r[1:] = p[1:] / p[:-1] - 1
    _, _, _, cr = grid_features(p, r)
    # 9.9 -> 10.0: L=10, 9.9 < 10 <= 10 -> +1 ; 10.0 -> 11.2: L=11 up ; 11.2 -> 10.9: L=11, 11.2 >= 11 > 10.9 -> -1
    assert list(cr[1:, 0]) == [1.0, 1.0, -1.0]


def test_missing_price_is_nan():
    p = np.array([[np.nan], [10.2], [10.4]])
    r = np.array([[np.nan], [np.nan], [10.4 / 10.2 - 1]])
    cents, rnd, logp, cr = grid_features(p, r)
    assert np.isnan(cents[0, 0]) and np.isnan(rnd[0, 0]) and np.isnan(logp[0, 0])
    assert np.isnan(cr[1, 0]) and cr[2, 0] == 0.0


def test_realized_vol():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, size=(30, 2))
    r[25, 1] = np.nan
    v = realized_vol(r, 21)
    assert np.all(np.isnan(v[:20]))
    assert v[20, 0] == pytest.approx(np.std(r[0:21, 0], ddof=1))
    assert v[29, 0] == pytest.approx(np.std(r[9:30, 0], ddof=1))
    assert np.all(np.isnan(v[25:, 1])) and np.isfinite(v[24, 1])
    assert np.all(np.isnan(realized_vol(r[:5], 21)))   # shorter than the window


def test_features_shape_and_no_lookahead():
    d = synthetic()
    X = build_features(d)
    T, N = len(d.dates), len(d.instruments)
    assert X.shape == (T, N, 6)
    # point-in-time: features on rows <= t do not use later closes (recompute with the future perturbed)
    t = T // 2
    close2 = np.array(d.close, dtype=np.float64)
    cents, rnd, logp, cr = grid_features(d.close, np.zeros((T, N)))
    close2[t + 1:] = close2[t + 1:] * 1.37
    c2, r2, l2, k2 = grid_features(close2, np.zeros((T, N)))
    for a, b in ((cents, c2), (rnd, r2), (logp, l2), (cr, k2)):
        assert np.array_equal(a[:t + 1], b[:t + 1], equal_nan=True)


def _small_view(rows=1500):
    """First `rows` rows of the synthetic market with a point-in-time universe (listed with a price)."""
    d = synthetic()
    T = len(d.dates)
    kw = {}
    for name, v in dataclasses.asdict(d).items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
            kw[name] = v[:rows]
        elif isinstance(v, dict):
            kw[name] = {k: (a[:rows] if isinstance(a, np.ndarray) and a.shape[:1] == (T,) else a)
                          for k, a in v.items()}
    small = dataclasses.replace(d, **kw)
    uni = np.asarray(small.listed, dtype=bool) & np.isfinite(np.asarray(small.close, dtype=np.float64))
    uni[:, 0] = False                                   # one column never a member
    return dataclasses.replace(small, universe=uni)


def test_book_mechanics_small_universe():
    d = _small_view()
    s = gbm_scores(d, {"n_estimators": 20, "top": 0.2})
    w = np.nan_to_num(np.asarray(book(s, d.universe, 0.2, min_names=10), dtype=np.float64))
    gross = np.abs(w).sum(axis=1)
    assert np.all(gross <= 1.0 + 1e-9)
    inv = gross > 0
    assert inv.any()                                                   # the model path produced a book
    assert np.allclose(w[inv].sum(axis=1), 0.0, atol=1e-9)             # dollar-neutral
    assert np.allclose(gross[inv], 1.0, atol=1e-9)                     # 0.5 per leg
    assert gross[0] == 0.0                                             # no model before the first refit
    assert np.all(w[~d.universe] == 0)                                 # members only
    # min_names floor: a floor above the member count gives a flat book
    w_hi = np.nan_to_num(np.asarray(book(s, d.universe, 0.2, min_names=10_000), dtype=np.float64))
    assert np.all(w_hi == 0)
    # top changes the leg size
    w_lo = np.nan_to_num(np.asarray(book(s, d.universe, 0.1, min_names=10), dtype=np.float64))
    assert ((w_lo != 0).sum(axis=1)[inv] <= (w != 0).sum(axis=1)[inv]).all()
    assert ((w_lo != 0).sum() < (w != 0).sum())


def test_n_estimators_changes_scores():
    d = _small_view()
    a = np.asarray(gbm_scores(d, {"n_estimators": 10, "top": 0.1}), dtype=np.float64)
    b = np.asarray(gbm_scores(d, {"n_estimators": 40, "top": 0.1}), dtype=np.float64)
    assert not np.array_equal(a, b, equal_nan=True)


def test_target_weights_flat_below_100_names():
    d = _small_view()                                   # fewer than 100 members -> flat on every row
    w = np.nan_to_num(np.asarray(target_weights(d, {"n_estimators": 10, "top": 0.1}), dtype=np.float64))
    assert w.shape == (len(d.dates), len(d.instruments))
    assert np.all(w == 0)


def test_requires_universe():
    d = synthetic()
    with pytest.raises(ValueError):
        target_weights(dataclasses.replace(d, universe=None), PARAMS)
