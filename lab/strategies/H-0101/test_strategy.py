import dataclasses

import numpy as np
import pytest

from strategy import PARAMS, target_weights
from strategy import (_book, _end_of_week, _entries, _features, _prev_std, _release_load_rank, _scores,
                      LOAD_WINDOW, HOLD, MAX_WEIGHT)
from lab.framework.data import synthetic


# ---------- rolling std strictly before t ----------
def test_prev_std_window_excludes_today():
    x = np.arange(10, dtype=np.float64)[:, None]
    sd = _prev_std(x, 3)
    assert np.all(np.isnan(sd[:3]))
    # rows 0..2 -> std of (0,1,2) = 1 at t = 3
    assert sd[3, 0] == pytest.approx(1.0)
    x2 = x.copy()
    x2[3, 0] = 1000.0                      # today's value must not enter today's std
    assert _prev_std(x2, 3)[3, 0] == pytest.approx(1.0)
    assert np.isnan(_prev_std(x2, 3)[4:7, 0]).sum() == 0


def test_prev_std_nan_in_window_gives_nan():
    x = np.ones((8, 1)) + np.arange(8)[:, None]
    x[2, 0] = np.nan
    sd = _prev_std(x, 3)
    assert np.isnan(sd[3, 0]) and np.isnan(sd[5, 0])
    assert np.isfinite(sd[6, 0])


def test_prev_std_short_view():
    sd = _prev_std(np.ones((2, 3)), 63)
    assert sd.shape == (2, 3) and np.all(np.isnan(sd))


# ---------- end-of-week flag ----------
def test_end_of_week_fridays():
    # day 0 = 1970-01-01, a Thursday; take two weeks of weekdays
    d = np.arange(0, 14).astype("datetime64[D]")
    wk = np.is_busday(d)
    d = d[wk]
    f = _end_of_week(d)
    wd = (d.astype(np.int64) + 3) % 7      # Monday = 0
    assert np.array_equal(f == 1.0, wd == 4)


# ---------- release-load percentile rank ----------
def test_release_load_rank():
    T, N, W = LOAD_WINDOW + 5, 4, LOAD_WINDOW
    uni = np.ones((T, N), dtype=bool)
    flag = np.zeros((T, N))
    flag[:3] = np.nan                           # no flag history in the first 3 sessions
    out = _release_load_rank(flag, uni)
    assert np.all(np.isnan(out[:W + 2]))        # needs 252 sessions of history: first at row 3 + 251
    assert out[W + 2] == pytest.approx(0.5)     # all counts tied at 0 -> average rank = middle
    flag2 = flag.copy()
    flag2[T - 1, :3] = 1.0                      # unique maximum on the last row
    out2 = _release_load_rank(flag2, uni)
    assert out2[T - 1] == pytest.approx(1.0)
    assert np.allclose(out2[:T - 1], out[:T - 1], equal_nan=True)   # no effect on earlier rows
    uni2 = uni.copy()
    uni2[T - 1, :3] = False                     # non-members are not counted
    assert _release_load_rank(flag2, uni2)[T - 1] == pytest.approx(0.5)
    flag3 = flag.copy()
    flag3[T - 1, 0] = 1.0
    flag3[T - 2, 0] = 1.0                       # two tied maxima: (250 + 0.5) / 251
    assert _release_load_rank(flag3, uni)[T - 1] == pytest.approx((W - 2 + 0.5) / (W - 1))


def test_release_load_rank_short_view():
    out = _release_load_rank(np.zeros((10, 3)), np.ones((10, 3), dtype=bool))
    assert np.all(np.isnan(out))


# ---------- entries: quantile over scored release rows of t-62..t ----------
def test_entries_quantile_window():
    T, N = 70, 5
    s = np.full((T, N), np.nan)
    s[0, :] = [10, 11, 12, 13, 14]              # high scores at row 0
    s[62, 0] = 5.0
    s[63, 0] = 5.0
    ent = _entries(s, 0.3)
    # row 0: own pool 10..14, 0.7 quantile = 12.8 -> 13, 14 enter
    assert ent[0].tolist() == [False, False, False, True, True]
    # row 62: pool includes row 0 (t-62) -> 5 is below the quantile
    assert not ent[62, 0]
    # row 63: row 0 has left the window, pool = {5, 5} -> enters
    assert ent[63, 0]
    # top changes the cut
    ent2 = _entries(s, 0.6)
    assert ent2[0].tolist() == [False, False, True, True, True]


def test_entries_no_future_dependence():
    rng = np.random.default_rng(0)
    s = np.where(rng.random((200, 8)) < 0.2, rng.normal(size=(200, 8)), np.nan)
    e = _entries(s, 0.3)
    s2 = s.copy()
    s2[120:] = rng.normal(size=(80, 8))
    e2 = _entries(s2, 0.3)
    assert np.array_equal(e[:120], e2[:120])


# ---------- book ----------
def test_book_hold_and_weights():
    T, N = 12, 15
    ent = np.zeros((T, N), dtype=bool)
    uni = np.ones((T, N), dtype=bool)
    ent[1, 0] = True
    w = _book(ent, uni)
    assert np.all(w[:1] == 0)
    assert np.all(w[1:1 + HOLD, 0] == MAX_WEIGHT)      # one name: capped at 0.10
    assert np.all(w[1 + HOLD:, 0] == 0)                # sold at open of t+6
    ent[2, 1:13] = True                                # 13 names on row 2 -> 1/13 each
    w = _book(ent, uni)
    assert w[2, 0] == pytest.approx(1 / 13)
    assert w[2].sum() == pytest.approx(1.0)
    assert np.all(w.sum(axis=1) <= 1 + 1e-12)
    uni[4, 5] = False                                  # leaves the universe -> closed
    w = _book(ent, uni)
    assert w[4, 5] == 0 and w[4, 6] == pytest.approx(1 / 12)
    assert np.all(w >= 0)


def test_book_reentry_extends():
    ent = np.zeros((12, 1), dtype=bool)
    uni = np.ones((12, 1), dtype=bool)
    ent[0, 0] = True
    ent[3, 0] = True
    w = _book(ent, uni)
    assert np.all(w[0:8, 0] == MAX_WEIGHT) and np.all(w[8:, 0] == 0)


# ---------- full pipeline on the synthetic market ----------
def _market(flag_rate):
    """Synthetic market with a universe (all listed names but MKT, one name leaving mid-sample) and a
    0/1 release flag."""
    data = synthetic()
    T, N = len(data.dates), len(data.instruments)
    uni = np.asarray(data.listed, dtype=bool).copy()
    uni[:, data.col("MKT")] = False
    uni[2500:, data.col("S03")] = False
    rng = np.random.default_rng(7)
    flag = (rng.random((T, N)) < flag_rate).astype(np.float32)
    flag[:40] = np.nan                                  # flag history starts late
    extras = dict(data.extras or {})
    extras["ea_reaction_day"] = flag
    # the synthetic dollar volume is equal across names (no z-score possible): give it dispersion
    dv = np.exp(rng.normal(16.0, 1.0, size=(1, N)) + 0.3 * rng.normal(size=(T, N)))
    dv = np.where(np.isfinite(np.asarray(data.dollar_volume)), dv, np.nan)
    return dataclasses.replace(data, universe=uni, extras=extras, dollar_volume=dv)


def _dense_flag_market():
    return _market(0.06)


def test_full_pipeline_no_releases():
    data = _market(0.0)
    w = target_weights(data, PARAMS)
    assert w.shape == (len(data.dates), len(data.instruments))
    assert np.all(w == 0)


def test_missing_flag_raises():
    data = synthetic()
    data = dataclasses.replace(data, universe=np.asarray(data.listed, dtype=bool), extras={})
    with pytest.raises(KeyError):
        target_weights(data, PARAMS)


def test_features_release_rows_only():
    data = _dense_flag_market()
    X, ok, uni = _features(data)
    T, N = len(data.dates), len(data.instruments)
    assert X.shape == (T, N, 6)
    flag = data.extras["ea_reaction_day"]
    assert not np.any(ok & ~uni)                        # members only
    assert not np.any(ok & (flag != 1.0))               # release rows only
    assert np.all(np.isfinite(X[ok]))
    assert np.all(np.isnan(X[~ok]))
    rel = uni & (flag == 1.0)
    dv = np.asarray(data.dollar_volume)
    info = (int(rel.sum()), int(np.isfinite(np.asarray(data.close))[rel].sum()), int(np.isfinite(dv)[rel].sum()),
            float(np.nanstd(dv[3000])), str(dv.dtype))
    assert ok.sum() > 1000, info
    # load feature: NaN before 252 sessions of flag history (history starts at row 40)
    assert not np.any(ok[:40 + LOAD_WINDOW - 1])
    # date-level features equal across a row; f6 is the Friday flag, f5 in [0, 1]
    r = np.nonzero(ok.any(axis=1))[0][0]
    assert np.unique(X[r][ok[r]][:, 4]).size == 1
    assert np.all((X[ok][:, 4] >= 0) & (X[ok][:, 4] <= 1))
    assert set(np.unique(X[ok][:, 5]).tolist()) <= {0.0, 1.0}


def test_features_point_in_time():
    data = _dense_flag_market()
    X, ok, _ = _features(data)
    cut = 3000
    co = np.array(data.ret_co, copy=True)
    oc = np.array(data.ret_oc, copy=True)
    dv = np.array(data.dollar_volume, copy=True)
    fl = np.array(data.extras["ea_reaction_day"], copy=True)
    co[cut + 1:] = 0.3
    oc[cut + 1:] = -0.2
    dv[cut + 1:] = 1.0
    fl[cut + 1:] = 1.0
    extras = dict(data.extras)
    extras["ea_reaction_day"] = fl
    data2 = dataclasses.replace(data, ret_co=co, ret_oc=oc, dollar_volume=dv, extras=extras)
    X2, ok2, _ = _features(data2)
    assert np.array_equal(ok[:cut + 1], ok2[:cut + 1])
    assert np.allclose(X[:cut + 1], X2[:cut + 1], equal_nan=True)


def test_model_path_weights_and_params():
    data = _dense_flag_market()
    score, uni = _scores(data, PARAMS)
    X, ok, _ = _features(data)
    assert not np.any(np.isfinite(score) & ~ok)         # only complete release rows are scored
    w = target_weights(data, PARAMS)
    assert np.all(w >= 0) and np.all(w.sum(axis=1) <= 1 + 1e-9)
    assert np.all(w[~uni] == 0)
    assert np.isfinite(score).any()
    first = np.nonzero(np.isfinite(score).any(axis=1))[0][0]
    assert first >= 40 + LOAD_WINDOW - 1
    assert np.all(w[:first] == 0)                        # nothing held before the first fit
    assert w.sum() > 0
    assert np.all(w[w > 0] <= MAX_WEIGHT + 1e-12)
    w_top = target_weights(data, dict(PARAMS, top=0.4))
    assert not np.array_equal(w, w_top)
    assert (w_top > 0).sum() > (w > 0).sum()             # a larger top admits more entries
    w_n = target_weights(data, dict(PARAMS, n_estimators=50))
    assert not np.array_equal(w, w_n)


def _cut(v, T, n):
    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
        return v[:n]
    if isinstance(v, dict):
        return {k: _cut(a, T, n) for k, a in v.items()}
    return v


def test_model_path_truncation_point_in_time():
    data = _dense_flag_market()
    w_full = target_weights(data, PARAMS)
    T = len(data.dates)
    cut = 3500
    n = cut + 1
    short = dataclasses.replace(
        data, dates=_cut(data.dates, T, n), ret_co=_cut(data.ret_co, T, n), ret_oc=_cut(data.ret_oc, T, n),
        tradable=_cut(data.tradable, T, n), listed=_cut(data.listed, T, n), delisting=_cut(data.delisting, T, n),
        close=_cut(data.close, T, n), dollar_volume=_cut(data.dollar_volume, T, n),
        universe=_cut(data.universe, T, n), extras=_cut(data.extras, T, n), series=_cut(data.series, T, n),
        cash_ret=_cut(data.cash_ret, T, n))
    w_cut = target_weights(short, PARAMS)
    assert np.allclose(w_cut, w_full[:cut + 1])
