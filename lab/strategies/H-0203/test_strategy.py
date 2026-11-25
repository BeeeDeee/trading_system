import numpy as np
import pytest

from strategy import PARAMS, target_weights
from strategy import (beta_ivol, book, cboe_features, compounded, daily_returns, dv_ratio,
                      market_mean, rolling_sum, weekly_days)
from strategy import new_entrants, surge_top_third
from strategy import features, model_score
import dataclasses

from lab.framework.data import synthetic


def _market(rows=1500, dispersed_volume=True):
    """Framework synthetic market cut to `rows` rows, with a member mask, Cboe series (pc ratio rounded to two
    decimals, so it repeats) and, optionally, dollar volume with cross-sectional dispersion."""
    data = synthetic()
    T = data.close.shape[0]
    rows = min(rows, T)
    rng = np.random.default_rng(5)
    kw = {}
    for name, v in dataclasses.asdict(data).items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
            kw[name] = v[:rows]
        elif isinstance(v, dict):
            kw[name] = {k: (a[:rows] if isinstance(a, np.ndarray) and a.shape[0] == T else a) for k, a in v.items()}
    ser = dict(kw.get("series") or {})
    ser["cboe_putcall_1d.equity_call"] = np.exp(rng.normal(13, 0.3, rows))
    ser["cboe_putcall_1d.equity_pc"] = np.round(rng.uniform(0.4, 0.9, rows), 2)
    kw["series"] = ser
    kw["universe"] = np.asarray(kw["listed"], dtype=bool).copy()
    if dispersed_volume:
        dv = np.asarray(kw["dollar_volume"], dtype=np.float64)
        kw["dollar_volume"] = np.where(np.isfinite(dv), np.exp(rng.normal(17, 1.0, dv.shape)), np.nan)
    return dataclasses.replace(data, **kw)


def test_model_path_scores_after_warmup_and_flat_below_50_names():
    data = _market()
    X, uni = features(data)
    assert X.shape[2] == 7
    # date-level features identical across members on a row
    row = 1000
    for f in (4, 5, 6):
        vals = X[row, uni[row], f]
        assert np.isfinite(vals).all() and np.allclose(vals, vals[0])
    assert np.isnan(X[~uni]).all()                          # non-members carry no features
    score, _ = model_score(data, PARAMS)
    rows = np.flatnonzero(np.isfinite(score).any(axis=1))
    assert rows.size > 0
    assert rows[0] > 252                                    # nothing before features and the first fit exist
    w = target_weights(data, PARAMS)
    dec = weekly_days(data.dates)
    assert np.isnan(w[~dec]).all()
    assert (np.nansum(np.abs(w), axis=1) <= 1.0 + 1e-9).all()
    assert (w[dec] == 0).all()                              # 31 instruments < 50 scored members -> flat


def test_params_and_fixed_settings_reach_walk_forward(monkeypatch):
    # on small samples min_child_samples=10000 allows no split, so check the call instead of the scores
    calls = []

    def fake(X, y, **kw):
        calls.append(kw)
        return np.zeros(X.shape[:2])

    monkeypatch.setattr("strategy.walk_forward", fake)
    data = _market(rows=600)
    for p in (PARAMS, {"n_estimators": 50, "num_leaves": 31}):
        model_score(data, p)
    assert calls[0]["params"] == {"n_estimators": 100, "num_leaves": 15, "learning_rate": 0.05,
                                  "min_child_samples": 10000}
    assert calls[1]["params"]["n_estimators"] == 50 and calls[1]["params"]["num_leaves"] == 31
    assert calls[0]["model"] == "gbm" and calls[0]["horizon"] == 5 and calls[0]["retrain_every"] == 63


def test_no_volume_dispersion_means_not_scored():
    data = _market(dispersed_volume=False)
    X, _ = features(data)
    assert not np.isfinite(X[..., 3]).any()                 # dv_ratio z-score undefined -> not scored


def _bdays(n):
    # business days starting on a Monday: day 0 of the epoch is a Thursday, two business days later is Monday
    first = np.busday_offset(np.zeros(1, dtype="datetime64[D]"), 2, roll="forward")[0]
    return np.busday_offset(first, np.arange(n), roll="forward")


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "num_leaves": 15}
    assert callable(target_weights)


def test_rolling_sum_short_view_and_values():
    x = np.array([1.0, 2.0, np.nan, 4.0])
    np.testing.assert_allclose(rolling_sum(x, 2), [1.0, 3.0, 2.0, 4.0])
    np.testing.assert_allclose(rolling_sum(x, 10), [1.0, 3.0, 3.0, 7.0])   # view shorter than window


def test_daily_returns_validity():
    ret_co = np.array([[0.0], [0.01], [0.0], [0.02]])
    ret_oc = np.array([[0.0], [0.02], [0.0], [-0.01]])
    close = np.array([[np.nan], [10.0], [10.0], [11.0]])
    listed = np.array([[False], [True], [True], [True]])
    r, v = daily_returns(ret_co, ret_oc, close, listed)
    assert v[:, 0].tolist() == [False, False, True, True]          # first priced row has no prior close
    assert r[2, 0] == 0.0
    assert r[3, 0] == pytest.approx(1.02 * 0.99 - 1.0)


def test_market_mean_members_only():
    r = np.array([[0.01, 0.03, 0.5], [np.nan, 0.02, 0.0]])
    v = np.isfinite(r)
    uni = np.array([[True, True, False], [True, True, False]])
    m = market_mean(r, v, uni)
    np.testing.assert_allclose(m, [0.02, 0.02])


def test_beta_exact_and_min_valid():
    rng = np.random.default_rng(0)
    T = 300
    m = rng.normal(0, 0.01, T)
    r = np.stack([2.0 * m + 0.001, 0.5 * m, rng.normal(0, 0.01, T)], axis=1)
    v = np.ones_like(r, dtype=bool)
    v[:, 2] = False
    v[-199:, 2] = True                      # only 199 valid in the last 252 rows
    beta, ivol = beta_ivol(r, v, m)
    assert np.isnan(beta[198, 0])           # 199 rows of history < 200
    assert beta[199, 0] == pytest.approx(2.0, abs=1e-8)
    assert beta[-1, 1] == pytest.approx(0.5, abs=1e-8)
    assert ivol[-1, 0] == pytest.approx(0.0, abs=1e-7)
    assert np.isnan(beta[-1, 2]) and np.isnan(ivol[-1, 2])


def test_ivol_matches_direct_residual_std():
    rng = np.random.default_rng(1)
    T = 260
    m = rng.normal(0, 0.01, T)
    r = (1.3 * m + rng.normal(0, 0.02, T))[:, None]
    v = np.ones_like(r, dtype=bool)
    v[-5, 0] = False                         # one gap in the 21-day window, 20 valid >= 18
    beta, ivol = beta_ivol(np.where(v, r, np.nan), v, m)
    win = np.arange(T - 21, T)
    win = win[v[win, 0]]
    e = r[win, 0] - beta[-1, 0] * m[win]
    assert ivol[-1, 0] == pytest.approx(np.std(e, ddof=1), rel=1e-9)
    # slope check against numpy polyfit over the 252-row window
    w = np.arange(T - 252, T)
    w = w[v[w, 0]]
    assert beta[-1, 0] == pytest.approx(np.polyfit(m[w], r[w, 0], 1)[0], rel=1e-9)
    v2 = v.copy()
    v2[-4:, 0] = False                       # 17 valid in the last 21 -> missing
    _, ivol2 = beta_ivol(np.where(v2, r, np.nan), v2, m)
    assert np.isnan(ivol2[-1, 0])


def test_compounded_requires_full_window():
    r = np.array([0.1, 0.1, np.nan, 0.1, 0.1, 0.1, 0.1, 0.1])
    out = compounded(r, 5)
    assert np.isnan(out[:7]).all()
    assert out[7] == pytest.approx(1.1 ** 5 - 1.0)


def test_dv_ratio():
    dv = np.ones((300, 2))
    dv[-21:, 1] = 2.0
    out = dv_ratio(dv)
    assert out[-1, 0] == pytest.approx(0.0)
    assert out[-1, 1] == pytest.approx(np.log(2.0 / ((231 + 42) / 252)))


def test_dv_ratio_min_counts():
    dv = np.ones((300, 1))
    assert np.isnan(dv_ratio(dv)[198, 0])    # 199 values in the 252 window
    assert np.isfinite(dv_ratio(dv)[199, 0])


def test_cboe_features_means_and_staleness():
    T = 150
    call = np.full(T, np.nan)
    pc = np.full(T, np.nan)
    obs = np.arange(1, 131)                  # 130 sessions observed on rows 10..139, then frozen
    call[10:140] = obs * 10.0
    pc[10:140] = 0.5 + 0.001 * obs
    call[140:] = call[139]                   # forward fill: no new session after row 139
    pc[140:] = pc[139]
    surge, dev = cboe_features(call, pc)
    assert np.isnan(surge[:10 + 125]).all()  # needs 126 sessions
    t = 139                                  # sessions 5..130 in the long window, 126..130 in the short one
    m5 = np.mean(np.arange(126, 131) * 10.0)
    m126 = np.mean(np.arange(5, 131) * 10.0)
    assert surge[t] == pytest.approx(np.log(m5 / m126))
    assert dev[t] == pytest.approx(0.001 * (np.mean(np.arange(126, 131)) - np.mean(np.arange(5, 131))))
    # latest value appeared on row 139: age 1..5 on rows 139..143 is usable, row 144 (age 6) is missing
    assert surge[143] == pytest.approx(surge[139])
    assert np.isnan(surge[144]) and np.isnan(dev[149])


def test_cboe_repeated_ratio_still_new_session():
    call = np.arange(1.0, 201.0)
    pc = np.full(200, 0.6)                   # ratio identical every day, volume changes
    surge, dev = cboe_features(call, pc)
    assert np.isfinite(surge[-1])
    assert dev[-1] == pytest.approx(0.0)


def test_weekly_days_are_fridays():
    d = _bdays(10)
    days = weekly_days(d)
    assert days.tolist() == [False, False, False, False, True] * 2


def test_book_legs_and_min_names():
    T, N = 2, 60
    score = np.tile(np.arange(N, dtype=float), (T, 1))
    uni = np.ones((T, N), dtype=bool)
    score[1, 15:] = np.nan                    # only 15 scored members on row 1 -> flat
    w = book(score, uni)
    assert np.abs(w[0]).sum() == pytest.approx(1.0)
    assert w[0, w[0] > 0].sum() == pytest.approx(0.5)
    assert w[0, w[0] < 0].sum() == pytest.approx(-0.5)
    assert (w[0, -12:] > 0).all() and (w[0, :12] < 0).all()
    assert (w[1] == 0).all()


def test_new_entrants():
    w = np.array([[0.5, -0.5, 0.0],
                  [9.0, 9.0, 9.0],          # not a decision row, ignored
                  [0.5, 0.0, -0.5],
                  [0.0, 0.5, -0.5]])
    days = np.array([True, False, True, True])
    side = new_entrants(w, days)
    assert side[0].tolist() == [1.0, -1.0, 0.0]
    assert side[1].tolist() == [0.0, 0.0, 0.0]
    assert side[2].tolist() == [0.0, 0.0, -1.0]   # stays long -> no event
    assert side[3].tolist() == [0.0, 1.0, 0.0]


def test_surge_top_third():
    s = np.concatenate([np.full(5, np.nan), np.linspace(0, 1, 300)])
    hot = surge_top_third(s)
    assert not hot[:5 + 251].any()
    assert hot[-1]                            # increasing series: latest is the max of its window
    s2 = s.copy()
    s2[-1] = 0.0
    assert not surge_top_third(s2)[-1]


def test_features_point_in_time_on_truncation():
    rng = np.random.default_rng(3)
    T, N = 400, 4
    m = rng.normal(0, 0.01, T)
    r = m[:, None] * np.array([0.5, 1.0, 1.5, 2.0]) + rng.normal(0, 0.01, (T, N))
    v = np.ones((T, N), dtype=bool)
    b_full, i_full = beta_ivol(r, v, m)
    b_cut, i_cut = beta_ivol(r[:300], v[:300], m[:300])
    np.testing.assert_array_equal(b_full[:300], b_cut)
    np.testing.assert_array_equal(i_full[:300], i_cut)
