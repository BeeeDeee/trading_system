import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

T = 540
N = 6
YEAR = 260          # 364 calendar days = 52 weeks = 260 weekdays on a pure weekday calendar


def weekday_dates(n, drop=()):
    raw = (np.arange(12000, 12000 + 2 * n + 50)).astype("datetime64[D]")
    d = raw[np.is_busday(raw)]
    if drop:
        d = np.delete(d, list(drop))
    return d[:n]


def make_view(dates, extras, universe, tradable=None):
    t, n = universe.shape
    if tradable is None:
        tradable = np.ones((t, n), dtype=bool)
    z = np.zeros((t, n))
    return DataView(
        dates=dates,
        instruments=tuple("E%d" % i for i in range(n)),
        asset_class=tuple("us_equity" for _ in range(n)),
        ret_co=z.copy(), ret_oc=z.copy(),
        tradable=tradable, listed=np.ones((t, n), dtype=bool),
        delisting=np.zeros((t, n), dtype=bool),
        close=np.ones((t, n)), dollar_volume=np.ones((t, n)),
        universe=universe, extras=extras, series={}, cash_ret=np.zeros(t),
    )


def base_case(drop=()):
    dates = weekday_dates(T, drop)
    E = np.zeros((T, N), dtype=np.float32)
    E[20, 0] = 1                      # stock 0: primary at row 20 -> event row 20 + 257 = 277
    E[40, 1] = 1                      # stock 1: primary at row 40 ...
    E[45, 1] = 1                      # ... duplicate 7 days later: not primary
    E[60, 2] = 1                      # stock 2: primary at row 60, trigger row 317 ...
    E[300, 2] = 1                     # ... but already announced 17 sessions before: no event
    E[80, 3] = 1                      # stock 3: event at row 337, leaves universe at row 339
    E[:, 4] = np.nan                  # stock 4: no filing data at all
    U = np.ones((T, N), dtype=bool)
    U[339:, 3] = False
    return dates, E, U


def test_params():
    assert PARAMS == {"window_days": 5}


def test_hand_case_weights():
    dates, E, U = base_case()
    W = target_weights(make_view(dates, {"ea_reaction_day": E}, U), PARAMS)
    assert W.shape == (T, N)
    # stock 0: window rows 277..281 (held for sessions 278..282 = p-2..p+2, p = row 280)
    t0 = 20 + YEAR - 3
    row = W[t0]
    assert row[0] == pytest.approx(0.05)
    assert np.allclose(row[1:], -0.01)
    assert np.isnan(W[t0 + 1:t0 + 5]).all()           # unchanged set: keep positions
    assert np.all(W[t0 + 5] == 0.0)                     # window closed: back to cash
    assert np.isnan(W[t0 - 1]).all()
    # stock 1: only the primary filing at row 40 creates an event
    t1 = 40 + YEAR - 3
    assert W[t1, 1] == pytest.approx(0.05)
    assert np.all(W[t1 + 5] == 0.0)
    assert np.all(np.nan_to_num(W[:, 1], nan=-1) <= 0.05)
    assert np.sum(np.nan_to_num(W[:, 1]) > 0) == 1
    # stock 2: already announced within 45 days -> no long anywhere
    assert not np.any(np.nan_to_num(W[:, 2]) > 0)
    # stock 3: event at 337, closed early when it leaves the universe at 339
    t3 = 80 + YEAR - 3
    assert W[t3, 3] == pytest.approx(0.05)
    assert np.allclose(W[t3, [0, 1, 2, 4, 5]], -0.01)
    assert np.isnan(W[t3 + 1]).all()
    assert np.all(W[339] == 0.0)
    assert np.all(W[340:, 3] == 0.0)                   # never held outside the universe
    # stock 4 (all NaN) is shorted but never long
    assert not np.any(np.nan_to_num(W[:, 4]) > 0)
    # first rows: nothing before any event
    assert np.isnan(W[:t0]).all()


def test_dollar_neutral_and_gross():
    dates, E, U = base_case()
    W = target_weights(make_view(dates, {"ea_reaction_day": E}, U), PARAMS)
    dec = ~np.isnan(W).all(axis=1)
    Wd = np.nan_to_num(W[dec])
    assert np.all(np.abs(Wd).sum(axis=1) <= 1.0 + 1e-9)
    assert np.allclose(Wd.sum(axis=1), 0.0)


def test_many_names_cap():
    n = 40
    dates = weekday_dates(T)
    E = np.zeros((T, n))
    E[10, :30] = 1
    U = np.ones((T, n), dtype=bool)
    W = target_weights(make_view(dates, {"ea_reaction_day": E}, U), PARAMS)
    r = W[10 + YEAR - 3]
    assert np.allclose(r[:30], 0.5 / 30)
    assert np.allclose(r[30:], -0.05)
    assert np.abs(r).sum() <= 1.0 + 1e-9


def test_window_param_changes_trigger_and_length():
    dates, E, U = base_case()
    view = make_view(dates, {"ea_reaction_day": E}, U)
    for wd in (3, 5, 7):
        h = (wd + 1) // 2
        W = target_weights(view, {"window_days": wd})
        t0 = 20 + YEAR - h
        assert W[t0, 0] == pytest.approx(0.05)
        assert np.isnan(W[t0 - 1]).all()
        assert np.isnan(W[t0 + 1:t0 + wd]).all()
        assert np.all(W[t0 + wd] == 0.0)


def test_holiday_on_trigger_day_fires_next_session():
    # drop the weekday that is 3 weekdays before p for stock 0 (row 277 of the full calendar)
    dates, E, U = base_case(drop=(277,))
    W = target_weights(make_view(dates, {"ea_reaction_day": E}, U), PARAMS)
    assert np.isnan(W[276]).all()
    assert W[277, 0] == pytest.approx(0.05)            # former row 278
    assert np.all(W[282] == 0.0)


def test_not_tradable_at_trigger_no_event():
    dates, E, U = base_case()
    tr = np.ones((T, N), dtype=bool)
    tr[277, 0] = False
    W = target_weights(make_view(dates, {"ea_reaction_day": E}, U, tr), PARAMS)
    assert not np.any(np.nan_to_num(W[:, 0]) > 0)


def test_not_member_at_trigger_no_event():
    dates, E, U = base_case()
    U[:300, 0] = False
    W = target_weights(make_view(dates, {"ea_reaction_day": E}, U), PARAMS)
    assert not np.any(np.nan_to_num(W[:, 0]) > 0)
    assert np.all(W[:300, 0][~np.isnan(W[:300, 0])] == 0.0)


def test_missing_dataset_raises():
    dates, E, U = base_case()
    with pytest.raises(KeyError):
        target_weights(make_view(dates, {}, U), PARAMS)
    with pytest.raises(ValueError):
        target_weights(make_view(dates, {"ea_reaction_day": np.full((T, N), np.nan)}, U), PARAMS)
    # a short panel that ends before the first filing is not an error: no events, no positions
    k = 200
    W = target_weights(make_view(dates[:k], {"ea_reaction_day": np.full((k, N), np.nan)}, U[:k]), PARAMS)
    assert np.isnan(W).all()


def test_point_in_time_truncation():
    dates, E, U = base_case()
    full = target_weights(make_view(dates, {"ea_reaction_day": E}, U), PARAMS)
    for k in (100, 276, 277, 278, 279, 300, 338, 339, 400):
        part = target_weights(make_view(dates[:k], {"ea_reaction_day": E[:k]}, U[:k]), PARAMS)
        np.testing.assert_array_equal(part, full[:k])


def quarterly_market(seed, t=900, n=30):
    """One reaction session per stock per quarter (+- a week jitter), random universe gaps."""
    rng = np.random.default_rng(seed)
    dates = weekday_dates(t)
    E = np.zeros((t, n))
    for j in range(n):
        r = int(rng.integers(0, 65))
        while r < t:
            E[r, j] = 1
            r += 65 + int(rng.integers(-5, 6))
        if j % 7 == 3:
            E[150 + j, j] = 1                         # an extra (non-primary) filing
    E[:40, :] = np.nan                                # data starts later than the panel
    U = rng.random((t, n)) > 0.05
    tr = rng.random((t, n)) > 0.02
    return dates, E, U, tr


def test_point_in_time_random_market():
    dates, E, U, tr = quarterly_market(1)
    t = len(dates)
    for wd in (3, 5, 7):
        p = {"window_days": wd}
        full = target_weights(make_view(dates, {"ea_reaction_day": E}, U, tr), p)
        assert np.nansum(full > 0) > 50                 # the event logic is exercised
        rng = np.random.default_rng(7)
        for k in list(rng.integers(300, t, 12)) + [t - 1]:
            k = int(k)
            part = target_weights(make_view(dates[:k], {"ea_reaction_day": E[:k]}, U[:k], tr[:k]), p)
            np.testing.assert_array_equal(part, full[:k])
            # perturb the future: rows >= k replaced by noise
            E2, U2, tr2 = E.copy(), U.copy(), tr.copy()
            E2[k:] = (rng.random((t - k, E.shape[1])) > 0.9).astype(float)
            U2[k:] = rng.random((t - k, E.shape[1])) > 0.5
            tr2[k:] = rng.random((t - k, E.shape[1])) > 0.5
            pert = target_weights(make_view(dates, {"ea_reaction_day": E2}, U2, tr2), p)
            np.testing.assert_array_equal(pert[:k], full[:k])
        dec = ~np.isnan(full).all(axis=1)
        Wd = np.nan_to_num(full[dec])
        assert np.all(np.abs(Wd).sum(axis=1) <= 1.0 + 1e-9)
        assert np.all(Wd[~U[dec]] == 0.0)
        assert np.all(np.nan_to_num(full)[~U] == 0.0)   # never a target outside the universe


def test_deterministic():
    dates, E, U = base_case()
    v = make_view(dates, {"ea_reaction_day": E}, U)
    np.testing.assert_array_equal(target_weights(v, PARAMS), target_weights(v, PARAMS))
