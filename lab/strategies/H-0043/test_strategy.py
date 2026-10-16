import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
import strategy as S

NAMES = ("SPY", "IEF", "HYG", "JNK", "LQD", "TLT", "SHY")


def make_arrays(T=900, seed=0, noise=0.001):
    """Synthetic panel: credit ETF returns are an exact linear function of the FRED changes plus small noise.

    FRED observation dated session d sits on series row d+1 (next_morning clock)."""
    rng = np.random.default_rng(seed)
    days = np.arange(12000, 12000 + 2 * T).astype("datetime64[D]")
    days = days[np.is_busday(days)][:T]
    N = len(NAMES)
    obs = np.zeros((T + 1, 4))           # obs[d + 1] = FRED value dated session d, obs[0] = dated session -1
    obs[0] = (4.0, 1.5, 3.0, 4.0)
    obs[1:] = obs[0] + np.cumsum(rng.normal(0, 0.05, (T, 4)), axis=0)
    dobs = np.diff(obs, axis=0)          # dobs[d] = change dated session d
    ret = rng.normal(0.0003, 0.01, (T, N))
    j = {n: i for i, n in enumerate(NAMES)}
    ret[:, j["HYG"]] = 0.0002 - 0.03 * dobs[:, 0] - 0.01 * dobs[:, 2] - 0.02 * dobs[:, 3] + rng.normal(0, noise, T)
    ret[:, j["JNK"]] = 0.0001 - 0.025 * dobs[:, 0] - 0.01 * dobs[:, 2] - 0.02 * dobs[:, 3] + rng.normal(0, noise, T)
    ret[:, j["LQD"]] = 0.0001 - 0.05 * dobs[:, 1] - 0.02 * dobs[:, 2] - 0.05 * dobs[:, 3] + rng.normal(0, noise, T)
    series_rows = np.zeros((T, 4))
    series_rows[:] = obs[:T]             # series row k = obs dated k-1
    return dict(days=days, ret=ret, series=series_rows, listed=np.ones((T, N), dtype=bool))


def view_from(a, T=None):
    T = len(a["days"]) if T is None else T
    N = len(NAMES)
    ret = a["ret"][:T]
    listed = a["listed"][:T]
    close = np.where(listed, 100.0 * np.cumprod(1.0 + ret, axis=0), np.nan)
    series = {}
    for i, k in enumerate(S.FRED_KEYS):
        series["fred_macro." + k] = a["series"][:T, i].astype(np.float32).astype(np.float64)
    return DataView(
        dates=a["days"][:T],
        instruments=NAMES,
        asset_class=tuple("us_etf" for _ in NAMES),
        ret_co=np.where(listed, ret * 0.3, 0.0),
        ret_oc=np.where(listed, ret * 0.7, 0.0),
        tradable=listed.copy(),
        listed=listed.copy(),
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e9, np.nan),
        universe=None,
        extras={},
        series=series,
        cash_ret=np.zeros(T),
    )


def col(name):
    return NAMES.index(name)


# ----------------------------------------------------------------------------------------------- helpers
def test_fred_changes_skip_stale_and_nan_rows():
    F = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.1, 2.0, 3.0, 4.2],
        [1.1, 2.0, 3.0, 4.2],   # forward-filled FRED holiday: skipped
        [1.5, 2.1, 3.0, 4.0],   # change taken against row 1
        [np.nan, 2.1, 3.0, 4.0],  # missing: skipped
        [1.4, 2.1, 3.1, 4.0],   # change against row 3
    ])
    dx, ok = S.fred_changes(F)
    assert ok.tolist() == [False, True, False, True, False, True]
    np.testing.assert_allclose(dx[1], [0.1, 0.0, 0.0, 0.2], atol=1e-12)
    np.testing.assert_allclose(dx[3], [0.4, 0.1, 0.0, -0.2], atol=1e-12)
    np.testing.assert_allclose(dx[5], [-0.1, 0.0, 0.1, 0.0], atol=1e-12)
    assert np.all(np.isnan(dx[[0, 2, 4]]))


def test_rolling_residuals_out_of_sample_and_min_valid():
    rng = np.random.default_rng(1)
    T = 60
    X = rng.normal(size=(T, 2))
    Z = np.column_stack([np.ones(T), X])
    y = 0.5 + 2.0 * X[:, 0] - 1.0 * X[:, 1]
    y[40] += 3.0                          # shock: residual is exactly the shock (window ends at row 39)
    s = np.ones(T, dtype=bool)
    s[10] = False                         # a skipped row
    e = S.rolling_ols_residuals(y, Z, s, window=20, min_valid=15)
    assert np.all(np.isnan(e[:16]))       # rows 0..15 have < 15 valid rows before them (row 10 skipped)
    assert np.isfinite(e[16])
    assert np.isnan(e[10])
    np.testing.assert_allclose(e[16:40], 0.0, atol=1e-9)
    assert e[40] == pytest.approx(3.0, abs=1e-9)
    # the shock is not in the coefficients of row 40, but it is in row 41's window
    assert abs(e[41]) > 1e-6


def test_gap_zscore_by_hand():
    e = np.array([np.nan, 1.0, -1.0, 1.0, -1.0, np.nan, 2.0, 1.0])
    z = S.gap_zscore(e, gap_days=2, window=6, min_valid=4)
    # row 4: residuals rows 1..4 -> [1,-1,1,-1], std (ddof 1) = sqrt(4/3); last 2 residuals sum 0
    assert z[4] == pytest.approx(0.0)
    # row 6: window rows 1..6 -> [1,-1,1,-1,2], mean .4, ss = 3*(.6^2)+... computed directly
    w = np.array([1.0, -1.0, 1.0, -1.0, 2.0])
    assert z[6] == pytest.approx((-1.0 + 2.0) / (np.sqrt(2) * w.std(ddof=1)))
    # row 7: window rows 2..7 -> [-1,1,-1,2,1]; last two valid residuals 2 + 1
    w = np.array([-1.0, 1.0, -1.0, 2.0, 1.0])
    assert z[7] == pytest.approx(3.0 / (np.sqrt(2) * w.std(ddof=1)))
    assert np.all(np.isnan(z[:4]))


def test_spaced_events_rule():
    z = np.array([0, -2.5, -3.0, 0, 0, 0, 0, 2.1, -2.0, 0, 0, 0, 0, 0, -1.99, 2.0])
    elig = np.ones(len(z), dtype=bool)
    elig[15] = False
    d, p = S.spaced_events(z, elig, z_entry=2.0, hold=5)
    # row 1 fires; row 2 blocked (within 5); row 7 is 6 rows later -> premium; row 8 blocked; row 14 below
    assert np.flatnonzero(d).tolist() == [1]
    assert np.flatnonzero(p).tolist() == [7]
    d2, p2 = S.spaced_events(z, elig, z_entry=2.4, hold=5)
    assert np.flatnonzero(d2).tolist() == [1]
    assert np.flatnonzero(p2).tolist() == []


def test_slot_book_weights():
    T = 20
    disc = np.zeros((T, 3), dtype=bool)
    disc[2, 0] = True       # HYG slot active rows 2..6
    disc[4, 1] = True       # JNK slot active rows 4..8
    disc[5, 2] = True       # LQD slot active rows 5..9
    active, credit, ief = S.slot_book(disc, hold=5)
    assert active[:, 0].tolist() == [i in range(2, 7) for i in range(T)]
    np.testing.assert_allclose(credit[2], [0.2, 0, 0])
    np.testing.assert_allclose(ief[2], 0.2)
    np.testing.assert_allclose(credit[4], [0.2, 0.2, 0])
    np.testing.assert_allclose(ief[4], 0.0, atol=1e-15)
    np.testing.assert_allclose(credit[5], [0.4 / 3] * 3)
    np.testing.assert_allclose(ief[5], 0.0, atol=1e-15)
    np.testing.assert_allclose(credit[7], [0, 0.2, 0.2])
    np.testing.assert_allclose(credit[9], [0, 0, 0.2])
    np.testing.assert_allclose(ief[10], 0.4)
    assert np.all(credit.sum(axis=1) + ief == pytest.approx(0.4))


# --------------------------------------------------------------------------------------------- full book
def shocked_arrays():
    a = make_arrays(T=900, seed=3, noise=0.001)
    h, l = col("HYG"), col("LQD")
    a["ret"][700:705, h] -= 0.01          # HYG falls 1 % a day more than its bonds on sessions 700..704
    a["ret"][800:805, l] += 0.01          # LQD premium (never traded)
    return a


def test_discount_event_opens_slot_and_premium_does_not():
    a = shocked_arrays()
    data = view_from(a)
    p = dict(PARAMS, z_entry=6.0)
    w = target_weights(data, p)
    ev = S.credit_events(data, p)
    # sessions 700..704 are known on rows 701..705 (next_morning clock): the event fires inside that range,
    # never before the first shocked session is known
    hyg = np.flatnonzero(ev["HYG"][0])
    assert len(hyg) == 1 and 702 <= hyg[0] <= 705
    lqd = np.flatnonzero(ev["LQD"][1])
    assert len(lqd) == 1 and 802 <= lqd[0] <= 805
    assert not ev["LQD"][0].any() and not ev["JNK"][0].any() and not ev["JNK"][1].any()
    t = hyg[0]
    assert w[t, col("HYG")] == pytest.approx(0.2)
    assert w[t, col("IEF")] == pytest.approx(0.2)
    assert w[t, col("SPY")] == pytest.approx(0.6)
    ms = S.month_start_mask(data.dates)
    for r in range(t + 1, t + 5):        # held: no decision unless it is a month reset (then the slot is kept)
        if ms[r]:
            assert w[r, col("HYG")] == pytest.approx(0.2)
        else:
            assert np.all(np.isnan(w[r]))
    assert w[t + 5, col("HYG")] == 0.0 and w[t + 5, col("IEF")] == pytest.approx(0.4)
    # premium on LQD changes nothing
    dec = ~np.isnan(w).any(axis=1)
    assert not dec[805] or w[805, col("LQD")] == 0.0
    assert np.nansum(w[:, col("LQD")]) == 0.0


def test_core_and_month_resets():
    data = view_from(make_arrays(T=400, seed=5))
    w = target_weights(data, dict(PARAMS, z_entry=50.0))     # no events
    dec = ~np.isnan(w).any(axis=1)
    ms = S.month_start_mask(data.dates)
    assert dec[0]
    assert np.array_equal(dec[1:], ms[1:])
    np.testing.assert_allclose(w[dec][:, col("SPY")], 0.6)
    np.testing.assert_allclose(w[dec][:, col("IEF")], 0.4)
    assert np.all(w[dec][:, [col("TLT"), col("SHY"), col("HYG"), col("JNK"), col("LQD")]] == 0.0)


def test_gross_long_only_and_point_in_time():
    a = make_arrays(T=900, seed=7, noise=0.002)
    data = view_from(a)
    w = target_weights(data, PARAMS)
    dec = ~np.isnan(w).any(axis=1)
    assert np.all(w[dec] >= 0)
    np.testing.assert_allclose(w[dec].sum(axis=1), 1.0)
    for cut in (650, 777, 860):
        wc = target_weights(view_from(a, T=cut), PARAMS)
        np.testing.assert_array_equal(np.isnan(wc), np.isnan(w[:cut]))
        np.testing.assert_allclose(np.nan_to_num(wc), np.nan_to_num(w[:cut]), atol=1e-12)
    w2 = target_weights(data, PARAMS)
    np.testing.assert_array_equal(np.nan_to_num(w, nan=-1), np.nan_to_num(w2, nan=-1))


def test_no_events_before_history_and_late_listing():
    a = make_arrays(T=1300, seed=11, noise=0.002)
    j = col("JNK")
    a["listed"][:500, j] = False
    data = view_from(a)
    ev = S.credit_events(data, dict(PARAMS, z_entry=0.5))
    d, p = ev["JNK"]
    # JNK needs 252 bars, and 200 residuals in the std window, which needs >= 200 sessions of regression first
    assert not (d | p)[:500 + 252].any()
    assert (d | p).any()
    dh, ph = ev["HYG"]
    assert not (dh | ph)[:252].any()
    assert (dh | ph).any()


def test_params_change_output():
    data = view_from(make_arrays(T=900, seed=13, noise=0.002))
    base = target_weights(data, PARAMS)
    for p in (dict(PARAMS, gap_days=3), dict(PARAMS, gap_days=10), dict(PARAMS, z_entry=1.5), dict(PARAMS, z_entry=2.5)):
        other = target_weights(data, p)
        assert not np.array_equal(np.nan_to_num(base, nan=-1), np.nan_to_num(other, nan=-1))


def test_missing_fred_series_fails_loudly():
    data = view_from(make_arrays(T=600, seed=2))
    del data.series["fred_macro.BAMLC0A0CM"]
    with pytest.raises(KeyError):
        target_weights(data, PARAMS)


def test_leading_nan_fred_gives_no_events_until_data_starts():
    a = make_arrays(T=1300, seed=17, noise=0.002)
    a["series"][:400, :] = np.nan
    ev = S.credit_events(view_from(a), dict(PARAMS, z_entry=0.5))
    for name in S.CREDIT:
        d, p = ev[name]
        assert not (d | p)[:800].any()      # 200 regression sessions + 200 residuals after the series starts
    assert any((ev[n][0] | ev[n][1]).any() for n in S.CREDIT)
