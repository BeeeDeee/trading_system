import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, decision_days, breakeven_z, cohort_signs, book_weights

NAMES = ("TIP", "SCHP", "IEF", "TLT", "SHY", "IEI")


def _bdays(start_offset, n):
    # business days from a Monday (1970-01-05 is a Monday = day 4)
    days = np.arange(4 + start_offset, 4 + start_offset + n * 2)
    days = days[((days + 3) % 7) < 5][:n]
    return days.astype("datetime64[D]")


def _view(be, schp_from=0, tip_missing=None):
    T = be.shape[0]
    N = len(NAMES)
    dates = _bdays(0, T)
    close = np.full((T, N), 100.0)
    listed = np.ones((T, N), dtype=bool)
    close[:schp_from, 1] = np.nan
    listed[:schp_from, 1] = False
    if tip_missing is not None:
        close[tip_missing, 0] = np.nan
    tradable = listed.copy()
    z = np.zeros((T, N))
    return DataView(
        dates=dates, instruments=NAMES, asset_class=tuple("us_etf" for _ in NAMES),
        ret_co=z.copy(), ret_oc=z.copy(), tradable=tradable, listed=listed,
        delisting=np.zeros((T, N), dtype=bool), close=close,
        dollar_volume=np.full((T, N), 1e8), universe=None, extras={},
        series={"fred_macro.T5YIE": be.astype(float)}, cash_ret=np.zeros(T),
    )


# ---------- calendar ----------

def test_decision_days_fridays_and_holiday_fallback():
    d = _bdays(0, 15)        # Mon..Fri x3
    dec = decision_days(d)
    wd = (d.astype(np.int64) + 3) % 7
    assert np.array_equal(dec, wd == 4)
    # drop the 2nd Friday (holiday): first row of week 3 takes the decision
    d2 = np.delete(d, 9)
    dec2 = decision_days(d2)
    assert list(np.flatnonzero(dec2)) == [4, 9, 13]
    # point-in-time: truncation does not change earlier rows
    for t in range(1, len(d2)):
        assert np.array_equal(decision_days(d2[:t]), dec2[:t])


# ---------- z-score ----------

def test_z_hand_computed():
    be = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    z = breakeven_z(be, 4)
    assert np.all(np.isnan(z[:3]))
    w = be[0:4]
    assert z[3] == pytest.approx((4.0 - w.mean()) / w.std(ddof=1))
    w = be[1:5]
    assert z[4] == pytest.approx((10.0 - w.mean()) / w.std(ddof=1))


def test_z_valid_share_and_ffill():
    be = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    z = breakeven_z(be, 10)
    # window ending row 9 has 8 valid of 10 -> 80 % ok; row 8 only partial window
    assert np.isnan(z[8])
    assert np.isfinite(z[9])
    be2 = be.copy()
    be2[2] = np.nan        # 7 valid of 10 at row 9 -> missing
    z2 = breakeven_z(be2, 10)
    assert np.isnan(z2[9])
    assert np.isfinite(z2[10])
    # gap is forward-filled for the statistics
    be3 = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    z3 = breakeven_z(be3, 5)
    x = np.array([1.0, 2.0, 2.0, 4.0, 5.0])
    assert z3[4] == pytest.approx((5.0 - x.mean()) / x.std(ddof=1))


def test_cohort_signs():
    s = cohort_signs(np.array([-1.5, -1.0, -0.99, 0.0, 0.99, 1.0, 2.0, np.nan]))
    assert list(s) == [1, 1, 0, 0, 0, -1, -1, 0]


# ---------- cohort book ----------

def test_book_cohorts_hand():
    H = 3
    signs = np.array([1, 1, 0, -1, -1, -1, 0.0])
    a = np.ones(7, dtype=bool)
    w = book_weights(signs, a, a, a, H)
    net = np.array([1, 2, 2, 0, -2, -3, -2.0])     # rolling sum of last 3 signs
    assert np.allclose(w[:, 0], 0.5 / H * net / 2)
    assert np.allclose(w[:, 1], 0.5 / H * net / 2)
    assert np.allclose(w[:, 2], -0.5 / H * net)
    assert np.all(np.abs(w).sum(axis=1) <= 1 + 1e-12)
    assert np.allclose(w.sum(axis=1), 0.0)


def test_book_tip_only_before_schp_and_fallbacks():
    H = 2
    signs = np.array([1.0, 1.0, 1.0, 1.0])
    a_tip = np.array([True, True, False, False])
    a_schp = np.array([False, True, True, False])
    a_ief = np.ones(4, dtype=bool)
    w = book_weights(signs, a_tip, a_schp, a_ief, H)
    leg = 0.5 / H
    # k0: one TIP-only cohort
    assert np.allclose(w[0], [leg, 0, -leg])
    # k1: TIP-only cohort (kept on TIP) + split cohort
    assert np.allclose(w[1], [leg + leg / 2, leg / 2, -2 * leg])
    # k2: TIP untradable -> both live cohorts' TIPS legs go to SCHP
    assert np.allclose(w[2], [0, 2 * leg, -2 * leg])
    # k3: no TIPS ETF -> cohorts flat (IEF leg too)
    assert np.allclose(w[3], [0, 0, 0])


# ---------- end to end ----------

def _be_path(T, seed=0):
    rng = np.random.default_rng(seed)
    return 2.0 + np.cumsum(rng.normal(0, 0.03, T))


def test_end_to_end_shape_gross_and_peers():
    T = 900
    be = _be_path(T)
    be[:5] = np.nan
    v = _view(be, schp_from=400)
    p = {"hold_weeks": 26, "lookback_days": 252}
    w = target_weights(v, p)
    assert w.shape == (T, len(NAMES))
    dec = decision_days(v.dates)
    assert np.all(np.isnan(w[~dec]))
    assert np.all(np.isfinite(w[dec]))
    assert np.all(w[dec][:, 3:] == 0.0)                 # TLT, SHY, IEI never traded
    assert np.all(np.abs(w[dec]).sum(axis=1) <= 1 + 1e-9)
    assert np.allclose(w[dec].sum(axis=1), 0.0)         # net zero
    assert np.all(w[dec][:, 1][np.flatnonzero(dec) < 400] == 0.0)   # no SCHP before it lists
    # warm-up flat
    first_z = np.flatnonzero(np.isfinite(breakeven_z(be, 252)))[0]
    assert np.all(w[dec & (np.arange(T) < first_z)] == 0.0)
    assert np.abs(w[dec]).sum() > 0


def test_params_change_output():
    T = 1200
    v = _view(_be_path(T, 3))
    base = target_weights(v, PARAMS)
    for k, alt in (("hold_weeks", 26), ("lookback_days", 504)):
        p = dict(PARAMS)
        p[k] = alt
        assert not np.allclose(np.nan_to_num(base), np.nan_to_num(target_weights(v, p)))


def test_missing_series_raises():
    v = _view(_be_path(300))
    v.series.pop("fred_macro.T5YIE")
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)


def test_new_cohort_sign_follows_z():
    T = 900
    be = _be_path(T, 1)
    v = _view(be)
    H = 26
    p = {"hold_weeks": H, "lookback_days": 252}
    w = target_weights(v, p)
    z = breakeven_z(be, 252)
    rows = np.flatnonzero(decision_days(v.dates))
    s = cohort_signs(z[rows])
    assert np.abs(s).sum() > 0
    # change in the TIPS leg between decisions = (new cohort - dropped cohort) * 0.5 / H
    tips = w[rows, 0] + w[rows, 1]
    dropped = np.concatenate([np.zeros(H), s[:-H]])
    expected = np.concatenate([[s[0]], s[1:] - dropped[1:]]) * 0.5 / H
    assert np.allclose(np.diff(np.concatenate([[0.0], tips])), expected)
