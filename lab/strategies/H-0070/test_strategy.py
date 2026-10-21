import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

NAMES = ("FXA", "FXB", "FXC", "FXE", "FXF", "FXY", "SPY")
T = 100


def make_view(ret_oc, tradable=None, universe=None):
    n_t, n = ret_oc.shape
    dates = np.arange(n_t).astype("timedelta64[D]") + np.zeros(1, dtype="datetime64[D]")
    if tradable is None:
        tradable = np.ones((n_t, n), dtype=bool)
    listed = np.ones((n_t, n), dtype=bool)
    close = np.full((n_t, n), 100.0)
    return DataView(
        dates=dates,
        instruments=NAMES,
        asset_class=tuple("us_etf" for _ in NAMES),
        ret_co=np.zeros((n_t, n)),
        ret_oc=ret_oc,
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((n_t, n), dtype=bool),
        close=close,
        dollar_volume=np.full((n_t, n), 1e7),
        universe=universe,
        extras={},
        series={},
        cash_ret=np.zeros(n_t),
    )


def base_returns():
    r = np.zeros((T, len(NAMES)))
    spy = NAMES.index("SPY")
    r[:, spy] = np.where(np.arange(T) % 2 == 0, 0.01, -0.01)  # sigma ~ 0.01
    return r


def col(name):
    return NAMES.index(name)


def shock(r, row, spy_ret, low, high):
    r[row, col("SPY")] = spy_ret
    r[row, col(low)] = -0.01
    r[row, col(high)] = 0.01


def test_single_cohort_weights_and_hold():
    r = base_returns()
    shock(r, 70, -0.02, "FXA", "FXY")
    w = target_weights(make_view(r), PARAMS)
    assert w.shape == (T, len(NAMES))
    assert w[70, col("FXA")] == pytest.approx(0.5)
    assert w[70, col("FXY")] == pytest.approx(-0.5)
    assert np.all(w[70, [col(n) for n in ("FXB", "FXC", "FXE", "FXF", "SPY")]] == 0)
    assert np.all(np.isnan(w[71:75]))          # held, no re-decision
    assert np.all(w[75] == 0)                  # closed at open of t+1+5
    assert np.all(w[:70] == 0)
    assert np.all(w[76:] == 0)


def test_hold_days_param():
    r = base_returns()
    shock(r, 70, -0.02, "FXA", "FXY")
    w = target_weights(make_view(r), {"hold_days": 3, "k": 1.0})
    assert np.all(np.isnan(w[71:73]))
    assert np.all(w[73] == 0)


def test_k_param():
    r = base_returns()
    shock(r, 70, -0.008, "FXC", "FXF")       # z ~ -0.8
    w1 = target_weights(make_view(r), {"hold_days": 5, "k": 1.0})
    w05 = target_weights(make_view(r), {"hold_days": 5, "k": 0.5})
    assert np.all(np.nan_to_num(w1) == 0)
    assert w05[70, col("FXC")] == pytest.approx(0.5)
    assert w05[70, col("FXF")] == pytest.approx(-0.5)


def test_overlapping_cohorts_split_and_net():
    r = base_returns()
    shock(r, 70, -0.02, "FXA", "FXY")
    shock(r, 71, -0.02, "FXB", "FXF")
    w = target_weights(make_view(r), PARAMS)
    assert w[71, col("FXA")] == pytest.approx(0.25)
    assert w[71, col("FXY")] == pytest.approx(-0.25)
    assert w[71, col("FXB")] == pytest.approx(0.25)
    assert w[71, col("FXF")] == pytest.approx(-0.25)
    assert np.all(np.isnan(w[72:75]))
    assert w[75, col("FXB")] == pytest.approx(0.5)
    assert w[75, col("FXF")] == pytest.approx(-0.5)
    assert w[75, col("FXA")] == 0
    assert np.all(w[76] == 0)


def test_netting_same_trust():
    r = base_returns()
    shock(r, 70, -0.02, "FXA", "FXY")
    shock(r, 71, -0.02, "FXA", "FXE")
    w = target_weights(make_view(r), PARAMS)
    assert w[71, col("FXA")] == pytest.approx(0.5)
    assert w[71, col("FXY")] == pytest.approx(-0.25)
    assert w[71, col("FXE")] == pytest.approx(-0.25)
    assert np.nansum(np.abs(w[71])) <= 1 + 1e-12


def test_relative_return_uses_valid_trusts_only():
    r = base_returns()
    r[70, col("SPY")] = -0.02
    r[70, :6] = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
    tr = np.ones((T, len(NAMES)), dtype=bool)
    tr[70, col("FXA")] = False               # FXA not valid: 5 of 6 left
    w = target_weights(make_view(r, tradable=tr), PARAMS)
    assert w[70, col("FXB")] == pytest.approx(0.5)
    assert w[70, col("FXY")] == pytest.approx(-0.5)
    assert w[70, col("FXA")] == 0


def test_fewer_than_five_valid_no_cohort():
    r = base_returns()
    shock(r, 70, -0.02, "FXA", "FXY")
    tr = np.ones((T, len(NAMES)), dtype=bool)
    tr[70, col("FXB")] = False
    tr[70, col("FXC")] = False
    w = target_weights(make_view(r, tradable=tr), PARAMS)
    assert np.all(np.nan_to_num(w) == 0)


def test_vol_window_needs_full_history_and_60_valid():
    r = base_returns()
    shock(r, 40, -0.05, "FXA", "FXY")        # before 63 sessions of history
    w = target_weights(make_view(r), PARAMS)
    assert np.all(np.nan_to_num(w) == 0)
    r = base_returns()
    shock(r, 70, -0.02, "FXA", "FXY")
    tr = np.ones((T, len(NAMES)), dtype=bool)
    tr[20:22, col("SPY")] = False             # rows 20,21 invalid, row 22 invalid (prev not tradable): 60 valid left
    w = target_weights(make_view(r, tradable=tr), PARAMS)
    assert w[70, col("FXA")] == pytest.approx(0.5)
    tr[20:23, col("SPY")] = False             # 4 invalid in window rows 7..69 -> 59 valid
    w = target_weights(make_view(r, tradable=tr), PARAMS)
    assert np.all(np.nan_to_num(w) == 0)


def test_spy_never_traded_and_gross():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, size=(T, len(NAMES)))
    w = target_weights(make_view(r), {"hold_days": 8, "k": 0.5})
    assert np.all(np.nan_to_num(w[:, col("SPY")]) == 0)
    assert np.all(np.nansum(np.abs(w), axis=1) <= 1 + 1e-12)
    assert np.all(np.abs(np.nansum(w, axis=1)) < 1e-12)   # dollar neutral


def test_missing_signal_instrument_raises():
    r = base_returns()[:, :6]
    n_t = r.shape[0]
    view = DataView(
        dates=np.arange(n_t).astype("timedelta64[D]") + np.zeros(1, dtype="datetime64[D]"),
        instruments=NAMES[:6],
        asset_class=tuple("us_etf" for _ in range(6)),
        ret_co=np.zeros((n_t, 6)),
        ret_oc=r,
        tradable=np.ones((n_t, 6), dtype=bool),
        listed=np.ones((n_t, 6), dtype=bool),
        delisting=np.zeros((n_t, 6), dtype=bool),
        close=np.full((n_t, 6), 100.0),
        dollar_volume=np.full((n_t, 6), 1e7),
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(n_t),
    )
    with pytest.raises(ValueError):
        target_weights(view, PARAMS)


def test_point_in_time_truncation():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, size=(T, len(NAMES)))
    full = target_weights(make_view(r), PARAMS)
    for cut in (65, 72, 80, 95):
        part = target_weights(make_view(r[: cut + 1]), PARAMS)
        np.testing.assert_array_equal(np.isnan(part), np.isnan(full[: cut + 1]))
        np.testing.assert_allclose(np.nan_to_num(part), np.nan_to_num(full[: cut + 1]))
