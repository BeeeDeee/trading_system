import math

import numpy as np
import pytest

from strategy import target_weights, PARAMS, unanchored_move, rolling_sigma, book, ASIA, HOME


# ---------------------------------------------------------------- unanchored move

def test_unanchored_single_day_is_ret_oc():
    ret_co = np.array([[0.01], [0.02], [0.03]])
    ret_oc = np.array([[0.05], [-0.02], [0.04]])
    anch = np.array([[True], [True], [True]])
    U = unanchored_move(ret_co, ret_oc, anch)
    assert np.allclose(U[:, 0], np.log1p(ret_oc[:, 0]))


def test_unanchored_accumulates_over_closure():
    ret_co = np.array([[0.01], [0.02], [0.03], [0.04]])
    ret_oc = np.array([[0.05], [-0.02], [0.04], [0.01]])
    anch = np.array([[True], [False], [False], [True]])
    U = unanchored_move(ret_co, ret_oc, anch)
    exp1 = math.log(1.05) + math.log(1.02) + math.log(0.98)
    exp2 = exp1 + math.log(1.03) + math.log(1.04)
    assert U[0, 0] == pytest.approx(math.log(1.05))
    assert U[1, 0] == pytest.approx(exp1)
    assert U[2, 0] == pytest.approx(exp2)
    assert U[3, 0] == pytest.approx(math.log(1.01))


def test_unanchored_nan_before_first_anchor():
    ret = np.full((3, 1), 0.01)
    anch = np.array([[False], [True], [False]])
    U = unanchored_move(ret, ret, anch)
    assert np.isnan(U[0, 0])
    assert np.isfinite(U[1, 0]) and np.isfinite(U[2, 0])


# ---------------------------------------------------------------- sigma

def test_rolling_sigma_window_and_presence():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.05, -0.01])[:, None]
    p = np.ones_like(x, dtype=bool)
    s = rolling_sigma(x, p, 5)            # needs ceil(0.8*5)=4 present rows
    assert np.isnan(s[0, 0]) and np.isnan(s[1, 0]) and np.isnan(s[2, 0])
    assert s[3, 0] == pytest.approx(np.std(x[0:4, 0], ddof=1))
    assert s[4, 0] == pytest.approx(np.std(x[0:5, 0], ddof=1))
    assert s[5, 0] == pytest.approx(np.std(x[1:6, 0], ddof=1))
    p2 = p.copy()
    p2[2, 0] = False
    p2[3, 0] = False
    s2 = rolling_sigma(x, p2, 5)          # rows 1..5 have only 3 present -> NaN
    assert np.isnan(s2[5, 0])
    assert np.isnan(s2[4, 0])


# ---------------------------------------------------------------- book

def _book_inputs(K=4, T=1):
    U = np.zeros((T, K))
    sigma = np.ones((T, K))
    ok = np.ones((T, K), dtype=bool)
    closed = np.zeros((T, K), dtype=bool)
    return U, sigma, ok, closed


def test_book_one_event_hedged_by_peers():
    U, sigma, ok, closed = _book_inputs(K=4)
    U[0, 0] = 1.0
    closed[0, 0] = True
    w, efa = book(U, sigma, ok, closed, np.array([True]), 0.5)
    assert w[0, 0] == pytest.approx(-0.5)
    assert np.allclose(w[0, 1:], 0.5 / 3)
    assert efa[0] == 0.0
    assert w.sum() + efa.sum() == pytest.approx(0.0)


def test_book_small_move_is_no_event():
    U, sigma, ok, closed = _book_inputs(K=4)
    U[0, 0] = 0.4
    closed[0, 0] = True
    w, efa = book(U, sigma, ok, closed, np.array([True]), 0.5)
    assert np.all(w == 0) and efa[0] == 0
    w, efa = book(U, sigma, ok, closed, np.array([True]), 0.25)   # min_z changes the outcome
    assert w[0, 0] == pytest.approx(-0.5)


def test_book_two_events_opposite_sides_net_zero():
    U, sigma, ok, closed = _book_inputs(K=4)
    U[0, 0] = 1.0
    U[0, 1] = -2.0
    closed[0, :2] = True
    w, efa = book(U, sigma, ok, closed, np.array([True]), 0.5)
    assert w[0, 0] == pytest.approx(-0.25)
    assert w[0, 1] == pytest.approx(0.25)
    assert np.allclose(w[0, 2:], 0.0)
    assert efa[0] == 0.0


def test_book_falls_back_to_efa():
    U, sigma, ok, closed = _book_inputs(K=4)
    U[0, :3] = [1.0, 1.0, -1.0]
    closed[0, :3] = True                   # only one open peer -> EFA
    w, efa = book(U, sigma, ok, closed, np.array([True]), 0.5)
    assert np.allclose(w[0, :3], [-0.5 / 3, -0.5 / 3, 0.5 / 3])
    assert w[0, 3] == 0.0
    assert efa[0] == pytest.approx(0.5 / 3)
    assert np.abs(w).sum() + abs(efa[0]) <= 1.0 + 1e-12


def test_book_flat_when_no_hedge_possible():
    U, sigma, ok, closed = _book_inputs(K=4)
    U[0, 0] = 1.0
    closed[0, :3] = True
    w, efa = book(U, sigma, ok, closed, np.array([False]), 0.5)
    assert np.all(w == 0) and efa[0] == 0


def test_book_missing_sigma_or_untradable_excluded():
    U, sigma, ok, closed = _book_inputs(K=5)
    U[0, 0] = 1.0
    U[0, 1] = 1.0
    closed[0, :2] = True
    sigma[0, 1] = np.nan                   # no history -> not an event
    ok[0, 2] = False                       # not tradable -> not a hedge leg
    w, efa = book(U, sigma, ok, closed, np.array([True]), 0.5)
    assert w[0, 0] == pytest.approx(-0.5)
    assert w[0, 1] == 0.0 and w[0, 2] == 0.0
    assert np.allclose(w[0, 3:], 0.25)


# ---------------------------------------------------------------- full DataView

INSTR = ASIA + ("EFA", "SPY")


def _view(T=120, seed=1, closed_rows=None, late=None):
    from lab.framework.data import DataView
    rng = np.random.default_rng(seed)
    N = len(INSTR)
    dates = np.arange(T).astype("datetime64[D]") + np.timedelta64(11000, "D")
    ret_co = rng.normal(0, 0.005, (T, N))
    ret_oc = rng.normal(0, 0.01, (T, N))
    listed = np.ones((T, N), dtype=bool)
    if late is not None:
        j, start = late
        listed[:start, j] = False
        ret_co[:start, j] = 0.0
        ret_oc[:start, j] = 0.0
    close = np.where(listed, 50.0, np.nan)
    series = {}
    mics = sorted(set(HOME.values()))
    for mic in mics:
        cn = np.zeros(T)
        an = np.ones(T)
        series["exchange_holidays_1d." + mic + ".closed_next"] = cn
        series["exchange_holidays_1d." + mic + ".anchored"] = an
    for (mic, r) in (closed_rows or []):
        series["exchange_holidays_1d." + mic + ".closed_next"][r] = 1.0
        series["exchange_holidays_1d." + mic + ".anchored"][r + 1] = 0.0
    return DataView(
        dates=dates,
        instruments=INSTR,
        asset_class=tuple("us_etf" for _ in INSTR),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=listed.copy(),
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e7, np.nan),
        universe=None,
        extras={},
        series=series,
        cash_ret=np.zeros(T),
    )


def test_full_view_event_row_only():
    T = 120
    data = _view(T=T, closed_rows=[("XTKS", 100)])
    j = data.col("EWJ")
    data.ret_oc[100, j] = 0.05            # big up move on the holiday eve -> short EWJ
    W = target_weights(data, PARAMS)
    assert W.shape == (T, len(INSTR))
    assert W[100, j] == pytest.approx(-0.5)
    others = [data.col(s) for s in ASIA if s != "EWJ"]
    assert np.allclose(W[100, others], 0.5 / len(others))
    assert W[100, data.col("SPY")] == 0.0
    assert W[100, data.col("EFA")] == 0.0
    assert W[100].sum() == pytest.approx(0.0)
    mask = np.ones(T, dtype=bool)
    mask[100] = False
    assert np.all(W[mask] == 0.0)          # flat on every other row (exit at the next open)
    assert np.all(np.abs(W).sum(axis=1) <= 1.0 + 1e-12)


def test_full_view_lunar_new_year_cluster_uses_efa():
    T = 120
    closed = [(m, 100) for m in ("XHKG", "XKRX", "XTAI", "XSES", "XKLS", "XASX", "XBKK",
                                  "XIDX", "XPHS", "XNZE", "XBOM")]
    data = _view(T=T, closed_rows=closed)   # only XTKS (EWJ) reopens -> 1 peer -> EFA
    for s in ("FXI", "EWY"):
        data.ret_oc[100, data.col(s)] = -0.05
    W = target_weights(data, PARAMS)
    assert W[100, data.col("FXI")] > 0 and W[100, data.col("EWY")] > 0
    assert W[100, data.col("EWJ")] == 0.0
    assert W[100, data.col("EFA")] == pytest.approx(-W[100, [data.col(s) for s in ASIA]].sum())
    assert W[100].sum() == pytest.approx(0.0)


def test_full_view_late_listing_not_event_nor_hedge():
    T = 120
    data = _view(T=T, closed_rows=[("XTKS", 100)], late=(INSTR.index("EIDO"), 80))
    data.ret_oc[100, data.col("EWJ")] = 0.05
    W = target_weights(data, PARAMS)      # EIDO has 20 rows < 51 needed for vol_days=63
    assert W[100, data.col("EIDO")] == 0.0
    W2 = target_weights(data, {**PARAMS, "vol_days": 21})   # vol_days changes the hedge basket
    assert W2[100, data.col("EIDO")] > 0.0


def test_full_view_multi_day_closure_holds_and_accumulates():
    T = 120
    data = _view(T=T, closed_rows=[("XKRX", 100)])
    s = data.series
    s["exchange_holidays_1d.XKRX.closed_next"][101] = 1.0   # closed two sessions
    s["exchange_holidays_1d.XKRX.anchored"][102] = 0.0
    j = data.col("EWY")
    data.ret_oc[100, j] = 0.04
    data.ret_co[101, j] = 0.0
    data.ret_oc[101, j] = 0.01
    W = target_weights(data, PARAMS)
    assert W[100, j] == pytest.approx(-0.5)
    assert W[101, j] == pytest.approx(-0.5)
    assert W[102, j] == 0.0


def test_missing_calendar_series_raises():
    data = _view(T=60)
    del data.series["exchange_holidays_1d.XTKS.closed_next"]
    with pytest.raises(KeyError):
        target_weights(data, PARAMS)


def test_nan_calendar_is_never_closed():
    T = 120
    data = _view(T=T, closed_rows=[("XTKS", 100)])
    data.series["exchange_holidays_1d.XTKS.closed_next"][:] = np.nan
    data.series["exchange_holidays_1d.XTKS.anchored"][:] = np.nan
    data.ret_oc[100, data.col("EWJ")] = 0.05
    W = target_weights(data, PARAMS)
    assert np.all(W == 0.0)


def test_deterministic_and_future_insensitive():
    T = 120
    data = _view(T=T, closed_rows=[("XTKS", 100), ("XHKG", 90)])
    data.ret_oc[100, data.col("EWJ")] = 0.05
    data.ret_oc[90, data.col("FXI")] = 0.05
    W = target_weights(data, PARAMS)
    W2 = target_weights(data, PARAMS)
    assert np.array_equal(W, W2)
    data.ret_oc[95:] = 0.3                 # change the future after row 94
    data.ret_co[95:] = -0.2
    W3 = target_weights(data, PARAMS)
    assert np.array_equal(W[:95], W3[:95])
