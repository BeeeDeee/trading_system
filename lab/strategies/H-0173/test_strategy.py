import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
from strategy import _oil_row_returns, _oil_session_returns, _rolling_slope, _rolling_std, _build_book, UNIVERSE, OIL_KEY


# ---------------------------------------------------------------------------------------------------------------
# hand-made inputs
# ---------------------------------------------------------------------------------------------------------------
def make_view(shock=0.19, shock_session=600, t_rows=700, late=None, lose=None, seed=1, keep=None):
    """32 ETFs with oil betas linspace(-1, 1) in UNIVERSE order, one big oil-specific move on shock_session."""
    rng = np.random.default_rng(seed)
    n = len(UNIVERSE)
    beta = np.linspace(-1.0, 1.0, n)
    common = rng.normal(0.0, 0.01, t_rows)
    e_true = rng.normal(0.0, 0.01, t_rows)
    common[shock_session] = 0.0
    e_true[shock_session] = shock
    noise = rng.normal(0.0, 0.001, (t_rows, n))
    r = common[:, None] + beta[None, :] * e_true[:, None] + noise
    r[0] = 0.0
    o_sess = e_true + 0.5 * common
    p_sess = 50.0 * np.cumprod(1.0 + o_sess)
    p_row = np.full(t_rows, np.nan)
    p_row[1:] = p_sess[:-1]                                  # next_morning: row r holds the value of session r-1
    close = 20.0 * np.cumprod(1.0 + r, axis=0)
    listed = np.ones((t_rows, n), dtype=bool)
    if late is not None:
        name, first = late
        j = UNIVERSE.index(name)
        listed[:first, j] = False
        close[:first, j] = np.nan
        r[:first, j] = 0.0
    if lose is not None:
        name, row = lose
        close[row, UNIVERSE.index(name)] = np.nan
    ret_co = np.zeros((t_rows, n))
    ret_oc = np.where(np.isfinite(close), r, 0.0)
    tradable = np.isfinite(close)
    dates = (np.arange(t_rows) + 11000).astype("datetime64[D]")
    k = t_rows if keep is None else keep                     # truncated copy: first k rows only
    return DataView(
        dates=dates[:k],
        instruments=tuple(UNIVERSE),
        asset_class=tuple("us_etf" for _ in UNIVERSE),
        ret_co=ret_co[:k],
        ret_oc=ret_oc[:k],
        tradable=tradable[:k],
        listed=listed[:k],
        delisting=np.zeros((k, n), dtype=bool),
        close=close[:k],
        dollar_volume=np.where(np.isfinite(close), 1e7, np.nan)[:k],
        universe=None,
        extras={},
        series={OIL_KEY: p_row[:k]},
        cash_ret=np.zeros(k),
    )


def cols(names):
    return [UNIVERSE.index(x) for x in names]


# ---------------------------------------------------------------------------------------------------------------
# rule (1): oil returns
# ---------------------------------------------------------------------------------------------------------------
def test_oil_row_returns_missing_nonpositive_winsor():
    p = np.array([np.nan, 50.0, 55.0, np.nan, 44.0, -1.0, 10.0, 12.0, 30.0])
    r = _oil_row_returns(p)
    assert np.isnan(r[0]) and np.isnan(r[1])          # no previous value
    assert r[2] == pytest.approx(0.1)
    assert np.isnan(r[3])                             # missing price
    assert r[4] == pytest.approx(-0.2)                # 44 / 55 - 1, vs the last available value
    assert np.isnan(r[5])                             # price <= 0
    assert np.isnan(r[6])                             # previous price <= 0
    assert r[7] == pytest.approx(0.2)
    assert r[8] == pytest.approx(0.2)                 # +150 % winsorized at +20 %


def test_oil_session_alignment():
    p = np.array([np.nan, 50.0, 55.0, 60.5])
    o = _oil_session_returns(p)
    assert o[1] == pytest.approx(0.1)                 # session 1 value appears at row 2
    assert o[2] == pytest.approx(0.1)
    assert np.isnan(o[3])                             # last session not yet known


# ---------------------------------------------------------------------------------------------------------------
# rolling estimates
# ---------------------------------------------------------------------------------------------------------------
def test_rolling_slope_and_min_obs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=30)
    y = 2.0 * x + 1.0
    y[3] = np.nan
    s = _rolling_slope(y, x, 10, 8)
    assert np.isnan(s[7])                             # rows 0..7 minus the NaN: 7 obs
    assert s[8] == pytest.approx(2.0)
    assert np.allclose(s[8:], 2.0)


def test_rolling_std():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0])
    s = _rolling_std(x, 3, 2)
    assert np.isnan(s[0])
    assert s[2] == pytest.approx(1.0)
    assert s[4] == pytest.approx(np.std([3.0, 4.0], ddof=1))
    assert s[5] == pytest.approx(np.std([4.0, 6.0], ddof=1))


def test_rolling_short_view():
    x = np.arange(5.0)
    assert np.isnan(_rolling_slope(x, x, 252, 200)).all()
    assert np.isnan(_rolling_std(x, 252, 200)).all()


# ---------------------------------------------------------------------------------------------------------------
# rule (6): the event book
# ---------------------------------------------------------------------------------------------------------------
def _book_inputs(t_rows=20, m=12):
    z = np.zeros(t_rows)
    expo = np.tile(np.arange(m, dtype=float), (t_rows, 1))
    eligible = np.ones((t_rows, m), dtype=bool)
    has_price = np.ones((t_rows, m), dtype=bool)
    return z, expo, eligible, has_price


def test_book_long_high_short_low_hold_five_then_flat():
    z, expo, el, hp = _book_inputs()
    z[3] = 2.0
    z[5] = 3.0                                        # book open: ignored
    z[8] = 3.0                                        # exit decided at 8, position still open: ignored
    z[9] = -1.6                                       # flat again: mirror book
    w, ev = _build_book(z, expo, el, hp, 3, 1.5)
    assert [e[0] for e in ev] == [3, 9]
    assert np.allclose(w[3, 9:], 0.5 / 3) and np.allclose(w[3, :3], -0.5 / 3) and np.allclose(w[3, 3:9], 0.0)
    assert np.isnan(w[4:8]).all()
    assert np.allclose(w[8], 0.0)
    assert np.allclose(w[9, :3], 0.5 / 3) and np.allclose(w[9, 9:], -0.5 / 3)
    assert np.allclose(w[14], 0.0)
    fin = np.isfinite(w).all(axis=1)
    assert (np.abs(w[fin]).sum(axis=1) <= 1.0 + 1e-12).all()
    assert np.allclose(w[fin].sum(axis=1), 0.0)


def test_book_threshold_and_min_eligible():
    z, expo, el, hp = _book_inputs()
    z[2] = 1.4
    z[4] = 2.0
    el[4, :3] = False                                 # 9 eligible: no event
    z[6] = 2.0
    el[6, 11] = False                                 # 11 eligible: event without column 11
    w, ev = _build_book(z, expo, el, hp, 3, 1.5)
    assert [e[0] for e in ev] == [6]
    assert np.allclose(w[6, 8:11], 0.5 / 3) and w[6, 11] == 0.0
    w2, ev2 = _build_book(z, expo, el, hp, 3, 1.25)   # lower threshold fires at row 2
    assert [e[0] for e in ev2] == [2]


def test_book_n_leg_and_lost_price():
    z, expo, el, hp = _book_inputs()
    z[3] = 2.0
    hp[5, 11] = False                                 # long name loses its price mid-slot
    w, ev = _build_book(z, expo, el, hp, 2, 1.5)
    assert np.allclose(w[3, 10:], 0.25) and np.allclose(w[3, :2], -0.25)
    assert np.isnan(w[4]).all()
    assert w[5, 11] == 0.0 and np.isnan(np.delete(w[5], 11)).all()
    assert np.allclose(w[8], 0.0)


# ---------------------------------------------------------------------------------------------------------------
# end to end on a DataView
# ---------------------------------------------------------------------------------------------------------------
def test_end_to_end_positive_shock_late_listing_and_lost_price():
    data = make_view(late=("INDA", 450), lose=("NORW", 603))
    w = target_weights(data, {"n_leg": 3, "shock_z": 4.0})
    assert w.shape == (700, len(UNIVERSE))
    rows = np.flatnonzero(np.isfinite(w).any(axis=1))
    assert rows[0] == 601                            # the only event: shock of session 600, seen at 601
    # INDA (highest beta) has < 200 observations in the window: not eligible
    assert np.allclose(w[601, cols(["NORW", "EPU", "EIS"])], 0.5 / 3)
    assert np.allclose(w[601, cols(["EWA", "EWC", "EWD"])], -0.5 / 3)
    assert w[601, cols(["INDA"])[0]] == 0.0
    assert abs(np.abs(w[601]).sum() - 1.0) < 1e-12
    assert np.isnan(w[602]).all()
    j = cols(["NORW"])[0]
    assert w[603, j] == 0.0 and np.isnan(np.delete(w[603], j)).all()
    assert np.allclose(w[606], 0.0)
    assert list(rows) == [601, 603, 606]


def test_end_to_end_negative_shock_mirror_and_params():
    data = make_view(shock=-0.19)
    w = target_weights(data, {"n_leg": 2, "shock_z": 4.0})
    assert np.flatnonzero(np.isfinite(w).any(axis=1))[0] == 601
    assert np.allclose(w[601, cols(["EWA", "EWC"])], 0.25)
    assert np.allclose(w[601, cols(["EIS", "INDA"])], -0.25)
    w3 = target_weights(data, dict(PARAMS))
    assert not np.array_equal(np.nan_to_num(w, nan=9.0), np.nan_to_num(w3, nan=9.0))


def test_no_decision_before_lookbacks_and_gross():
    data = make_view()
    w = target_weights(data, dict(PARAMS))
    assert not np.isfinite(w[:400]).any()           # b needs 200 obs, then std of e / exposures 200 more
    fin = np.isfinite(w)
    assert (np.abs(np.where(fin, w, 0.0)).sum(axis=1) <= 1.0 + 1e-12).all()


def test_truncation_point_in_time():
    data = make_view()
    full = target_weights(data, dict(PARAMS))
    for cut in (450, 601, 603, 640):
        part = target_weights(make_view(keep=cut + 1), dict(PARAMS))
        assert np.array_equal(np.nan_to_num(full[: cut + 1], nan=9.0), np.nan_to_num(part, nan=9.0))
