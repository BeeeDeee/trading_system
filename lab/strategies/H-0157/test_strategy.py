import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import PARAMS, events, target_weights

ETFS = ("EWJ", "EWH", "FXI", "EWY", "EWT", "EWS", "EWM", "EWA")
EXCH = ("XTKS", "XHKG", "XKRX", "XTAI", "XSES", "XKLS", "XASX")
N_STOCKS = 60
T_FULL = 420
EVENT_ROW = 380


def _dates(T):
    base = (np.zeros(1, dtype=np.int64) + 13000).astype("datetime64[D]")[0]
    return np.busday_offset(base, np.arange(T), roll="forward").astype("datetime64[D]")


def make_view(T=T_FULL, linked=(0,), move=0.05, closure_exch="XTKS", closure_row=EVENT_ROW,
              cut=None, drop_etf=None, drop_series=None, leave_row=None, leave_col=0, seed=0):
    """Hand-made market: stocks S00..S59 plus the 8 country ETFs (never members).

    Raw return of stock j = f + idio_j; of ETF k = f + e_k. Stocks in `linked` load on EWJ's country
    shock e_EWJ, so their excess return correlates ~0.95 with EWJ's (assigned to XTKS), all others ~0.
    On `closure_row` every linked stock gets an extra `move`. `closure_exch` has a two-session closure
    whose first unanchored session is `closure_row`.
    """
    rng = np.random.default_rng(seed)
    names = tuple("S%02d" % j for j in range(N_STOCKS)) + ETFS
    N = len(names)
    f = rng.normal(0.0, 0.01, T)
    e = rng.normal(0.0, 0.01, (T, len(ETFS)))
    idio = rng.normal(0.0, 0.01, (T, N_STOCKS))
    r = np.zeros((T, N))
    r[:, :N_STOCKS] = f[:, None] + idio
    r[:, N_STOCKS:] = f[:, None] + e
    for j in linked:
        r[:, j] = f + e[:, 0] + 0.003 * idio[:, j]
        r[closure_row, j] += move
    uni = np.zeros((T, N), dtype=bool)
    uni[:, :N_STOCKS] = True
    if leave_row is not None:
        uni[leave_row:, leave_col] = False
    dv = np.zeros((T, N))
    dv[:, :N_STOCKS] = 1e6 * (np.arange(N_STOCKS) + 1.0)[None, :]
    dv[:, N_STOCKS:] = 1e9
    series = {}
    for x in EXCH:
        a = np.ones(T)
        c = np.zeros(T)
        if x == closure_exch:
            a[closure_row] = 0.0          # no home session since the previous US session
            a[closure_row + 1] = 0.0      # still shut
            c[closure_row - 1] = 1.0      # the day before: nothing before the next US session
            c[closure_row] = 1.0          # still shut before the next US session
        series["exchange_holidays_1d." + x + ".anchored"] = a
        series["exchange_holidays_1d." + x + ".closed_next"] = c
    if drop_series is not None:
        del series[drop_series]
    keep = np.ones(N, dtype=bool)
    if drop_etf is not None:
        keep[names.index(drop_etf)] = False
    rows = slice(0, T if cut is None else cut + 1)
    names_k = tuple(n for n, k in zip(names, keep) if k)
    return DataView(
        dates=_dates(T)[rows],
        instruments=names_k,
        asset_class=tuple("us_equity" if n.startswith("S") else "us_etf" for n in names_k),
        ret_co=np.zeros((T, N))[rows][:, keep],
        ret_oc=r[rows][:, keep],
        tradable=np.ones((T, N), dtype=bool)[rows][:, keep],
        listed=np.ones((T, N), dtype=bool)[rows][:, keep],
        delisting=np.zeros((T, N), dtype=bool)[rows][:, keep],
        close=np.full((T, N), 10.0)[rows][:, keep],
        dollar_volume=dv[rows][:, keep],
        universe=uni[rows][:, keep],
        extras={},
        series={k: v[rows] for k, v in series.items()},
        cash_ret=np.zeros(T)[rows],
    )


def _w(view, **kw):
    p = dict(PARAMS)
    p.update(kw)
    return target_weights(view, p)


def test_event_slot_and_hedge():
    v = make_view()
    w = _w(v)
    assert w.shape == (T_FULL, N_STOCKS + len(ETFS))
    # fade the positive unanchored move: short 0.20 on decision rows 380..382, flat from 383
    assert np.allclose(w[EVENT_ROW:EVENT_ROW + 3, 0], -0.20)
    assert w[EVENT_ROW + 3, 0] == 0.0
    # hedge: +0.20 split over the 50 most liquid unassigned members without a slot (S10..S59)
    for u in range(EVENT_ROW, EVENT_ROW + 3):
        assert np.allclose(w[u, 10:N_STOCKS], 0.20 / 50)
        assert np.all(w[u, 1:10] == 0.0)
        assert abs(w[u].sum()) < 1e-12
        assert np.isclose(np.abs(w[u]).sum(), 0.40)
    # cash otherwise, ETFs never held
    other = np.ones(T_FULL, dtype=bool)
    other[EVENT_ROW:EVENT_ROW + 3] = False
    assert np.all(w[other] == 0.0)
    assert np.all(w[:, N_STOCKS:] == 0.0)


def test_negative_move_goes_long():
    w = _w(make_view(move=-0.05))
    assert np.allclose(w[EVENT_ROW:EVENT_ROW + 3, 0], 0.20)
    assert np.allclose(w[EVENT_ROW, 10:N_STOCKS], -0.20 / 50)


def test_hold_days_param():
    w = _w(make_view(), hold_days=5)
    assert np.allclose(w[EVENT_ROW:EVENT_ROW + 5, 0], -0.20)
    assert w[EVENT_ROW + 5, 0] == 0.0
    w2 = _w(make_view(), hold_days=2)
    assert np.allclose(w2[EVENT_ROW:EVENT_ROW + 2, 0], -0.20)
    assert w2[EVENT_ROW + 2, 0] == 0.0


def test_z_min_param():
    assert np.any(_w(make_view(), z_min=0.25) != 0.0)
    assert np.all(_w(make_view(), z_min=1e6) == 0.0)
    # a tiny move passes z_min = 0 but not z_min = 0.5
    v = make_view(move=0.0)
    mask0, _ = events(v, {"hold_days": 3, "z_min": 0.0})
    assert mask0[EVENT_ROW, 0]
    big_z = 1e6
    mask1, _ = events(v, {"hold_days": 3, "z_min": big_z})
    assert not mask1.any()


def test_only_first_unanchored_session_and_home_exchange():
    mask, side = events(make_view(), PARAMS)
    assert mask[EVENT_ROW, 0] and side[EVENT_ROW, 0] == -1.0
    assert mask.sum() == 1                      # not on row 381 (anchored(t-1) = 0)
    # a closure of another exchange does not trigger a XTKS-assigned stock
    mask2, _ = events(make_view(closure_exch="XHKG"), PARAMS)
    assert not mask2.any()


def test_unassigned_stock_has_no_event():
    w = _w(make_view(linked=()))
    assert np.all(w == 0.0)


def test_no_event_before_assignment_history():
    # closure at row 200: no month start with 252 sessions of history yet
    w = _w(make_view(closure_row=200))
    assert np.all(w == 0.0)


def test_early_exit_when_leaving_universe():
    w = _w(make_view(leave_row=EVENT_ROW + 1, leave_col=0))
    assert np.isclose(w[EVENT_ROW, 0], -0.20)
    assert np.all(w[EVENT_ROW + 1:, 0] == 0.0)
    assert np.all(w[EVENT_ROW + 1:] == 0.0)


def test_slot_gross_cap():
    # four XTKS-linked stocks fire together: slot gross 0.8 -> scaled to 0.5
    w = _w(make_view(linked=(0, 1, 2, 3)))
    u = EVENT_ROW
    assert np.allclose(w[u, :4], -0.125)
    assert np.isclose(np.abs(w[u, :4]).sum(), 0.5)
    assert np.isclose(w[u, 4:N_STOCKS].sum(), 0.5)
    # hedge names: the 50 highest dollar volume among S04..S59 -> S10..S59
    assert np.allclose(w[u, 10:N_STOCKS], 0.5 / 50)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)


def test_point_in_time_truncation():
    full = _w(make_view(linked=(0, 1)))
    for cut in (EVENT_ROW - 1, EVENT_ROW, EVENT_ROW + 1, EVENT_ROW + 3):
        part = _w(make_view(linked=(0, 1), cut=cut))
        assert np.array_equal(part, full[:cut + 1])


def test_short_view_does_not_crash():
    w = _w(make_view(cut=10, closure_row=5))
    assert np.all(w == 0.0)


def test_missing_inputs_raise():
    with pytest.raises(Exception):
        _w(make_view(drop_etf="EWT"))
    with pytest.raises(Exception):
        _w(make_view(drop_series="exchange_holidays_1d.XKLS.closed_next"))


def test_deterministic_and_diagnostic_matches_book():
    v = make_view(linked=(0, 1, 2))
    assert np.array_equal(_w(v), _w(v))
    mask, side = events(v, PARAMS)
    w = _w(v)
    ev_t, ev_j = np.nonzero(mask)
    for t, j in zip(ev_t, ev_j):
        assert np.sign(w[t, j]) == side[t, j]
