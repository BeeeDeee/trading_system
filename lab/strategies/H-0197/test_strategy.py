import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
from strategy import anniversary_source_rows, book_weights, compute_events, release_sessions


def weekday_dates(n, start_day=12002):
    """n consecutive weekdays starting near day start_day since the epoch (no holidays)."""
    d = np.arange(start_day, start_day + 2 * n).astype("datetime64[D]")
    d = d[np.is_busday(d)]
    return d[:n]


def make_view(T, N, rng, ea=None, universe=None, tradable=None, close=None, sigma=0.01):
    dates = weekday_dates(T)
    names = tuple("E%d" % (i + 1) for i in range(N))
    ret_co = (rng.standard_normal((T, N)) * sigma * 0.3).astype(np.float32)
    ret_oc = (rng.standard_normal((T, N)) * sigma).astype(np.float32)
    ret_co[0] = 0.0
    ret_oc[0] = 0.0
    if close is None:
        close = np.full((T, N), 50.0, dtype=np.float32)
    if universe is None:
        universe = np.ones((T, N), dtype=bool)
    if tradable is None:
        tradable = np.ones((T, N), dtype=bool)
    if ea is None:
        ea = np.zeros((T, N), dtype=np.float32)
    return DataView(
        dates=dates,
        instruments=names,
        asset_class=tuple("us_equity" for _ in names),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=tradable,
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.full((T, N), 1e8, dtype=np.float32),
        universe=universe,
        extras={"ea_reaction_day": ea.astype(np.float32), "liq_rank": np.ones((T, N), dtype=np.float32)},
        series={},
        cash_ret=np.zeros(T),
    )


# ---------------------------------------------------------------- rule (1)
def test_release_dedup():
    ea = np.zeros((200, 2), dtype=np.float32)
    ea[:, 1] = np.nan
    ea[10, 0] = 1     # release
    ea[25, 0] = 1     # 15 after a flag -> ignored
    ea[40, 0] = 1     # 15 after the ignored flag -> ignored (any earlier flag counts)
    ea[61, 0] = 1     # 21 after the last flag -> release
    ea[81, 0] = 1     # exactly 20 after -> ignored
    ea[150, 1] = 1    # NaN elsewhere is no release
    rel = release_sessions(ea)
    assert list(np.nonzero(rel[:, 0])[0]) == [10, 61]
    assert list(np.nonzero(rel[:, 1])[0]) == [150]


# ---------------------------------------------------------------- rule (4)
def test_anniversary_rows_weekday_calendar():
    dates = weekday_dates(600)
    pos, has = anniversary_source_rows(dates)
    d = dates.astype(np.int64)
    # 364 days = 52 weeks = 260 weekdays; t + 2 weekdays - 260 -> row t - 258
    for t in (258, 300, 599):
        assert has[t]
        assert pos[t] == t - 258
        assert d[pos[t]] + 364 == np.busday_offset(dates[t], 2).astype(np.int64)
    assert not has[:258].any()


def test_anniversary_rows_with_gap_in_past_calendar():
    dates = weekday_dates(600)
    dates = np.delete(dates, 100)          # a past holiday: that date has no session
    pos, has = anniversary_source_rows(dates)
    d = dates.astype(np.int64)
    for t in np.nonzero(has)[0]:
        assert d[pos[t]] + 364 == np.busday_offset(dates[t], 2).astype(np.int64)
    # the row whose source would be the removed date has no anniversary
    # weekday positions 258..599 have a source (342), minus the one whose source (weekday 100) is gone
    assert has.sum() == 341


# ---------------------------------------------------------------- rules (2)-(5) on a hand-built panel
def _panel(rng, N=60, T=700, quarter=65, big=None):
    """N stocks releasing every `quarter` sessions (13 weeks) from staggered offsets."""
    ea = np.zeros((T, N), dtype=np.float32)
    offs = {}
    for j in range(N):
        o = 70 + (j % 13)
        offs[j] = o
        ea[o::quarter, j] = 1.0
    view = make_view(T, N, rng, ea=ea)
    return view, offs


def _set_reaction(view, j, R, move):
    view.ret_oc[R, j] = move
    view.ret_oc[R + 1, j] = 0.0
    view.ret_co[R, j] = 0.0
    view.ret_co[R + 1, j] = 0.0


def test_events_long_and_short():
    rng = np.random.default_rng(1)
    view, offs = _panel(rng)
    # stock 0: releases at 70, 135, 200, 265, 330, 395 ...; anniversary of L=70 is row 330 -> t = 328,
    # Rprev = 265 (63 sessions before t). Make reaction at 265 extreme positive.
    _set_reaction(view, 0, 265, 0.25)
    # stock 13 has offset 70 as well: extreme negative reaction at 265
    _set_reaction(view, 13, 265, -0.25)
    side = compute_events(view, PARAMS)
    assert side[328, 0] == 1.0
    assert side[328, 13] == -1.0
    # events only on rows t = L + 258 of each stock
    for j in range(view.ret_oc.shape[1]):
        rows = np.nonzero(side[:, j])[0]
        for t in rows:
            assert (t + 2 - offs[j]) % 65 == 0
    # events need >= 200 reference values: no event before enough releases have accumulated
    assert not side[:300].any()


def test_event_requirements():
    rng = np.random.default_rng(2)
    view, offs = _panel(rng)
    _set_reaction(view, 0, 265, 0.25)
    _set_reaction(view, 13, 265, 0.25)
    _set_reaction(view, 26, 265, 0.25)
    _set_reaction(view, 39, 265, 0.25)
    view.universe[328, 13] = False            # not a member at t
    view.tradable[328, 26] = False            # not tradable at t
    view.close[328, 39] = np.nan              # missing return data at t
    side = compute_events(view, PARAMS)
    assert side[328, 0] == 1.0
    assert side[328, 13] == 0.0
    assert side[328, 26] == 0.0
    assert side[328, 39] == 0.0


def test_early_release_blocks_event():
    rng = np.random.default_rng(3)
    view, offs = _panel(rng)
    _set_reaction(view, 0, 265, 0.25)
    view.extras["ea_reaction_day"][320, 0] = 1.0   # current release came early (and 55 after Rprev)
    side = compute_events(view, PARAMS)
    assert side[328, 0] == 0.0


def test_prev_release_window():
    rng = np.random.default_rng(4)
    view, offs = _panel(rng)
    # stock 0: remove the releases at 135, 200, 265 -> most recent release before 328 is 70 (258 back)
    view.extras["ea_reaction_day"][135, 0] = 0.0
    view.extras["ea_reaction_day"][200, 0] = 0.0
    view.extras["ea_reaction_day"][265, 0] = 0.0
    _set_reaction(view, 0, 70, 0.25)
    side = compute_events(view, PARAMS)
    assert side[328, 0] == 0.0


def test_leg_fraction_changes_events():
    rng = np.random.default_rng(5)
    view, offs = _panel(rng)
    a = compute_events(view, {"hold_days": 5, "leg_fraction": 0.15})
    b = compute_events(view, {"hold_days": 5, "leg_fraction": 0.35})
    assert (a != 0).sum() < (b != 0).sum()
    assert np.all((a != 0) <= (b != 0))


def test_reaction_needs_40_returns():
    rng = np.random.default_rng(6)
    view, offs = _panel(rng)
    _set_reaction(view, 0, 265, 0.25)
    # rows 204..263 (R-61 .. R-2): no price on 204..224 -> returns 204..225 invalid, 38 valid remain
    view.close[204:225, 0] = np.nan
    side = compute_events(view, PARAMS)
    assert side[328, 0] == 0.0
    view2, _ = _panel(np.random.default_rng(6))
    _set_reaction(view2, 0, 265, 0.25)
    view2.close[204:223, 0] = np.nan     # 40 valid returns remain (rows 224..263)
    assert compute_events(view2, PARAMS)[328, 0] == 1.0


# ---------------------------------------------------------------- rules (6), (7)
def test_book_weights_single_long():
    T, N = 12, 5
    side = np.zeros((T, N))
    side[2, 0] = 1.0
    member = np.ones((T, N), dtype=bool)
    tradable = np.ones((T, N), dtype=bool)
    w = book_weights(side, member, tradable, 5)
    assert np.allclose(w[:2], 0.0)
    for t in range(2, 7):
        assert w[t, 0] == pytest.approx(0.05)
        assert np.allclose(w[t, 1:], -0.05 / 4)
        assert abs(w[t].sum()) < 1e-12
    assert np.allclose(w[7:], 0.0)


def test_book_weights_restart_universe_exit_and_caps():
    T, N = 20, 100
    side = np.zeros((T, N))
    side[1, 0] = 1.0
    side[4, 0] = 1.0          # restart: held rows 1..8 with hold 5
    side[1, 1] = -1.0
    member = np.ones((T, N), dtype=bool)
    member[3, 1] = False      # leaves LIQ-500: closed from row 3 on, also after rejoining
    tradable = np.ones((T, N), dtype=bool)
    tradable[5, 50] = False   # untradable member: not in the basket that row
    w = book_weights(side, member, tradable, 5)
    assert np.all(w[1:9, 0] > 0) and w[9, 0] <= 0
    assert w[2, 1] < 0 and w[3, 1] == 0 and w[4, 1] <= 0
    assert w[5, 50] == 0.0
    # 30 longs, 5 shorts: longs 0.5/30 each, shorts 0.05 each, net 0.25 offset by the basket
    side2 = np.zeros((T, N))
    side2[0, :30] = 1.0
    side2[0, 30:35] = -1.0
    w2 = book_weights(side2, np.ones((T, N), dtype=bool), np.ones((T, N), dtype=bool), 3)
    assert np.allclose(w2[0, :30], 0.5 / 30)
    assert np.allclose(w2[0, 30:35], -0.05)
    assert np.allclose(w2[0, 35:], -0.25 / 65)
    assert abs(w2[0].sum()) < 1e-12
    assert np.abs(w2[0]).sum() <= 1.0 + 1e-12


def test_hold_days_changes_output():
    T, N = 20, 10
    side = np.zeros((T, N))
    side[2, 3] = -1.0
    m = np.ones((T, N), dtype=bool)
    w3 = book_weights(side, m, m, 3)
    w8 = book_weights(side, m, m, 8)
    assert (w3[:, 3] < 0).sum() == 3
    assert (w8[:, 3] < 0).sum() == 8


# ---------------------------------------------------------------- whole strategy
def test_full_strategy_gross_neutral_and_point_in_time():
    rng = np.random.default_rng(7)
    view, offs = _panel(rng)
    w = target_weights(view, PARAMS)
    assert w.shape == view.ret_oc.shape
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-9)
    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-9)
    assert (w != 0).any()
    # truncation: weights on rows <= cut do not change
    for cut in (330, 400, 520):
        v2, _ = _panel(np.random.default_rng(7))
        v2 = make_view(cut + 1, v2.ret_oc.shape[1], np.random.default_rng(0),
                       ea=v2.extras["ea_reaction_day"][:cut + 1].copy())
        v2.ret_oc[:] = view.ret_oc[:cut + 1]
        v2.ret_co[:] = view.ret_co[:cut + 1]
        w2 = target_weights(v2, PARAMS)
        assert np.allclose(w2, w[:cut + 1])


def test_missing_input_raises():
    rng = np.random.default_rng(8)
    view = make_view(50, 3, rng)
    del view.extras["ea_reaction_day"]
    with pytest.raises(KeyError):
        target_weights(view, PARAMS)
