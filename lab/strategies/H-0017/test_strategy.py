import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS


def make_view(member, close, tradable=None):
    T, N = member.shape
    dates = (np.arange(T).astype("timedelta64[D]") + np.zeros(1, dtype="datetime64[D]")[0])
    if tradable is None:
        tradable = np.isfinite(close)
    zeros = np.zeros((T, N))
    return DataView(
        dates=dates,
        instruments=tuple("S%02d" % i for i in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=zeros.copy(),
        ret_oc=zeros.copy(),
        tradable=np.asarray(tradable, dtype=bool),
        listed=np.isfinite(close),
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(np.isfinite(close), 1e7, np.nan),
        universe=member,
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


def base_panel(T=60, N=6, add_day=40, add_col=5):
    member = np.ones((T, N), dtype=bool)
    member[:add_day, add_col] = False          # S05 joins on add_day
    member[:, 4] = False                        # S04 never a member (gives non-member prices)
    close = np.full((T, N), 50.0)
    return member, close


def test_basic_event_weights_and_hold():
    member, close = base_panel()
    dv = make_view(member, close)
    w = target_weights(dv, {"hold_days": 5})
    # before the event: cash
    assert np.all(w[:40] == 0)
    # event rows 40..44 active (5 rows), row 45 back to cash
    for t in range(40, 45):
        assert w[t, 5] == pytest.approx(-0.5)
        # longs: members S00..S03 (4 names), S04 not a member
        assert np.allclose(w[t, :4], 0.125)
        assert w[t, 4] == 0
        assert np.abs(w[t]).sum() == pytest.approx(1.0)
        assert w[t].sum() == pytest.approx(0.0)
    assert np.all(w[45:] == 0)


def test_hold_days_param_changes_window():
    member, close = base_panel()
    dv = make_view(member, close)
    for h in (3, 5, 10):
        w = target_weights(dv, {"hold_days": h})
        active_rows = np.where(w[:, 5] < 0)[0]
        assert list(active_rows) == list(range(40, 40 + h))


def test_no_event_on_first_row_and_default_params():
    member, close = base_panel()
    dv = make_view(member, close)
    w = target_weights(dv, PARAMS)
    assert PARAMS["hold_days"] == 5
    assert np.all(w[0] == 0)


def test_seasoning_filter_excludes_new_listing():
    member, close = base_panel()
    close[:35, 5] = np.nan       # only 5 valid closes in the 30 rows before day 40
    dv = make_view(member, close)
    w = target_weights(dv, {"hold_days": 5})
    assert np.all(w == 0)


def test_seasoning_boundary_21_of_30():
    member, close = base_panel()
    close[:19, 5] = np.nan       # rows 19..39 valid: exactly 21 valid in rows 10..39
    w = target_weights(make_view(member, close), {"hold_days": 5})
    assert w[40, 5] == pytest.approx(-0.5)
    close[19, 5] = np.nan        # 20 valid -> no event
    w = target_weights(make_view(member, close), {"hold_days": 5})
    assert np.all(w == 0)


def test_seasoning_dropped_when_no_premembership_prices():
    member, close = base_panel()
    close[~member] = np.nan      # panel holds prices only for members
    w = target_weights(make_view(member, close), {"hold_days": 5})
    assert w[40, 5] == pytest.approx(-0.5)


def test_leaving_index_or_untradable_drops_from_short_set():
    member, close = base_panel()
    tradable = np.ones_like(member)
    tradable[42, 5] = False
    member[44:, 5] = False
    w = target_weights(make_view(member, close, tradable), {"hold_days": 5})
    assert w[41, 5] == pytest.approx(-0.5)
    assert w[42, 5] == 0 and np.all(w[42] == 0)      # A empty on 42 -> cash
    assert w[43, 5] == pytest.approx(-0.5)
    assert np.all(w[44] == 0)


def test_missing_close_excluded_from_long_leg():
    member, close = base_panel()
    close[41, 0] = np.nan
    w = target_weights(make_view(member, close), {"hold_days": 5})
    assert w[41, 0] == 0
    assert np.allclose(w[41, 1:4], 0.5 / 3)
    assert np.abs(w[41]).sum() == pytest.approx(1.0)


def test_two_overlapping_events():
    member, close = base_panel(T=60, N=7)
    member[:42, 6] = False       # S06 joins on day 42
    w = target_weights(make_view(member, close), {"hold_days": 5})
    assert w[41, 5] == pytest.approx(-0.5) and w[41, 6] == 0
    assert w[42, 5] == pytest.approx(-0.25) and w[42, 6] == pytest.approx(-0.25)
    assert np.allclose(w[42, :4], 0.125)
    # after its hold window S05 is an ordinary member in the long leg (S00..S03, S05)
    assert w[45, 5] == pytest.approx(0.1) and w[45, 6] == pytest.approx(-0.5)
    assert np.all(np.abs(w).sum(axis=1) <= 1 + 1e-12)


def test_point_in_time_truncation():
    member, close = base_panel()
    full = target_weights(make_view(member, close), {"hold_days": 5})
    for cut in (39, 40, 42, 50):
        part = target_weights(make_view(member[:cut + 1], close[:cut + 1]), {"hold_days": 5})
        assert np.array_equal(part, full[:cut + 1])
