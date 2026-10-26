import math

import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, compute_events

T = 100


def make_view(n_pairs=2, T=T):
    """Pairs (2k, 2k+1) with opposite returns, so the equal-weight member mean is exactly 0 and a = r.
    Base pattern +-1 % alternating; flags: nonearn 0, ea 0 everywhere."""
    N = 2 * n_pairs
    x = np.where(np.arange(T) % 2 == 0, 0.01, -0.01)
    rco = np.zeros((T, N))
    for k in range(n_pairs):
        rco[:, 2 * k] = x
        rco[:, 2 * k + 1] = -x
    v = dict(
        dates=np.arange(12000, 12000 + T).astype("datetime64[D]"),
        instruments=tuple("S%02d" % j for j in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=rco,
        ret_oc=np.zeros((T, N)),
        tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.ones((T, N)),
        dollar_volume=np.ones((T, N)) * 1e8,
        universe=np.ones((T, N), dtype=bool),
        extras={"ea_reaction_day": np.zeros((T, N), dtype=np.float32),
                "nonearn_8k_reaction_day": np.zeros((T, N), dtype=np.float32)},
        series={},
        cash_ret=np.zeros(T),
    )
    return v


def build(v):
    return DataView(**v)


def shock(v, t, pair, size):
    """Stock 2*pair gets +size on row t, its twin -size (mean stays 0), both flagged as non-earnings 8-K."""
    v["ret_co"][t, 2 * pair] = size
    v["ret_co"][t, 2 * pair + 1] = -size
    v["extras"]["nonearn_8k_reaction_day"][t, 2 * pair] = 1
    v["extras"]["nonearn_8k_reaction_day"][t, 2 * pair + 1] = 1


def test_basic_event_long_and_short_slots():
    v = make_view()
    shock(v, 70, 0, 0.03)       # sigma = 0.01*sqrt(60/59), z = +-2.975
    w = target_weights(build(v), PARAMS)
    assert np.all(w[:70] == 0)
    assert np.allclose(w[70:75, 0], 0.10)
    assert np.allclose(w[70:75, 1], -0.10)
    assert np.all(w[75:] == 0)          # exit at the open of t+1+hold_days
    assert np.all(w[:, 2:] == 0)        # unflagged pair untouched


def test_z_value_and_threshold():
    v = make_view()
    shock(v, 70, 0, 0.03)
    z = 0.03 / (0.01 * math.sqrt(60 / 59))
    for thr, expect in [(1.5, True), (2.0, True), (2.5, True), (z + 1e-6, False), (z - 1e-6, True)]:
        ev, sd = compute_events(build(v), {"hold_days": 5, "z_threshold": thr})
        assert bool(ev[70, 0]) == expect and bool(ev[70, 1]) == expect
        if expect:
            assert sd[70, 0] == 1 and sd[70, 1] == -1
    # a 2.2 % shock: z = 2.18 -> event at 2.0, none at 2.5
    v = make_view()
    shock(v, 70, 0, 0.022)
    assert compute_events(build(v), {"hold_days": 5, "z_threshold": 2.0})[0][70, 0]
    assert not compute_events(build(v), {"hold_days": 5, "z_threshold": 2.5})[0][70, 0]


def test_hold_days_param():
    v = make_view()
    shock(v, 70, 0, 0.03)
    for hold in (3, 5, 7):
        w = target_weights(build(v), {"hold_days": hold, "z_threshold": 2.0})
        assert np.allclose(w[70:70 + hold, 0], 0.10)
        assert np.all(w[70 + hold:, 0] == 0)


def test_earnings_exclusion_window_and_nan_flags():
    # ea = 1 on t-2 kills the event; on t-3 it does not; NaN in the window kills it; NaN nonearn is no event
    for row, val, expect in [(70, 1.0, False), (69, 1.0, False), (68, 1.0, False), (67, 1.0, True),
                             (69, np.nan, False), (66, np.nan, True)]:
        v = make_view()
        shock(v, 70, 0, 0.03)
        v["extras"]["ea_reaction_day"][row, 0] = val
        ev, _ = compute_events(build(v), PARAMS)
        assert bool(ev[70, 0]) == expect, (row, val)
        assert ev[70, 1]                 # twin unaffected
    v = make_view()
    shock(v, 70, 0, 0.03)
    v["extras"]["nonearn_8k_reaction_day"][70, 0] = np.nan
    ev, _ = compute_events(build(v), PARAMS)
    assert not ev[70, 0] and ev[70, 1]


def test_large_move_without_filing_is_no_event():
    v = make_view()
    shock(v, 70, 0, 0.03)
    v["extras"]["nonearn_8k_reaction_day"][70, :] = 0
    assert np.all(target_weights(build(v), PARAMS) == 0)


def test_non_member_on_event_day_is_no_event():
    v = make_view()
    shock(v, 70, 0, 0.03)
    v["universe"][70, 0] = False
    ev, _ = compute_events(build(v), PARAMS)
    assert not ev[70, 0]


def test_short_history_needs_40_valid_values():
    v = make_view()
    # pair 1 lists on row 40: at row 70 the window 10..69 has 30 valid values, at row 85 it has 45
    v["listed"][:40, 2:] = False
    v["close"][:40, 2:] = np.nan
    v["ret_co"][:40, 2:] = 0.0
    v["universe"][:40, 2:] = False
    shock(v, 70, 1, 0.03)
    shock(v, 85, 1, 0.03)
    ev, _ = compute_events(build(v), PARAMS)
    assert not ev[70, 2] and not ev[70, 3]
    assert ev[85, 2] and ev[85, 3]
    # early rows: event impossible before 40 rows of history even for a full-history stock
    v = make_view()
    shock(v, 30, 0, 0.10)
    ev, _ = compute_events(build(v), PARAMS)
    assert not ev.any()


def test_new_event_replaces_slot():
    v = make_view()
    shock(v, 70, 0, 0.03)
    shock(v, 72, 0, -0.06)              # opposite side two rows later
    ev, sd = compute_events(build(v), PARAMS)
    assert ev[72, 0] and sd[72, 0] == -1
    w = target_weights(build(v), PARAMS)
    assert np.allclose(w[70:72, 0], 0.10)
    assert np.allclose(w[72:77, 0], -0.10)
    assert np.all(w[77:, 0] == 0)
    assert np.allclose(w[72:77, 1], 0.10)


def test_universe_exit_and_untradable_close_the_slot():
    v = make_view()
    shock(v, 70, 0, 0.03)
    v["universe"][72, 0] = False        # leaves the universe for a day, slot closed for good
    v["tradable"][71, 1] = False        # not tradable: slot closed for good
    w = target_weights(build(v), PARAMS)
    assert np.allclose(w[70:72, 0], 0.10) and np.all(w[72:, 0] == 0)
    assert np.allclose(w[70, 1], -0.10) and np.all(w[71:, 1] == 0)


def test_gross_cap_pro_rata():
    v = make_view(n_pairs=12)           # 24 simultaneous events -> gross 2.4 -> each 1/24
    for p in range(12):
        shock(v, 70, p, 0.03)
    w = target_weights(build(v), PARAMS)
    g = np.abs(w).sum(axis=1)
    assert np.all(g <= 1.0 + 1e-12)
    assert np.allclose(np.abs(w[70:75]), 1.0 / 24)
    # 4 events -> gross 0.4, slots keep 0.10
    v = make_view(n_pairs=12)
    for p in range(2):
        shock(v, 70, p, 0.03)
    w = target_weights(build(v), PARAMS)
    assert np.isclose(np.abs(w[70]).sum(), 0.4)


def test_point_in_time_truncation():
    v = make_view(n_pairs=3)
    shock(v, 70, 0, 0.03)
    shock(v, 80, 1, -0.03)
    shock(v, 90, 2, 0.04)
    full = target_weights(build(v), PARAMS)
    for cut in (65, 71, 74, 81, 91):
        vt = {}
        for key, val in v.items():
            if key == "extras":
                vt[key] = {k: a[:cut].copy() for k, a in val.items()}
            elif key in ("instruments", "asset_class", "series"):
                vt[key] = val
            else:
                vt[key] = val[:cut].copy()
        part = target_weights(build(vt), PARAMS)
        assert np.array_equal(part, full[:cut])


def test_determinism_and_event_count():
    v = make_view(n_pairs=3)
    shock(v, 70, 0, 0.03)
    shock(v, 80, 1, -0.03)
    d = build(v)
    assert np.array_equal(target_weights(d, PARAMS), target_weights(d, PARAMS))
    ev, sd = compute_events(d, PARAMS)
    assert ev.sum() == 4
    assert sd[70, 0] == 1 and sd[70, 1] == -1 and sd[80, 2] == -1 and sd[80, 3] == 1


def test_missing_flag_raises():
    v = make_view()
    del v["extras"]["nonearn_8k_reaction_day"]
    with pytest.raises(KeyError):
        target_weights(build(v), PARAMS)
