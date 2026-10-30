import numpy as np
import pytest
from lab.framework.data import DataView, synthetic
from strategy import target_weights, PARAMS, month_end_decisions, events


def weekday_dates(n_days=130):
    base = synthetic().dates[0]                     # a Monday at the start of a month
    d = base + np.arange(n_days)
    return d[np.is_busday(d)]


def month_end_rows_by_hand(dates):
    # in a calendar of all weekdays the last row of each month (before the last data row) is its last weekday
    m = dates.astype("datetime64[M]")
    return [t for t in range(len(dates) - 1) if m[t + 1] != m[t]]


def make_view(dates, nry, uni=None, tradable=None, listed=None, mcap=None):
    T, N = nry.shape
    if mcap is None:
        mcap = np.full((T, N), 100.0)
    ncf = -nry * mcap
    ones = np.ones((T, N), dtype=bool)
    return DataView(
        dates=dates,
        instruments=tuple("E%d" % (j + 1) for j in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=np.zeros((T, N)), ret_oc=np.zeros((T, N)),
        tradable=ones.copy() if tradable is None else tradable,
        listed=ones.copy() if listed is None else listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=np.ones((T, N)), dollar_volume=np.ones((T, N)),
        universe=ones.copy() if uni is None else uni,
        extras={"sf1_art_ncfcommon": ncf.astype(np.float32), "sf1_art_marketcap": mcap.astype(np.float32)},
        series={}, cash_ret=None,
    )


def base_nry(T, N=10):
    # NRY descending with column: E1 highest ... E10 lowest
    return np.tile(np.linspace(0.10, -0.08, N), (T, 1))


# ---------------------------------------------------------------- calendar

def test_decisions_on_last_weekday_of_month():
    dates = weekday_dates()
    dec = month_end_decisions(dates)
    assert list(np.flatnonzero(dec)) == month_end_rows_by_hand(dates)
    assert len(month_end_rows_by_hand(dates)) == 4


def test_holiday_on_last_weekday_moves_decision_to_first_row_of_next_month():
    dates = weekday_dates()
    ends = month_end_rows_by_hand(dates)
    hol = ends[1]                                   # drop the last weekday of the second month
    d2 = np.delete(dates, hol)
    dec = np.flatnonzero(month_end_decisions(d2))
    # the row before the holiday is not a decision; the first row of the next month is (index hol in d2)
    assert hol - 1 not in dec
    assert hol in dec
    assert list(dec) == [ends[0], hol, ends[2] - 1, ends[3] - 1]


def test_decisions_are_point_in_time_under_truncation():
    dates = weekday_dates(400)
    full = month_end_decisions(dates)
    for cut in (5, 21, 22, 23, 100, 250):
        assert np.array_equal(month_end_decisions(dates[:cut]), full[:cut])


# ---------------------------------------------------------------- book

def test_first_cohort_top_quintile_equal_weight_and_nan_between():
    dates = weekday_dates()
    T = len(dates)
    v = make_view(dates, base_nry(T))
    w = target_weights(v, PARAMS)
    ends = month_end_rows_by_hand(dates)
    t0 = ends[0]
    assert np.isnan(w[:t0]).all()
    assert np.allclose(w[t0], [0.5, 0.5] + [0.0] * 8)        # floor(0.2 * 10) = 2 names
    assert np.isnan(w[t0 + 1:ends[1]]).all()                   # no trades between decision dates
    # same ranking every month: every cohort is {E1, E2}, the sum stays 0.5 / 0.5
    for t in ends:
        assert np.allclose(w[t], [0.5, 0.5] + [0.0] * 8)


def test_overlapping_cohorts_and_expiry():
    dates = weekday_dates()
    T = len(dates)
    ends = month_end_rows_by_hand(dates)
    nry = base_nry(T)
    # after the first decision, E9 / E10 become the top repurchasers
    nry[ends[0] + 1:, 8] = 0.5
    nry[ends[0] + 1:, 9] = 0.4
    v = make_view(dates, nry)
    w = target_weights(v, {"hold_months": 2, "quantile": 0.2})
    exp1 = np.zeros(10); exp1[[8, 9]] = 0.25; exp1[[0, 1]] = 0.25
    assert np.allclose(w[ends[1]], exp1)                       # two live cohorts, half each
    exp2 = np.zeros(10); exp2[[8, 9]] = 0.5
    assert np.allclose(w[ends[2]], exp2)                       # first cohort expired after 2 months
    w12 = target_weights(v, {"hold_months": 12, "quantile": 0.2})
    exp3 = np.zeros(10); exp3[[0, 1]] = 1 / 6; exp3[[8, 9]] = 1 / 3
    assert np.allclose(w12[ends[2]], exp3)                     # 3 live cohorts: {E1,E2}, {E9,E10}, {E9,E10}


def test_ties_broken_by_instrument_order_and_min_one_name():
    dates = weekday_dates()
    T = len(dates)
    nry = np.zeros((T, 4))                                      # all tied
    v = make_view(dates, nry)
    w = target_weights(v, {"hold_months": 12, "quantile": 0.2})  # floor(0.8) -> at least 1 name
    t0 = month_end_rows_by_hand(dates)[0]
    assert np.allclose(w[t0], [1.0, 0.0, 0.0, 0.0])


def test_quantile_changes_cohort_size():
    dates = weekday_dates()
    T = len(dates)
    v = make_view(dates, base_nry(T))
    t0 = month_end_rows_by_hand(dates)[0]
    w1 = target_weights(v, {"hold_months": 12, "quantile": 0.1})
    w3 = target_weights(v, {"hold_months": 12, "quantile": 0.3})
    assert np.allclose(w1[t0], [1.0] + [0.0] * 9)
    assert np.allclose(w3[t0], [1 / 3] * 3 + [0.0] * 7)


def test_ineligible_nan_fields_nonpositive_mcap_and_non_members():
    dates = weekday_dates()
    T = len(dates)
    nry = base_nry(T)
    mcap = np.full((T, 10), 100.0)
    mcap[:, 0] = np.nan                     # E1 unknown market cap
    mcap[:, 1] = 0.0                        # E2 non-positive market cap
    uni = np.ones((T, 10), dtype=bool)
    uni[:, 2] = False                       # E3 not a member
    v = make_view(dates, nry, uni=uni, mcap=mcap)
    v.extras["sf1_art_ncfcommon"][:, 3] = np.nan   # E4 unknown issuance
    t0 = month_end_rows_by_hand(dates)[0]
    w = target_weights(v, PARAMS)
    # 6 eligible (E5..E10): floor(1.2) = 1 name -> E5
    exp = np.zeros(10); exp[4] = 1.0
    assert np.allclose(w[t0], exp)


def test_universe_exit_sold_midmonth_and_dropped_from_cohort():
    dates = weekday_dates()
    T = len(dates)
    ends = month_end_rows_by_hand(dates)
    uni = np.ones((T, 10), dtype=bool)
    s = ends[0] + 5
    uni[s:s + 3, 0] = False                 # E1 leaves for three days mid-month, then returns
    v = make_view(dates, base_nry(T), uni=uni)
    w = target_weights(v, {"hold_months": 2, "quantile": 0.2})
    assert w[s, 0] == 0.0 and np.isnan(w[s, 1])               # sold at the next open, E2 kept
    assert np.isnan(w[ends[0] + 1:s]).all()
    assert np.isnan(w[s + 3:ends[1], 0]).all()                # not bought back before the decision date
    # at the next decision: cohort 1 = {E2} only (E1 dropped), cohort 2 = {E1, E2}
    exp = np.zeros(10); exp[0] = 0.25; exp[1] = 0.75
    assert np.allclose(w[ends[1]], exp)


def test_not_tradable_on_decision_date_dropped_and_gross_bounds():
    dates = weekday_dates()
    T = len(dates)
    ends = month_end_rows_by_hand(dates)
    tr = np.ones((T, 10), dtype=bool)
    tr[ends[0], 1] = False
    v = make_view(dates, base_nry(T), tradable=tr)
    w = target_weights(v, PARAMS)
    assert np.allclose(w[ends[0]], [1.0] + [0.0] * 9)
    dec = np.isfinite(w).all(axis=1)
    g = np.nansum(np.abs(w), axis=1)
    assert (g <= 1 + 1e-12).all() and (np.nan_to_num(w) >= 0).all()
    assert np.allclose(g[dec], 1.0)


def test_empty_universe_gives_cash():
    dates = weekday_dates()
    T = len(dates)
    v = make_view(dates, base_nry(T), uni=np.zeros((T, 10), dtype=bool))
    w = target_weights(v, PARAMS)
    t0 = month_end_rows_by_hand(dates)[0]
    assert np.allclose(w[t0], 0.0)


def test_truncation_point_in_time():
    dates = weekday_dates(400)
    T = len(dates)
    rng = np.random.default_rng(3)
    nry = rng.normal(size=(T, 12))
    uni = rng.random((T, 12)) > 0.1
    v = make_view(dates, nry, uni=uni)
    full = target_weights(v, PARAMS)
    for cut in (30, 64, 65, 66, 200):
        vc = make_view(dates[:cut], nry[:cut], uni=uni[:cut])
        assert np.array_equal(np.isnan(target_weights(vc, PARAMS)), np.isnan(full[:cut]))
        assert np.allclose(target_weights(vc, PARAMS), full[:cut], equal_nan=True)


def test_missing_input_raises():
    dates = weekday_dates()
    v = make_view(dates, base_nry(len(dates)))
    del v.extras["sf1_art_marketcap"]
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)


# ---------------------------------------------------------------- diagnostic

def test_events_new_top_and_bottom_entries():
    dates = weekday_dates()
    T = len(dates)
    ends = month_end_rows_by_hand(dates)
    nry = base_nry(T)
    # after the first decision E9 jumps to the top, E1 falls to the bottom
    nry[ends[0] + 1:, 8] = 0.5
    nry[ends[0] + 1:, 0] = -0.5
    v = make_view(dates, nry)
    mask, side = events(v, PARAMS)
    assert not mask[:ends[1]].any()                            # no events on the first decision date
    # second decision: top2 = {E9, E2} (E9 new); bottom2 = {E10, E1} (E1 new), E10 was bottom before
    assert list(np.flatnonzero(mask[ends[1]])) == [0, 8]
    assert side[ends[1], 8] == 1.0 and side[ends[1], 0] == -1.0
    assert not mask[ends[2]].any()                             # unchanged sets -> no new entries
    assert mask.sum() == 2


def test_events_quantile_param_and_tradable():
    dates = weekday_dates()
    T = len(dates)
    ends = month_end_rows_by_hand(dates)
    nry = base_nry(T)
    nry[ends[0] + 1:, 7] = 0.5               # E8 to the top
    tr = np.ones((T, 10), dtype=bool)
    v = make_view(dates, nry, tradable=tr)
    m1, _ = events(v, {"hold_months": 12, "quantile": 0.1})
    m3, _ = events(v, {"hold_months": 12, "quantile": 0.3})
    assert m1[ends[1], 7] and m3[ends[1], 7]
    assert m1.sum() != m3.sum()
    tr[ends[1], 7] = False
    m4, _ = events(make_view(dates, nry, tradable=tr), PARAMS)
    assert not m4[ends[1], 7]
