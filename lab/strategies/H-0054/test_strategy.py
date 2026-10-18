import dataclasses

import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import (rebalance_rows, cumulate, block_returns, score_from_blocks, legs, leg_gross,
                      FIRST_ROW, N_BLOCKS, BLOCK, events)
from lab.framework.data import synthetic


def _day(y, m, d):
    base = np.array([(y - 1970) * 12 + m - 1]).astype("datetime64[M]").astype("datetime64[D]")
    return (base + np.array([d - 1]).astype("timedelta64[D]"))[0]


def _sessions(y0, m0, y1, m1, closed=()):
    days = np.arange(_day(y0, m0, 1), _day(y1, m1, 1))
    hol = np.array(list(closed), dtype="datetime64[D]") if closed else None
    keep = np.is_busday(days, holidays=hol) if hol is not None else np.is_busday(days)
    return days[keep]


def _flagged(dates):
    return list(dates[rebalance_rows(dates)])


# ----------------------------------------------------------------------------------------------- calendar
def test_calendar_good_friday_month_end():
    gf = _day(2018, 3, 30)          # Good Friday 2018 = last weekday of March
    mem = _day(2018, 5, 28)
    d = _sessions(2018, 1, 2018, 8, closed=[gf, mem])
    assert _flagged(d) == [_day(2018, 1, 31), _day(2018, 2, 28), _day(2018, 3, 29), _day(2018, 4, 30),
                           _day(2018, 5, 31), _day(2018, 6, 29), _day(2018, 7, 31)]


def test_calendar_memorial_day_month_end():
    mem = _day(2021, 5, 31)         # Memorial Day 2021 = Monday May 31
    gf = _day(2021, 4, 2)
    d = _sessions(2021, 4, 2021, 7, closed=[gf, mem])
    assert _flagged(d) == [_day(2021, 4, 30), _day(2021, 5, 28), _day(2021, 6, 30)]


def test_calendar_fallback_for_unknown_closure():
    # an unforeseen closure on the last weekday of October: the first session of November is flagged instead
    d = _sessions(2012, 10, 2012, 12, closed=[_day(2012, 10, 31)])
    assert _flagged(d) == [_day(2012, 11, 1), _day(2012, 11, 30)]


def test_calendar_point_in_time():
    d = _sessions(2010, 1, 2012, 1)
    full = rebalance_rows(d)
    for cut in (5, 40, 100, 300):
        assert np.array_equal(rebalance_rows(d[:cut]), full[:cut])


# ----------------------------------------------------------------------------------------------- blocks
def test_block_returns_compounding_and_validity():
    T = FIRST_ROW + 1
    r = np.full((T, 4), 0.01)
    valid = np.ones((T, 4), dtype=bool)
    t = T - 1
    valid[t - 6:t + 1, 1] = False          # block 1: 14 valid sessions -> missing
    valid[t - 5:t + 1, 2] = False          # block 1: 15 valid sessions -> valid
    r[t - 30, 3] = -1.0                    # wipe-out in block 2
    C, CV, CW = cumulate(r, valid)
    br, bvalid, start = block_returns(t, C, CV, CW)
    assert br.shape == (N_BLOCKS, 4)
    assert start[0] == t - BLOCK + 1 and start[-1] == 0
    assert np.isclose(br[0, 0], 1.01 ** 21 - 1)
    assert np.isclose(br[35, 0], 1.01 ** 21 - 1)
    assert not bvalid[0, 1] and bvalid[1, 1]
    assert bvalid[0, 2] and np.isclose(br[0, 2], 1.01 ** 15 - 1)
    assert br[1, 3] == -1.0 and np.isclose(br[0, 3], 1.01 ** 21 - 1)


def _ref_score(m, r, fm):
    beta, alpha = np.polyfit(m, r, 1)
    e = r - alpha - beta * m
    ef = e[1:fm]
    return ef.sum() / ef.std(ddof=1), beta


def test_score_matches_reference_ols():
    rng = np.random.default_rng(1)
    br = rng.normal(0.01, 0.06, size=(N_BLOCKS, 6))
    bvalid = np.ones_like(br, dtype=bool)
    mem = np.ones_like(bvalid)
    elig = np.ones(6, dtype=bool)
    for fm in (9, 12, 15):
        score, beta = score_from_blocks(br, bvalid, mem, elig, fm)
        m = br.mean(axis=1)
        for j in range(6):
            s_ref, b_ref = _ref_score(m, br[:, j], fm)
            assert np.isclose(score[j], s_ref) and np.isclose(beta[j], b_ref)
    s9 = score_from_blocks(br, bvalid, mem, elig, 9)[0]
    s12 = score_from_blocks(br, bvalid, mem, elig, 12)[0]
    assert not np.allclose(s9, s12)


def test_score_market_uses_first_session_members_only():
    rng = np.random.default_rng(2)
    br = rng.normal(0.0, 0.05, size=(N_BLOCKS, 5))
    bvalid = np.ones_like(br, dtype=bool)
    mem = np.ones_like(bvalid)
    mem[:, 4] = False                                   # stock 4 never a member at block starts
    elig = np.ones(5, dtype=bool)
    score, beta = score_from_blocks(br, bvalid, mem, elig, 12)
    m = br[:, :4].mean(axis=1)
    s_ref, b_ref = _ref_score(m, br[:, 4], 12)
    assert np.isclose(score[4], s_ref) and np.isclose(beta[4], b_ref)


def test_score_missing_data_rules():
    rng = np.random.default_rng(3)
    br = rng.normal(0.0, 0.05, size=(N_BLOCKS, 8))
    bvalid = np.ones_like(br, dtype=bool)
    mem = np.ones_like(bvalid)
    elig = np.ones(8, dtype=bool)
    bvalid[5, 0] = False                  # formation block k = 6 missing -> no score
    bvalid[0, 1] = False                  # block 1 (skipped month) missing -> still scored
    bvalid[20:33, 2] = False              # 23 valid blocks -> no beta, no score
    bvalid[20:32, 3] = False              # 24 valid blocks -> scored
    elig[4] = False                       # not member / not tradable at t
    score, beta = score_from_blocks(br, bvalid, mem, elig, 12)
    assert np.isnan(score[0]) and np.isfinite(beta[0])
    assert np.isfinite(score[1])
    assert np.isnan(score[2]) and np.isnan(beta[2])
    assert np.isfinite(score[3])
    assert np.isnan(score[4]) and np.isnan(beta[4])
    # block 12 is only in the 12-month formation window, not in the 9-month one
    bvalid2 = np.ones_like(bvalid)
    bvalid2[11, 5] = False
    s9 = score_from_blocks(br, bvalid2, mem, np.ones(8, bool), 9)[0]
    s12 = score_from_blocks(br, bvalid2, mem, np.ones(8, bool), 12)[0]
    assert np.isfinite(s9[5]) and np.isnan(s12[5])


# ----------------------------------------------------------------------------------------------- legs, gross
def test_legs_sizes_and_order():
    score = np.arange(50, dtype=float)
    li, si = legs(score, 0.2)
    assert sorted(li) == list(range(40, 50)) and sorted(si) == list(range(0, 10))
    score = np.arange(200, dtype=float)
    li, si = legs(score, 0.1)
    assert len(li) == 20 and len(si) == 20
    li, si = legs(score, 0.3)
    assert len(li) == 60 and min(li) == 140 and max(si) == 59
    score = np.full(30, np.nan)
    score[:25] = np.arange(25.0)
    li, si = legs(score, 0.2)            # floor(5) < 10 -> 10 per leg
    assert len(li) == 10 and len(si) == 10 and set(li).isdisjoint(si)
    score = np.arange(19.0)
    li, si = legs(score, 0.2)            # cannot form two legs of 10
    assert len(li) == 0 and len(si) == 0


def test_leg_gross_beta_neutral_and_fallbacks():
    beta = np.ones(20)
    li, si = np.arange(10), np.arange(10, 20)
    beta[10:] = 1.5
    g_l, g_s = leg_gross(beta, li, si)
    assert np.isclose(g_l, 0.6) and np.isclose(g_s, 0.4)
    assert np.isclose(g_l * 1.0, g_s * 1.5) and np.isclose(g_l + g_s, 1.0)
    beta[10:] = 2.0
    beta[:10] = 0.8                       # g_L = 2 / 2.8 = 0.714 > 0.65 -> dollar neutral
    assert leg_gross(beta, li, si) == (0.5, 0.5)
    beta[:10] = 0.25
    beta[10:] = 0.3                       # long leg beta < 0.3 -> dollar neutral
    assert leg_gross(beta, li, si) == (0.5, 0.5)


# ----------------------------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def view():
    v = synthetic()
    if v.universe is not None:
        return v
    # synthetic() has no universe: membership = listed, minus a deterministic churn block for 3 stocks
    uni = np.asarray(v.listed, dtype=bool).copy()
    uni[3000:3400, :3] = False
    return dataclasses.replace(v, universe=uni)


def test_exit_on_universe_or_tradable_loss(view):
    w = target_weights(view, PARAMS)
    full = [t for t in np.flatnonzero(np.all(np.isfinite(w), axis=1)) if np.any(w[t] != 0)]
    t, t2 = full[0], full[1]
    held = np.flatnonzero(w[t] != 0)
    j, k = held[0], held[1]
    uni = np.asarray(view.universe, dtype=bool).copy()
    tr = np.asarray(view.tradable, dtype=bool).copy()
    uni[t + 3, j] = False                # leaves the universe for one day, then returns
    uni[t + 4:t2, j] = True
    tr[t + 5, k] = False                 # one non-tradable open
    w2 = target_weights(dataclasses.replace(view, universe=uni, tradable=tr), PARAMS)
    assert np.array_equal(np.nan_to_num(w2[:t + 1], nan=9.0), np.nan_to_num(w[:t + 1], nan=9.0))
    assert w2[t + 3, j] == 0.0 and np.all(np.isnan(w2[t + 1:t + 3, j])) and np.all(np.isnan(w2[t + 4:t2, j]))
    assert w2[t + 5, k] == 0.0 and np.all(np.isnan(w2[t + 1:t + 5, k])) and np.all(np.isnan(w2[t + 6:t2, k]))
    # the other held stocks keep their weight (NaN = hold) on those rows
    others = np.setdiff1d(held, [j, k])
    assert np.all(np.isnan(w2[t + 1:t2][:, others]))


def test_end_to_end_mechanics(view):
    w = target_weights(view, PARAMS)
    T, N = len(view.dates), len(view.instruments)
    assert w.shape == (T, N)
    flags = rebalance_rows(view.dates)
    full_rows = np.flatnonzero(np.all(np.isfinite(w), axis=1))
    assert len(full_rows) > 0
    assert all(flags[t] and t >= FIRST_ROW for t in full_rows)
    assert np.all(np.isnan(w[:FIRST_ROW]))
    U = np.asarray(view.universe, dtype=bool) if view.universe is not None else None
    for t in full_rows:
        row = w[t]
        assert np.abs(row).sum() <= 1.0 + 1e-9
        if np.any(row != 0):
            assert np.isclose(row[row > 0].sum() + (-row[row < 0]).sum(), 1.0)
            assert (row > 0).sum() >= 10 and (row < 0).sum() >= 10
            assert len(set(np.round(row[row > 0], 12))) == 1 and len(set(np.round(row[row < 0], 12))) == 1
            if U is not None:
                assert np.all(U[t][row != 0])
    # partial rows (between rebalances) only close positions
    part = np.flatnonzero(np.any(np.isfinite(w), axis=1) & ~np.all(np.isfinite(w), axis=1))
    for t in part:
        assert np.all(w[t][np.isfinite(w[t])] == 0.0)


def test_params_change_output(view):
    base = target_weights(view, PARAMS)
    # synthetic() has ~31 scored stocks, so leg_quantile 0.1..0.3 all hit the 10-name floor; 0.4 gives 12
    for key, val in (("formation_months", 9), ("formation_months", 15), ("leg_quantile", 0.4)):
        p = dict(PARAMS)
        p[key] = val
        assert not np.array_equal(np.nan_to_num(base, nan=9.0), np.nan_to_num(target_weights(view, p), nan=9.0))


def test_truncation_point_in_time(view):
    full = target_weights(view, PARAMS)
    T = len(view.dates)
    for cut in (FIRST_ROW + 30, (FIRST_ROW + T) // 2):
        if cut >= T:
            continue
        sub = _truncate(view, cut)
        part = target_weights(sub, PARAMS)
        assert np.array_equal(np.nan_to_num(part, nan=9.0), np.nan_to_num(full[:cut], nan=9.0))


def _truncate(view, cut):
    uni = None if view.universe is None else view.universe[:cut]
    return dataclasses.replace(view, dates=view.dates[:cut], ret_co=view.ret_co[:cut], ret_oc=view.ret_oc[:cut],
                               tradable=view.tradable[:cut], listed=view.listed[:cut], close=view.close[:cut],
                               universe=uni)


def test_diagnostic_marks_leg_entries(view):
    mask, side = events(view, PARAMS)
    w = target_weights(view, PARAMS)
    rows = np.flatnonzero(mask.any(axis=1))
    assert len(rows) > 0
    prev = None
    for t in np.flatnonzero(np.all(np.isfinite(w), axis=1)):
        long_now = w[t] > 0
        short_now = w[t] < 0
        if prev is None:
            exp_long, exp_short = long_now, short_now
        else:
            exp_long, exp_short = long_now & ~prev[0], short_now & ~prev[1]
        assert np.array_equal(mask[t] & (side[t] > 0), exp_long)
        assert np.array_equal(mask[t] & (side[t] < 0), exp_short)
        prev = (long_now, short_now)
