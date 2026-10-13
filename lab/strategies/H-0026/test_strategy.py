import numpy as np
import pytest

from strategy import target_weights, PARAMS, overdue_events
from lab.framework.data import DataView

# Calendar of weekdays only: A + 364 calendar days is the same weekday, i.e. exactly 260 rows later.
ROWS_PER_YEAR = 260


def bdays(n):
    days = np.arange(12600, 12600 + 2 * n).astype("datetime64[D]")
    return days[np.is_busday(days)][:n]


def view(ea, uni=None, trad=None, dates=None):
    T, N = ea.shape
    if dates is None:
        dates = bdays(T)
    if uni is None:
        uni = np.ones((T, N), dtype=bool)
    if trad is None:
        trad = np.ones((T, N), dtype=bool)
    z = np.zeros((T, N))
    return DataView(
        dates[:T], tuple("E%d" % (k + 1) for k in range(N)), tuple("us_equity" for _ in range(N)),
        z.copy(), z.copy(), trad.copy(), np.ones((T, N), dtype=bool), np.zeros((T, N), dtype=bool),
        np.full((T, N), 10.0), np.full((T, N), 1e6),
        universe=uni.copy(), extras={"ea_reaction_day": ea.astype(np.float32)},
    )


def base_market(T=400, N=6):
    """Stock 0: releases at rows 10 (anchor A) and 210 (previous quarter), none around E = 270.
    Stocks 1..N-1: no filings (NaN) -> never overdue, but valid hedge names."""
    ea = np.full((T, N), np.nan)
    ea[:, 0] = 0.0
    ea[10, 0] = 1.0
    ea[210, 0] = 1.0
    return ea


A, E = 10, 10 + ROWS_PER_YEAR          # E(A) = row 270
P = dict(PARAMS)


def test_event_timing_primary():
    ev = overdue_events(view(base_market()), 3)
    assert ev.sum() == 1 and ev[E + 3, 0]


@pytest.mark.parametrize("d", [1, 3, 5])
def test_event_timing_delay(d):
    ev = overdue_events(view(base_market()), d)
    assert ev.sum() == 1 and ev[E + d, 0]


def test_weights_primary():
    W = target_weights(view(base_market()), P)
    t = E + 3
    assert np.all(W[0] == 0.0)                         # explicit flat start
    assert W[t, 0] == pytest.approx(-0.125)            # enters after the close of t
    assert np.allclose(W[t, 1:], 0.125 / 5)
    assert np.all(np.isnan(W[t + 1:t + 5]))            # S unchanged: keep positions
    assert np.all(W[t + 5] == 0.0)                     # exit at the open after t + hold_days
    other = np.ones(W.shape[0], dtype=bool)
    other[[0, t, t + 5]] = False
    assert np.all(np.isnan(W[other]))


def test_hold_days_param():
    q = dict(P, hold_days=3)
    W = target_weights(view(base_market()), q)
    t = E + 3
    assert W[t, 0] == pytest.approx(-0.125)
    assert np.all(np.isnan(W[t + 1:t + 3]))
    assert np.all(W[t + 3] == 0.0)


def test_delay_param_changes_entry():
    W1 = target_weights(view(base_market()), dict(P, delay_sessions=1))
    assert W1[E + 1, 0] == pytest.approx(-0.125)
    assert np.isnan(W1[E, 0])


@pytest.mark.parametrize("row,expect", [(E - 25, 0), (E - 10, 0), (E + 2, 0), (E + 3, 0),
                                        (E - 26, 1), (E - 60, 1), (E + 4, 1)])
def test_quiet_window(row, expect):
    ea = base_market()
    ea[row, 0] = 1.0
    ev = overdue_events(view(ea), 3)
    assert int(ev[E + 3, 0]) == expect


def test_nan_in_quiet_window_blocks():
    ea = base_market()
    ea[E - 5, 0] = np.nan
    assert overdue_events(view(ea), 3).sum() == 0


def test_needs_previous_quarter_release():
    ea = base_market()
    ea[210, 0] = 0.0
    assert overdue_events(view(ea), 3).sum() == 0
    ea[E - 125, 0] = 1.0                              # edge of window (ii)
    assert overdue_events(view(ea), 3)[E + 3, 0]
    ea[E - 125, 0] = 0.0
    ea[E - 126, 0] = 1.0                              # outside window (ii)
    assert overdue_events(view(ea), 3).sum() == 0


def test_member_and_tradable_at_t():
    ea = base_market()
    uni = np.ones(ea.shape, dtype=bool)
    uni[E + 3, 0] = False
    assert overdue_events(view(ea, uni=uni), 3).sum() == 0
    trad = np.ones(ea.shape, dtype=bool)
    trad[E + 3, 0] = False
    assert overdue_events(view(ea, trad=trad), 3).sum() == 0


def test_dedupe_double_anchor():
    ea = base_market()
    ea[11, 0] = 1.0                                   # second anchor one session later
    ev = overdue_events(view(ea), 3)
    assert ev.sum() == 1 and ev[E + 3, 0]


def test_leaves_universe_while_short():
    ea = base_market()
    uni = np.ones(ea.shape, dtype=bool)
    t = E + 3
    uni[t + 2:, 0] = False
    W = target_weights(view(ea, uni=uni), P)
    assert W[t, 0] == pytest.approx(-0.125)
    assert np.all(np.isnan(W[t + 1]))
    assert np.all(W[t + 2] == 0.0)                    # S empty -> book in cash
    assert np.all(np.isnan(W[t + 3:]))


def test_hedge_only_tradable_members_and_drop_on_quiet_row():
    ea = base_market()
    uni = np.ones(ea.shape, dtype=bool)
    trad = np.ones(ea.shape, dtype=bool)
    t = E + 3
    trad[t, 5] = False                                # not tradable at t: not in the hedge
    uni[t + 2:, 4] = False                            # hedge name leaves the universe during the hold
    W = target_weights(view(ea, uni=uni, trad=trad), P)
    assert np.allclose(W[t, 1:4], 0.125 / 4) and W[t, 4] == pytest.approx(0.125 / 4) and W[t, 5] == 0.0
    assert W[t + 2, 4] == 0.0 and np.all(np.isnan(np.delete(W[t + 2], 4)))


def test_more_than_four_names():
    T, N = 400, 12
    ea = np.full((T, N), np.nan)
    ea[:, :6] = 0.0
    ea[10, :6] = 1.0
    ea[210, :6] = 1.0
    W = target_weights(view(ea), P)
    t = E + 3
    assert np.allclose(W[t, :6], -0.5 / 6)
    assert np.allclose(W[t, 6:], 0.5 / 6)
    assert np.abs(W[t]).sum() == pytest.approx(1.0)


def random_market(seed, T=1400, N=40):
    rng = np.random.default_rng(seed)
    ea = np.zeros((T, N))
    for j in range(N):
        start = int(rng.integers(0, 60))
        r = start
        while r < T:
            ea[r, j] = 1.0
            r += 65 + int(rng.integers(-3, 4)) + (int(rng.integers(5, 15)) if rng.random() < 0.25 else 0)
        if j % 7 == 0:
            ea[: int(rng.integers(100, 400)), j] = np.nan
    uni = rng.random((T, N)) < 0.9
    trad = rng.random((T, N)) < 0.97
    return ea, uni, trad


def test_random_market_has_events_and_valid_weights():
    ea, uni, trad = random_market(1)
    ev = overdue_events(view(ea, uni, trad), 3)
    assert ev.sum() >= 10
    W = target_weights(view(ea, uni, trad), P)
    dec = ~np.all(np.isnan(W), axis=1)
    Wd = np.nan_to_num(W[dec])
    assert np.all(np.abs(Wd).sum(axis=1) <= 1.0 + 1e-9)
    assert np.allclose(Wd.sum(axis=1), 0.0)           # dollar-neutral
    assert np.all(Wd[~uni[dec]] == 0.0)               # never targets a non-member


@pytest.mark.parametrize("seed", [2, 3])
def test_point_in_time_truncation_and_perturbation(seed):
    ea, uni, trad = random_market(seed)
    dates = bdays(ea.shape[0])
    full = target_weights(view(ea, uni, trad, dates), P)
    rng = np.random.default_rng(seed + 100)
    cuts = list(rng.integers(300, ea.shape[0] - 5, 15))
    changes = np.nonzero(np.any(np.nan_to_num(full, nan=9.0) != 9.0, axis=1))[0]
    cuts += [int(c) for c in changes[changes > 300][:10]]
    for c in cuts:
        c = int(c)
        tr = target_weights(view(ea[: c + 1], uni[: c + 1], trad[: c + 1], dates), P)
        assert np.array_equal(tr, full[: c + 1], equal_nan=True)
        ea2 = ea.copy()
        ea2[c + 1:] = (rng.random(ea2[c + 1:].shape) < 0.05).astype(float)
        uni2 = uni.copy()
        uni2[c + 1:] = rng.random(uni2[c + 1:].shape) < 0.5
        pert = target_weights(view(ea2, uni2, trad, dates), P)
        assert np.array_equal(pert[: c + 1], full[: c + 1], equal_nan=True)


def test_deterministic():
    ea, uni, trad = random_market(4)
    a = target_weights(view(ea, uni, trad), P)
    b = target_weights(view(ea, uni, trad), P)
    assert np.array_equal(a, b, equal_nan=True)
