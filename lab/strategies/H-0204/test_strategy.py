import math

import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import PARAMS, build_book, compute_events, reaction_z, target_weights

T = 400
N = 12
R_OLD = 40            # last year's release row; dates are consecutive weekdays, so
PRED = R_OLD + 260    # date(R_OLD) + 364 days = 52 weeks = 260 weekdays later = row 300
T_EV = PRED - 2       # event row for lead_sessions = 2
R_LAST = T_EV - 60    # last quarter's release, gap 60


def _weekdays(n):
    days = np.arange(12000, 12000 + 2 * n).astype("datetime64[D]")
    days = days[np.is_busday(days)]
    return days[:n]


def _view(ret, flag, universe=None, listed=None, slicer=slice(None)):
    t, n = ret.shape
    if universe is None:
        universe = np.ones((t, n), dtype=bool)
    if listed is None:
        listed = np.ones((t, n), dtype=bool)
    s = slicer
    return DataView(
        dates=_weekdays(t)[s],
        instruments=tuple("E%d" % (j + 1) for j in range(n)),
        asset_class=tuple("us_equity" for _ in range(n)),
        ret_co=np.zeros((t, n))[s],
        ret_oc=ret[s],
        tradable=listed[s].copy(),
        listed=listed[s].copy(),
        delisting=np.zeros((t, n), dtype=bool)[s],
        close=np.where(listed, 10.0, np.nan)[s],
        dollar_volume=np.where(listed, 1e8, np.nan)[s],
        universe=universe[s].copy(),
        extras={"ea_reaction_day": flag[s].astype(np.float32)},
        series={},
        cash_ret=np.zeros(t)[s],
    )


TARGET_Z = {0: 3.0, 1: -3.0, 2: 3.0, 3: 3.0, 4: 1.8}


def _scenario():
    """Every stock alternates +1 % / -1 % daily; the reacting stocks get a jump on R_LAST + 1
    (after-close release) solved so that z(R_LAST) equals TARGET_Z exactly.

    stock 0: z = +3 -> short; stock 1: z = -3 -> long;
    stock 2: like 0 but NaN flag at row 20 (inside t-290 .. t-1) -> no event;
    stock 3: like 0 but an extra release 20 sessions before t -> ineligible;
    stock 4: z = +1.8 -> event only for z_min 1.5; stocks 5..11: no releases."""
    sign = np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    ret = np.tile((0.01 * sign)[:, None], (1, N))
    flag = np.zeros((T, N))
    sd = np.std(ret[R_LAST - 64:R_LAST - 1, 0], ddof=1)
    r_r = ret[R_LAST, 0]
    c = (1 + r_r) * (1 + ret[R_LAST + 1, 0]) - 1          # two-day return of a quiet stock
    k = {j: zt * math.sqrt(2.0) * sd for j, zt in TARGET_Z.items()}
    k_quiet = -sum(k.values()) / (N - len(k))             # deviations from the mean sum to 0
    mean = c - k_quiet
    for j in TARGET_Z:
        flag[R_OLD, j] = 1.0
        flag[R_LAST, j] = 1.0
        ret[R_LAST + 1, j] = (1 + mean + k[j]) / (1 + r_r) - 1
    flag[20, 2] = np.nan
    flag[T_EV - 20, 3] = 1.0
    return ret, flag


def _reference_z(ret, valid_members, R, j):
    """Straight loop version of rule (1)."""
    g = (1 + ret[R]) * (1 + ret[R + 1]) - 1
    mem = valid_members[R - 1]
    m = np.mean(g[mem])
    sd = np.std(ret[R - 64:R - 1, j], ddof=1)
    return (g[j] - m) / (math.sqrt(2.0) * sd)


def test_reaction_z_matches_reference_and_sign():
    ret, flag = _scenario()
    v = _view(ret, flag)
    z = reaction_z(v)
    uni = np.ones((T, N), dtype=bool)
    for j in range(5):
        assert z[R_LAST, j] == pytest.approx(_reference_z(ret, uni, R_LAST, j), rel=1e-9)
    for j, zt in TARGET_Z.items():
        assert z[R_LAST, j] == pytest.approx(zt, rel=1e-9)
    # R_OLD = 40 has fewer than 40 valid returns in rows R-64 .. R-2 -> no z
    assert np.isnan(z[R_OLD, 0])
    # rows without a release carry no z
    assert np.isnan(z[R_LAST + 1, 0])


def test_events_sides_and_eligibility():
    ret, flag = _scenario()
    v = _view(ret, flag)
    ev, side, absz = compute_events(v, PARAMS)
    assert np.nonzero(ev[:, 0])[0].tolist() == [T_EV]
    assert side[T_EV, 0] == -1.0          # big positive previous reaction -> short
    assert np.nonzero(ev[:, 1])[0].tolist() == [T_EV]
    assert side[T_EV, 1] == 1.0           # big negative previous reaction -> long
    assert not ev[:, 2].any()             # NaN flag inside the needed window
    assert not ev[:, 3].any()             # release 20 sessions before t
    assert not ev[:, 4].any()             # |z| < 2
    assert not ev[:, 5:].any()
    assert absz[T_EV, 0] == pytest.approx(abs(reaction_z(v)[R_LAST, 0]))


def test_nan_outside_window_is_ignored():
    ret, flag = _scenario()
    flag[5, 0] = np.nan                   # row 5 < t - 290 = 8
    ev, _, _ = compute_events(_view(ret, flag), PARAMS)
    assert ev[T_EV, 0]
    flag[8, 0] = np.nan                   # row t - 290 is needed
    ev, _, _ = compute_events(_view(ret, flag), PARAMS)
    assert not ev[:, 0].any()


def test_weights_slot_mechanics():
    ret, flag = _scenario()
    w = target_weights(_view(ret, flag), PARAMS)
    assert w.shape == (T, N)
    assert np.all(w[:T_EV] == 0)
    assert np.allclose(w[T_EV:T_EV + 5, 0], -0.04)
    assert np.allclose(w[T_EV:T_EV + 5, 1], 0.04)
    assert np.all(w[T_EV + 5:] == 0)
    assert np.all(w[:, 2:] == 0)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)


def test_params_change_output():
    ret, flag = _scenario()
    v = _view(ret, flag)
    w15 = target_weights(v, dict(PARAMS, z_min=1.5))
    assert np.allclose(w15[T_EV:T_EV + 5, 4], -0.04)
    w25 = target_weights(v, dict(PARAMS, z_min=2.5))
    assert np.allclose(w25[T_EV:T_EV + 5, 0], -0.04)
    w35 = target_weights(v, dict(PARAMS, z_min=3.5))
    assert np.all(w35 == 0)
    w1 = target_weights(v, dict(PARAMS, lead_sessions=1))
    assert np.all(w1[:T_EV + 1] == 0)
    assert np.allclose(w1[T_EV + 1:T_EV + 6, 0], -0.04)
    w3 = target_weights(v, dict(PARAMS, lead_sessions=3))
    assert np.allclose(w3[T_EV - 1:T_EV + 4, 1], 0.04)


def test_universe_required_on_event_day_and_exit_on_leave():
    ret, flag = _scenario()
    uni = np.ones((T, N), dtype=bool)
    uni[T_EV, 1] = False                  # not a member on t -> no event
    uni[T_EV + 2, 0] = False              # leaves during the slot -> closed
    w = target_weights(_view(ret, flag, universe=uni), PARAMS)
    assert np.all(w[:, 1] == 0)
    assert np.allclose(w[T_EV:T_EV + 2, 0], -0.04)
    assert np.all(w[T_EV + 2:, 0] == 0)


def test_point_in_time_truncation():
    ret, flag = _scenario()
    full = target_weights(_view(ret, flag), PARAMS)
    for cut in (T_EV, T_EV + 1, T_EV + 3, R_LAST + 1, 350):
        part = target_weights(_view(ret, flag, slicer=slice(0, cut + 1)), PARAMS)
        assert np.array_equal(part, full[:cut + 1])
    # future noise does not change the past
    rng = np.random.default_rng(1)
    ret2 = ret.copy()
    ret2[T_EV + 1:] = rng.normal(0, 0.05, ret2[T_EV + 1:].shape)
    flag2 = flag.copy()
    flag2[T_EV + 1:] = (rng.random(flag2[T_EV + 1:].shape) < 0.1).astype(float)
    other = target_weights(_view(ret2, flag2), PARAMS)
    assert np.array_equal(other[:T_EV + 1], full[:T_EV + 1])


def test_book_one_slot_per_stock_and_renewal():
    t, n = 12, 2
    ev = np.zeros((t, n), dtype=bool)
    side = np.zeros((t, n))
    absz = np.zeros((t, n))
    ev[[0, 2, 5], 0] = True
    side[[0, 2, 5], 0] = 1.0
    absz[[0, 2, 5], 0] = 2.5
    ones = np.ones((t, n), dtype=bool)
    w = build_book(ev, side, absz, ones, ones)
    assert np.allclose(w[0:5, 0], 0.04)   # event on row 2 ignored, no extension
    assert np.allclose(w[5:10, 0], 0.04)  # new slot after the first closed
    assert np.all(w[10:, 0] == 0)
    listed = ones.copy()
    listed[7, 0] = False                  # stops trading -> slot closed
    w = build_book(ev, side, absz, ones, listed)
    assert np.allclose(w[5:7, 0], 0.04)
    assert np.all(w[7:, 0] == 0)


def test_book_gross_cap_largest_z_first():
    t, n = 8, 40
    ev = np.zeros((t, n), dtype=bool)
    side = np.zeros((t, n))
    absz = np.zeros((t, n))
    ev[0, :20] = True
    side[0, :20] = -1.0
    absz[0, :20] = 3.0
    ev[1, 20:30] = True
    side[1, 20:30] = 1.0
    absz[1, 20:30] = np.arange(10) + 2.0  # stocks 25..29 have the largest |z|
    ev[5, 30:40] = True
    side[5, 30:40] = 1.0
    absz[5, 30:40] = 2.0
    ones = np.ones((t, n), dtype=bool)
    w = build_book(ev, side, absz, ones, ones)
    gross = np.abs(w).sum(axis=1)
    assert np.all(gross <= 1.0 + 1e-12)
    assert gross[1] == pytest.approx(1.0)
    assert np.all(w[1:6, 20:25] == 0)
    assert np.allclose(w[1:6, 25:30], 0.04)
    assert np.allclose(w[1:5, 0:20], -0.04)  # existing slots never scaled down
    assert np.allclose(w[5:8, 30:40], 0.04)  # room again after row-0 slots closed


def test_missing_flag_raises():
    ret, flag = _scenario()
    v = _view(ret, flag)
    v.extras.pop("ea_reaction_day")
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)


def test_deterministic():
    ret, flag = _scenario()
    a = target_weights(_view(ret, flag), PARAMS)
    b = target_weights(_view(ret, flag), PARAMS)
    assert np.array_equal(a, b)
