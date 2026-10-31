import dataclasses
from typing import Any

import numpy as np
import pytest

from strategy import target_weights, PARAMS, formation_rows, legs_at, id_ranks, LOOKBACK

# Day numbers since 1970-01-01 (no date literals). 18628 = 2021-01-01, 18778 = Monday 2021-05-31 (Memorial Day).
DAY0 = 18628
MEMORIAL = 18778


def weekdays(start, stop, drop=()):
    d = np.arange(start, stop).astype("datetime64[D]")
    keep = np.is_busday(d)
    for x in drop:
        keep &= d != np.arange(x, x + 1).astype("datetime64[D]")[0]
    return d[keep]


@dataclasses.dataclass
class FakeData:
    dates: Any
    instruments: Any
    ret_co: Any
    ret_oc: Any
    tradable: Any
    listed: Any
    universe: Any
    asset_class: Any = None


def make_data(T=420, N=45, seed=0):
    rng = np.random.default_rng(seed)
    dates = weekdays(DAY0 - 700, DAY0 + 700)[:T]
    vols = np.linspace(0.005, 0.03, N)
    ret_oc = rng.normal(0.0004, 1.0, (T, N)) * vols
    ret_co = np.zeros((T, N))
    inst = tuple("E%d" % (100 + j) for j in range(N))
    ones = np.ones((T, N), dtype=bool)
    return FakeData(dates, inst, ret_co, ret_oc, ones.copy(), ones.copy(), ones.copy())


# ---------------------------------------------------------------- calendar

def test_formation_last_weekday_and_holiday_fallback():
    d = weekdays(DAY0, DAY0 + 200, drop=(MEMORIAL,))
    form = formation_rows(d)
    month = d.astype("datetime64[M]")
    fallback_row = -1
    for m in np.unique(month)[:-1]:
        rows = np.flatnonzero(month == m)
        flagged = rows[form[rows]]
        if d[rows[-1]].astype(int) == MEMORIAL - 3:   # May 2021: last session is Friday the 28th
            assert flagged.size == 0
            fallback_row = np.flatnonzero(month > m)[0]
            assert form[fallback_row]                  # fallback: first June session
        elif rows[0] == fallback_row:
            assert list(flagged) == [rows[0], rows[-1]]
        else:
            assert list(flagged) == [rows[-1]]
    assert fallback_row > 0
    # point-in-time: flags of rows <= t do not depend on later rows
    for t in (10, 57, 103):
        assert np.array_equal(formation_rows(d[: t + 1]), form[: t + 1])


# ---------------------------------------------------------------- cells and legs

def cell_panel(per_cell=5, pth_levels=None, T=300):
    """Stocks laid out cell by cell: MOM tercile a, VOL tercile b, rank q inside the cell."""
    if pth_levels is None:
        pth_levels = np.linspace(0.6, 1.0, per_cell)
    N = 9 * per_cell
    f = T - 1
    r = np.zeros((T, N))
    P = np.ones((T, N))
    alt = np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    layout = []
    for a in range(3):
        for b in range(3):
            for q in range(per_cell):
                j = len(layout)
                layout.append((a, b, q))
                r[:, j] = alt * (0.01 * (b + 1) + 1e-5 * q)
                mom = (-0.2, 0.1, 0.5)[a] + 0.001 * q
                P[f - LOOKBACK, j] = 1.0           # MOM base
                P[f - LOOKBACK + 1:f, j] = 0.5     # below every P_f, so the high sits at f-10
                P[f, j] = 1.0 + mom
                P[f - 10, j] = (1.0 + mom) / pth_levels[q]
    valid = np.ones((T, N), dtype=bool)
    member = np.ones((T, N), dtype=bool)
    return f, P, r, valid, member, layout


def test_one_near_and_far_per_cell():
    f, P, r, valid, member, layout = cell_panel()
    near, far = legs_at(f, P, r, valid, member, member, 252, 0.2, np.arange(P.shape[1], dtype=float))
    assert sorted(layout[j][:2] for j in near) == [(a, b) for a in range(3) for b in range(3)]
    assert all(layout[j][2] == 4 for j in near)    # highest PTH
    assert all(layout[j][2] == 0 for j in far)     # lowest PTH


def test_leg_fraction_rounds_down():
    f, P, r, valid, member, layout = cell_panel(per_cell=10)
    ids = np.arange(P.shape[1], dtype=float)
    for frac, k in ((0.1, 1), (0.2, 2), (0.3, 3)):
        near, far = legs_at(f, P, r, valid, member, member, 252, frac, ids)
        assert near.size == 9 * k and far.size == 9 * k
        assert all(layout[j][2] >= 10 - k for j in near)
        assert all(layout[j][2] < k for j in far)


def test_small_cell_skipped_and_warmup():
    f, P, r, valid, member, layout = cell_panel()
    ids = np.arange(P.shape[1], dtype=float)
    member = member.copy()
    # drop one stock of cells (0,0), (1,1), (2,2): terciles stay intact, those cells have 4 -> skipped
    for j, (a, b, q) in enumerate(layout):
        if a == b and q == 0:
            member[f, j] = False
    near, far = legs_at(f, P, r, valid, member, member, 252, 0.2, ids)
    cells = {layout[j][:2] for j in near}
    assert cells == {(a, b) for a in range(3) for b in range(3) if a != b}
    assert {layout[j][:2] for j in far} == cells
    assert legs_at(LOOKBACK - 1, P, r, valid, member, member, 252, 0.2, ids) is None


def test_min_valid_and_tradable():
    f, P, r, valid, member, layout = cell_panel(per_cell=6)
    ids = np.arange(P.shape[1], dtype=float)
    valid = valid.copy()
    top = 5                       # cell (0,0), best PTH
    valid[f - 30:f - 17, top] = False   # 13 missing of 252 -> 239 valid -> not eligible
    near, _ = legs_at(f, P, r, valid, member, member, 252, 0.2, ids)
    assert top not in near
    valid[f - 30, top] = True     # 12 missing -> 240 valid -> eligible
    near, _ = legs_at(f, P, r, valid, member, member, 252, 0.2, ids)
    assert top in near
    trad = member.copy()
    trad[f, top] = False
    near, _ = legs_at(f, P, r, valid, member, trad, 252, 0.2, ids)
    assert top not in near


def test_high_window_changes_pth():
    f, P, r, valid, member, layout = cell_panel()
    ids = np.arange(P.shape[1], dtype=float)
    P = P.copy()
    # cell (0,0) = stocks 0..4: all at their high (PTH 1) within 189 sessions;
    # stock 1 has an older high at f-220 (inside 252, outside 189)
    for q in range(5):
        P[f - 10, q] = P[f, q]
    P[f - 220, 1] = P[f, 1] / 0.5
    near, far = legs_at(f, P, r, valid, member, member, 252, 0.2, ids)
    assert 1 in far and 0 in near
    near189, far189 = legs_at(f, P, r, valid, member, member, 189, 0.2, ids)
    assert 1 not in far189 and 4 in far189 and 0 in near189


def test_ties_smaller_vol_then_id():
    f, P, r, valid, member, layout = cell_panel(pth_levels=np.array([0.6, 0.7, 0.8, 1.0, 1.0]))
    ids = np.arange(P.shape[1], dtype=float)
    near, _ = legs_at(f, P, r, valid, member, member, 252, 0.2, ids)
    # q=3 and q=4 tie at PTH=1; q=3 has the smaller VOL
    assert all(layout[j][2] == 3 for j in near)
    # equal VOL as well: the lower id wins
    r2 = r.copy()
    for j, (a, b, q) in enumerate(layout):
        if q == 4:
            r2[:, j] = r[:, j - 1]
    ids_rev = ids[::-1].copy()
    near, _ = legs_at(f, P, r2, valid, member, member, 252, 0.2, ids)
    assert all(layout[j][2] == 3 for j in near)
    near, _ = legs_at(f, P, r2, valid, member, member, 252, 0.2, ids_rev)
    assert all(layout[j][2] == 4 for j in near)


def test_id_ranks():
    assert list(id_ranks(("E3", "E1", "E2"))) == [2.0, 0.0, 1.0]


# ---------------------------------------------------------------- weights

def test_weights_book():
    data = make_data()
    W = target_weights(data, PARAMS)
    T, N = W.shape
    assert W.shape == (420, 45)
    form = np.flatnonzero(formation_rows(data.dates))
    dec = [f for f in form if f >= LOOKBACK]
    assert dec
    assert np.all(np.isnan(W[: dec[0]]))
    for f in dec:
        row = W[f]
        assert not np.isnan(row).any()
        held = row > 0
        assert held.sum() >= 1
        assert np.allclose(row[held], 1.0 / held.sum())
        assert abs(row.sum() - 1.0) < 1e-9
    other = np.setdiff1d(np.arange(dec[0], T), dec)
    assert np.all(np.isnan(W[other]))       # no intra-month trading when nothing leaves
    assert np.all(np.nansum(np.abs(W), axis=1) <= 1.0 + 1e-9)
    assert np.all(np.nan_to_num(W) >= 0)


def test_exit_when_leaving_universe_stays_cash():
    data = make_data()
    W0 = target_weights(data, PARAMS)
    form = [f for f in np.flatnonzero(formation_rows(data.dates)) if f >= LOOKBACK]
    f, g = form[0], form[1]
    j = int(np.flatnonzero(W0[f] > 0)[0])
    k = int(np.flatnonzero(W0[f] > 0)[1])
    data.universe[f + 3, j] = False          # leaves for one day, comes back
    data.tradable[f + 5, k] = False          # halted one day
    W = target_weights(data, PARAMS)
    assert np.array_equal(W[: f + 3], W0[: f + 3], equal_nan=True)
    assert np.all(np.isnan(W[f + 1:f + 3, j]))
    assert np.all(W[f + 3:g, j] == 0.0)      # sold and kept in cash until the next formation
    assert np.all(np.isnan(W[f + 1:f + 5, k]))
    assert np.all(W[f + 5:g, k] == 0.0)
    others = np.setdiff1d(np.arange(W.shape[1]), [j, k])
    assert np.all(np.isnan(W[f + 1:g][:, others]))


def test_point_in_time_and_determinism():
    data = make_data(seed=3)
    W = target_weights(data, PARAMS)
    assert np.array_equal(W, target_weights(data, PARAMS), equal_nan=True)
    rng = np.random.default_rng(9)
    for t in (260, 300, 333, 380):
        cut = make_data(seed=3)
        cut.ret_oc[t + 1:] = rng.normal(0, 0.05, cut.ret_oc[t + 1:].shape)
        cut.universe[t + 1:] = rng.random(cut.universe[t + 1:].shape) > 0.5
        Wc = target_weights(cut, PARAMS)
        assert np.array_equal(W[: t + 1], Wc[: t + 1], equal_nan=True)


def test_params_change_output():
    data = make_data(seed=5)
    W = target_weights(data, PARAMS)
    W1 = target_weights(data, dict(PARAMS, leg_fraction=0.3))
    W2 = target_weights(data, dict(PARAMS, high_window_days=189))
    assert not np.array_equal(W, W1, equal_nan=True)
    assert not np.array_equal(W, W2, equal_nan=True)


def test_missing_universe_fails_loudly():
    data = make_data()
    data.universe = None
    with pytest.raises(ValueError):
        target_weights(data, PARAMS)
