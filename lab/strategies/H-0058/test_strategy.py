import numpy as np
import pytest
from lab.framework.data import DataView

from strategy import target_weights, PARAMS, sunday_rows, Book

BASE_DAY = 18000      # a Sunday (day number since the datetime64 epoch)
T = 120
OLD = ["O1USDT.P", "O2USDT.P", "O3USDT.P"]
YOUNG = ["Y1USDT.P", "Y2USDT.P", "Y3USDT.P"]
SPOT_START_YOUNG = 5   # spot pair of the young coins starts on row 5
PERP_START_YOUNG = 20  # their perp starts on row 20
T_DEC = 84             # a Sunday whose 60-day beta window (rows 25..84) lies after all perps exist


def make_view(young_scale=1.0, with_spot=True, universe_edit=None, young_perp_start=PERP_START_YOUNG,
              n_rows=T, extra_perp_nan=False):
    rng = np.random.default_rng(7)
    g = rng.normal(0.0, 0.03, n_rows)          # common log-return factor
    perps = OLD + YOUNG + (["Z1USDT.P"] if extra_perp_nan else [])
    spots = [p[:-2] for p in OLD + YOUNG] if with_spot else []
    names = perps + spots
    N = len(names)
    ac = ["crypto_perp"] * len(perps) + ["crypto_spot"] * len(spots)
    start = {}
    for p in OLD:
        start[p] = 0
        start[p[:-2]] = 0
    for p in YOUNG:
        start[p] = young_perp_start
        start[p[:-2]] = SPOT_START_YOUNG
    lr = np.zeros((n_rows, N))
    has = np.zeros((n_rows, N), dtype=bool)
    for j, nm in enumerate(names):
        if nm == "Z1USDT.P":
            continue
        s = start[nm]
        has[s:, j] = True
        scale = young_scale if nm.startswith("Y") else 1.0
        lr[s + 1:, j] = scale * g[s + 1:]
    ret_oc = np.where(has, np.exp(lr) - 1.0, 0.0)
    ret_co = np.zeros((n_rows, N))
    close = np.where(has, 10.0 * np.exp(np.cumsum(lr, axis=0)), np.nan)
    universe = np.zeros((n_rows, N), dtype=bool)
    for j, nm in enumerate(perps):
        universe[:, j] = has[:, j]
    if extra_perp_nan:
        universe[:, names.index("Z1USDT.P")] = True
    if universe_edit is not None:
        universe_edit(universe, names)
    dates = (BASE_DAY + np.arange(n_rows)).astype("datetime64[D]")
    return DataView(
        dates=dates, instruments=tuple(names), asset_class=tuple(ac),
        ret_co=ret_co, ret_oc=ret_oc, tradable=has.copy(), listed=has.copy(),
        delisting=np.zeros((n_rows, N), dtype=bool), close=close,
        dollar_volume=np.where(has, 1e6, np.nan), universe=universe,
    )


def cols(view, names):
    return [view.col(n) for n in names]


def test_sundays():
    dates = (BASE_DAY + np.arange(14)).astype("datetime64[D]")
    s = sunday_rows(dates)
    assert list(np.flatnonzero(s)) == [0, 7]


def test_equal_betas_give_half_half():
    v = make_view(young_scale=1.0)
    w = target_weights(v, PARAMS)
    row = w[T_DEC]
    assert np.allclose(row[cols(v, OLD)], 0.5 / 3)
    assert np.allclose(row[cols(v, YOUNG)], -0.5 / 3)
    spot = [v.col(p[:-2]) for p in OLD + YOUNG]
    assert np.all(row[spot] == 0.0)
    assert abs(np.nansum(np.abs(row)) - 1.0) < 1e-9


def test_beta_neutral_split():
    # young log returns = 2 g, old = g; market = 1.5 g -> beta_young 4/3, beta_old 2/3
    # shrunk: 7/6 and 5/6 -> G_L = 7/12, G_S = 5/12
    v = make_view(young_scale=2.0)
    row = target_weights(v, PARAMS)[T_DEC]
    assert np.allclose(row[cols(v, OLD)], (7 / 12) / 3, atol=1e-9)
    assert np.allclose(row[cols(v, YOUNG)], -(5 / 12) / 3, atol=1e-9)
    net_beta = 3 * (7 / 36) * (5 / 6) - 3 * (5 / 36) * (7 / 6)
    assert abs(net_beta) < 1e-12


def test_only_sundays_decide_and_nan_between():
    v = make_view()
    w = target_weights(v, PARAMS)
    sun = sunday_rows(v.dates)
    assert np.all(np.isnan(w[~sun]))
    assert np.all(np.isfinite(w[sun]))


def test_flat_before_lookback():
    v = make_view()
    w = target_weights(v, PARAMS)
    # row 28: old coins have 28 returns (< 30), young none -> flat
    assert np.all(w[28] == 0.0)
    # row 49: old eligible, young have 29 perp returns -> S empty -> flat
    assert np.all(w[49] == 0.0)
    # row 56: young have 36 returns and 36 beta obs, age 51 -> book on
    assert np.all(w[56][cols(v, YOUNG)] < 0)


def test_spot_listing_date_is_used():
    # age at T_DEC: from spot (row 5) = 79 days, from perp (row 20) = 64 days
    v = make_view()
    p = dict(PARAMS, young_max_days=70)
    assert np.all(target_weights(v, p)[T_DEC] == 0.0)        # 79 > 70: not young -> S empty -> flat
    v2 = make_view(with_spot=False, young_perp_start=PERP_START_YOUNG)
    # without spot data at all the card's input is missing: fail loudly
    with pytest.raises(ValueError):
        target_weights(v2, p)


def test_min_age_boundary():
    v = make_view()
    on = target_weights(v, dict(PARAMS, min_age_days=79))[T_DEC]
    off = target_weights(v, dict(PARAMS, min_age_days=80))[T_DEC]
    assert np.all(on[cols(v, YOUNG)] < 0)
    assert np.all(off == 0.0)


def test_young_max_boundary():
    v = make_view()
    on = target_weights(v, dict(PARAMS, young_max_days=79))[T_DEC]
    off = target_weights(v, dict(PARAMS, young_max_days=78))[T_DEC]
    assert np.all(on[cols(v, YOUNG)] < 0)
    assert np.all(off == 0.0)


def test_fewer_than_three_in_a_leg_is_flat():
    def drop_y3(u, names):
        u[:, names.index("Y3USDT.P")] = False
    v = make_view(universe_edit=drop_y3)
    assert np.all(target_weights(v, PARAMS)[T_DEC] == 0.0)

    def drop_o3(u, names):
        u[:, names.index("O3USDT.P")] = False
    v = make_view(universe_edit=drop_o3)
    assert np.all(target_weights(v, PARAMS)[T_DEC] == 0.0)


def test_universe_exit_closes_position_until_next_sunday():
    def leave(u, names):
        u[T_DEC + 2:T_DEC + 4, names.index("O1USDT.P")] = False
    v = make_view(universe_edit=leave)
    w = target_weights(v, PARAMS)
    o1 = v.col("O1USDT.P")
    assert w[T_DEC, o1] > 0
    assert np.isnan(w[T_DEC + 1]).all()
    assert w[T_DEC + 2, o1] == 0.0
    others = [j for j in range(w.shape[1]) if j != o1]
    assert np.isnan(w[T_DEC + 2, others]).all()
    assert np.isnan(w[T_DEC + 3]).all()          # already closed: nothing more to do
    # back in the universe at the next Sunday: reset to full targets
    assert w[T_DEC + 7, o1] > 0


def test_member_without_listing_date_is_not_eligible():
    v = make_view(extra_perp_nan=True)
    w = target_weights(v, PARAMS)
    z = v.col("Z1USDT.P")
    assert np.all(np.nan_to_num(w[:, z]) == 0.0)
    assert np.all(w[T_DEC][cols(v, YOUNG)] < 0)


def test_gross_and_long_short_signs():
    v = make_view(young_scale=1.7)
    w = target_weights(v, PARAMS)
    g = np.nansum(np.abs(w), axis=1)
    assert np.all(g <= 1.0 + 1e-9)


def test_point_in_time_truncation():
    v = make_view(young_scale=1.5)
    full = target_weights(v, PARAMS)
    for cut in (60, 84, 90, 100):
        vt = make_view(young_scale=1.5, n_rows=cut + 1)
        wt = target_weights(vt, PARAMS)
        assert np.array_equal(np.isnan(wt), np.isnan(full[:cut + 1]))
        assert np.allclose(np.nan_to_num(wt), np.nan_to_num(full[:cut + 1]))


def test_deterministic():
    v = make_view(young_scale=1.3)
    a = target_weights(v, PARAMS)
    b = target_weights(v, PARAMS)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_short_leg_definition():
    # S (the diagnostic's event set uses the same Book.legs) holds exactly the young coins at T_DEC
    v = make_view(young_scale=2.0)
    S, L, b = Book(v, PARAMS).legs(T_DEC)
    assert list(np.flatnonzero(S)) == cols(v, YOUNG)
    assert list(np.flatnonzero(L)) == cols(v, OLD)
    assert np.allclose(b[S], 7 / 6) and np.allclose(b[L], 5 / 6)


def test_censored_spot_listing_is_old():
    # old coins' spot pairs start on row 0 of the data: censored, treated as old (> 730 days) at once
    v = make_view()
    S, L, _ = Book(v, PARAMS).legs(35)
    assert all(L[j] for j in cols(v, OLD))
