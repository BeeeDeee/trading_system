import numpy as np
import pytest
from lab.framework.data import DataView, synthetic
from strategy import target_weights, PARAMS
import strategy as S

T = 420
N = 24
T0 = 400          # the event row in most tests


def _dates():
    return synthetic().dates[:T]


def make(seed=0, groups=True, ea_only_t0=True):
    """24 stocks in two correlated groups of 12 (factor + noise); ret_co = 0 so r = ret_oc."""
    rng = np.random.default_rng(seed)
    fa = rng.normal(0, 0.01, T)
    fb = rng.normal(0, 0.01, T)
    noise = rng.normal(0, 0.005, (T, N))
    r = noise.copy()
    if groups:
        r[:, :12] += fa[:, None]
        r[:, 12:] += fb[:, None]
    if ea_only_t0:
        ea = np.full((T, N), np.nan)
        ea[T0] = 0.0
    else:
        ea = np.zeros((T, N))
    return DataView(
        dates=_dates(),
        instruments=tuple("E%d" % (i + 1) for i in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=np.zeros((T, N)),
        ret_oc=r,
        tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.full((T, N), 50.0),
        dollar_volume=np.full((T, N), 1e8),
        universe=np.ones((T, N), dtype=bool),
        extras={"ea_reaction_day": ea},
        series={},
        cash_ret=np.zeros(T),
    )


def shock(d, j, x, t=T0):
    d.ret_oc[t, j] = x
    return d


def test_return_formula():
    d = make()
    d.ret_co[5, 3] = 0.1
    d.ret_oc[5, 3] = -0.2
    d.tradable[6, 4] = False
    r, valid = S._returns(d)
    assert r[5, 3] == pytest.approx(1.1 * 0.8 - 1.0)
    assert np.isnan(r[6, 4]) and not valid[6, 4]


def test_peers_are_the_correlated_group_and_use_only_past_rows():
    d = make()
    r, valid = S._returns(d)
    s = S._month_start_rows(d.dates)
    s = int(s[s >= 300][0])
    cols, peers = S._peers(r, valid, d.universe[s], s)
    assert peers.shape == (N, 10)
    for c, p in zip(cols, peers):
        assert c not in p
        assert np.all((p < 12) == (c < 12))
    # changing the return of row s (and later) does not change the peers decided on row s
    r2 = r.copy()
    r2[s:] = np.random.default_rng(9).normal(0, 1, r2[s:].shape)
    cols2, peers2 = S._peers(r2, valid, d.universe[s], s)
    assert np.array_equal(cols, cols2) and np.array_equal(peers, peers2)


def test_too_few_common_observations_means_no_peers():
    d = make()
    d.listed[:330, 5] = False
    d.tradable[:330, 5] = False
    r, valid = S._returns(d)
    s = S._month_start_rows(d.dates)
    s = int(s[(s > 330) & (s < 330 + 126)][0])
    cols, peers = S._peers(r, valid, d.universe[s], s)
    assert 5 not in cols
    assert not np.any(peers == 5)


def test_group_wide_move_is_not_a_shock():
    d = make(ea_only_t0=True)
    d.ret_oc[T0, :12] += 0.30            # whole group A jumps: peer-adjusted shock ~ 0
    shock(d, 15, -0.25)                  # single-name drop in group B
    C = S.cohorts(d, PARAMS)
    assert C[T0, 15] > 0
    assert np.all(C[T0, :12] == 0)


def test_single_cohort_weights_and_hold():
    d = make()
    shock(d, 3, -0.3)
    shock(d, 17, 0.3)
    W = target_weights(d, PARAMS)
    H = PARAMS["hold_days"]
    # n = 24 eligible, k = max(1, round(0.02*24)) = 1
    assert np.all(W[:T0] == 0)
    for t in range(T0, T0 + H):
        assert W[t, 3] == pytest.approx(1 / (2 * H))
        assert W[t, 17] == pytest.approx(-1 / (2 * H))
        assert np.count_nonzero(W[t]) == 2
    assert np.all(W[T0 + H:] == 0)


def test_hold_days_param_changes_hold_and_size():
    d = make()
    shock(d, 3, -0.3)
    shock(d, 17, 0.3)
    p = dict(PARAMS, hold_days=3)
    W = target_weights(d, p)
    assert W[T0, 3] == pytest.approx(1 / 6)
    assert W[T0 + 2, 3] == pytest.approx(1 / 6)
    assert W[T0 + 3, 3] == 0


def test_leg_pct_param_changes_leg_size():
    d = make()
    shock(d, 3, -0.3)
    shock(d, 4, -0.2)
    shock(d, 17, 0.3)
    shock(d, 18, 0.2)
    p = dict(PARAMS, leg_pct=0.1)        # round(2.4) = 2 names per leg
    C = S.cohorts(d, p)
    H = PARAMS["hold_days"]
    assert C[T0, 3] == pytest.approx(1 / (2 * H * 2)) and C[T0, 4] == pytest.approx(1 / (2 * H * 2))
    assert C[T0, 17] == pytest.approx(-1 / (2 * H * 2)) and C[T0, 18] == pytest.approx(-1 / (2 * H * 2))
    assert np.count_nonzero(C[T0]) == 4
    assert np.count_nonzero(S.cohorts(d, PARAMS)[T0]) == 2


def test_news_exclusion_on_t_t_minus_1_and_nan():
    for how in ("t", "t-1", "nan"):
        d = make()
        shock(d, 3, -0.3)
        shock(d, 4, -0.2)
        ea = d.extras["ea_reaction_day"]
        if how == "t":
            ea[T0, 3] = 1.0
        elif how == "t-1":
            ea[T0 - 1, 3] = 1.0
        else:
            ea[T0, 3] = np.nan
        C = S.cohorts(d, PARAMS)
        assert C[T0, 3] == 0, how
        assert C[T0, 4] > 0, how
    # a release two sessions earlier does not exclude
    d = make()
    shock(d, 3, -0.3)
    d.extras["ea_reaction_day"][T0 - 2, 3] = 1.0
    assert S.cohorts(d, PARAMS)[T0, 3] > 0


def test_exit_from_universe_closes_and_slot_stays_cash():
    d = make()
    shock(d, 3, -0.3)
    shock(d, 17, 0.3)
    d.universe[T0 + 2, 3] = False        # leaves for one day, comes back
    d.tradable[T0 + 1, 17] = False       # trading halt on one day
    W = target_weights(d, PARAMS)
    assert W[T0, 3] > 0 and W[T0 + 1, 3] > 0
    assert np.all(W[T0 + 2:, 3] == 0)
    assert W[T0, 17] < 0
    assert np.all(W[T0 + 1:, 17] == 0)


def test_history_requirement_and_late_listing():
    d = make(ea_only_t0=False)
    d.listed[:250, 7] = False
    d.tradable[:250, 7] = False
    shock(d, 7, -0.5, t=T0)              # only 150 sessions of history at T0: not eligible
    C = S.cohorts(d, PARAMS)
    assert C[T0, 7] == 0
    assert np.all(C[:251] == 0)          # 252 sessions of history (incl. t) first reached on row 251
    assert np.any(C[300:] != 0)


def test_gross_and_neutrality_with_overlapping_cohorts():
    d = make(ea_only_t0=False)
    for p in (PARAMS, dict(PARAMS, hold_days=7, leg_pct=0.03), dict(PARAMS, hold_days=3, leg_pct=0.01)):
        W = target_weights(d, p)
        assert np.all(np.abs(W).sum(axis=1) <= 1 + 1e-12)
        assert np.allclose(W.sum(axis=1), 0.0)
        assert np.any(W != 0)


def test_point_in_time_truncation():
    d = make(ea_only_t0=False)
    W = target_weights(d, PARAMS)
    for cut in (300, 350, 401):
        dc = DataView(
            dates=d.dates[:cut], instruments=d.instruments, asset_class=d.asset_class,
            ret_co=d.ret_co[:cut], ret_oc=d.ret_oc[:cut], tradable=d.tradable[:cut], listed=d.listed[:cut],
            delisting=d.delisting[:cut], close=d.close[:cut], dollar_volume=d.dollar_volume[:cut],
            universe=d.universe[:cut], extras={"ea_reaction_day": d.extras["ea_reaction_day"][:cut]},
            series={}, cash_ret=d.cash_ret[:cut])
        assert np.array_equal(target_weights(dc, PARAMS), W[:cut])


def test_missing_input_raises():
    d = make()
    d.extras.pop("ea_reaction_day")
    with pytest.raises(KeyError):
        target_weights(d, PARAMS)
