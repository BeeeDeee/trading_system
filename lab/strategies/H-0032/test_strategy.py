import numpy as np
import pytest
from lab.framework.data import DataView, synthetic

from strategy import target_weights, PARAMS

N = 120
T = 700
R0 = 10                       # last year's release row
LONGS = [0, 1, 2, 3, 30]      # low accruals -> high score (stock 30 long only with q = 0.3)
SHORTS = [110, 111, 112, 113, 90]   # high accruals -> low score (stock 90 short only with q = 0.3)


def _dates():
    return synthetic().dates[:T]


def make_market(release_rows=None, n=N, gp_nan=(), universe=None, delisting=None):
    dates = _dates()
    ea = np.zeros((T, n), dtype=np.float32)
    if release_rows is None:
        release_rows = {j: [R0] for j in LONGS + SHORTS}
    for j, rows in release_rows.items():
        for r in rows:
            ea[r, j] = 1.0
    assets = np.full((T, n), 100.0, dtype=np.float32)          # AG = 0 for all (tie)
    ncfo = np.zeros((T, n), dtype=np.float32)
    netinc = np.tile(np.arange(n, dtype=np.float32), (T, 1))    # ACC = j / 100, rank increasing in j
    gp = np.full((T, n), 30.0, dtype=np.float32)                # GPA tie
    for j in gp_nan:
        gp[:, j] = np.nan
    ones = np.ones((T, n), dtype=bool)
    uni = ones.copy() if universe is None else universe
    dl = np.zeros((T, n), dtype=bool) if delisting is None else delisting
    return DataView(
        dates=dates,
        instruments=tuple("E%d" % (1000 + j) for j in range(n)),
        asset_class=tuple("us_equity" for _ in range(n)),
        ret_co=np.zeros((T, n)),
        ret_oc=np.zeros((T, n)),
        tradable=ones.copy(),
        listed=ones.copy(),
        delisting=dl,
        close=np.ones((T, n)),
        dollar_volume=np.ones((T, n)),
        universe=uni,
        extras={"ea_reaction_day": ea, "sf1_art_netinc": netinc, "sf1_art_ncfo": ncfo,
                "sf1_art_assets": assets, "sf1_art_gp": gp},
        series={},
        cash_ret=np.zeros(T),
    )


def expected_decision_row(dates, r, hw):
    """First row t whose weekday count to P = date(r) + 364 days is <= hw + 1."""
    p = (dates[r].astype(np.int64) + 364).astype(dates.dtype)
    for t in range(len(dates)):
        if np.busday_count(dates[t], p) <= hw + 1:
            return t
    return None


def test_params_are_card_primary():
    assert PARAMS == {"half_window": 2, "leg_quantile": 0.3}


def test_window_and_weights():
    d = make_market()
    w = target_weights(d, PARAMS)
    te = expected_decision_row(d.dates, R0, 2)
    assert te is not None and te > 252
    rows = np.nonzero(np.abs(w).sum(axis=1) > 0)[0]
    assert list(rows) == list(range(te, te + 5))          # 2*hw+1 sessions
    for t in rows:
        assert np.allclose(w[t, LONGS], 0.1)
        assert np.allclose(w[t, SHORTS], -0.1)
        assert abs(w[t].sum()) < 1e-12
        assert abs(np.abs(w[t]).sum() - 1.0) < 1e-12
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)


def test_half_window_changes_window():
    d = make_market()
    w = target_weights(d, {"half_window": 1, "leg_quantile": 0.3})
    te = expected_decision_row(d.dates, R0, 1)
    rows = np.nonzero(np.abs(w).sum(axis=1) > 0)[0]
    assert list(rows) == list(range(te, te + 3))
    te2 = expected_decision_row(d.dates, R0, 2)
    assert te == te2 + 1                                    # opens one session later


def test_leg_quantile_changes_legs():
    d = make_market()
    w = target_weights(d, {"half_window": 2, "leg_quantile": 0.2})
    te = expected_decision_row(d.dates, R0, 2)
    assert w[te, 30] == 0.0 and w[te, 90] == 0.0           # middle names drop out
    assert np.allclose(w[te, [0, 1, 2, 3]], 0.125)
    assert np.allclose(w[te, [110, 111, 112, 113]], -0.125)


def test_leg_with_fewer_than_three_names_is_flat():
    d = make_market(release_rows={0: [R0], 1: [R0], 2: [R0], 110: [R0], 111: [R0]})
    w = target_weights(d, PARAMS)
    assert np.all(w == 0.0)


def test_recent_release_blocks_event():
    te = expected_decision_row(_dates(), R0, 2)
    rel = {j: [R0] for j in LONGS + SHORTS}
    rel[0] = [R0, te - 29]          # inside the 30 sessions up to t
    rel[1] = [R0, te - 30]          # just outside
    d = make_market(release_rows=rel)
    w = target_weights(d, PARAMS)
    assert w[te, 0] == 0.0
    assert w[te, 1] == pytest.approx(0.125)
    assert w[te, 2] == pytest.approx(0.125)


def test_second_release_within_ten_sessions_ignored():
    rel = {j: [R0] for j in LONGS + SHORTS}
    rel[0] = [R0, R0 + 10]          # within 10 sessions: dropped
    rel[1] = [R0, R0 + 11]          # 11 sessions later: its own predicted window
    d = make_market(release_rows=rel)
    w = target_weights(d, PARAMS)
    te = expected_decision_row(d.dates, R0, 2)
    te10 = expected_decision_row(d.dates, R0 + 10, 2)
    te11 = expected_decision_row(d.dates, R0 + 11, 2)
    assert np.all(w[te + 5:, 0] == 0.0)
    assert te10 > te + 5
    # stock 1 opens a second window, but its long leg alone has 1 name -> book flat
    assert np.all(w[te11:, :] == 0.0)
    assert w[te, 1] > 0.0


def test_too_few_scored_members_means_no_events():
    d = make_market(gp_nan=tuple(range(40, 61)))           # 99 members with all inputs
    assert np.all(target_weights(d, PARAMS) == 0.0)
    d = make_market(gp_nan=tuple(range(40, 60)))           # 100 members
    assert np.abs(target_weights(d, PARAMS)).sum() > 0.0


def test_missing_fundamental_means_no_event():
    d = make_market(gp_nan=(0,))
    w = target_weights(d, PARAMS)
    te = expected_decision_row(d.dates, R0, 2)
    assert w[te, 0] == 0.0
    assert w[te, 1] == pytest.approx(0.125)


def test_universe_exit_closes_position_and_reweights():
    te = expected_decision_row(_dates(), R0, 2)
    uni = np.ones((T, N), dtype=bool)
    uni[te + 2:, 0] = False
    d = make_market(universe=uni)
    w = target_weights(d, PARAMS)
    assert w[te + 1, 0] == pytest.approx(0.1)
    assert np.all(w[te + 2:, 0] == 0.0)
    assert w[te + 2, 1] == pytest.approx(0.125)
    assert np.all(w[~uni] == 0.0)


def test_not_member_at_decision_means_no_event():
    te = expected_decision_row(_dates(), R0, 2)
    uni = np.ones((T, N), dtype=bool)
    uni[te, 0] = False
    d = make_market(universe=uni)
    w = target_weights(d, PARAMS)
    assert np.all(w[:, 0] == 0.0)


def test_delisting_closes_position():
    te = expected_decision_row(_dates(), R0, 2)
    dl = np.zeros((T, N), dtype=bool)
    dl[te + 1, 110] = True
    d = make_market(delisting=dl)
    w = target_weights(d, PARAMS)
    assert w[te, 110] == pytest.approx(-0.1)
    assert np.all(w[te + 1:, 110] == 0.0)
    assert w[te + 1, 111] == pytest.approx(-0.125)


def test_missing_input_raises():
    d = make_market()
    del d.extras["sf1_art_gp"]
    with pytest.raises(KeyError):
        target_weights(d, PARAMS)


def _truncate(d, k):
    return DataView(
        dates=d.dates[:k], instruments=d.instruments, asset_class=d.asset_class,
        ret_co=d.ret_co[:k], ret_oc=d.ret_oc[:k], tradable=d.tradable[:k], listed=d.listed[:k],
        delisting=d.delisting[:k], close=d.close[:k], dollar_volume=d.dollar_volume[:k],
        universe=d.universe[:k], extras={key: v[:k] for key, v in d.extras.items()},
        series={}, cash_ret=d.cash_ret[:k],
    )


def test_point_in_time_truncation_and_perturbation():
    rng = np.random.default_rng(7)
    rel = {j: [R0, R0 + 63, R0 + 126] for j in LONGS + SHORTS}
    d = make_market(release_rows=rel)
    full = target_weights(d, PARAMS)
    assert np.abs(full).sum() > 0.0
    te = expected_decision_row(d.dates, R0, 2)
    for k in [te - 1, te, te + 1, te + 3, te + 64, te + 130, T - 1]:
        part = target_weights(_truncate(d, k), PARAMS)
        assert np.array_equal(part, full[:k])
        # perturb the future
        e = make_market(release_rows=rel)
        for key in ("sf1_art_netinc", "sf1_art_ncfo", "sf1_art_gp", "sf1_art_assets"):
            e.extras[key][k:] = rng.normal(100.0, 30.0, size=(T - k, N)).astype(np.float32)
        e.extras["ea_reaction_day"][k:] = (rng.random((T - k, N)) < 0.05).astype(np.float32)
        e.universe[k:] = rng.random((T - k, N)) < 0.8
        pert = target_weights(e, PARAMS)
        assert np.array_equal(pert[:k], full[:k])
