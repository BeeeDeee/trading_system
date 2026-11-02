import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import accrual_events


class View:
    """Minimal stand-in for DataView with the fields the strategy reads."""

    def __init__(self, ni, cf, at, uni=None, trad=None):
        T, N = ni.shape
        self.extras = {
            "sf1_arq_netinc": ni.astype(np.float32),
            "sf1_arq_ncfo": cf.astype(np.float32),
            "sf1_arq_assets": at.astype(np.float32),
        }
        self.universe = np.ones((T, N), dtype=bool) if uni is None else uni
        self.tradable = np.ones((T, N), dtype=bool) if trad is None else trad
        self.instruments = tuple("E%d" % j for j in range(N))

    def head(self, t):
        return View(self.extras["sf1_arq_netinc"][:t], self.extras["sf1_arq_ncfo"][:t],
                    self.extras["sf1_arq_assets"][:t], self.universe[:t], self.tradable[:t])


def base(T=130, N=120):
    # acc_j = j / 1000, loaded on row 0 (warm-up), no change afterwards
    ni = np.tile(np.arange(N, dtype=np.float64), (T, 1))
    cf = np.zeros((T, N))
    at = np.full((T, N), 1000.0)
    return ni, cf, at


def P(**kw):
    p = dict(PARAMS)
    p.update(kw)
    return p


def test_params_primary():
    assert PARAMS == {"hold_days": 5, "leg_quantile": 0.2}


def test_sides_and_hold():
    ni, cf, at = base()
    ni[70:, 0] = -5.0      # lowest acc -> long
    ni[70:, 119] = 500.0   # highest -> short
    ni[70:, 60] = 60.5     # middle -> no event
    v = View(ni, cf, at)
    filing, side = accrual_events(v, PARAMS)
    assert filing[70, 0] and filing[70, 119] and filing[70, 60]
    assert filing.sum() == 3
    assert side[70, 0] == 1 and side[70, 119] == -1 and side[70, 60] == 0
    w = target_weights(v, PARAMS)
    assert w[69, 0] == 0
    assert np.allclose(w[70:75, 0], 0.04)
    assert np.allclose(w[70:75, 119], -0.04)
    assert np.all(w[75:, 0] == 0) and np.all(w[75:, 119] == 0)
    assert np.all(w[:, 60] == 0)
    assert np.abs(w).sum() == pytest.approx(0.04 * 10)
    w3 = target_weights(v, P(hold_days=3))
    assert np.allclose(w3[70:73, 0], 0.04) and w3[73, 0] == 0
    w8 = target_weights(v, P(hold_days=8))
    assert np.allclose(w8[70:78, 0], 0.04) and w8[78, 0] == 0


def test_leg_quantile_changes_membership():
    ni, cf, at = base()
    ni[70:, 30] = 30.01    # rank 30 of 120: lo(0.2) = 23.8, lo(0.3) = 35.7 (per 1000)
    v = View(ni, cf, at)
    assert np.all(target_weights(v, P(leg_quantile=0.2)) == 0)
    assert np.all(target_weights(v, P(leg_quantile=0.1)) == 0)
    w = target_weights(v, P(leg_quantile=0.3))
    assert np.allclose(w[70:75, 30], 0.04)


def test_change_in_any_field_counts():
    ni, cf, at = base()
    cf[70:, 0] = 1.0       # ncfo change: acc 0 -> -0.001, lowest
    at[70:, 119] = 900.0   # assets change: acc 0.119 -> 0.1322, highest
    v = View(ni, cf, at)
    _, side = accrual_events(v, PARAMS)
    assert side[70, 0] == 1 and side[70, 119] == -1


def test_warmup():
    ni, cf, at = base()
    ni[62:, 0] = -5.0
    ni[63:, 119] = 500.0
    _, side = accrual_events(View(ni, cf, at), PARAMS)
    assert side[62, 0] == 0 and side[:, 0].sum() == 0
    assert side[63, 119] == -1


def test_forty_session_rule():
    ni, cf, at = base()
    ni[70:, 0] = -5.0
    ni[100:, 0] = -6.0     # 30 sessions later: rejected, does not reset the clock
    ni[110:, 0] = -7.0     # 40 sessions after the accepted one: rejected
    ni[111:, 0] = -8.0     # 41 sessions after: accepted
    filing, side = accrual_events(View(ni, cf, at), PARAMS)
    assert list(np.nonzero(filing[:, 0])[0]) == [70, 111]
    assert side[111, 0] == 1
    w = target_weights(View(ni, cf, at), PARAMS)
    assert np.allclose(w[111:116, 0], 0.04) and w[116, 0] == 0 and w[100, 0] == 0


def test_nan_to_value_is_a_filing():
    ni, cf, at = base()
    ni[:80, 0] = np.nan
    cf[:80, 0] = np.nan
    at[:80, 0] = np.nan
    ni[80:, 0] = -5.0
    _, side = accrual_events(View(ni, cf, at), PARAMS)
    assert side[80, 0] == 1
    # missing one field: no event, and the stock is not in the cross-section
    ni2, cf2, at2 = base()
    cf2[:, 0] = np.nan
    ni2[80:, 0] = -5.0
    f2, s2 = accrual_events(View(ni2, cf2, at2), PARAMS)
    assert not f2[:, 0].any()


def test_nonpositive_assets_no_event():
    ni, cf, at = base()
    ni[70:, 0] = -5.0
    at[70:, 0] = 0.0
    f, _ = accrual_events(View(ni, cf, at), PARAMS)
    assert not f[:, 0].any()


def test_min_members():
    ni, cf, at = base()
    ni[70:, 0] = -5.0
    uni = np.ones(ni.shape, dtype=bool)
    uni[:, 99:] = False    # 99 members
    f, side = accrual_events(View(ni, cf, at, uni=uni), PARAMS)
    assert f[70, 0] and side[70, 0] == 0
    uni[:, 99] = True      # 100 members
    _, side = accrual_events(View(ni, cf, at, uni=uni), PARAMS)
    assert side[70, 0] == 1


def test_non_member_filing_is_not_an_event():
    ni, cf, at = base()
    ni[70:, 0] = -5.0
    uni = np.ones(ni.shape, dtype=bool)
    uni[70, 0] = False
    f, _ = accrual_events(View(ni, cf, at, uni=uni), PARAMS)
    assert not f[:, 0].any()


def test_exit_on_universe_or_tradable():
    ni, cf, at = base()
    ni[70:, 0] = -5.0
    ni[70:, 119] = 500.0
    uni = np.ones(ni.shape, dtype=bool)
    uni[72, 0] = False
    trad = np.ones(ni.shape, dtype=bool)
    trad[71, 119] = False
    w = target_weights(View(ni, cf, at, uni=uni, trad=trad), PARAMS)
    assert np.allclose(w[70:72, 0], 0.04) and np.all(w[72:, 0] == 0)
    assert w[70, 119] == pytest.approx(-0.04) and np.all(w[71:, 119] == 0)


def test_untradable_event_row_still_enters():
    ni, cf, at = base()
    ni[70:, 0] = -5.0
    trad = np.ones(ni.shape, dtype=bool)
    trad[70, 0] = False
    v = View(ni, cf, at, trad=trad)
    w = target_weights(v, PARAMS)
    assert np.allclose(w[70:75, 0], 0.04)


def test_gross_cap_scales_new_entries():
    ni, cf, at = base(T=130, N=200)
    ni[70:, :40] += 0.1           # 40 long events (lo = 39.82/1000)
    ni[71:, 199] = 500.0          # short event when the book is full
    v = View(ni, cf, at)
    _, side = accrual_events(v, PARAMS)
    assert side[70, :40].sum() == 40
    w = target_weights(v, PARAMS)
    assert np.allclose(w[70, :40], 0.025)
    assert np.abs(w[70]).sum() <= 1.0 and np.abs(w[70]).sum() == pytest.approx(1.0)
    assert np.all(w[71:, 199] == 0)   # no room, the event is dropped
    assert np.all(w[75, :40] == 0)


def test_gross_cap_with_existing_book():
    ni, cf, at = base(T=130, N=200)
    ni[70:, :15] += 0.1           # 15 long events: gross 0.6
    ni[72:, 180:] += 0.1          # 20 short events: 0.8 more -> scaled by 0.5
    w = target_weights(View(ni, cf, at), PARAMS)
    assert np.allclose(w[72, :15], 0.04)
    assert np.allclose(w[72, 180:], -0.02)
    assert np.abs(w[72]).sum() == pytest.approx(1.0) and np.abs(w[72]).sum() <= 1.0
    # weights stay at their entry values (no re-equalizing) after the first group exits
    assert np.all(w[75, :15] == 0) and np.allclose(w[75:77, 180:], -0.02) and np.all(w[77, 180:] == 0)


def random_view(seed=0, T=260, N=150):
    rng = np.random.default_rng(seed)
    ni = rng.normal(0, 10, (T, N))
    cf = rng.normal(0, 10, (T, N))
    at = rng.uniform(100, 1000, (T, N))
    # values change only on filing rows (every ~60 rows with random offsets, plus some re-filings)
    for j in range(N):
        rows = [0] + sorted(set(rng.integers(1, T, 6).tolist()))
        for a, b in zip(rows, rows[1:] + [T]):
            ni[a:b, j] = ni[a, j]
            cf[a:b, j] = cf[a, j]
            at[a:b, j] = at[a, j]
    ni[:30, :20] = np.nan
    uni = rng.random((T, N)) > 0.05
    trad = rng.random((T, N)) > 0.02
    return View(ni, cf, at, uni=uni, trad=trad)


def test_random_invariants_and_point_in_time():
    v = random_view()
    for p in (PARAMS, P(hold_days=8, leg_quantile=0.3), P(hold_days=3, leg_quantile=0.1)):
        w = target_weights(v, p)
        assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
        assert np.all(w[~v.universe] == 0)
        assert np.all(w[:63] == 0)
        assert np.abs(w).sum() > 0
        assert np.array_equal(w, target_weights(v, p))
        for t in (70, 101, 150, 200):
            assert np.array_equal(target_weights(v.head(t), p), w[:t])
    w5 = target_weights(v, PARAMS)
    assert not np.array_equal(w5, target_weights(v, P(hold_days=3)))
    assert not np.array_equal(w5, target_weights(v, P(leg_quantile=0.3)))


def test_missing_universe_raises():
    ni, cf, at = base()
    v = View(ni, cf, at)
    v.universe = None
    with pytest.raises(ValueError):
        target_weights(v, PARAMS)
