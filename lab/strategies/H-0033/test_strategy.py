import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

T = 90
EV = 65          # event row in most tests


def make_view(ret_oc, universe=None, tradable=None, ea=None, mc=None, extras=None, no_universe=False):
    T_, N = ret_oc.shape
    names = tuple("S%02d" % j for j in range(N))
    if universe is None:
        universe = np.ones((T_, N), dtype=bool)
    if tradable is None:
        tradable = np.ones((T_, N), dtype=bool)
    if ea is None:
        ea = np.zeros((T_, N))
    if mc is None:
        mc = np.tile(100.0 * (np.arange(N) + 1.0), (T_, 1))
    if extras is None:
        extras = {"ea_reaction_day": ea.astype(np.float32), "sf1_arq_marketcap": mc.astype(np.float32)}
    return DataView(
        dates=np.arange(12000, 12000 + T_).astype("datetime64[D]"),
        instruments=names,
        asset_class=tuple("us_equity" for _ in names),
        ret_co=np.zeros((T_, N)),
        ret_oc=ret_oc.astype(np.float64),
        tradable=tradable,
        listed=np.ones((T_, N), dtype=bool),
        delisting=np.zeros((T_, N), dtype=bool),
        close=np.full((T_, N), 50.0),
        dollar_volume=np.full((T_, N), 1e8),
        universe=None if no_universe else universe,
        extras=extras,
        series={},
        cash_ret=np.zeros(T_),
    )


def alt_returns(T_, N, amp=0.01):
    """Stock 0 alternates +amp/-amp, every other stock has zero return."""
    r = np.zeros((T_, N))
    r[:, 0] = np.where(np.arange(T_) % 2 == 0, amp, -amp)
    return r


def core_weights(r, mc_row, t):
    """Hand formula: constant market caps from row 0, grown with the product of (1+r) after row 0."""
    g = np.prod(1.0 + r[1:t + 1], axis=0)
    cap = mc_row * g
    return cap / cap.sum()


def test_single_event_slot_and_exit():
    N = 4
    r = alt_returns(T, N)
    r[EV, 0] = 0.10
    ea = np.zeros((T, N)); ea[EV, 0] = 1
    W = target_weights(make_view(r, ea=ea), PARAMS)
    mc_row = 100.0 * (np.arange(N) + 1.0)
    # row 0 is a decision row with the pure core
    assert np.allclose(W[0], mc_row / mc_row.sum())
    # nothing changes between row 1 and the event: no decision
    assert np.isnan(W[1:EV]).all()
    c = core_weights(r, mc_row, EV)
    exp = 0.96 * c
    exp[0] += 0.04
    assert np.allclose(W[EV], exp)
    assert abs(W[EV].sum() - 1.0) < 1e-12
    # held for hold_days = 5 rows (EV..EV+4), closed in row EV+5
    assert np.isnan(W[EV + 1:EV + 5]).all()
    assert np.allclose(W[EV + 5], core_weights(r, mc_row, EV + 5))
    assert np.isnan(W[EV + 6:]).all()


def test_hold_days_param_moves_exit():
    N = 4
    r = alt_returns(T, N)
    r[EV, 0] = 0.10
    ea = np.zeros((T, N)); ea[EV, 0] = 1
    W = target_weights(make_view(r, ea=ea), dict(PARAMS, hold_days=3))
    assert np.isnan(W[EV + 1:EV + 3]).all()
    assert not np.isnan(W[EV + 3]).any()
    assert np.isnan(W[EV + 4:]).all()


def test_max_name_weight_param():
    N = 4
    r = alt_returns(T, N)
    r[EV, 0] = 0.10
    ea = np.zeros((T, N)); ea[EV, 0] = 1
    W = target_weights(make_view(r, ea=ea), dict(PARAMS, max_name_weight=0.02))
    c = core_weights(r, 100.0 * (np.arange(N) + 1.0), EV)
    exp = 0.98 * c
    exp[0] += 0.02
    assert np.allclose(W[EV], exp)


def test_z_threshold_param():
    N = 4
    r = alt_returns(T, N)
    # sigma of a over rows EV-63..EV-1: stock 0's own returns (others are 0, so the peer mean is 0)
    sigma = np.std(r[EV - 63:EV, 0], ddof=1)
    r[EV, 0] = 2.2 * sigma
    ea = np.zeros((T, N)); ea[EV, 0] = 1
    W20 = target_weights(make_view(r, ea=ea), dict(PARAMS, z_threshold=2.0))
    W25 = target_weights(make_view(r, ea=ea), dict(PARAMS, z_threshold=2.5))
    assert not np.isnan(W20[EV]).any()          # event at z = 2.2 >= 2.0
    assert np.isnan(W25[EV]).all()              # no event at 2.5: nothing changes, no decision


def test_negative_reaction_and_missing_flag_no_event():
    N = 4
    r = alt_returns(T, N)
    r[EV, 0] = -0.10
    r[EV + 2, 0] = 0.10
    ea = np.zeros((T, N)); ea[EV, 0] = 1; ea[EV + 2, 0] = np.nan
    W = target_weights(make_view(r, ea=ea), PARAMS)
    assert np.isnan(W[1:]).all()


def test_no_event_without_flag():
    N = 4
    r = alt_returns(T, N)
    r[EV, 0] = 0.10
    W = target_weights(make_view(r), PARAMS)
    assert np.isnan(W[1:]).all()


def test_min_observations_edge():
    N = 4
    r = alt_returns(T, N)
    ea = np.zeros((T, N))
    r[39, 0] = 0.10; ea[39, 0] = 1     # window rows 0..38 = 39 obs < 40: no event
    W = target_weights(make_view(r, ea=ea), PARAMS)
    assert np.isnan(W[39]).all()
    r = alt_returns(T, N)
    ea = np.zeros((T, N))
    r[40, 0] = 0.10; ea[40, 0] = 1     # window rows 0..39 = 40 obs: event
    W = target_weights(make_view(r, ea=ea), PARAMS)
    assert W[40, 0] > 0.04


def test_zero_variance_history_no_event():
    N = 4
    r = np.zeros((T, N))
    r[EV, 0] = 0.10
    ea = np.zeros((T, N)); ea[EV, 0] = 1
    W = target_weights(make_view(r, ea=ea), PARAMS)
    assert np.isnan(W[1:]).all()


def test_leave_one_out_abnormal_return():
    # all four stocks jump by the same 10 %: the abnormal return is 0, no event
    N = 4
    r = alt_returns(T, N)
    r[EV, :] = 0.10
    ea = np.zeros((T, N)); ea[EV, 0] = 1
    W = target_weights(make_view(r, ea=ea), PARAMS)
    assert np.isnan(W[EV]).all()


def test_budget_split_and_full():
    N = 40
    r = np.zeros((T, N))
    r[:, :] = np.where((np.arange(T)[:, None] + np.arange(N)[None, :]) % 2 == 0, 0.01, -0.01)
    ea = np.zeros((T, N))
    # day EV: 12 events -> 12 x 0.04 = 0.48
    r[EV, 0:12] = 0.20; ea[EV, 0:12] = 1
    # day EV+1: 2 events -> room 0.02 / 2 = 0.01 each
    r[EV + 1, 12:14] = 0.20; ea[EV + 1, 12:14] = 1
    # day EV+2: budget full -> skipped
    r[EV + 2, 14] = 0.20; ea[EV + 2, 14] = 1
    W = target_weights(make_view(r, ea=ea), PARAMS)
    core1 = W[EV + 1] - np.r_[np.full(12, 0.04), np.full(2, 0.01), np.zeros(N - 14)]
    assert abs(W[EV + 1].sum() - 1.0) < 1e-12
    assert abs(core1.sum() - 0.5) < 1e-12          # core scaled to 1 - 0.50
    assert np.isnan(W[EV + 2]).all()                # skipped event: no change
    # first 12 slots close at EV+5, the next 2 at EV+6
    assert not np.isnan(W[EV + 5]).any()
    core5 = W[EV + 5] - np.r_[np.zeros(12), np.full(2, 0.01), np.zeros(N - 14)]
    assert abs(core5.sum() - 0.98) < 1e-12
    assert not np.isnan(W[EV + 6]).any()
    assert abs(W[EV + 6].sum() - 1.0) < 1e-12


def test_many_events_same_evening_split_budget():
    N = 40
    r = np.where((np.arange(T)[:, None] + np.arange(N)[None, :]) % 2 == 0, 0.01, -0.01)
    ea = np.zeros((T, N))
    r[EV, 0:20] = 0.30; ea[EV, 0:20] = 1
    W = target_weights(make_view(r, ea=ea), PARAMS)
    core = W[EV] - np.r_[np.full(20, 0.025), np.zeros(N - 20)]   # min(0.04, 0.5/20)
    assert abs(core.sum() - 0.5) < 1e-12
    assert (core >= -1e-15).all()


def test_second_event_does_not_add_slot():
    N = 4
    r = alt_returns(T, N)
    r[EV, 0] = 0.10; r[EV + 2, 0] = 0.12
    ea = np.zeros((T, N)); ea[EV, 0] = 1; ea[EV + 2, 0] = 1
    W = target_weights(make_view(r, ea=ea), PARAMS)
    assert np.isnan(W[EV + 2]).all()            # no new slot, no change
    assert not np.isnan(W[EV + 5]).any()        # old slot exits on schedule
    assert np.isnan(W[EV + 7]).all()


def test_slot_closed_when_stock_leaves_universe():
    N = 4
    r = alt_returns(T, N)
    r[EV, 0] = 0.10
    ea = np.zeros((T, N)); ea[EV, 0] = 1
    uni = np.ones((T, N), dtype=bool); uni[EV + 2:, 0] = False
    W = target_weights(make_view(r, ea=ea, universe=uni), PARAMS)
    assert W[EV + 2, 0] == 0.0
    assert abs(W[EV + 2].sum() - 1.0) < 1e-12
    assert np.isnan(W[EV + 5]).all()            # nothing left to close


def test_slot_closed_when_stock_stops_trading():
    N = 4
    r = alt_returns(T, N)
    r[EV, 0] = 0.10
    ea = np.zeros((T, N)); ea[EV, 0] = 1
    trad = np.ones((T, N), dtype=bool); trad[EV + 1, 0] = False
    W = target_weights(make_view(r, ea=ea, tradable=trad), PARAMS)
    assert W[EV + 1, 0] == 0.0


def test_missing_marketcap_no_core_but_tilt_allowed_and_late_listing():
    N = 4
    r = alt_returns(T, N)
    r[EV, 0] = 0.10
    ea = np.zeros((T, N)); ea[EV, 0] = 1
    mc = np.tile(100.0 * (np.arange(N) + 1.0), (T, 1))
    mc[:, 0] = np.nan                   # stock 0 has no cap: no core weight
    mc[:30, 3] = np.nan                 # stock 3 gets its first filing in row 30
    W = target_weights(make_view(r, ea=ea, mc=mc), PARAMS)
    assert W[0, 0] == 0.0 and W[0, 3] == 0.0
    assert np.allclose(W[0, 1:3], np.array([200.0, 300.0]) / 500.0)
    assert not np.isnan(W[30]).any() and W[30, 3] > 0     # core membership change
    assert abs(W[EV, 0] - 0.04) < 1e-12                    # tilt only
    assert abs(W[EV].sum() - 1.0) < 1e-12


def test_cap_reanchors_on_new_filing_value():
    N = 2
    r = np.zeros((T, N))
    r[1:11, 0] = 0.05                   # stock 0 grows by 1.05**10
    mc = np.tile(np.array([100.0, 100.0]), (T, 1))
    mc[20:, 1] = 300.0                  # stock 1 new filing value in row 20
    uni = np.ones((T, N), dtype=bool); uni[20, 1] = False; uni[21, 1] = True  # force a decision row
    W = target_weights(make_view(r, mc=mc, universe=uni), PARAMS)
    cap0 = 100.0 * 1.05 ** 10
    assert np.allclose(W[21], np.array([cap0, 300.0]) / (cap0 + 300.0))


def test_long_only_gross_one_and_point_in_time():
    N = 40
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.02, (T + 60, N))
    ea = (rng.random((T + 60, N)) < 0.08).astype(float)
    ea[rng.random((T + 60, N)) < 0.05] = np.nan
    uni = rng.random((T + 60, N)) < 0.9
    v = make_view(r, ea=ea, universe=uni)
    W = target_weights(v, PARAMS)
    dec = ~np.isnan(W).all(axis=1)
    assert dec.sum() > 5
    assert (W[dec] >= 0).all()
    assert (np.abs(W[dec].sum(axis=1) - 1.0) < 1e-9).all()
    assert not (W[dec][~uni[dec]] > 0).any()   # only members held
    cut = T
    vt = make_view(r[:cut], ea=ea[:cut], universe=uni[:cut])
    Wt = target_weights(vt, PARAMS)
    assert np.array_equal(np.isnan(Wt), np.isnan(W[:cut]))
    assert np.allclose(np.nan_to_num(Wt), np.nan_to_num(W[:cut]))


def test_missing_inputs_raise():
    N = 4
    r = alt_returns(T, N)
    with pytest.raises(KeyError):
        target_weights(make_view(r, extras={"sf1_arq_marketcap": np.ones((T, N))}), PARAMS)
    with pytest.raises(KeyError):
        target_weights(make_view(r, extras={"ea_reaction_day": np.zeros((T, N))}), PARAMS)
    with pytest.raises(ValueError):
        target_weights(make_view(r, no_universe=True), PARAMS)
