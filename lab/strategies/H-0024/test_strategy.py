import numpy as np
import pytest
from dataclasses import dataclass, field

from strategy import target_weights, PARAMS
from strategy import compute_events


@dataclass
class Fake:
    dates: np.ndarray
    instruments: tuple
    universe: np.ndarray
    listed: np.ndarray
    tradable: np.ndarray
    delisting: np.ndarray
    dollar_volume: np.ndarray
    extras: dict = field(default_factory=dict)


T0 = 200
EV = 153          # event row (no background filing of stocks 0..6, 8, 9 on this row)


def make(n=40, t=T0, seed=0):
    """Background: every stock files every 10 rows (rows with (t + j) % 10 == 0) a TTM net income change
    uniform in [-1, 1] with market cap 100 (S in [-0.01, 0.01]); dollar volume constant 1 (no spikes)."""
    rng = np.random.default_rng(seed)
    ni = np.full((t, n), 100.0)
    for j in range(n):
        for r in range(1, t):
            if (r + j) % 10 == 0:
                ni[r:, j] += rng.uniform(-1.0, 1.0)
    mc = np.full((t, n), 100.0)
    d = Fake(
        dates=np.arange(t).astype("datetime64[D]"),
        instruments=tuple("E%d" % (j + 1) for j in range(n)),
        universe=np.ones((t, n), dtype=bool),
        listed=np.ones((t, n), dtype=bool),
        tradable=np.ones((t, n), dtype=bool),
        delisting=np.zeros((t, n), dtype=bool),
        dollar_volume=np.ones((t, n)),
        extras={"sf1_art_netinc": ni.astype(np.float32), "sf1_art_marketcap": mc.astype(np.float32)},
    )
    return d


def filing(d, row, j, delta, spike_row=None, spike=3.0):
    d.extras["sf1_art_netinc"][row:, j] += np.float32(delta)
    if spike_row is not None:
        d.dollar_volume[spike_row, j] = spike


def test_long_short_events_and_hold_window():
    d = make()
    filing(d, EV, 0, +5.0, spike_row=EV)          # S = +0.05, spike on t -> long
    filing(d, EV, 1, -5.0, spike_row=EV - 1)      # S = -0.05, spike on t-1 -> short
    filing(d, EV, 2, +5.0)                        # extreme but no volume spike -> no event
    ev = compute_events(d, PARAMS)
    assert ev[EV, 0] == 1 and ev[EV, 1] == -1 and ev[EV, 2] == 0
    assert ev.sum() == 0 and np.abs(ev).sum() == 2
    w = target_weights(d, PARAMS)
    assert w[EV, 0] == pytest.approx(0.10)
    assert w[EV, 1] == pytest.approx(-0.10)
    assert np.allclose(w[EV, 2:], 0.0)            # net 0 -> no hedge
    assert np.isnan(w[EV + 1:EV + 5]).all()       # active set unchanged on rows t+1 .. t+4
    assert np.allclose(w[EV + 5], 0.0)            # exit at the open of t+6
    assert np.allclose(w[0], 0.0)                 # flat before any event
    assert np.isnan(w[1:EV]).all()


def test_hedge_is_equal_weight_over_other_members():
    d = make()
    filing(d, EV, 0, +5.0, spike_row=EV)
    w = target_weights(d, PARAMS)
    assert w[EV, 0] == pytest.approx(0.10)
    assert np.allclose(w[EV, 1:], -0.10 / 39)
    assert np.nansum(w[EV]) == pytest.approx(0.0)
    # a hedge name that leaves the universe is exited without recomputing the others
    d.universe[EV + 2, 10] = False
    w = target_weights(d, PARAMS)
    assert w[EV + 2, 10] == 0.0
    assert np.isnan(np.delete(w[EV + 2], 10)).all()


def test_side_cap_scales_to_half():
    d = make()
    for j in range(6):
        filing(d, EV, j, +5.0 + j, spike_row=EV)
    w = target_weights(d, PARAMS)
    assert np.allclose(w[EV, :6], 0.5 / 6)
    assert np.allclose(w[EV, 6:], -0.5 / 34)
    assert np.abs(w[EV]).sum() == pytest.approx(1.0)


def test_new_event_replaces_old_and_restarts_clock():
    d = make()
    filing(d, EV, 0, +5.0, spike_row=EV)
    filing(d, EV + 2, 0, -10.0, spike_row=EV + 2)   # S = -0.1 two rows later -> short
    w = target_weights(d, PARAMS)
    assert w[EV, 0] == pytest.approx(0.10)
    assert w[EV + 2, 0] == pytest.approx(-0.10)
    assert np.allclose(w[EV + 2, 1:], 0.10 / 39)
    assert np.isnan(w[EV + 3:EV + 7]).all()
    assert np.allclose(w[EV + 7], 0.0)


def test_exit_when_leaving_universe_or_not_trading():
    d = make()
    filing(d, EV, 0, +5.0, spike_row=EV)
    d.universe[EV + 2:, 0] = False
    w = target_weights(d, PARAMS)
    assert np.allclose(w[EV + 2], 0.0)
    d = make()
    filing(d, EV, 0, +5.0, spike_row=EV)
    d.tradable[EV + 3, 0] = False
    w = target_weights(d, PARAMS)
    assert np.allclose(w[EV + 3], 0.0)
    assert np.isnan(w[EV + 4:EV + 6]).all()       # not re-entered when it trades again


def test_value_after_missing_stretch_is_not_an_event():
    d = make()
    d.extras["sf1_art_netinc"][100:EV, 3] = np.nan
    filing(d, EV, 3, +5.0, spike_row=EV)
    assert compute_events(d, PARAMS)[EV, 3] == 0
    d = make()
    d.extras["sf1_art_marketcap"][EV - 1, 4] = np.nan
    filing(d, EV, 4, +5.0, spike_row=EV)
    assert compute_events(d, PARAMS)[EV, 4] == 0
    d = make()
    d.extras["sf1_art_marketcap"][EV:, 5] = -1.0  # marketcap <= 0 -> skipped
    filing(d, EV, 5, +5.0, spike_row=EV)
    assert compute_events(d, PARAMS)[EV, 5] == 0


def test_marketcap_change_alone_is_a_filing_with_zero_surprise():
    d = make()
    d.extras["sf1_art_marketcap"][EV:, 0] = 120.0
    d.dollar_volume[EV, 0] = 3.0
    assert compute_events(d, PARAMS)[EV, 0] == 0   # S = 0 is in the middle of the pool


def test_pool_needs_200_filings():
    d = make(n=10)                                # about 150 filings in the 252-row pool
    filing(d, EV, 0, +5.0, spike_row=EV)
    assert np.abs(compute_events(d, PARAMS)).sum() == 0
    d = make(n=40)
    filing(d, EV, 0, +5.0, spike_row=EV)
    assert compute_events(d, PARAMS)[EV, 0] == 1


def test_volume_window_needs_40_values_and_excludes_t_minus_1():
    d = make()
    filing(d, EV, 0, +5.0, spike_row=EV)
    d.dollar_volume[EV - 64:EV - 1, 0] = np.nan
    d.dollar_volume[EV - 30:EV - 1, 0] = 1.0      # 29 valid values < 40
    assert compute_events(d, PARAMS)[EV, 0] == 0
    d.dollar_volume[EV - 45:EV - 1, 0] = 1.0      # 44 valid values
    assert compute_events(d, PARAMS)[EV, 0] == 1


def test_vol_mult_param():
    d = make()
    filing(d, EV, 0, +5.0, spike_row=EV, spike=2.5)
    assert compute_events(d, dict(PARAMS, vol_mult=2.0))[EV, 0] == 1
    assert compute_events(d, dict(PARAMS, vol_mult=1.5))[EV, 0] == 1
    assert compute_events(d, dict(PARAMS, vol_mult=3.0))[EV, 0] == 0


def test_quantile_param():
    d = make()
    filing(d, EV, 0, +0.7, spike_row=EV)          # S = 0.007, about the 85th percentile of the pool
    assert compute_events(d, dict(PARAMS, quantile=0.1))[EV, 0] == 0
    assert compute_events(d, dict(PARAMS, quantile=0.2))[EV, 0] == 1
    assert compute_events(d, dict(PARAMS, quantile=0.3))[EV, 0] == 1
    filing(d, EV + 3, 1, -0.7, spike_row=EV + 3)  # mirror on the short side
    assert compute_events(d, dict(PARAMS, quantile=0.1))[EV + 3, 1] == 0
    assert compute_events(d, dict(PARAMS, quantile=0.3))[EV + 3, 1] == -1


def test_non_member_filing_not_an_event():
    d = make()
    filing(d, EV, 0, +5.0, spike_row=EV)
    d.universe[EV, 0] = False
    assert compute_events(d, PARAMS)[EV, 0] == 0


def busy(seed=1):
    d = make(seed=seed)
    rng = np.random.default_rng(seed + 10)
    for r in range(80, T0 - 5, 3):
        for j in rng.choice(40, size=4, replace=False):
            if (r + j) % 10 != 0:
                filing(d, r, j, rng.normal(0, 5), spike_row=r)
    return d


def test_gross_and_neutrality():
    d = busy()
    w = target_weights(d, PARAMS)
    rows = ~np.isnan(w).all(axis=1)
    g = np.nansum(np.abs(w[rows]), axis=1)
    assert (g <= 1.0 + 1e-9).all()
    full = rows & ~np.isnan(w).any(axis=1)
    assert np.allclose(np.sum(w[full], axis=1), 0.0)
    assert (np.nansum(np.abs(w[rows]), axis=1) > 0).sum() > 10


def test_point_in_time_truncation():
    d = busy()
    w = target_weights(d, PARAMS)
    for cut in (120, 150, 171):
        dt = Fake(
            dates=d.dates[:cut], instruments=d.instruments, universe=d.universe[:cut], listed=d.listed[:cut],
            tradable=d.tradable[:cut], delisting=d.delisting[:cut], dollar_volume=d.dollar_volume[:cut],
            extras={k: v[:cut] for k, v in d.extras.items()},
        )
        wt = target_weights(dt, PARAMS)
        assert np.array_equal(np.isnan(wt), np.isnan(w[:cut]))
        assert np.allclose(np.nan_to_num(wt), np.nan_to_num(w[:cut]))


def test_missing_input_raises():
    d = make()
    del d.extras["sf1_art_marketcap"]
    with pytest.raises(KeyError):
        target_weights(d, PARAMS)
    d = make()
    d.universe = None
    with pytest.raises(ValueError):
        target_weights(d, PARAMS)
