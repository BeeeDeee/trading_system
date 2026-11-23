import numpy as np
import pytest
from lab.framework.api import total_return_index
from lab.framework.data import DataView

from strategy import PARAMS, target_weights, _events, _core, _hold_state, _book

T0 = 300


def _dates(T):
    d = np.arange(12000, 12000 + 2 * T)
    d = d[((d + 3) % 7) < 5][:T]          # weekdays only (1970-01-01 was a Thursday)
    return d.astype("datetime64[D]")


def _view(r, ea, mc, member, listed=None):
    T, N = r.shape
    if listed is None:
        listed = np.ones((T, N), dtype=bool)
    r = np.where(listed, r, 0.0)
    gross = np.cumprod(1.0 + r, axis=0)
    close = np.where(listed, 100.0 * gross, np.nan)
    tradable = listed.copy()
    return DataView(
        dates=_dates(T),
        instruments=tuple("E%d" % (i + 1) for i in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=np.zeros((T, N)),
        ret_oc=r.astype(np.float64),
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e8, np.nan),
        universe=member,
        extras={"ea_reaction_day": ea.astype(np.float32), "sf1_arq_marketcap": mc.astype(np.float32)},
        series={},
        cash_ret=np.zeros(T),
    )


def _scenario(T=330, N=10, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.005, (T, N))
    ea = np.zeros((T, N))
    r[:, 0] += 0.002                   # uptrend: near its 52-week high
    r[:, 1] -= 0.002                   # downtrend: near its 52-week low
    r[T0, 0] = 0.08
    ea[T0, 0] = 1                      # positive event
    r[T0, 1] = -0.08
    ea[T0, 1] = 1                      # negative event
    r[T0, 2] = 0.001
    ea[T0, 2] = 1                      # release without a large reaction
    r[100, 3] = 0.08
    ea[100, 3] = 1                     # before the 240-session anchor history
    r[:281, 4] = 0.001                 # 12 % below its high: event only with band 0.05
    r[281:291, 4] = -0.012
    r[291:, 4] = 0.0
    r[T0, 4] = 0.085
    ea[T0, 4] = 1
    listed = np.ones((T, N), dtype=bool)
    listed[:150, 8] = False            # late listing
    r[T0, 8] = 0.08
    ea[T0, 8] = 1
    member = listed.copy()
    member[200:, 9] = False            # leaves the index
    mc = np.tile(np.arange(1, N + 1, dtype=np.float64) * 1e10, (T, 1))
    mc[:, 7] = np.nan                  # no marketcap: gets the median
    mc[:150, 8] = np.nan
    return r, ea, mc, member, listed


@pytest.fixture(scope="module")
def scen():
    r, ea, mc, member, listed = _scenario()
    return _view(r, ea, mc, member, listed)


def test_events(scen):
    pos, neg, _ = _events(scen, PARAMS["band"], PARAMS["z_min"])
    assert pos[T0, 0] and not neg[T0, 0]
    assert neg[T0, 1] and not pos[T0, 1]
    assert not pos[T0, 2] and not neg[T0, 2]
    assert not pos[100, 3]                  # anchor history < 240 sessions
    assert not pos[T0, 4]                   # 3.8 % below the high, band 3 %
    assert not pos[T0, 8]                   # listed 150 sessions only
    assert pos.sum() == 1 and neg.sum() == 1
    pos5, _, _ = _events(scen, 0.05, PARAMS["z_min"])
    assert pos5[T0, 4] and pos5[T0, 0]
    posz, negz, _ = _events(scen, PARAMS["band"], 100.0)
    assert posz.sum() == 0 and negz.sum() == 0


def test_weights_tilt(scen):
    w = target_weights(scen, PARAMS)
    c, ms = _core(scen)
    for t in range(T0, T0 + 5):
        if np.isnan(w[t]).all():
            continue
        assert t == T0 or ms[t]
        c0, c1 = c[t, 0], c[t, 1]
        assert w[t, 0] == pytest.approx(c0 + 0.05, abs=1e-9)
        assert w[t, 1] == 0.0
        k = (1.0 - c0 - 0.05) / (1.0 - c0 - c1)
        others = [2, 3, 4, 5, 6, 7, 8]
        assert np.allclose(w[t, others], c[t, others] * k)
        assert w[t].sum() == pytest.approx(1.0, abs=1e-9)
    assert not np.isnan(w[T0]).any()
    # removed at the open of t+6: row t+5 is a decision row back to the core
    t = T0 + 5
    assert not np.isnan(w[t]).any()
    assert np.allclose(w[t], c[t])
    # the row before the event has no tilt
    assert np.isnan(w[T0 - 1]).all() or np.allclose(w[T0 - 1], c[T0 - 1])


def test_decision_rows_and_universe(scen):
    w = target_weights(scen, PARAMS)
    dec = ~np.isnan(w).all(axis=1)
    _, ms = _core(scen)
    assert dec[0]
    assert dec[150] and dec[200]           # membership changes
    assert dec[T0] and dec[T0 + 5]
    assert not dec[T0 + 1] or ms[T0 + 1]
    assert dec.sum() < ms.sum() + 6
    # partially-NaN rows never occur
    assert np.all(np.isnan(w[~dec]))
    d = w[dec]
    assert np.all(d >= 0)
    assert np.all(d.sum(axis=1) <= 1.0 + 1e-12)
    assert np.allclose(d.sum(axis=1), 1.0)
    member = np.asarray(scen.universe)
    assert np.all(d[~member[dec]] == 0)
    assert np.all(w[200:, 9][dec[200:]] == 0)
    assert np.all(w[:150, 8][dec[:150]] == 0)
    assert w[150, 8] == 0 or ms[150]        # joined mid-month: no core until the next month start
    first_ms = np.flatnonzero(ms & (np.arange(len(ms)) > 150))[0]
    assert w[first_ms, 8] > 0


def test_core_median_fill_and_month_drift():
    T, N = 60, 4
    r = np.zeros((T, N))
    r[:, 0] = 0.01
    ea = np.zeros((T, N))
    mc = np.tile(np.array([1.0, 2.0, np.nan, 5.0]) * 1e9, (T, 1))
    mc[10:, 3] = 1e10                       # new filing mid-month
    member = np.ones((T, N), dtype=bool)
    v = _view(r, ea, mc, member)
    c, ms = _core(v)
    # row 0: caps 1, 2, median(1, 2, 5) = 2, 5
    assert np.allclose(c[0], np.array([1, 2, 2, 5]) / 10.0)
    tri = np.asarray(total_return_index(v), dtype=np.float64)
    m2 = np.flatnonzero(ms)[1]
    assert m2 > 10
    # inside the first month the core drifts with total return only
    t = m2 - 1
    raw = np.array([1.0 * tri[t, 0] / tri[0, 0], 2, 2, 5])
    assert np.allclose(c[t], raw / raw.sum())
    # next month start: the new filing (10e9, anchored at its filing row) enters
    cap0 = 1.0 * tri[m2, 0] / tri[0, 0]
    med = np.median([cap0, 2.0, 10.0])
    raw2 = np.array([cap0, 2.0, med, 10.0])
    assert np.allclose(c[m2], raw2 / raw2.sum())


def test_overweight_cap():
    T, N = 330, 30
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.005, (T, N))
    ea = np.zeros((T, N))
    r[:, :11] += 0.002
    r[T0, :11] = 0.08
    ea[T0, :11] = 1
    mc = np.tile(np.linspace(1, 3, N) * 1e10, (T, 1))
    member = np.ones((T, N), dtype=bool)
    v = _view(r, ea, mc, member)
    pos, neg, _ = _events(v, PARAMS["band"], PARAMS["z_min"])
    assert pos[T0, :11].all() and pos.sum() == 11
    w = target_weights(v, PARAMS)
    c, _ = _core(v)
    over = w[T0, :11] - c[T0, :11]
    assert np.allclose(over, 0.5 / 11)
    assert w[T0].sum() == pytest.approx(1.0)
    # the same book with fewer events: no scaling
    w2 = _book(c[T0:T0 + 1], np.where(np.arange(N) < 3, 1, 0).astype(np.int8)[None, :],
               member[T0:T0 + 1], 0.05)
    assert np.allclose(w2[0, :3] - c[T0, :3], 0.05)
    assert w2[0].sum() == pytest.approx(1.0)


def test_hold_state_latest_event_wins():
    T, N = 12, 2
    pos = np.zeros((T, N), dtype=bool)
    neg = np.zeros((T, N), dtype=bool)
    pos[2, 0] = True
    pos[4, 0] = True                        # second event extends, does not stack
    neg[3, 1] = True
    member = np.ones((T, N), dtype=bool)
    member[6, 1] = False
    s = _hold_state(pos, neg, member)
    assert list(s[:, 0]) == [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert list(s[:, 1]) == [0, 0, 0, -1, -1, -1, 0, -1, 0, 0, 0, 0]


def test_params_change_output(scen):
    base = target_weights(scen, PARAMS)
    w_band = target_weights(scen, {"band": 0.05, "z_min": 2.0})
    w_z = target_weights(scen, {"band": 0.03, "z_min": 100.0})
    c, _ = _core(scen)
    assert w_band[T0, 4] == pytest.approx(c[T0, 4] + 0.05)
    assert base[T0, 4] < c[T0, 4] + 0.01
    assert np.isnan(w_z[T0]).all() or np.allclose(w_z[T0], c[T0])
    w0 = target_weights(scen, {"band": 0.03, "z_min": 2.0, "tilt_w": 0.0})
    assert w0[T0, 0] == pytest.approx(c[T0, 0])          # core only, still zeroed negative
    assert w0[T0, 1] == 0.0
    assert w0[T0, 2] == pytest.approx(c[T0, 2] * (1 - c[T0, 0]) / (1 - c[T0, 0] - c[T0, 1]))


@pytest.mark.parametrize("cut", [120, 260, 299, 300, 302, 305])
def test_point_in_time_truncation(cut):
    r, ea, mc, member, listed = _scenario()
    full = target_weights(_view(r, ea, mc, member, listed), PARAMS)
    k = cut + 1
    tr = target_weights(_view(r[:k], ea[:k], mc[:k], member[:k], listed[:k]), PARAMS)
    assert np.array_equal(np.isnan(full[:k]), np.isnan(tr))
    assert np.allclose(np.nan_to_num(full[:k]), np.nan_to_num(tr))
    # noise in the future does not change the past
    r2 = r.copy()
    r2[k:] = np.random.default_rng(5).normal(0, 0.05, r2[k:].shape)
    ea2 = ea.copy()
    ea2[k:] = 1
    pert = target_weights(_view(r2, ea2, mc, member, listed), PARAMS)
    assert np.array_equal(np.isnan(full[:k]), np.isnan(pert[:k]))
    assert np.allclose(np.nan_to_num(full[:k]), np.nan_to_num(pert[:k]))


def test_short_view():
    r, ea, mc, member, listed = _scenario()
    for k in (1, 2, 18):
        w = target_weights(_view(r[:k], ea[:k], mc[:k], member[:k], listed[:k]), PARAMS)
        assert w.shape == (k, 10)
        assert not np.isnan(w[0]).any()


def test_never_member_column_is_zero():
    r, ea, mc, member, listed = _scenario()
    member = member.copy()
    member[:, 5] = False
    w = target_weights(_view(r, ea, mc, member, listed), PARAMS)
    dec = ~np.isnan(w).all(axis=1)
    assert np.all(w[dec, 5] == 0)
    assert np.allclose(w[dec].sum(axis=1), 1.0)
    pos, neg, _ = _events(_view(r, ea, mc, member, listed), PARAMS["band"], PARAMS["z_min"])
    assert pos[T0, 0] and neg[T0, 1]


def test_missing_inputs_raise(scen):
    r, ea, mc, member, listed = _scenario()
    v = _view(r, ea, mc, None, listed)
    with pytest.raises(Exception):
        target_weights(v, PARAMS)
