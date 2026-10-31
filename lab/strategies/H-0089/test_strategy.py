import dataclasses

import numpy as np
import pytest

from strategy import target_positions, PARAMS, signals

T = 3100
N = 7
T0 = 3000          # event candle used in the tests (baseline full from row 2883)


@dataclasses.dataclass
class View:
    times: np.ndarray
    instruments: tuple
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    qv: np.ndarray
    nt: np.ndarray
    tbq: np.ndarray
    funding: np.ndarray
    real: np.ndarray
    tradable: np.ndarray
    listed: np.ndarray
    universe: np.ndarray
    bars_per_day: int = 96

    def col(self, name):
        return self.instruments.index(name)

    def head(self, n):
        kw = {}
        for name, v in dataclasses.asdict(self).items():
            kw[name] = v[:n] if isinstance(v, np.ndarray) else v
        return View(**kw)


def make_view(trend=None, seed=0):
    """Flat flow: qv = 100 per candle, tbq = 50 + common noise (same for every symbol, so the
    baselines are identical across symbols). Price trends by trend[j] * 1e-4 per candle."""
    rng = np.random.default_rng(seed)
    if trend is None:
        trend = np.ones(N)
    e = rng.normal(0.0, 2.0, size=T)
    tbq = np.repeat((50.0 + e)[:, None], N, axis=1)
    qv = np.full((T, N), 100.0)
    t = np.arange(T)[:, None]
    c = 100.0 * np.exp(np.asarray(trend)[None, :] * 1e-4 * t)
    ones = np.ones((T, N), dtype=bool)
    return View(
        times=(np.arange(T) * 15).astype("datetime64[m]"),
        instruments=tuple("S%dUSDT" % j for j in range(N)),
        o=c.copy(), h=c.copy(), l=c.copy(), c=c, qv=qv, nt=np.ones((T, N)), tbq=tbq,
        funding=np.zeros((T, N)), real=ones.copy(), tradable=ones.copy(), listed=ones.copy(),
        universe=ones.copy(),
    )


def inject(v, j, t0, buy_share, absorbed=True):
    """One heavy candle at t0 (qv 4000) with a taker-buy share buy_share; the price at t0 is moved
    against the flow (absorbed) or with the flow (not absorbed)."""
    v.qv[t0, j] = 4000.0
    v.tbq[t0, j] = 4000.0 * buy_share
    flow = 1.0 if buy_share > 0.5 else -1.0
    move = -flow if absorbed else flow
    v.c[t0, j] *= 1.0 + 0.01 * move


def ref_z(v, t, j):
    """Direct (loop-free slicing) reference of the card's z at candle t for symbol j."""
    def I_at(s):
        QV = v.qv[s - 3:s + 1, j].sum()
        TB = v.tbq[s - 3:s + 1, j].sum()
        return 2 * TB / QV - 1, QV
    base = [I_at(s) for s in range(t - 2883, t - 3)]
    assert len(base) == 2880
    Is = np.array([b[0] for b in base])
    QVs = np.array([b[1] for b in base])
    I, QV = I_at(t)
    return (I - Is.mean()) / Is.std(), QV, QVs.mean()


def test_z_matches_reference():
    v = make_view()
    inject(v, 2, T0, 0.9)
    _, _, absz = signals(v, PARAMS)
    for t in (2890, T0, T0 + 2, T0 + 50):
        z, _, _ = ref_z(v, t, 2)
        assert absz[t, 2] == pytest.approx(abs(z), rel=1e-7)


def test_absorbed_buying_gives_short_held_H():
    v = make_view(trend=np.ones(N))
    inject(v, 3, T0, 0.9)          # takers bought, price fell -> absorbed buying -> short
    z, QV, QVbar = ref_z(v, T0, 3)
    assert z > PARAMS["imbalance_z"] and QV >= 2 * QVbar
    w = target_positions(v, PARAMS)
    H = PARAMS["hold_candles"]
    assert np.all(w[:T0] == 0)
    assert np.all(w[T0:T0 + H, 3] == pytest.approx(-0.2))
    assert np.all(w[T0 + H:, 3] == 0)
    assert np.count_nonzero(w) == H


def test_absorbed_selling_gives_long():
    v = make_view(trend=-np.ones(N))
    inject(v, 1, T0, 0.1)          # takers sold, price rose -> long
    w = target_positions(v, PARAMS)
    H = PARAMS["hold_candles"]
    assert np.all(w[T0:T0 + H, 1] == pytest.approx(0.2))
    assert np.count_nonzero(w) == H


def test_flow_driven_hour_is_not_an_event():
    v = make_view(trend=np.ones(N))
    inject(v, 3, T0, 0.9, absorbed=False)   # price moved with the takers
    w = target_positions(v, PARAMS)
    assert np.all(w == 0)


def test_small_volume_is_not_an_event():
    v = make_view(trend=np.ones(N))
    # strong imbalance but QV of the hour below 2x its mean: 4 candles at 100 with tbq = 90
    v.tbq[T0 - 3:T0 + 1, 3] = 90.0
    v.c[T0, 3] *= 0.99
    ev, _, absz = signals(v, PARAMS)
    assert absz[T0, 3] > PARAMS["imbalance_z"]
    assert not ev[T0, 3]


def test_slots_ranked_by_abs_z_rest_dropped():
    v = make_view(trend=np.ones(N))
    shares = [0.70, 0.95, 0.75, 0.90, 0.80, 0.85, 0.72]
    for j, sh in enumerate(shares):
        inject(v, j, T0, sh)
    ev, _, absz = signals(v, PARAMS)
    assert ev[T0].all()
    w = target_positions(v, PARAMS)
    held = set(np.nonzero(w[T0])[0].tolist())
    assert held == {1, 3, 5, 4, 2}     # five largest buy shares = five largest |z|
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12
    # dropped, not queued: symbols 0 and 6 never trade
    assert np.all(w[:, 0] == 0) and np.all(w[:, 6] == 0)


def test_no_pyramiding_and_slot_reuse_after_close():
    v = make_view(trend=np.ones(N))
    H = PARAMS["hold_candles"]
    inject(v, 2, T0, 0.9)
    inject(v, 2, T0 + 10, 0.9)        # ignored: symbol still open
    inject(v, 2, T0 + H + 5, 0.9)     # after the close: new trade
    w = target_positions(v, PARAMS)
    assert np.all(w[T0:T0 + H, 2] == pytest.approx(-0.2))
    assert np.all(w[T0 + H:T0 + H + 5, 2] == 0)
    assert np.all(w[T0 + H + 5:T0 + 2 * H + 5, 2] == pytest.approx(-0.2))
    assert np.all(w[T0 + 2 * H + 5:, 2] == 0)


def test_early_exit_on_universe_and_real():
    v = make_view(trend=np.ones(N))
    inject(v, 2, T0, 0.9)
    inject(v, 4, T0, 0.9)
    v.universe[T0 + 5, 2] = False
    v.real[T0 + 7, 4] = False
    w = target_positions(v, PARAMS)
    assert np.all(w[T0:T0 + 5, 2] == pytest.approx(-0.2)) and np.all(w[T0 + 5:, 2] == 0)
    assert np.all(w[T0:T0 + 7, 4] == pytest.approx(-0.2)) and np.all(w[T0 + 7:, 4] == 0)


def test_not_in_universe_or_short_history_no_event():
    v = make_view(trend=np.ones(N))
    inject(v, 2, T0, 0.9)
    v.universe[T0, 2] = False
    inject(v, 3, 2000, 0.9)           # baseline would start before row 0
    inject(v, 4, T0, 0.9)
    v.real[1000:1600, 4] = False      # only 2280 real candles in the baseline
    w = target_positions(v, PARAMS)
    assert np.all(w == 0)


def test_params_change_output():
    v = make_view(trend=np.ones(N))
    inject(v, 2, T0, 0.9)
    w24 = target_positions(v, PARAMS)
    w12 = target_positions(v, dict(PARAMS, hold_candles=12))
    assert np.count_nonzero(w24) == 24 and np.count_nonzero(w12) == 12
    # imbalance threshold: an event with |z| between 2.0 and 3.0
    v2 = make_view(trend=np.ones(N))
    v2.qv[T0, 2] = 4000.0
    _, _, absz0 = signals(v2, PARAMS)
    sd_scale = None
    # choose a buy share giving |z| about 2.5 using the reference
    for sh in np.linspace(0.5, 0.6, 201):
        v2.tbq[T0, 2] = 4000.0 * sh
        z, _, _ = ref_z(v2, T0, 2)
        if z >= 2.6:
            sd_scale = sh
            break
    assert sd_scale is not None
    v2.c[T0, 2] *= 0.99
    lo = target_positions(v2, dict(PARAMS, imbalance_z=2.0))
    hi = target_positions(v2, dict(PARAMS, imbalance_z=3.0))
    assert np.count_nonzero(lo[:, 2]) > 0 and np.count_nonzero(hi[:, 2]) == 0


def test_point_in_time_truncation():
    v = make_view(trend=np.ones(N))
    for j, sh in enumerate([0.9, 0.1, 0.8, 0.85, 0.95, 0.7, 0.75]):
        inject(v, j, T0 + 3 * j, sh)
    full = target_positions(v, PARAMS)
    for cut in (T0 - 1, T0 + 1, T0 + 7, T0 + 30):
        part = target_positions(v.head(cut), PARAMS)
        assert np.array_equal(part, full[:cut])


def test_deterministic_and_gross():
    v = make_view(trend=np.ones(N))
    for j in range(N):
        inject(v, j, T0, 0.6 + 0.05 * j)
    a = target_positions(v, PARAMS)
    b = target_positions(v, PARAMS)
    assert np.array_equal(a, b)
    assert np.abs(a).sum(axis=1).max() <= 1.0 + 1e-12


def test_synthetic_view_runs():
    from lab.framework.intraday import synthetic_view
    v = synthetic_view()
    w = target_positions(v, PARAMS)
    assert w.shape == np.asarray(v.c).shape
    assert np.all(np.isfinite(w))
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12


def test_missing_universe_fails_loudly():
    v = make_view()
    v.universe = None
    with pytest.raises(Exception):
        target_positions(v, PARAMS)
