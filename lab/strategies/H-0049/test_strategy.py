import numpy as np
import pytest

from lab.framework.data import DataView, synthetic
from strategy import target_weights, PARAMS

NAMES = ("BTCUSDT.P", "ETHUSDT.P", "C1USDT.P", "C2USDT.P", "C3USDT.P", "C4USDT.P", "C5USDT.P", "C6USDT.P")
T = 60
BASE_F = 0.0003


def base():
    n = len(NAMES)
    sign = np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    r = 0.01 * np.repeat(sign[:, None], n, axis=1)
    r[0] = 0.0
    f = np.full((T, n), BASE_F)
    univ = np.ones((T, n), dtype=bool)
    return r, f, univ


def view(r, f, univ):
    n = len(NAMES)
    close = 100.0 * np.cumprod(1.0 + r, axis=0)
    return DataView(
        dates=np.arange(T).astype("datetime64[D]"),
        instruments=NAMES,
        asset_class=tuple("crypto_perp" for _ in NAMES),
        ret_co=np.zeros((T, n)),
        ret_oc=r.copy(),
        tradable=np.ones((T, n), dtype=bool),
        listed=np.ones((T, n), dtype=bool),
        delisting=np.zeros((T, n), dtype=bool),
        close=close,
        dollar_volume=np.full((T, n), 1e8),
        universe=univ,
        extras={"funding": f, "basis": np.zeros((T, n))},
        series={},
        cash_ret=np.zeros(T),
    )


C1, BTC, ETH = NAMES.index("C1USDT.P"), 0, 1


def run(r, f, univ, **kw):
    p = dict(PARAMS)
    p.update(kw)
    return target_weights(view(r, f, univ), p)


def test_params():
    assert PARAMS == {"hold_days": 3, "k_sigma": 2.0}


def test_no_event_all_zero():
    r, f, u = base()
    w = run(r, f, u)
    assert w.shape == (T, len(NAMES))
    assert np.all(w == 0.0)


def test_flush_long_hedged_held_hold_days():
    r, f, u = base()
    r[40, C1] = -0.10
    f[40, C1] = -0.001          # negative and a new 7-day low
    w = run(r, f, u)
    for t in (40, 41, 42):
        assert w[t, C1] == pytest.approx(0.10)
        assert w[t, BTC] == pytest.approx(-0.05)
        assert w[t, ETH] == pytest.approx(-0.05)
    assert np.all(w[:40] == 0.0) and np.all(w[43:] == 0.0)
    w5 = run(r, f, u, hold_days=5)
    assert np.all(w5[40:45, C1] == pytest.approx(0.10)) and w5[45, C1] == 0.0


def test_unflushed_short_pinned_funding():
    r, f, u = base()
    r[40, C1] = -0.10           # funding stays at the default rate: >= median, never FLUSH
    w = run(r, f, u)
    assert w[40, C1] == pytest.approx(-0.10)
    assert w[40, BTC] == pytest.approx(0.05) and w[40, ETH] == pytest.approx(0.05)
    assert np.sum(w[40]) == pytest.approx(0.0)


def test_neither_class_no_event():
    r, f, u = base()
    r[40, C1] = -0.10
    f[40, C1] = 0.0001          # fell below median but not negative
    assert np.all(run(r, f, u) == 0.0)
    f[40, C1] = -0.001
    f[36, C1] = -0.002          # negative but not a new 7-day low; below median -> no event
    assert np.all(run(r, f, u) == 0.0)


def test_k_sigma_threshold():
    r, f, u = base()
    r[40, C1] = -0.022          # sigma = 0.01*sqrt(30/29) = 0.01017: -2 sigma = -0.0203, -2.5 sigma = -0.0254
    assert run(r, f, u, k_sigma=2.0)[40, C1] == pytest.approx(-0.10)
    assert run(r, f, u, k_sigma=2.5)[40, C1] == 0.0
    assert run(r, f, u, k_sigma=1.5)[40, C1] == pytest.approx(-0.10)


def test_missing_funding_no_event():
    r, f, u = base()
    r[40, C1] = -0.10
    f[37, C1] = np.nan
    assert np.all(run(r, f, u) == 0.0)
    f[37, C1] = BASE_F
    f[40, C1] = np.nan
    assert np.all(run(r, f, u) == 0.0)


def test_sigma_needs_30_valid_days():
    r, f, u = base()
    r[30, C1] = -0.10           # window rows 0..29, row 0 has no return -> 29 valid
    assert np.all(run(r, f, u)[30] == 0.0)
    r, f, u = base()
    r[31, C1] = -0.10           # window rows 1..30 complete
    assert run(r, f, u)[31, C1] == pytest.approx(-0.10)


def test_second_event_on_open_slot_ignored():
    r, f, u = base()
    r[40, C1] = -0.10
    f[40, C1] = -0.001
    r[41, C1] = -0.10
    f[41, C1] = 0.0003          # would be UNFLUSHED, ignored: slot still open
    w = run(r, f, u)
    assert np.all(w[40:43, C1] == pytest.approx(0.10))
    assert w[43, C1] == 0.0


def test_universe_exit_closes_slot():
    r, f, u = base()
    r[40, C1] = -0.10
    f[40, C1] = -0.001
    u[41, C1] = False
    w = run(r, f, u)
    assert w[40, C1] == pytest.approx(0.10)
    assert np.all(w[41:] == 0.0)


def test_tercile_rule():
    r, f, u = base()
    for name, x in (("C1USDT.P", -0.10), ("C2USDT.P", -0.09), ("C3USDT.P", -0.08), ("C4USDT.P", -0.05)):
        r[40, NAMES.index(name)] = x
    # 33.3rd pct of 8 values = -0.08 + 0.331 * 0.03 = -0.0701: C1..C3 qualify, C4 does not
    w = run(r, f, u)
    for name in ("C1USDT.P", "C2USDT.P", "C3USDT.P"):
        assert w[40, NAMES.index(name)] == pytest.approx(-0.10)
    assert w[40, NAMES.index("C4USDT.P")] == 0.0
    assert w[40, BTC] == pytest.approx(0.15)


def test_cap_at_half_and_gross():
    r, f, u = base()
    for j in range(2, len(NAMES)):
        r[40, j] = -0.10
        f[40, j] = -0.001
    w = run(r, f, u)
    assert np.all(w[40, 2:] == pytest.approx(0.5 / 6))
    assert w[40, BTC] == pytest.approx(-0.25) and w[40, ETH] == pytest.approx(-0.25)
    assert np.abs(w[40]).sum() == pytest.approx(1.0)


def test_hedges_never_slots():
    r, f, u = base()
    r[40, BTC] = -0.10
    f[40, BTC] = -0.001
    assert np.all(run(r, f, u) == 0.0)


def test_point_in_time_truncation():
    r, f, u = base()
    r[40, C1] = -0.10
    f[40, C1] = -0.001
    r[45, 3] = -0.10
    full = run(r, f, u)
    for cut in (35, 41, 44, 50):
        rr, ff, uu = r.copy(), f.copy(), u.copy()
        rr[cut + 1:] = -0.3
        ff[cut + 1:] = -0.01
        w = run(rr, ff, uu)
        assert np.array_equal(w[:cut + 1], full[:cut + 1])
