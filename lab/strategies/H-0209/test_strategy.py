import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
from strategy import rolling_sum, pair_zscores, first_crossings, select_entries, pair_active, pair_signals

NAMES = ("GDX", "GDXJ", "SIL", "XOP", "XLE", "FCG", "GLD", "SLV", "USO", "UNG", "SPY")
PRODUCERS = ("GDX", "GDXJ", "SIL", "XOP", "XLE", "FCG")
STRICT = {"formation_days": 5, "z_entry": 4.0}   # only the injected shocks cross (random 5-day sums do not)


def make_view(r, listed=None):
    T, N = r.shape
    if listed is None:
        listed = np.ones((T, N), dtype=bool)
    r = np.where(listed, r, 0.0)
    close = np.where(listed, 100.0 * np.cumprod(1.0 + r, axis=0), np.nan)
    dates = (np.arange(T) + 14000).astype("datetime64[D]")
    return DataView(
        dates=dates,
        instruments=NAMES,
        asset_class=tuple("us_etf" for _ in NAMES),
        ret_co=np.zeros((T, N)),
        ret_oc=r.astype(np.float64),
        tradable=listed.copy(),
        listed=listed.copy(),
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e8, np.nan),
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


def random_market(T=700, seed=1):
    rng = np.random.default_rng(seed)
    N = len(NAMES)
    r = np.zeros((T, N))
    col = {n: i for i, n in enumerate(NAMES)}
    mkt = rng.normal(0, 0.01, T)
    r[:, col["SPY"]] = mkt
    for c in ("GLD", "SLV", "USO", "UNG"):
        r[:, col[c]] = 0.2 * mkt + rng.normal(0, 0.012, T)
    for p, c in (("GDX", "GLD"), ("GDXJ", "GLD"), ("SIL", "SLV"), ("XOP", "USO"), ("XLE", "USO"), ("FCG", "UNG")):
        r[:, col[p]] = 1.5 * r[:, col[c]] + 0.8 * mkt + rng.normal(0, 0.01, T)
    return r, col


def test_rolling_sum():
    x = np.arange(1.0, 7.0)
    out = rolling_sum(x, 3)
    assert np.isnan(out[:2]).all()
    assert np.allclose(out[2:], [6, 9, 12, 15])


def test_betas_recovered_and_min_obs():
    rng = np.random.default_rng(0)
    T = 600
    xc = rng.normal(0, 0.01, T)
    xm = rng.normal(0, 0.01, T)
    y = 0.001 + 1.5 * xc + 0.7 * xm + rng.normal(0, 0.002, T)
    valid = np.ones(T, dtype=bool)
    z, bc, bm, s = pair_zscores(y, xc, xm, valid, 5)
    assert np.isnan(z[:251]).all()           # window of 252 rows not full
    assert np.isfinite(z[251:]).all()
    assert abs(bc[-1] - 1.5) < 0.05 and abs(bm[-1] - 0.7) < 0.05
    assert abs(s[-1] - 0.002) < 0.0004
    # gap uses betas of row t and no intercept
    t = 400
    e = np.sum(y[t - 4:t + 1] - bc[t] * xc[t - 4:t + 1] - bm[t] * xm[t - 4:t + 1])
    assert np.isclose(z[t], e / (s[t] * np.sqrt(5)))
    # fewer than 200 valid sessions in the window -> inactive
    valid2 = valid.copy()
    valid2[300:360] = False                  # windows covering these rows have < 200 valid rows
    z2, _, _, _ = pair_zscores(y, xc, xm, valid2, 5)
    assert np.isnan(z2[359:552]).all()       # window holds all 60 invalid rows: 192 valid < 200
    assert np.isfinite(z2[299])              # window before the gap
    assert np.isfinite(z2[599])              # window holds 12 invalid rows: 240 valid


def test_first_crossings():
    z = np.array([np.nan, 0.0, -1.6, -2.0, -1.0, 1.5, 1.7, np.nan, 2.0])
    dn, up = first_crossings(z, 1.5)
    assert list(np.flatnonzero(dn)) == [2]
    assert list(np.flatnonzero(up)) == [5]   # 1.5 >= 1.5 crosses from -1.0; row 8 follows NaN -> no entry


def test_select_entries_blocks_reentry():
    T = 20
    dn = np.zeros(T, dtype=bool)
    up = np.zeros(T, dtype=bool)
    dn[[3, 5, 8]] = True
    up[7] = True
    up[12] = True
    side = select_entries(dn, up, 5)
    # 3 enters; 5 and 7 fall inside its slot; 8 = 3 + 5 enters; 12 falls inside the second slot
    assert list(np.flatnonzero(side)) == [3, 8]
    assert side[3] == 1 and side[8] == 1 and side[12] == 0
    side2 = select_entries(dn, up, 3)
    assert list(np.flatnonzero(side2)) == [3, 7, 12]
    assert side2[7] == -1 and side2[12] == -1


def test_pair_active_history():
    T = 300
    b = np.ones(T, dtype=bool)
    late = np.zeros(T, dtype=bool)
    late[20:] = True
    act = pair_active(b, late, b)
    assert not act[:20 + 251].any()
    assert act[20 + 251:].all()


def test_single_pair_slot_weights():
    r, col = random_market(T=700, seed=3)
    listed = np.ones(r.shape, dtype=bool)
    for p in PRODUCERS[1:]:
        listed[:, col[p]] = False            # only GDX -> GLD is active
    t0 = 500
    r[t0, col["GDX"]] -= 0.15                # producer drops while GLD and SPY do not
    data = make_view(r, listed)
    W = target_weights(data, STRICT)
    assert W.shape == r.shape
    assert np.abs(W[:251]).sum() == 0
    for p in PRODUCERS[1:]:
        assert np.all(W[:, col[p]] == 0)
    assert W[t0, col["GDX"]] > 0             # long the producer that lagged
    assert W[t0, col["GLD"]] < 0 and W[t0, col["SPY"]] < 0
    assert np.isclose(np.abs(W[t0]).sum(), 0.4)
    for k in range(5):
        assert np.allclose(W[t0 + k], W[t0])  # fixed weights for 5 sessions
    # weights proportional to (1, -b_c, -b_m): hedge ratio near the true betas
    w = W[t0, col["GDX"]]
    assert 1.2 < -W[t0, col["GLD"]] / w < 1.8
    assert 0.5 < -W[t0, col["SPY"]] / w < 1.1
    # the only slot: rows t0..t0+4, closed by the target of row t0+5 (open of t0+6)
    assert np.all(W[t0 + 5] == 0)
    assert np.flatnonzero(W[:, col["GDX"]]).tolist() == list(range(t0, t0 + 5))


def test_short_side():
    r, col = random_market(T=700, seed=5)
    listed = np.ones(r.shape, dtype=bool)
    for p in PRODUCERS:
        if p != "XOP":
            listed[:, col[p]] = False
    t0 = 450
    r[t0, col["XOP"]] += 0.15
    W = target_weights(make_view(r, listed), STRICT)
    assert W[t0, col["XOP"]] < 0 and W[t0, col["USO"]] > 0 and W[t0, col["SPY"]] > 0


def test_late_listing():
    r, col = random_market(T=700, seed=7)
    listed = np.ones(r.shape, dtype=bool)
    listed[:200, col["SIL"]] = False
    W = target_weights(make_view(r, listed), PARAMS)
    assert np.all(W[:200 + 251, col["SIL"]] == 0)


def test_gross_cap_and_params():
    r, col = random_market(T=900, seed=11)
    data = make_view(r)
    W = target_weights(data, PARAMS)
    g = np.abs(W).sum(axis=1)
    assert g.max() <= 1.0 + 1e-9
    assert g.max() > 0.0
    W2 = target_weights(data, {**PARAMS, "z_entry": 1.0})
    W3 = target_weights(data, {**PARAMS, "formation_days": 10})
    assert not np.allclose(W, W2)
    assert not np.allclose(W, W3)
    assert np.all(np.isfinite(W))


def test_cap_scales_three_slots():
    r, col = random_market(T=700, seed=13)
    t0 = 500
    for p in ("GDX", "SIL", "XOP"):
        r[t0, col[p]] -= 0.25
    W = target_weights(make_view(r), STRICT)
    # three slots of gross 0.4 on distinct producers and commodities, SPY hedges add up: scaled to gross 1.0
    assert np.isclose(np.abs(W[t0]).sum(), 1.0)
    assert W[t0, col["GDX"]] > 0 and W[t0, col["SIL"]] > 0 and W[t0, col["XOP"]] > 0
    assert W[t0, col["XLE"]] == 0 and W[t0, col["UNG"]] == 0
    # one slot alone has gross 0.4 (the GDX slot computed without the other two shocks)
    r1, _ = random_market(T=700, seed=13)
    r1[t0, col["GDX"]] -= 0.25
    W1 = target_weights(make_view(r1), STRICT)
    assert np.isclose(W[t0, col["GDX"]], W1[t0, col["GDX"]] / 1.2, rtol=1e-6)


def test_point_in_time_truncation():
    r, col = random_market(T=800, seed=17)
    full = target_weights(make_view(r), PARAMS)
    cut = 600
    part = target_weights(make_view(r[:cut]), PARAMS)
    assert np.allclose(full[:cut], part)


def test_signal_pipeline_crossings():
    # the pipeline the diagnostic copies: first crossings only on producers, right side
    r, col = random_market(T=700, seed=3)
    t0 = 500
    r[t0, col["GDX"]] -= 0.15
    r[t0, col["XLE"]] += 0.15
    sig = pair_signals(make_view(r), STRICT)
    by_prod = {jp: (jc, dn, up) for jp, jc, _z, _bc, _bm, dn, up in sig}
    assert sorted(by_prod) == sorted(col[p] for p in PRODUCERS)
    jc, dn, up = by_prod[col["GDX"]]
    assert jc == col["GLD"] and dn[t0] and np.flatnonzero(dn | up).tolist() == [t0]
    jc, dn, up = by_prod[col["XLE"]]
    assert jc == col["USO"] and up[t0] and np.flatnonzero(dn | up).tolist() == [t0]
    for jp, (jc, dn, up) in by_prod.items():
        assert not (dn | up)[:251].any()
