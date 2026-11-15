import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import PARAMS, target_weights

NAMES = ("SPY", "IWM", "EFA", "EEM", "EWJ", "IEF", "TLT", "GLD", "SLV", "USO", "DBA", "DBB",
         "FXE", "FXY", "FXB", "FXA", "FXC", "FXF")
SECTOR_OF = [0] * 5 + [1] * 2 + [2] * 5 + [3] * 6


def make_view(R, bar, names=NAMES):
    """Close-to-close returns R carried in ret_oc (ret_co = 0); bar = listed and tradable."""
    T, N = R.shape
    R = np.where(bar, R, 0.0)
    price = np.cumprod(1.0 + R, axis=0)
    return DataView(
        dates=np.arange(T).astype("datetime64[D]"),
        instruments=tuple(names),
        asset_class=("us_etf",) * N,
        ret_co=np.zeros((T, N)),
        ret_oc=R,
        tradable=bar.copy(),
        listed=bar.copy(),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.where(bar, price, np.nan),
        dollar_volume=np.where(bar, 1e8, np.nan),
        universe=None,
        extras={},
        series={},
        cash_ret=None,
    )


# ---------------------------------------------------------------- slow reference (explicit loops)
def reference(R, bar, L, k, hold=5):
    T, N = R.shape
    R = np.where(bar, R, 0.0)
    vr = np.zeros((T, N), dtype=bool)
    vr[1:] = bar[1:] & bar[:-1]
    s = np.full((T, N), np.nan)
    sig = np.full((T, N), np.nan)
    for d in range(T):
        for j in range(N):
            if d >= L:
                s[d, j] = np.sign(np.sum(np.log1p(R[d - L + 1:d + 1, j])))
            lo = max(0, d - 62)
            x = R[lo:d + 1, j][vr[lo:d + 1, j]]
            if len(x) >= 2:
                sig[d, j] = np.std(x, ddof=1)
    elig = np.zeros((T, N), dtype=bool)
    P = np.full(T, np.nan)
    for t in range(1, T):
        for j in range(N):
            elig[t, j] = (vr[t, j] and vr[:t, j].sum() >= L + 1 and vr[max(0, t - 63):t, j].sum() >= 50
                          and np.isfinite(s[t - 1, j]) and s[t - 1, j] != 0
                          and np.isfinite(sig[t - 1, j]) and sig[t - 1, j] > 0)
        avgs = []
        for sec in range(4):
            us = [s[t - 1, j] * R[t, j] / sig[t - 1, j] for j in range(N) if SECTOR_OF[j] == sec and elig[t, j]]
            if us:
                avgs.append(np.mean(us))
        if elig[t].sum() >= 6 and len(avgs) >= 3:
            P[t] = np.mean(avgs)
    cohorts = {}
    for t in range(T):
        if not np.isfinite(P[t]):
            continue
        w = P[max(0, t - 252):t]
        w = w[np.isfinite(w)]
        if len(w) < 200:
            continue
        sd = np.std(w, ddof=1)
        if sd <= 0 or P[t] / sd > -k:
            continue
        by = [j for j in range(N) if elig[t, j] and s[t - 1, j] * R[t, j] >= 0]
        if by:
            v = np.zeros(N)
            for j in by:
                v[j] = -s[t - 1, j] / len(by)
            cohorts[t] = v
    W = np.zeros((T, N))
    for t in range(T):
        act = [cohorts[c] for c in cohorts if c <= t <= c + hold - 1]
        if act:
            W[t] = np.mean(act, axis=0)
    return W, cohorts


def random_market(T=420, seed=7):
    rng = np.random.default_rng(seed)
    N = len(NAMES)
    R = rng.standard_t(4, size=(T, N)) * 0.01 + rng.normal(0, 0.0005, size=(1, N))
    R += rng.standard_t(4, size=(T, 1)) * 0.006        # common factor so the trend book has fat loss days
    bar = np.ones((T, N), dtype=bool)
    bar[:200, 17] = False                                # FXF lists late
    bar[100:105, 3] = False                              # EEM has a 5-day gap
    bar[250, 9] = False                                  # USO misses one day
    return R, bar


def test_matches_slow_reference_on_random_market():
    R, bar = random_market()
    for L, k in ((20, 1.0), (30, 1.25)):
        W = target_weights(make_view(R, bar), {"trend_lookback_days": L, "shock_k": k})
        Wr, cohorts = reference(R, bar, L, k)
        assert len(cohorts) >= 3
        np.testing.assert_allclose(W, Wr, atol=1e-10)
        assert np.all(np.abs(W).sum(axis=1) <= 1.0 + 1e-12)


# ---------------------------------------------------------------- hand-made shock scenario
def trend_market(T=400):
    """Each ETF trends steadily (sign alternates); small noise. No natural shock day."""
    rng = np.random.default_rng(1)
    N = len(NAMES)
    sgn = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(N)])
    R = sgn[None, :] * 0.003 + rng.normal(0, 0.001, size=(T, N))
    return R, np.ones((T, N), dtype=bool), sgn


BYST = [NAMES.index(x) for x in ("SPY", "IEF", "GLD", "FXE")]   # one bystander per sector


def inject_shock(R, sgn, t):
    for j in range(R.shape[1]):
        R[t, j] = sgn[j] * 0.0005 if j in BYST else -sgn[j] * 0.02


def test_single_shock_cohort_rows_and_weights():
    R, bar, sgn = trend_market()
    inject_shock(R, sgn, 350)
    W = target_weights(make_view(R, bar), {"trend_lookback_days": 20, "shock_k": 3.0})
    expected = np.zeros(len(NAMES))
    expected[BYST] = -sgn[BYST] / 4.0
    for t in range(350, 355):                 # decided after close 350, held opens 351..355
        np.testing.assert_allclose(W[t], expected, atol=1e-12)
    assert np.all(W[:350] == 0.0)             # nothing before the shock (pre-warm-up rows too)
    assert np.all(W[355:] == 0.0)             # closed at the open of t+6
    assert np.isclose(np.abs(W[350]).sum(), 1.0)
    # drivers are never traded
    drivers = [j for j in range(len(NAMES)) if j not in BYST]
    assert np.all(W[:, drivers] == 0.0)


def test_overlapping_cohorts_average():
    R, bar, sgn = trend_market()
    inject_shock(R, sgn, 350)
    inject_shock(R, sgn, 352)
    W = target_weights(make_view(R, bar), {"trend_lookback_days": 20, "shock_k": 3.0})
    v = np.zeros(len(NAMES))
    v[BYST] = -sgn[BYST] / 4.0
    np.testing.assert_allclose(W[350], v, atol=1e-12)
    np.testing.assert_allclose(W[351], v, atol=1e-12)
    for t in range(352, 355):                 # two active cohorts, same vector: average = v
        np.testing.assert_allclose(W[t], v, atol=1e-12)
    np.testing.assert_allclose(W[355], v, atol=1e-12)
    np.testing.assert_allclose(W[356], v, atol=1e-12)
    assert np.all(W[357:] == 0.0)


def test_late_listing_not_a_bystander():
    R, bar, sgn = trend_market()
    j = NAMES.index("FXY")
    bar[:340, j] = False                      # listed 10 rows before the shock: fewer than L+1 returns
    inject_shock(R, sgn, 350)
    R[350, j] = sgn[j] * 0.0005               # it did not lose, but it is not eligible
    W = target_weights(make_view(R, bar), {"trend_lookback_days": 20, "shock_k": 3.0})
    assert np.all(W[:, j] == 0.0)
    assert np.isclose(np.abs(W[350]).sum(), 1.0)


def test_shock_threshold_parameter():
    R, bar, sgn = trend_market()
    inject_shock(R, sgn, 350)
    W_lo = target_weights(make_view(R, bar), {"trend_lookback_days": 20, "shock_k": 3.0})
    W_hi = target_weights(make_view(R, bar), {"trend_lookback_days": 20, "shock_k": 1e6})
    assert np.abs(W_lo).sum() > 0
    assert np.all(W_hi == 0.0)


def test_parameters_change_output_on_random_market():
    R, bar = random_market()
    base = target_weights(make_view(R, bar), {"trend_lookback_days": 20, "shock_k": 1.0})
    other_k = target_weights(make_view(R, bar), {"trend_lookback_days": 20, "shock_k": 1.5})
    other_L = target_weights(make_view(R, bar), {"trend_lookback_days": 40, "shock_k": 1.0})
    assert not np.allclose(base, other_k)
    assert not np.allclose(base, other_L)


def test_point_in_time_truncation():
    R, bar = random_market()
    p = {"trend_lookback_days": 20, "shock_k": 1.0}
    full = target_weights(make_view(R, bar), p)
    for cut in (260, 300, 377):
        part = target_weights(make_view(R[:cut], bar[:cut]), p)
        np.testing.assert_array_equal(part, full[:cut])


def test_primary_params_run_and_gross():
    R, bar = random_market(T=700, seed=3)
    W = target_weights(make_view(R, bar), PARAMS)
    assert W.shape == R.shape
    assert np.all(np.isfinite(W))
    assert np.all(np.abs(W).sum(axis=1) <= 1.0 + 1e-12)
    assert PARAMS == {"shock_k": 1.25, "trend_lookback_days": 126}


def test_missing_instrument_raises():
    R, bar = random_market()
    names = list(NAMES)
    names[0] = "XXX"
    with pytest.raises(Exception):
        target_weights(make_view(R, bar, names=tuple(names)), PARAMS)

