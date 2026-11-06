import numpy as np
import pytest
from lab.framework.data import DataView

from strategy import target_weights, PARAMS
from strategy import strategy as S
from strategy import diagnostic as D


def _view(T=900, N=30, seed=0, release_p=0.08, late=None, leave=None):
    rng = np.random.default_rng(seed)
    days = np.arange(12419, 12419 + 2 * T).astype("datetime64[D]")
    dates = days[np.is_busday(days)][:T]
    ret_co = rng.normal(0, 0.005, size=(T, N))
    ret_oc = rng.normal(0, 0.012, size=(T, N))
    dv = np.exp(rng.normal(17, 0.5, size=(T, N))) * np.linspace(1, 3, N)[None, :]
    close = 50 * np.cumprod((1 + ret_co) * (1 + ret_oc), axis=0)
    listed = np.ones((T, N), dtype=bool)
    uni = np.ones((T, N), dtype=bool)
    if late is not None:
        j, t0 = late
        listed[:t0, j] = False
        uni[:t0 + 70, j] = False
    if leave is not None:
        j, t0, t1 = leave
        uni[t0:t1, j] = False
    ret_co = np.where(listed, ret_co, 0.0)
    ret_oc = np.where(listed, ret_oc, 0.0)
    close = np.where(listed, close, np.nan)
    dv = np.where(listed, dv, np.nan)
    ea = (rng.random((T, N)) < release_p).astype(np.float32)
    ea[:40] = np.nan
    tradable = listed.copy()
    delisting = np.zeros((T, N), dtype=bool)
    names = tuple("E%d" % (100 + j) for j in range(N))
    return DataView(dates=dates, instruments=names, asset_class=tuple("us_equity" for _ in names),
                    ret_co=ret_co, ret_oc=ret_oc, tradable=tradable, listed=listed, delisting=delisting,
                    close=close, dollar_volume=dv, universe=uni,
                    extras={"ea_reaction_day": ea, "liq_rank": np.ones((T, N), dtype=np.float32)},
                    series={}, cash_ret=np.zeros(T))


# ---------------------------------------------------------------- features

def test_features_match_hand_computation():
    d = _view(T=200, N=5)
    feats, med = S.features(d)
    gap_z, session_z, vol_shock, runup_z, log_sigma, log_med = feats
    r = (1 + d.ret_co) * (1 + d.ret_oc) - 1
    t, j = 150, 2
    sig = np.std(r[t - 63:t, j], ddof=1)
    m = np.median(d.dollar_volume[t - 63:t, j])
    run = np.prod(1 + r[t - 21:t, j]) - 1
    assert gap_z[t, j] == pytest.approx(d.ret_co[t, j] / sig, rel=1e-6)
    assert session_z[t, j] == pytest.approx(d.ret_oc[t, j] / sig, rel=1e-6)
    assert vol_shock[t, j] == pytest.approx(np.log(d.dollar_volume[t, j] / m), rel=1e-6)
    assert runup_z[t, j] == pytest.approx(run / (sig * np.sqrt(21)), rel=1e-6)
    assert log_sigma[t, j] == pytest.approx(np.log(sig), rel=1e-6)
    assert log_med[t, j] == pytest.approx(np.log(m), rel=1e-6)
    assert med[t, j] == pytest.approx(m, rel=1e-9)


def test_features_nan_before_full_lookback_and_late_listing():
    d = _view(T=300, N=4, late=(1, 100))
    feats, med = S.features(d)
    # sigma needs 63 returns t-63..t-1 (row 0 has no return): first at row 64, late listing at 100+64
    for i in (0, 1, 3, 4):
        f = feats[i]
        assert np.all(np.isnan(f[:64, :]))
        assert np.isfinite(f[64, 0])
        assert np.all(np.isnan(f[:164, 1]))
        assert np.isfinite(f[164, 1])
    # the dollar-volume median needs 63 prices t-63..t-1: first at row 63, late listing at 163
    for i in (2, 5):
        f = feats[i]
        assert np.all(np.isnan(f[:63, :]))
        assert np.isfinite(f[63, 0])
        assert np.all(np.isnan(f[:163, 1]))
        assert np.isfinite(f[163, 1])


def test_features_do_not_use_today_in_windows():
    d = _view(T=200, N=3)
    f0, med0 = S.features(d)
    d.dollar_volume[150, 0] *= 100.0
    f1, med1 = S.features(d)
    assert med1[150, 0] == med0[150, 0]               # median over t-63..t-1 excludes t
    assert f1[4][150, 0] == f0[4][150, 0]             # sigma excludes t
    assert f1[2][150, 0] != f0[2][150, 0]             # vol_shock uses dv[t]


# ---------------------------------------------------------------- entry rule

def test_entries_threshold_and_minimum():
    T, N = 10, 20
    sc = np.full((T, N), np.nan)
    sc[0, :] = np.arange(20)          # 20 rows
    sc[5, :10] = np.arange(10) + 100  # 10 rows
    e = S.entries(sc, 0.2)
    assert not e.any()                # never 100 scored rows in the window
    sc = np.full((T, N), np.nan)
    for t in range(5):
        sc[t, :] = np.arange(20) + t * 20      # 0..99 over rows 0..4
    sc[5, :3] = [79.0, 80.0, 79.3]
    e = S.entries(sc, 0.2)
    pool = np.concatenate([np.arange(100.0), [79.0, 80.0, 79.3]])
    thr = np.quantile(pool, 0.8)
    assert np.array_equal(e[5, :3], np.array([79.0, 80.0, 79.3]) >= thr)
    assert e[4].sum() == 20 - int(np.sum(np.arange(80, 100) < np.quantile(np.arange(100.0), 0.8)))
    assert not e[:4].any()            # fewer than 100 rows before row 4


def test_entries_trailing_window_252():
    T, N = 400, 1
    sc = np.full((T, N), np.nan)
    sc[:100, 0] = 1.0                 # 100 low scores in rows 0..99
    sc[350, 0] = 0.5                  # row 350: window is rows 99..350 -> only 2 scored rows
    e = S.entries(sc, 0.2)
    assert not e[350, 0]
    sc[99, 0] = 1.0
    e = S.entries(sc, 0.2)
    assert e[99, 0]                   # 100 rows, ties at the threshold count ("at or above")


# ---------------------------------------------------------------- book

def test_sleeve_slots_hold_close_and_cap():
    T, N = 20, 15
    uni = np.ones((T, N), dtype=bool)
    uni[8, 1] = False                 # stock 1 leaves on row 8 and comes back
    L = S.member_run_length(uni)
    core = np.full((T, N), 1.0 / N) * uni
    ent = np.zeros((T, N), dtype=bool)
    ent[5, 0] = True
    ent[6, 1] = True
    w = S.book(ent, core, L)
    # stock 0: slot on rows 5..9 (opened at open of 6, closed at open of 11)
    s = 0.05 * ((np.arange(T) >= 5) & (np.arange(T) <= 9)) + 0.05 * ((np.arange(T) >= 6) & (np.arange(T) <= 7))
    expect0 = (1 - s) / N + 0.05 * ((np.arange(T) >= 5) & (np.arange(T) <= 9))
    assert np.allclose(w[:, 0], expect0)
    # stock 1: slot rows 6, 7, then closed at row 8 (non-member) and not reopened on 9, 10
    assert w[7, 1] == pytest.approx((1 - 0.10) / N + 0.05)
    assert w[8, 1] == 0.0
    assert w[9, 1] == pytest.approx((1 - 0.05) / N)
    assert np.all(np.abs(w).sum(axis=1) <= 1 + 1e-12)
    # cap: 14 entries on one row -> 0.70 raw, scaled to 0.50
    ent = np.zeros((T, N), dtype=bool)
    ent[3, :14] = True
    core = np.full((T, N), 1.0 / N)
    w = S.book(ent, core, S.member_run_length(np.ones((T, N), dtype=bool)))
    assert w[3, 0] == pytest.approx(0.5 / N + 0.5 / 14)
    assert w[3, 14] == pytest.approx(0.5 / N)
    assert np.abs(w[3]).sum() == pytest.approx(1.0)
    assert np.all(w >= 0)


def test_core_monthly_fixed_fractions_and_leaver():
    T, N = 12, 3
    uni = np.ones((T, N), dtype=bool)
    uni[6:, 2] = False
    uni[:4, 1] = False               # joins on row 4 (mid period)
    med = np.tile(np.array([1.0, 2.0, 3.0]), (T, 1))
    med[5:] = np.array([3.0, 3.0, 3.0])
    starts = np.zeros(T, dtype=bool)
    starts[[0, 8]] = True
    L = S.member_run_length(uni)
    core = S.core_weights(uni, med, starts, L)
    assert np.allclose(core[0], [0.25, 0.0, 0.75])
    assert np.allclose(core[5], [0.25, 0.0, 0.75])     # fixed between refreshes, joiner waits
    assert np.allclose(core[6], [1.0, 0.0, 0.0])       # leaver dropped, survivors rescaled to gross 1
    assert np.allclose(core[8], [0.5, 0.5, 0.0])       # refresh with the new medians and members


def test_month_start_mask_point_in_time():
    d = _view(T=80, N=2)
    m = S.month_start_mask(d.dates)
    months = d.dates.astype("datetime64[M]")
    expect = np.zeros(80, dtype=bool)
    expect[1:] = months[1:] != months[:-1]
    assert np.array_equal(m[1:], expect[1:])


# ---------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def dense():
    d = _view(T=900, N=60, seed=3, late=(4, 200), leave=(7, 600, 640))
    return d, target_weights(d, PARAMS)


def test_weights_schema(dense):
    d, w = dense
    assert w.shape == (900, 60)
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0)
    assert np.all(w.sum(axis=1) <= 1 + 1e-9)
    assert np.all(w[~d.universe] == 0)


def test_sleeve_only_after_first_fit_and_on_release_rows(dense):
    d, w = dense
    uni = d.universe
    rel = S.release_rows(d, uni)
    score, _ = S.model_scores(d, uni, rel, "gbm", dict(S.GBM_FIXED, n_estimators=PARAMS["n_estimators"]))
    assert np.all(np.isnan(score[~rel]))
    fin = np.isfinite(score)
    first = np.argmax(fin.any(axis=1))
    assert fin.any(), "no scores: rel=%d" % rel.sum()
    assert first > 63
    ent = S.entries(score, PARAMS["top_q"])
    assert not ent[:first].any()
    assert ent.any(), "first=%d scored=%d" % (first, fin.sum())
    # without entries the book is exactly the core: the sleeve appears only after entries
    core_only = S.book(np.zeros_like(ent), S.core_weights(uni, _, S.month_start_mask(d.dates),
                                                          S.member_run_length(uni)), S.member_run_length(uni))
    assert np.allclose(w[:first], core_only[:first])
    assert not np.allclose(w[first:], core_only[first:])


def test_top_q_changes_output(dense):
    d, w = dense
    w2 = target_weights(d, dict(PARAMS, top_q=0.1))
    assert not np.allclose(w, w2)


def test_truncation_point_in_time():
    d = _view(T=900, N=60, seed=5)
    w = target_weights(d, PARAMS)
    cut = 780
    dt = DataView(dates=d.dates[:cut], instruments=d.instruments, asset_class=d.asset_class,
                  ret_co=d.ret_co[:cut], ret_oc=d.ret_oc[:cut], tradable=d.tradable[:cut],
                  listed=d.listed[:cut], delisting=d.delisting[:cut], close=d.close[:cut],
                  dollar_volume=d.dollar_volume[:cut], universe=d.universe[:cut],
                  extras={k: v[:cut] for k, v in d.extras.items()}, series={}, cash_ret=d.cash_ret[:cut])
    wt = target_weights(dt, PARAMS)
    assert np.allclose(w[:cut], wt)


def test_missing_flag_raises():
    d = _view(T=120, N=5)
    d.extras.pop("ea_reaction_day")
    with pytest.raises(KeyError):
        target_weights(d, PARAMS)


def test_diagnostic_sides(dense):
    d, _ = dense
    mask, side = D.events(d, PARAMS)
    assert mask.shape == side.shape == (900, 60)
    assert mask.any()
    rel = S.release_rows(d, d.universe)
    assert not (mask & ~rel).any()
    assert set(np.unique(side[mask])) <= {1.0, -1.0}
    assert np.all(side[~mask] == 0)
