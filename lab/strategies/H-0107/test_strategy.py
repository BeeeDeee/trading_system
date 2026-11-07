import numpy as np
import pytest
from lab.framework.data import DataView

from strategy import target_weights, PARAMS
from strategy import (build_book, top_half_events, wrapper_flow_days, assign_wrappers, flow_pipeline,
                      _beta_resid, _prior_median, _prior_std, _rsum)

WR = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SMH", "IBB", "IYR", "IGV")


# ------------------------------------------------------------------------------------------ rolling helpers
def test_rolling_sum_windows_and_short_views():
    x = np.arange(1.0, 11.0)[:, None]
    s = _rsum(x, 3)
    assert s[0, 0] == 1 and s[1, 0] == 3 and s[2, 0] == 6 and s[9, 0] == 8 + 9 + 10
    p = _rsum(x, 3, lag=1)                 # rows t-3..t-1
    assert p[0, 0] == 0 and p[1, 0] == 1 and p[4, 0] == 2 + 3 + 4
    short = _rsum(x[:2], 126)              # view shorter than the window
    assert short.shape == (2, 1) and short[1, 0] == 3


def test_prior_median_and_std_match_naive():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(150, 3))
    x[10:40, 1] = np.nan
    med = _prior_median(x, 63, 50)
    sd = _prior_std(x, 63, 50)
    for t in (0, 49, 50, 51, 80, 149):
        for j in range(3):
            win = x[max(0, t - 63):t, j]
            win = win[np.isfinite(win)]
            if len(win) >= 50:
                assert med[t, j] == pytest.approx(np.median(win), rel=1e-5, abs=1e-6)
                assert sd[t, j] == pytest.approx(np.std(win, ddof=1), rel=1e-6)
            else:
                assert np.isnan(med[t, j]) and np.isnan(sd[t, j])


def test_beta_residual_needs_100_returns():
    rng = np.random.default_rng(2)
    rm = rng.normal(0, 0.01, 200)
    rm[0] = np.nan
    r = (1.5 * rm + 0.0)[:, None]
    e = _beta_resid(r, rm)
    assert np.all(np.isnan(e[:100]))       # rows 1..99 hold only 99 valid returns
    assert np.all(np.abs(e[100:]) < 1e-12)


def test_assignment_to_most_correlated_wrapper():
    rng = np.random.default_rng(3)
    T = 300
    ek = rng.normal(0, 0.01, (T, 3))
    ej = np.column_stack([0.8 * ek[:, 1] + 0.003 * rng.normal(size=T),
                          rng.normal(0, 0.01, T)])
    ej[:30, 0] = np.nan
    assigned, kk, corr, b = assign_wrappers(ej, ek)
    assert not assigned[128, 0]            # only 99 valid pairs in t-125..t
    assert assigned[129, 0]
    assert assigned[299, 0] and kk[299, 0] == 1 and corr[299, 0] > 0.9
    assert b[299, 0] == pytest.approx(0.8, abs=0.1)
    assert not assigned[299, 1] and np.isnan(corr[299, 1]) and np.isnan(b[299, 1])


# ------------------------------------------------------------------------------------------ flow days
def _wrapper_inputs(e_today, dv_today):
    T = 71
    ek = (0.01 * (-1.0) ** np.arange(T))[:, None].copy()
    dv = np.full((T, 1), 100.0)
    ek[70, 0] = e_today
    dv[70, 0] = dv_today
    return ek, dv


def test_down_and_up_flow_day_rules():
    d, u, inten = wrapper_flow_days(*_wrapper_inputs(-0.022, 200.0), 2.0)
    assert d[70, 0] and not u[70, 0] and inten[70, 0] == pytest.approx(np.log(2.0))
    d, u, _ = wrapper_flow_days(*_wrapper_inputs(-0.022, 200.0), 2.5)   # flow_sigma matters
    assert not d[70, 0]
    d, u, _ = wrapper_flow_days(*_wrapper_inputs(-0.022, 199.0), 2.0)   # volume rule
    assert not d[70, 0]
    d, u, _ = wrapper_flow_days(*_wrapper_inputs(0.022, 300.0), 2.0)    # mirror
    assert u[70, 0] and not d[70, 0]
    ek, dv = _wrapper_inputs(-0.05, np.nan)                              # no volume on t
    d, u, _ = wrapper_flow_days(ek, dv, 2.0)
    assert not d[70, 0]
    ek, dv = _wrapper_inputs(np.nan, 300.0)                              # no price on t
    d, u, _ = wrapper_flow_days(ek, dv, 2.0)
    assert not d[70, 0] and not u[70, 0]
    d, u, _ = wrapper_flow_days(ek[:40], dv[:40], 2.0)                   # baseline not full yet
    assert not d.any() and not u.any()


def test_top_half_events():
    cand = np.zeros((3, 6), dtype=bool)
    score = np.full((3, 6), np.nan)
    cand[0, :5] = True
    score[0, :5] = [0.1, 0.5, 0.3, 0.9, np.nan]      # 4 scored -> top 2
    cand[1, :3] = True
    score[1, :3] = [0.2, 0.1, 0.3]                   # 3 -> top 2 (rounded up)
    cand[2, 0] = True
    score[2, 0] = 1.0                                # fewer than 2 -> none
    score[2, 1] = 2.0                                # scored but not a candidate
    ev = top_half_events(cand, score)
    assert list(np.nonzero(ev[0])[0]) == [1, 3]
    assert list(np.nonzero(ev[1])[0]) == [0, 2]
    assert not ev[2].any()


# ------------------------------------------------------------------------------------------ book
def test_book_core_slots_renewal_and_expiry():
    T, N = 12, 4
    uni = np.ones((T, N), dtype=bool)
    dv = np.tile(np.array([1.0, 3.0, 2.0, 2.0]), (T, 1))
    core_days = np.zeros(T, dtype=bool)
    core_days[0] = True
    ev = np.zeros((T, N), dtype=bool)
    ev[2, [0, 1]] = True
    ev[4, 0] = True                                   # renewal of col 0
    W = build_book(ev, uni, dv, core_days)
    c = np.array([1, 3, 2, 2]) / 8.0
    assert np.allclose(W[0], c) and np.allclose(W[1], c)
    assert np.allclose(W[2], 0.94 * c + np.array([0.03, 0.03, 0, 0]))
    assert np.allclose(W[6], 0.94 * c + np.array([0.03, 0.03, 0, 0]))      # held t..t+4
    assert np.allclose(W[7], 0.97 * c + np.array([0.03, 0, 0, 0]))         # col 1 sold at the open of t+6
    assert np.allclose(W[8], 0.97 * c + np.array([0.03, 0, 0, 0]))         # renewed, not doubled
    assert np.allclose(W[9], c)
    assert np.allclose(W.sum(axis=1), 1.0)


def test_book_cap_leavers_and_monthly_core():
    T, N = 8, 30
    uni = np.ones((T, N), dtype=bool)
    dv = np.tile(np.arange(1.0, N + 1), (T, 1))
    core_days = np.zeros(T, dtype=bool)
    core_days[0] = True
    core_days[6] = True
    dv[3:, 0] = 1000.0                                # visible only from the monthly recompute (row 6)
    base0 = dv[0].copy()
    ev = np.zeros((T, N), dtype=bool)
    ev[0, :20] = True                                 # 20 x 3 % > 50 % -> scaled to 2.5 % each
    ev[1, 20:25] = True                               # no room left -> no slots
    uni[3, 29] = False                                # leaver: dropped, rest renormalized
    uni[3, 0] = False                                 # slot of a leaver is closed
    W = build_book(ev, uni, dv, core_days)
    assert np.allclose(W[0, :20] - 0.5 * dv[0, :20] / dv[0].sum(), 0.025)
    assert np.allclose(W[1, 20:25], 0.5 * dv[1, 20:25] / dv[1].sum())
    assert W[3, 29] == 0 and W[3, 0] == 0
    base = base0.copy()
    base[[0, 29]] = 0
    slots3 = 19 * 0.025
    assert np.allclose(W[3, 1:20], (1 - slots3) * base[1:20] / base.sum() + 0.025)
    assert np.allclose(W[4, 0], (1 - slots3) * base0[0] / base0.sum())   # back in the core, slot stays closed
    assert np.allclose(W[5], base0 / base0.sum())     # all slots expired, core still from row 0
    med6 = np.array([1000.0] + list(range(2, 31)))     # rows 0..6 of col 0: four of seven are 1000
    assert np.allclose(W[6], med6 / med6.sum())        # monthly core recompute
    assert np.all(W >= 0) and np.all(W.sum(axis=1) <= 1 + 1e-12)


# ------------------------------------------------------------------------------------------ full pipeline
def make_view(T=440, cut=None):
    rng = np.random.default_rng(7)
    names = ["SPY"] + list(WR) + ["E%d" % (1000 + j) for j in range(26)]
    N = len(names)
    rm = rng.normal(0, 0.01, T)
    f = rng.normal(0, 0.005, (T, 13))
    f[400, 2] = -0.05                                   # XLF residual crash on row 400
    f[410, 4] = 0.05                                    # XLK residual surge on row 410
    r = np.zeros((T, N))
    r[:, 0] = rm
    r[:, 1:14] = rm[:, None] + f
    for j in range(26):
        g = (2, 4, 1)[j % 3] if j < 24 else None
        idio = rng.normal(0, 0.004, T)
        r[:, 14 + j] = rm + (0.8 * f[:, g] if g is not None else 0.0) + idio
    close = 50.0 * np.cumprod(1 + r, axis=0)
    dv = np.zeros((T, N))
    dv[:, :14] = 1e9 * (1 + 0.05 * rng.random((T, 14)))
    dv[400, 3] = 3e9
    dv[410, 5] = 3e9
    dv[:, 14:] = (np.arange(1, 27) * 1e6)[None, :] * (1 + 0.05 * rng.random((T, 26)))
    uni = np.zeros((T, N), dtype=bool)
    uni[:, 14:] = True
    uni[:300, 14 + 23] = False                          # late joiner
    uni[405:, 14 + 22] = False                          # leaver
    ea = np.zeros((T, N))
    ne = np.zeros((T, N))
    ea[400, 14 + 0] = 1.0                               # own earnings 8-K on the flow day
    ne[400, 14 + 3] = np.nan                            # NaN flag = no filing
    ne[410, 14 + 1] = 1.0                               # own non-earnings 8-K on the up-flow day
    ac = tuple(["us_etf"] * 14 + ["us_equity"] * 26)
    ret_co = r * 0.3
    ret_oc = (1 + r) / (1 + ret_co) - 1
    k = T if cut is None else cut
    ext = {"ea_reaction_day": ea[:k], "nonearn_8k_reaction_day": ne[:k], "liq_rank": np.ones((k, N)),
           "sp500_member": uni[:k].astype(float)}
    return DataView(dates=np.arange(k).astype("datetime64[D]"), instruments=tuple(names), asset_class=ac,
                    ret_co=ret_co[:k], ret_oc=ret_oc[:k], tradable=np.ones((k, N), dtype=bool),
                    listed=np.ones((k, N), dtype=bool), delisting=np.zeros((k, N), dtype=bool),
                    close=close[:k], dollar_volume=dv[:k], universe=uni[:k], extras=ext, series={},
                    cash_ret=np.zeros(k))


def test_flow_rows_follow_assignment_and_8k_exclusion():
    v = make_view()
    ev, flow, cand, score = flow_pipeline(v, PARAMS)
    xlf = [v.col("E%d" % (1000 + j)) for j in range(24) if j % 3 == 0]
    xlk = [v.col("E%d" % (1000 + j)) for j in range(24) if j % 3 == 1 and j != 22]   # E1022 leaves at 405
    other = [v.col("E%d" % (1000 + j)) for j in range(24) if j % 3 == 2]
    assert not flow[400, xlf[0]]                        # earnings 8-K excluded
    assert all(flow[400, c] for c in xlf[1:])            # incl. the NaN-flag stock
    assert all(cand[400, c] for c in xlf[1:])            # down-push candidates
    assert not any(flow[400, c] for c in xlk + other)
    assert not flow[410, xlk[0]]                        # non-earnings 8-K excluded
    assert all(flow[410, c] for c in xlk[1:]) and not any(cand[410, c] for c in xlk)
    assert not flow[400, v.col("E1024")] and not flow[400, v.col("E1025")]   # unassigned
    assert not flow[:, :14].any()
    ev2, flow2, cand2, _ = flow_pipeline(v, {"flow_sigma": 50.0, "n_estimators": 100})
    assert not flow2.any() and not cand2.any()           # flow_sigma reaches the rule


def test_weights_core_only_before_first_fit_and_point_in_time():
    v = make_view()
    W = target_weights(v, PARAMS)
    assert W.shape == (440, 40)
    assert np.all(W >= 0) and np.all(W.sum(axis=1) <= 1 + 1e-9)
    assert np.allclose(W.sum(axis=1), 1.0)               # always invested, no model fit yet -> core only
    assert np.all(W[:, :14] == 0)                        # ETFs are signal only
    assert np.all(W[~np.asarray(v.universe)] == 0)       # members only
    late = v.col("E1023")
    assert W[302, late] == 0 and W[304, late] > 0        # joins at the next month start (row 304)
    leaver = v.col("E1022")
    assert W[404, leaver] > 0 and W[405, leaver] == 0
    assert np.allclose(W[405].sum(), 1.0)
    dv0 = np.asarray(v.dollar_volume[0, 14:], dtype=float)
    dv0[23] = 0
    assert np.allclose(W[0, 14:], dv0 / dv0.sum())       # first row is a core date (1 observation)
    Wc = target_weights(make_view(cut=380), PARAMS)
    assert np.allclose(Wc, W[:380])


def test_missing_instrument_fails_loudly():
    v = make_view()
    v2 = DataView(dates=v.dates, instruments=tuple(n if n != "XLF" else "XXX" for n in v.instruments),
                  asset_class=v.asset_class, ret_co=v.ret_co, ret_oc=v.ret_oc, tradable=v.tradable,
                  listed=v.listed, delisting=v.delisting, close=v.close, dollar_volume=v.dollar_volume,
                  universe=v.universe, extras=v.extras, series={}, cash_ret=v.cash_ret)
    with pytest.raises(Exception):
        target_weights(v2, PARAMS)
