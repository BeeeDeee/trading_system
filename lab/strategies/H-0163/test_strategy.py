import numpy as np
import pytest

from strategy import target_weights, PARAMS, _session_dv, _book, _events, PAIRS, WINDOW

INSTR = ("SPY", "QQQ", "IWM", "EEM", "EWZ", "FXI", "XLE", "GDX", "SLV", "GLD", "USO", "FXE")
VOLKEYS = ["fred_macro.VIXCLS"] + [k for _, k in PAIRS]


class View:
    """Minimal stand-in with the DataView fields the strategy reads."""

    def __init__(self, ret_co, ret_oc, close, series, listed=None):
        T, N = close.shape
        self.dates = np.arange(T).astype("datetime64[D]")
        self.instruments = INSTR
        self.asset_class = tuple("us_etf" for _ in INSTR)
        self.ret_co = ret_co
        self.ret_oc = ret_oc
        self.close = close
        self.listed = np.isfinite(close) if listed is None else listed
        self.tradable = self.listed.copy()
        self.delisting = np.zeros((T, N), bool)
        self.dollar_volume = np.where(np.isfinite(close), 1e8, np.nan)
        self.universe = None
        self.extras = {}
        self.series = series
        self.cash_ret = np.zeros(T)

    def col(self, name):
        return self.instruments.index(name)

    def cut(self, t):
        v = View(self.ret_co[:t + 1], self.ret_oc[:t + 1], self.close[:t + 1],
                 {k: s[:t + 1] for k, s in self.series.items()}, self.listed[:t + 1])
        return v


def make_market(T=700, seed=0, spikes=()):
    """Random market where each vol index = 0.8*VIX change - 2*residual return + noise; spikes add shocks."""
    rng = np.random.default_rng(seed)
    N = len(INSTR)
    rm = rng.normal(0, 0.01, T)
    r = 0.9 * rm[:, None] + rng.normal(0, 0.01, (T, N))
    r[:, 0] = rm
    ret_co = r * 0.3
    ret_oc = (1 + r) / (1 + ret_co) - 1
    close = 100 * np.cumprod(1 + r, axis=0)
    dvix = -3.0 * rm + rng.normal(0, 0.03, T)
    series = {}
    # session k's value sits at row k+1 (next_morning); row 0 holds the value of an earlier session
    lvix = np.log(20) + np.cumsum(dvix)
    series["fred_macro.VIXCLS"] = np.exp(np.r_[np.log(20), lvix[:-1]])
    for p, (etf, key) in enumerate(PAIRS):
        j = INSTR.index(etf)
        dvi = 0.8 * dvix - 2.0 * (r[:, j] - 0.9 * r[:, 0]) - 1.5 * r[:, j] + rng.normal(0, 0.02, T)
        for (sp, sk, size) in spikes:
            if sp == p:
                dvi[sk] += size
        lv = np.log(25) + np.cumsum(dvi)
        series[key] = np.exp(np.r_[np.log(25), lv[:-1]])
    return View(ret_co, ret_oc, close, series)


# ---------------------------------------------------------------- vol-change alignment

def test_session_dv_alignment_missing_and_repeat():
    v = np.array([10.0, 11.0, np.nan, 12.0, 12.0, 13.0])
    d = _session_dv(v)
    # row r holds session r-1: session 0 = ln11-ln10; session 1 missing; session 2 = ln12-ln11
    assert d[0] == pytest.approx(np.log(11) - np.log(10))
    assert np.isnan(d[1])
    assert d[2] == pytest.approx(np.log(12) - np.log(11))
    assert np.isnan(d[3])           # repeated value = carried, no new observation
    assert d[4] == pytest.approx(np.log(13) - np.log(12))
    assert np.isnan(d[5])           # last session: value arrives only on the next row
    assert np.all(np.isnan(_session_dv(np.array([np.nan, np.nan, 5.0]))))


# ---------------------------------------------------------------- book mechanics

def _book_inputs(T=30, P=11):
    fire = np.zeros((T, P), bool)
    side = np.zeros((T, P))
    ae = np.zeros((T, P))
    beta = np.zeros((T, P))
    cols = np.arange(1, P + 1)
    has = np.ones((T, P + 1), bool)
    return fire, side, ae, beta, cols, has


def test_single_slot_hold_and_hedge():
    fire, side, ae, beta, cols, has = _book_inputs()
    fire[10, 0], side[10, 0], ae[10, 0], beta[10, 0] = True, -1.0, 2.5, 1.0
    fire[12, 0], side[12, 0], ae[12, 0], beta[12, 0] = True, 1.0, 3.0, 1.0   # ignored: slot open
    W = _book(fire, side, ae, beta, cols, 0, has, 5, 12)
    assert np.all(W[:10] == 0)
    assert np.allclose(W[10:15, 1], -0.30)
    assert np.allclose(W[10:15, 0], 0.30)
    assert np.all(W[15:] == 0)


def test_reentry_after_close_and_hold_param():
    fire, side, ae, beta, cols, has = _book_inputs()
    fire[10, 0], side[10, 0], ae[10, 0], beta[10, 0] = True, 1.0, 2.5, 0.5
    fire[13, 0], side[13, 0], ae[13, 0], beta[13, 0] = True, 1.0, 2.5, 0.5
    W = _book(fire, side, ae, beta, cols, 0, has, 3, 12)
    assert np.allclose(W[10:16, 1], 0.30)       # 10..12 then new slot 13..15
    assert np.allclose(W[10:16, 0], -0.15)
    assert W[16, 1] == 0
    W7 = _book(fire, side, ae, beta, cols, 0, has, 7, 12)
    assert np.allclose(W7[10:17, 1], 0.30) and W7[17, 1] == 0   # event at 13 ignored


def test_slot_cap_largest_eps_and_gross_scaling():
    fire, side, ae, beta, cols, has = _book_inputs()
    for p, e in zip(range(4), (2.1, 3.0, 2.5, 2.2)):
        fire[5, p], side[5, p], ae[5, p], beta[5, p] = True, 1.0, e, 1.5
    W = _book(fire, side, ae, beta, cols, 0, has, 5, 12)
    held = set(np.flatnonzero(W[5, 1:]) + 0)
    assert held == {1, 2, 3}                   # pairs 1,2,3 (eps 3.0, 2.5, 2.2), not pair 0
    raw_gross = 3 * 0.30 + 3 * 1.5 * 0.30
    assert np.abs(W[5]).sum() == pytest.approx(1.0)
    assert W[5, 2] == pytest.approx(0.30 / raw_gross)
    assert W[5, 0] == pytest.approx(-1.35 / raw_gross)


def test_spy_legs_net_and_no_entry_without_price():
    fire, side, ae, beta, cols, has = _book_inputs()
    fire[5, 0], side[5, 0], ae[5, 0], beta[5, 0] = True, 1.0, 2.5, 1.0
    fire[5, 1], side[5, 1], ae[5, 1], beta[5, 1] = True, -1.0, 2.4, 1.0
    fire[5, 2], side[5, 2], ae[5, 2], beta[5, 2] = True, 1.0, 2.3, 1.0
    has[5, cols[2]] = False                     # ETF of pair 2 has no bar on t
    W = _book(fire, side, ae, beta, cols, 0, has, 5, 12)
    assert W[5, 0] == pytest.approx(0.0)        # hedges cancel
    assert W[5, cols[2]] == 0
    assert np.abs(W[5]).sum() == pytest.approx(0.6)


# ---------------------------------------------------------------- signal

def test_spike_fires_event_on_next_row_with_feared_side():
    # QQQ vol (pair 0) falls with the ETF's own return (-1.5*r) -> negative corr -> feared side -1
    sk = 600
    d = make_market(spikes=[(0, sk, 0.5)])
    fire, side, ae, beta, cols, spy = _events(d, 2.0)
    assert fire[sk + 1, 0]                      # session sk is observed at decision sk+1
    assert side[sk + 1, 0] == -1.0              # excess rise -> toward the feared (down) side
    assert not fire[sk, 0] and not fire[sk + 2, 0]
    d2 = make_market(spikes=[(0, sk, -0.5)])
    fire2, side2, _, _, _, _ = _events(d2, 2.0)
    assert fire2[sk + 1, 0] and side2[sk + 1, 0] == 1.0
    W = target_weights(d, PARAMS)
    assert W[sk + 1, cols[0]] < 0 and W[sk + 1, spy] > 0


def test_eps_matches_independent_ols():
    sk = 600
    d = make_market(seed=3)
    fire, side, ae, beta, cols, spy = _events(d, 0.0)
    p = 1
    j = cols[p]
    r = (1 + d.ret_co) * (1 + d.ret_oc) - 1
    # independent beta on session sk (rows sk-251..sk; row 0 has no return)
    rs, ri = r[sk - WINDOW + 1:sk + 1, 0], r[sk - WINDOW + 1:sk + 1, j]
    b_sk = np.clip(np.cov(rs, ri, bias=True)[0, 1] / np.var(rs), 0, 1.5)
    assert beta[sk + 1, p] == pytest.approx(b_sk, rel=1e-8)
    lv = np.log(d.series[PAIRS[p][1]])
    lm = np.log(d.series["fred_macro.VIXCLS"])
    dvi = lv[1:] - lv[:-1]                       # session k = index k
    dvm = lm[1:] - lm[:-1]
    bet = np.full(len(r), np.nan)
    for k in range(WINDOW, sk + 1):
        a, b_ = r[k - WINDOW + 1:k + 1, 0], r[k - WINDOW + 1:k + 1, j]
        bet[k] = np.clip(np.cov(a, b_, bias=True)[0, 1] / np.var(a), 0, 1.5)
    x = r[:, j] - bet * r[:, 0]
    ks = np.arange(sk - WINDOW, sk)
    ks = ks[np.isfinite(x[ks])]
    A = np.column_stack([np.ones(len(ks)), dvm[ks], x[ks]])
    coef, *_ = np.linalg.lstsq(A, dvi[ks], rcond=None)
    res = dvi[ks] - A @ coef
    sd = np.sqrt(res @ res / (len(ks) - 3))
    eps = (dvi[sk] - coef @ np.array([1.0, dvm[sk], x[sk]])) / sd
    assert ae[sk + 1, p] == pytest.approx(abs(eps), rel=1e-6)


def test_no_event_before_warmup_and_late_listing():
    d = make_market(spikes=[(0, 300, 0.8), (1, 600, 0.8)])
    # IWM (pair 1) not listed before row 450 -> beta/regression not ready at session 600
    j = INSTR.index("IWM")
    d.close[:450, j] = np.nan
    d.listed[:450, j] = False
    fire, *_ = _events(d, 2.0)
    # beta needs 200 returns (rows 1..200), the regression 200 observations with x (sessions 200..399):
    # the first eps is on session 400, acted on at decision row 401; the big spike at 300 is not an event
    assert not fire[:401].any()
    assert not fire[:, 1].any()
    W = target_weights(d, PARAMS)
    assert np.all(W[:450, j] == 0)


def test_missing_vol_value_no_double_event():
    sk = 600
    d = make_market(spikes=[(0, sk, 0.6)])
    key = PAIRS[0][1]
    d.series[key] = d.series[key].copy()
    d.series[key][sk + 2] = np.nan              # session sk+1 has no value
    fire, *_ = _events(d, 2.0)
    assert fire[sk + 1, 0]
    assert not fire[sk + 2, 0]                  # s = sk again: already used


def test_params_change_output_and_gross():
    d = make_market(T=900, seed=5, spikes=[(p, 500 + 30 * p, 0.4) for p in range(11)])
    base = target_weights(d, PARAMS)
    assert np.abs(base).sum(axis=1).max() <= 1.0 + 1e-12
    assert np.abs(base).sum() > 0
    for k, v in (("hold_days", 7), ("z_entry", 1.5)):
        q = dict(PARAMS)
        q[k] = v
        w = target_weights(d, q)
        assert not np.allclose(w, base)
        assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12


def test_point_in_time_truncation():
    d = make_market(T=800, seed=7, spikes=[(p, 520 + 20 * p, 0.4) for p in range(11)])
    full = target_weights(d, PARAMS)
    for t in (480, 541, 600, 701, 799):
        part = target_weights(d.cut(t), PARAMS)
        assert np.array_equal(part, full[:t + 1])

