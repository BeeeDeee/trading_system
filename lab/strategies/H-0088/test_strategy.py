import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, COUNTRIES, decision_rows, events

ETFS = [c[0] for c in COUNTRIES]
INSTR = ETFS
SLOPES = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0]) * 0.01


def _dates(n_days=420):
    d = (np.arange(n_days) + 10957).astype("datetime64[D]")  # calendar days from 2000-01-01
    return d[np.is_busday(d)]


def make_view(dates=None, slopes=SLOPES, tradable=None, fx_override=None):
    if dates is None:
        dates = _dates()
    T, N = len(dates), len(INSTR)
    t = np.arange(T, dtype=float)
    series = {}
    for k, (etf, key, sgn) in enumerate(COUNTRIES):
        lp = slopes[k] * t
        series["fred_macro." + key] = np.exp(sgn * lp)
    if fx_override:
        for key, arr in fx_override.items():
            series["fred_macro." + key] = arr
    if tradable is None:
        tradable = np.ones((T, N), dtype=bool)
    zeros = np.zeros((T, N))
    return DataView(
        dates=dates,
        instruments=tuple(INSTR),
        asset_class=tuple(["us_etf"] * N),
        ret_co=zeros.copy(),
        ret_oc=zeros.copy(),
        tradable=tradable,
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.ones((T, N)),
        dollar_volume=np.full((T, N), 1e8),
        universe=None,
        extras={},
        series=series,
        cash_ret=np.zeros(T),
    )


P1 = {"hold_months": 3, "lookback_months": 1, "n_per_leg": 2}


def test_decision_rows_last_weekday_of_month():
    d = _dates()
    dec = decision_rows(d)
    months = d.astype("datetime64[M]")
    for i in np.flatnonzero(dec):
        assert i == len(d) - 1 or months[i + 1] != months[i]
    # every complete month has exactly one decision; the last month in the data is incomplete
    assert not dec[-1]                         # 2001-02-23 is a Friday but not the month's last weekday
    assert dec.sum() == len(np.unique(months)) - 1
    assert dec[np.flatnonzero(months[1:] != months[:-1])].all()


def test_decision_holiday_fallback():
    d = _dates()
    dec = decision_rows(d)
    i = np.flatnonzero(dec)[2]           # remove the last weekday of the 3rd month (holiday)
    d2 = np.delete(d, i)
    dec2 = decision_rows(d2)
    assert not dec2[i - 1]               # previous day is not the last weekday
    assert dec2[i]                       # first session of next month takes the decision
    assert dec2.sum() == dec.sum()


def test_weights_first_cohort_and_ranking():
    v = make_view()
    W = target_weights(v, P1)
    dec = decision_rows(v.dates)
    rows = np.flatnonzero(dec)
    assert np.all(np.isnan(W[~dec]))
    # decision rows before the 21-session window is full: cash
    first_full = rows[rows >= 20][0]
    for r in rows[rows < 20]:
        assert np.all(W[r] == 0)
    w = W[first_full]
    unit = 0.5 / 2 / 3
    # falling currencies (most negative slope) are cheapest -> long; rising -> short
    assert w[v.col("EWA")] == pytest.approx(unit)
    assert w[v.col("EWC")] == pytest.approx(unit)
    assert w[v.col("ENZL")] == pytest.approx(-unit)
    assert w[v.col("NORW")] == pytest.approx(-unit)
    assert np.sum(np.abs(w)) == pytest.approx(4 * unit)
    # after hold_months decisions the book is full: gross 1, dollar-neutral
    wf = W[rows[rows >= 20][3]]
    assert np.sum(np.abs(wf)) == pytest.approx(1.0)
    assert np.sum(wf) == pytest.approx(0.0)
    assert np.all(np.nansum(np.abs(W), axis=1) <= 1 + 1e-12)


def test_cohort_sum_identity():
    # with changing rankings, H=2 book = mean of the last two H=1 cohorts
    d = _dates()
    T = len(d)
    rng = np.random.default_rng(0)
    over = {}
    for etf, key, sgn in COUNTRIES:
        over[key] = np.exp(np.cumsum(rng.normal(0, 0.01, T)))
    v = make_view(dates=d, fx_override=over)
    p1 = dict(P1, hold_months=1)
    p2 = dict(P1, hold_months=2)
    W1, W2 = target_weights(v, p1), target_weights(v, p2)
    rows = np.flatnonzero(decision_rows(d))
    rows = rows[rows >= 20]
    changed = False
    for a, b in zip(rows[:-1], rows[1:]):
        assert np.allclose(W2[b], (W1[a] + W1[b]) / 2)
        changed |= not np.allclose(W1[a], W1[b])
    assert changed


def test_too_few_eligible_no_cohort():
    d = _dates()
    T = len(d)
    nan = np.full(T, np.nan)
    over = {COUNTRIES[k][1]: nan for k in range(4)}   # 6 eligible left
    v = make_view(dates=d, fx_override=over)
    W3 = target_weights(v, dict(P1, n_per_leg=3))     # needs 7
    assert np.all(np.nan_to_num(W3) == 0)
    W2 = target_weights(v, dict(P1, n_per_leg=2))     # needs 5
    assert np.nansum(np.abs(W2)) > 0
    # missing countries never get a weight
    for k in range(4):
        assert np.all(np.nan_to_num(W2[:, v.col(COUNTRIES[k][0])]) == 0)


def test_staleness_and_forward_fill():
    d = _dates()
    T = len(d)
    rows = np.flatnonzero(decision_rows(d))
    r = rows[rows >= 80][0]
    x = np.exp(SLOPES[0] * np.arange(T))       # EWA (P = x) is the cheapest currency
    p = dict(P1, lookback_months=3, hold_months=1)   # 63-session window: coverage holds, staleness binds
    stale = x.copy()
    stale[r - 10:] = stale[r - 11]             # carried forward: latest observation 11 sessions old
    ok = x.copy()
    ok[r - 9:] = ok[r - 10]                    # latest observation 10 sessions old -> still eligible
    j = 0
    W1s = target_weights(make_view(dates=d, fx_override={"DEXUSAL": stale}), p)
    W1o = target_weights(make_view(dates=d, fx_override={"DEXUSAL": ok}), p)
    assert W1s[r, j] == 0
    assert W1o[r, j] == pytest.approx(0.25)
    nanned = x.copy()
    nanned[r - 10:] = np.nan                   # NaN gap behaves like carried values
    W1n = target_weights(make_view(dates=d, fx_override={"DEXUSAL": nanned}), p)
    assert W1n[r, j] == 0
    nanned_ok = x.copy()
    nanned_ok[r - 9:] = np.nan
    W1m = target_weights(make_view(dates=d, fx_override={"DEXUSAL": nanned_ok}), p)
    assert W1m[r, j] == pytest.approx(0.25)


def test_coverage_rule():
    d = _dates()
    T = len(d)
    rows = np.flatnonzero(decision_rows(d))
    r = rows[rows >= 60][0]
    x = np.exp(SLOPES[0] * np.arange(T))
    gappy = x.copy()
    gappy[r - 20:r - 14] = np.nan              # 6 of 21 missing -> 15/21 < 80 %
    W = target_weights(make_view(dates=d, fx_override={"DEXUSAL": gappy}), dict(P1, hold_months=1))
    assert W[r, 0] == 0
    gappy2 = x.copy()
    gappy2[r - 20:r - 16] = np.nan             # 4 missing -> 17/21 >= 80 %
    W = target_weights(make_view(dates=d, fx_override={"DEXUSAL": gappy2}), dict(P1, hold_months=1))
    assert W[r, 0] == pytest.approx(0.25)


def test_untradable_slots_go_to_cash_until_expiry():
    d = _dates()
    T, N = len(d), len(INSTR)
    rows = np.flatnonzero(decision_rows(d))
    rows = rows[rows >= 20]
    tr = np.ones((T, N), dtype=bool)
    tr[rows[1], 0] = False                     # EWA not tradable on the 2nd decision date only
    v = make_view(dates=d, tradable=tr)
    W = target_weights(v, P1)
    unit = 0.5 / 2 / 3
    assert W[rows[0], 0] == pytest.approx(unit)
    assert W[rows[1], 0] == 0                  # old cohort's slot to cash, EWA not eligible
    assert W[rows[1], 1] == pytest.approx(2 * unit)
    assert W[rows[2], 0] == pytest.approx(unit)          # only the new cohort at rows[2]
    assert W[rows[3], 0] == pytest.approx(2 * unit)      # cohort of rows[0] expired; rows[1] cohort lacked EWA
    assert W[rows[4], 0] == pytest.approx(3 * unit)


def test_late_listing():
    d = _dates()
    T, N = len(d), len(INSTR)
    rows = np.flatnonzero(decision_rows(d))
    rows = rows[rows >= 20]
    tr = np.ones((T, N), dtype=bool)
    j = INSTR.index("ENZL")
    tr[: rows[2], j] = False
    W = target_weights(make_view(dates=d, tradable=tr), P1)
    assert W[rows[0], j] == 0 and W[rows[1], j] == 0
    assert W[rows[2], j] < 0


def test_params_change_output():
    d = _dates()
    T = len(d)
    rng = np.random.default_rng(1)
    over = {key: np.exp(np.cumsum(rng.normal(0, 0.01, T))) for _, key, _ in COUNTRIES}
    v = make_view(dates=d, fx_override=over)
    base = target_weights(v, P1)
    for k, alt in (("hold_months", 2), ("lookback_months", 2), ("n_per_leg", 1)):
        other = target_weights(v, dict(P1, **{k: alt}))
        assert not np.allclose(np.nan_to_num(base), np.nan_to_num(other)), k
    assert set(PARAMS) == {"hold_months", "lookback_months", "n_per_leg"}


def test_missing_series_raises():
    v = make_view()
    del v.series["fred_macro.DEXUSEU"]
    with pytest.raises(KeyError):
        target_weights(v, P1)


def test_diagnostic_events():
    v = make_view()
    mask, side = events(v, P1)
    rows = np.flatnonzero(mask.any(axis=1))
    assert len(rows) > 0
    assert np.all(decision_rows(v.dates)[rows])
    r = rows[0]
    assert mask[r].sum() == 4
    assert side[r, v.col("EWA")] == 1 and side[r, v.col("ENZL")] == -1
    # 6 eligible < 2*2+3 -> no events
    nan = np.full(len(v.dates), np.nan)
    v2 = make_view(fx_override={COUNTRIES[k][1]: nan for k in range(4)})
    m2, _ = events(v2, P1)
    assert not m2.any()
