import numpy as np
import pytest

from strategy import target_weights, PARAMS, ffill_limited, value_scores, month_end_rows
from lab.framework.data import DataView

# dates as day numbers since 1970-01-01 (no date literals allowed)
D_2001_01_01 = 11323   # a Monday
D_2021_05_24 = 18771   # a Monday
D_2021_05_31 = 18778
D_2021_06_01 = 18779


def days(start, n):
    return np.busday_offset(np.array([start]).astype("datetime64[D]"), np.arange(n), roll="forward")

ETFS = ("FXE", "FXY", "FXB", "FXA", "FXC", "FXF")
KEYS = ("DEXUSEU", "DEXJPUS", "DEXUSUK", "DEXUSAL", "DEXCAUS", "DEXSZUS")
INVERT = (False, True, False, False, True, True)


def make_view(T=1400, seed=0, series_override=None, tradable_override=None, start_offset=0):
    rng = np.random.default_rng(seed)
    dates = days(D_2001_01_01, T + start_offset)[start_offset:]
    N = len(ETFS)
    series = {}
    for k, key in enumerate(KEYS):
        lvl = np.exp(np.cumsum(rng.normal(0, 0.005, T)))
        series["fred_macro." + key] = lvl
    if series_override:
        series.update(series_override)
    tradable = np.ones((T, N), dtype=bool)
    if tradable_override is not None:
        tradable = tradable_override
    z = np.zeros((T, N))
    return DataView(
        dates=dates, instruments=ETFS, asset_class=tuple(["us_etf"] * N),
        ret_co=z.copy(), ret_oc=z.copy(), tradable=tradable, listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool), close=np.ones((T, N)), dollar_volume=np.ones((T, N)) * 1e7,
        universe=None, extras={}, series=series, cash_ret=np.zeros(T),
    )


SMALL = {"lookback_years": 1, "n_per_leg": 2}  # window 252 rows


def test_ffill_limited():
    x = np.array([[1.0], [np.nan], [np.nan], [4.0], [np.nan], [np.nan], [np.nan]])
    out = ffill_limited(x, 2)
    assert out[:, 0].tolist()[:6] == [1.0, 1.0, 1.0, 4.0, 4.0, 4.0]
    assert np.isnan(out[6, 0])
    lead = ffill_limited(np.array([[np.nan], [2.0]]), 5)
    assert np.isnan(lead[0, 0]) and lead[1, 0] == 2.0


def test_value_scores_by_hand():
    lp = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0]])
    v, n = value_scores(lp, 3)
    # row 4: window rows 2..4 -> values 4, 5 (row 2 missing) -> mean 4.5, v = 4.5 - 5
    assert v[4, 0] == pytest.approx(-0.5)
    assert n[4, 0] == 2
    # row 1: rows 0..1 -> mean 1.5 - 2
    assert v[1, 0] == pytest.approx(-0.5)
    assert np.isnan(v[2, 0])


def test_month_end_rows():
    d = days(D_2021_05_24, 12)
    flags = month_end_rows(d)
    # 31 May 2021 is a weekday (Memorial Day, but weekday arithmetic ignores holidays) -> flagged
    assert d[flags].astype(np.int64).tolist() == [D_2021_05_31]
    # drop 31 May (holiday): 28 May is not flagged, 1 June becomes the one-session-late fallback
    d2 = d[d.astype(np.int64) != D_2021_05_31]
    f2 = month_end_rows(d2)
    assert d2[f2].astype(np.int64).tolist() == [D_2021_06_01]


def test_weights_mechanics_and_point_in_time():
    T = 1400
    view = make_view(T)
    w = target_weights(view, SMALL)
    assert w.shape == (T, 6)
    rows = np.flatnonzero(~np.isnan(w).all(axis=1))
    assert len(rows) > 30
    # no decisions before the window has 90% coverage (row 226 = first with 227 fixings)
    assert rows[0] >= int(0.9 * 252) - 1
    for t in rows:
        r = w[t]
        assert np.sum(np.abs(r)) == pytest.approx(1.0)
        assert np.sum(r) == pytest.approx(0.0)
        assert (r > 0).sum() == 2 and (r < 0).sum() == 2
        assert np.allclose(r[r > 0], 0.25) and np.allclose(r[r < 0], -0.25)
    # truncation invariance
    cut = 900
    vt = make_view(T)
    short = DataView(
        dates=vt.dates[:cut], instruments=vt.instruments, asset_class=vt.asset_class,
        ret_co=vt.ret_co[:cut], ret_oc=vt.ret_oc[:cut], tradable=vt.tradable[:cut], listed=vt.listed[:cut],
        delisting=vt.delisting[:cut], close=vt.close[:cut], dollar_volume=vt.dollar_volume[:cut],
        universe=None, extras={}, series={k: s[:cut] for k, s in vt.series.items()}, cash_ret=vt.cash_ret[:cut],
    )
    ws = target_weights(short, SMALL)
    assert np.array_equal(np.isnan(ws), np.isnan(w[:cut]))
    assert np.allclose(np.nan_to_num(ws), np.nan_to_num(w[:cut]))


def test_ranking_by_hand():
    T = 600
    t_idx = np.arange(T)
    # constant levels except a final drift so the value ranking is known:
    # EUR falls (cheap, highest v), GBP rises (rich, lowest v) etc.
    drift = {"DEXUSEU": -0.002, "DEXJPUS": +0.001, "DEXUSUK": +0.002,
             "DEXUSAL": -0.001, "DEXCAUS": 0.0, "DEXSZUS": -0.0005}
    over = {}
    for key, inv in zip(KEYS, INVERT):
        p = np.exp(drift[key] * t_idx)  # USD price of the currency path
        over["fred_macro." + key] = 1.0 / p if inv else p
    view = make_view(T, series_override=over)
    w = target_weights(view, SMALL)
    t = np.flatnonzero(~np.isnan(w).all(axis=1))[-1]
    # P drifts: EUR -, JPY +, GBP +, AUD -, CAD 0, CHF -  -> v high for falling P
    # v ranking: EUR (-0.002) > AUD (-0.001) > CHF (-0.0005) > CAD > JPY > GBP
    assert w[t, ETFS.index("FXE")] == pytest.approx(0.25)
    assert w[t, ETFS.index("FXA")] == pytest.approx(0.25)
    assert w[t, ETFS.index("FXY")] == pytest.approx(-0.25)
    assert w[t, ETFS.index("FXB")] == pytest.approx(-0.25)
    assert w[t, ETFS.index("FXC")] == 0 and w[t, ETFS.index("FXF")] == 0
    w1 = target_weights(view, {"lookback_years": 1, "n_per_leg": 1})
    assert w1[t, ETFS.index("FXE")] == pytest.approx(0.5) and w1[t, ETFS.index("FXB")] == pytest.approx(-0.5)
    w3 = target_weights(view, {"lookback_years": 1, "n_per_leg": 3})
    assert w3[t, ETFS.index("FXF")] == pytest.approx(0.5 / 3) and w3[t, ETFS.index("FXC")] == pytest.approx(-0.5 / 3)


def test_lookback_changes_output():
    view = make_view(2000, seed=3)
    a = target_weights(view, {"lookback_years": 1, "n_per_leg": 2})
    b = target_weights(view, {"lookback_years": 2, "n_per_leg": 2})
    assert not np.array_equal(np.nan_to_num(a, nan=9), np.nan_to_num(b, nan=9))


def test_excluded_currency_and_error():
    T = 800
    tr = np.ones((T, 6), dtype=bool)
    tr[-30:, ETFS.index("FXF")] = False  # one ETF not tradable at the end: left out
    view = make_view(T, tradable_override=tr)
    w = target_weights(view, SMALL)
    rows = np.flatnonzero(~np.isnan(w).all(axis=1))
    last = rows[rows >= T - 30][-1]
    assert w[last, ETFS.index("FXF")] == 0
    assert np.sum(np.abs(w[last])) == pytest.approx(1.0)
    tr2 = np.ones((T, 6), dtype=bool)
    tr2[-30:, :2] = False  # two missing -> fewer than 5 -> rebalance skipped (NaN row, positions kept)
    w2 = target_weights(make_view(T, tradable_override=tr2), SMALL)
    assert np.isnan(w2[-30:]).all()
    assert not np.isnan(w2[: T - 30]).all()


def test_stale_fixings_drop_currency():
    T = 800
    rng = np.random.default_rng(5)
    s = np.exp(np.cumsum(rng.normal(0, 0.005, T)))
    s[-10:] = np.nan  # more than 5 sessions without a fixing -> unusable
    view = make_view(T, series_override={"fred_macro.DEXUSAL": s})
    w = target_weights(view, SMALL)
    rows = np.flatnonzero(~np.isnan(w).all(axis=1))
    if rows[-1] >= T - 4:
        assert w[rows[-1], ETFS.index("FXA")] == 0


def test_missing_series_raises():
    view = make_view(400)
    del view.series["fred_macro.DEXUSEU"]
    with pytest.raises(KeyError):
        target_weights(view, SMALL)


def test_params_primary():
    assert PARAMS == {"lookback_years": 5, "n_per_leg": 2}
