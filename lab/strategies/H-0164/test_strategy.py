import numpy as np
import pytest

from strategy import target_weights, PARAMS, book_from_arrays, trailing_vol, daily_total_returns

# column order used in the array tests
NAMES = ("SPY", "IEF", "SHY", "IEI", "TLH", "TLT", "TIP")
C = {n: i for i, n in enumerate(NAMES)}
BUCKET_COLS = [C["SHY"], C["IEI"], C["IEF"], C["TLH"], C["TLT"], C["TIP"]]
B = {"b0_3": 0, "b3_7": 1, "b7_10": 2, "b10_20": 3, "b20_30": 4, "tips": 5}


def _book(flags, sigma, close=None, tilt=0.75):
    T = flags.shape[0]
    if close is None:
        close = np.ones((T, len(NAMES)))
    return book_from_arrays(flags, close, sigma, C["SPY"], C["IEF"], C["SHY"], BUCKET_COLS, tilt)


def _sigma(T, **vols):
    s = np.full((T, len(NAMES)), 0.10)
    for n, v in vols.items():
        s[:, C[n]] = v
    return s


def test_empty_and_ief_only_keep_core():
    flags = np.zeros((2, 6), dtype=bool)
    flags[1, B["b7_10"]] = True
    w, key = _book(flags, _sigma(2))
    for t in range(2):
        assert w[t, C["SPY"]] == pytest.approx(0.60)
        assert w[t, C["IEF"]] == pytest.approx(0.40)
        assert w[t].sum() == pytest.approx(1.0)
    assert np.all(key == 0)


def test_single_long_bucket_duration_matched():
    # TLT twice as volatile as IEF: 0.30 share -> 0.15 TLT + 0.15 SHY, IEF keeps 0.10
    flags = np.zeros((1, 6), dtype=bool)
    flags[0, B["b20_30"]] = True
    w, _ = _book(flags, _sigma(1, IEF=0.07, TLT=0.14))
    assert w[0, C["SPY"]] == pytest.approx(0.60)
    assert w[0, C["IEF"]] == pytest.approx(0.10)
    assert w[0, C["TLT"]] == pytest.approx(0.15)
    assert w[0, C["SHY"]] == pytest.approx(0.15)
    assert w[0].sum() == pytest.approx(1.0)
    assert np.all(w >= 0)


def test_short_bucket_full_share():
    # SHY less volatile than IEF: ratio capped at 1, full 0.30 to SHY
    flags = np.zeros((1, 6), dtype=bool)
    flags[0, B["b0_3"]] = True
    w, _ = _book(flags, _sigma(1, IEF=0.07, SHY=0.02))
    assert w[0, C["SHY"]] == pytest.approx(0.30)
    assert w[0, C["IEF"]] == pytest.approx(0.10)


def test_split_with_ief_bucket_and_tips():
    # B = {IEF, TLT, TIP}: k = 3, share 0.10 each; IEF share stays in IEF
    flags = np.zeros((1, 6), dtype=bool)
    flags[0, [B["b7_10"], B["b20_30"], B["tips"]]] = True
    w, key = _book(flags, _sigma(1, IEF=0.06, TLT=0.12, TIP=0.05))
    assert w[0, C["IEF"]] == pytest.approx(0.10 + 0.10)
    assert w[0, C["TLT"]] == pytest.approx(0.05)
    assert w[0, C["TIP"]] == pytest.approx(0.10)
    assert w[0, C["SHY"]] == pytest.approx(0.05)
    assert w[0].sum() == pytest.approx(1.0)
    assert key[0, C["TLT"]] == pytest.approx(1 / 3)


def test_tilt_fraction_changes_weights():
    flags = np.zeros((1, 6), dtype=bool)
    flags[0, B["b3_7"]] = True
    w1, _ = _book(flags, _sigma(1), tilt=0.5)
    w2, _ = _book(flags, _sigma(1), tilt=1.0)
    assert w1[0, C["IEI"]] == pytest.approx(0.20)
    assert w2[0, C["IEI"]] == pytest.approx(0.40)
    assert w2[0, C["IEF"]] == pytest.approx(0.0)


def test_missing_price_or_vol_means_not_purchased():
    flags = np.zeros((3, 6), dtype=bool)
    flags[:, B["b10_20"]] = True
    sigma = _sigma(3)
    close = np.ones((3, len(NAMES)))
    close[0, C["TLH"]] = np.nan          # no price on d
    sigma[1, C["TLH"]] = np.nan          # too short history
    sigma[2, C["IEF"]] = np.nan          # IEF vol unknown -> no ratio, no tilt
    w, key = _book(flags, sigma, close=close)
    for t in range(3):
        assert w[t, C["IEF"]] == pytest.approx(0.40)
        assert w[t, C["TLH"]] == 0.0
    assert np.all(key == 0)


def test_trailing_vol_window_and_short_views():
    T = 6
    r = np.zeros((T, 1))
    r[:, 0] = [0.0, 0.01, -0.01, 0.02, 0.0, 0.03]
    valid = np.ones((T, 1), dtype=bool)
    valid[0] = False
    s = trailing_vol(r, valid, 3)
    assert np.isnan(s[:3, 0]).all()            # window rows 1..3 needs row 3; row 0 invalid
    assert s[3, 0] == pytest.approx(np.std([0.01, -0.01, 0.02], ddof=1) * np.sqrt(252))
    assert s[5, 0] == pytest.approx(np.std([0.02, 0.0, 0.03], ddof=1) * np.sqrt(252))
    # shorter than the window: all NaN, no crash
    assert np.isnan(trailing_vol(r[:2], valid[:2], 3)).all()
    # point-in-time: prefix unchanged by future rows
    s2 = trailing_vol(r[:4], valid[:4], 3)
    assert np.allclose(s2[3], s[3])


def test_daily_total_returns_validity():
    co = np.array([[0.0], [0.01], [0.0]])
    oc = np.array([[0.0], [0.02], [0.0]])
    close = np.array([[1.0], [1.0], [np.nan]])
    r, v = daily_total_returns(co, oc, close)
    assert r[1, 0] == pytest.approx(1.01 * 1.02 - 1)
    assert list(v[:, 0]) == [False, True, False]


# ---- end-to-end on a hand-made DataView -------------------------------------------

def _view(T=90, ops=None):
    from lab.framework.data import DataView
    rng = np.random.default_rng(1)
    first = np.array([14613], dtype="int64").astype("datetime64[D]")[0]   # a Monday (day number)
    dates = np.busday_offset(first, np.arange(T), roll="forward")
    N = len(NAMES)
    vols = np.array([0.012, 0.004, 0.001, 0.003, 0.006, 0.008, 0.003])
    ret_co = rng.normal(0, 1, (T, N)) * vols * 0.3
    ret_oc = rng.normal(0, 1, (T, N)) * vols
    close = 100 * np.cumprod(1 + ret_co, axis=0) * np.cumprod(1 + ret_oc, axis=0)
    series = {}
    for key in B:
        s = np.zeros(T)
        series["nyfed_tsy_operations_1d.buy_next." + key] = s
    if ops:
        for t, key in ops:
            series["nyfed_tsy_operations_1d.buy_next." + key][t] = 1.0
    series["nyfed_tsy_operations_1d.buy_next.tips"][:3] = np.nan
    ones = np.ones((T, N), dtype=bool)
    return DataView(
        dates=dates, instruments=NAMES, asset_class=tuple("us_etf" for _ in NAMES),
        ret_co=ret_co, ret_oc=ret_oc, tradable=ones.copy(), listed=ones.copy(),
        delisting=np.zeros((T, N), dtype=bool), close=close, dollar_volume=close * 1e6,
        universe=None, extras={}, series=series, cash_ret=np.zeros(T),
    )


def test_end_to_end_one_session_hold_and_decisions():
    p = {"tilt_fraction": 0.75, "vol_lookback": 42}
    data = _view(ops=[(60, "b20_30"), (61, "b20_30"), (70, "b3_7"), (75, "b7_10")])
    w = target_weights(data, p)
    T, N = w.shape
    assert (T, N) == (90, 7)
    dec = ~np.isnan(w).all(axis=1)
    assert dec[0]
    full = w[dec]
    assert np.all(full >= -1e-12)
    assert np.allclose(full.sum(axis=1), 1.0)
    # TLT tilt decided on 60, kept (no new decision) on 61, back to core on 62
    assert w[60, C["TLT"]] > 0
    assert w[60, C["SPY"]] == pytest.approx(0.60)
    month_start = np.zeros(T, dtype=bool)
    m = data.dates.astype("datetime64[M]")
    month_start[1:] = m[1:] != m[:-1]
    if not month_start[61]:
        assert np.isnan(w[61]).all()
    assert w[62, C["TLT"]] == 0.0 and w[62, C["IEF"]] == pytest.approx(0.40)
    assert w[70, C["IEI"]] > 0 and w[71, C["IEI"]] == 0.0
    # IEF-only bucket: no book change, no decision unless month start
    if not month_start[75]:
        assert np.isnan(w[75]).all()
    # before the lookback is full no tilt can happen
    data2 = _view(ops=[(10, "b20_30")])
    w2 = target_weights(data2, p)
    assert not (~np.isnan(w2[10])).any() or w2[10, C["TLT"]] == 0.0


def test_vol_lookback_changes_output_and_pit():
    data = _view(ops=[(50, "b20_30"), (80, "b10_20")])
    a = target_weights(data, {"tilt_fraction": 0.75, "vol_lookback": 42})
    b = target_weights(data, {"tilt_fraction": 0.75, "vol_lookback": 63})
    assert a[50, C["TLT"]] > 0
    assert np.isnan(b[50]).all() or b[50, C["TLT"]] == 0.0      # 63 returns not yet available at row 50
    assert a[80, C["TLH"]] != pytest.approx(b[80, C["TLH"]])
    # missing series raises
    del data.series["nyfed_tsy_operations_1d.buy_next.b0_3"]
    with pytest.raises(KeyError):
        target_weights(data, PARAMS)
