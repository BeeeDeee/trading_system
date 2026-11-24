import dataclasses

import numpy as np
import pytest
from lab.framework.data import DataView

from strategy import PARAMS, target_weights, avg_rank01, ols_residual, n_cohorts, formation_signal, AA_KEYS

PREFIX = "coinmetrics_adractcnt_1d."
BASE = 17532          # day number of 2018-01-01
T = 600
N = 9
# nine spot pairs whose keys are in the card's list (<ID>USDT -> <ID>USDT.P)
LISTED = ("ADAUSDT", "BCHUSDT", "BTCUSDT", "DASHUSDT", "EOSUSDT", "ETCUSDT", "ETHUSDT", "LTCUSDT", "NEOUSDT")
UNLISTED = ("XRPUSDT", "BUSDUSDT")     # pairs whose key is not in the card's list
ROW_JAN19, ROW_FEB19, ROW_MAR19, ROW_APR19 = 365, 396, 424, 455


def make_view(a=None, b=None, unlisted=(), universe=None, aa_override=None, close_override=None,
              missing_key=None):
    d = np.arange(T, dtype=float)[:, None]
    a = np.arange(N) * 0.001 if a is None else np.asarray(a, float)
    b = np.full(N, 0.001) if b is None else np.asarray(b, float)
    aa = 1000.0 * np.exp(a[None, :] * d)
    close = 10.0 * np.exp(b[None, :] * d)
    if aa_override is not None:
        aa = aa_override
    if close_override is not None:
        close = close_override
    names = list(LISTED)
    for i, j in enumerate(unlisted):
        names[j] = UNLISTED[i]
    names = tuple(names)
    series = {PREFIX + k: np.ones(T) for k in AA_KEYS if k != missing_key}
    for j, nm in enumerate(names):
        # an unlisted pair gets a served series too: the card's list, not the dataset, decides
        series[PREFIX + nm + ".P"] = aa[:, j].copy()
        if missing_key is not None and nm + ".P" == missing_key:
            del series[PREFIX + nm + ".P"]
    dates = np.arange(BASE, BASE + T).astype("datetime64[D]")
    ones = np.ones((T, N), dtype=bool)
    return DataView(
        dates=dates, instruments=names, asset_class=tuple("crypto_spot" for _ in names),
        ret_co=np.zeros((T, N)), ret_oc=np.zeros((T, N)), tradable=ones.copy(), listed=ones.copy(),
        delisting=np.zeros((T, N), dtype=bool), close=close, dollar_volume=np.ones((T, N)),
        universe=ones.copy() if universe is None else universe, extras={}, series=series,
        cash_ret=np.zeros(T))


P = {"hold_days": 91, "lookback_days": 60}       # K = 3


def test_cohort_count():
    assert n_cohorts(126) == 4 and n_cohorts(182) == 6 and n_cohorts(273) == 9 and n_cohorts(91) == 3


def test_avg_rank_ties_and_scale():
    assert np.allclose(avg_rank01(np.array([3.0, 1.0, 2.0, 2.0])), [1.0, 0.0, 0.5, 0.5])


def test_ols_residual():
    x = np.array([0.0, 0.25, 0.5, 1.0])
    assert np.allclose(ols_residual(2 * x + 1, x), 0.0)
    y = np.array([0.0, 1.0, 0.0, 1.0])
    e = ols_residual(y, x)
    assert abs(e.sum()) < 1e-12 and abs((e * x).sum()) < 1e-12


def test_dates_rows():
    v = make_view()
    days = v.dates.astype(np.int64)
    # 2019-01-01 is day 17897, 2019-04-01 is day 17987 after the epoch
    assert days[ROW_JAN19] == 17897 and days[ROW_APR19] == 17987
    months = v.dates.astype("datetime64[M]")
    for row in (ROW_JAN19, ROW_FEB19, ROW_MAR19, ROW_APR19):
        assert months[row] != months[row - 1]


def test_cohort_ramp_and_long_group():
    w = target_weights(make_view(), P)
    # before seasoning (address series younger than 365 days): formation rows are cash
    assert np.all(w[0] == 0) and np.all(w[ROW_JAN19 - 31] == 0)
    # growth highest for coins 6, 7, 8; price runs equal -> residual = rank(g) - mean
    for row, n_open in ((ROW_JAN19, 1), (ROW_FEB19, 2), (ROW_MAR19, 3), (ROW_APR19, 3)):
        assert np.allclose(w[row, 6:], n_open / 9.0)
        assert np.all(w[row, :6] == 0)
    assert np.isclose(np.nansum(w[ROW_APR19]), 1.0)
    # in between: no decision
    assert np.all(np.isnan(w[ROW_JAN19 + 1])) and np.all(np.isnan(w[ROW_APR19 - 1]))
    assert np.all(np.nansum(np.abs(w), axis=1) <= 1 + 1e-12)
    assert np.all(np.nan_to_num(w) >= 0)


def test_residual_removes_price_run():
    # coin 8 has the highest address growth but also by far the highest price run;
    # coin 0 lowest growth, lowest run. ranks: g = j, r = (0,0,...,0,1 for coin 8 high)
    b = np.full(N, 0.001)
    b[8] = 0.01
    b[0] = -0.01
    w = target_weights(make_view(b=b), P)
    v = make_view(b=b)
    # hand computation of the residual
    g = np.arange(N) / 8.0
    r = avg_rank01(np.log(v.close[ROW_JAN19] / v.close[ROW_JAN19 - 60]))
    e = ols_residual(g, r)
    top = np.lexsort((np.arange(N), -e))[:3]
    expect = np.zeros(N)
    expect[top] = 1.0 / 9.0
    assert np.allclose(w[ROW_JAN19], expect)
    assert 8 not in top or e[8] >= np.sort(e)[-3]


def test_fewer_than_eight_eligible_is_cash():
    w = target_weights(make_view(unlisted=(0, 1)), P)      # 7 coins with a listed key
    assert np.all(np.nan_to_num(w) == 0)
    w8 = target_weights(make_view(unlisted=(0,)), P)       # 8 coins: ceil(8/3) = 3
    assert np.all(np.nan_to_num(w8[:, 0]) == 0)
    assert np.allclose(w8[ROW_JAN19, 6:], 1.0 / 9.0)


def test_late_address_series_not_seasoned():
    d = np.arange(T, dtype=float)[:, None]
    aa = 1000.0 * np.exp((np.arange(N) * 0.001)[None, :] * d)
    aa[:100, 8] = np.nan       # coin 8 starts 2018-04-11: seasoned only from 2019-04-11
    w = target_weights(make_view(aa_override=aa), P)
    assert w[ROW_JAN19, 8] == 0 and np.allclose(w[ROW_JAN19, 5:8], 1.0 / 9.0)
    assert w[ROW_APR19 + 30, 8] > 0      # 2019-05-01


def test_sparse_address_window_ineligible():
    d = np.arange(T, dtype=float)[:, None]
    aa = 1000.0 * np.exp((np.arange(N) * 0.001)[None, :] * d)
    aa[ROW_JAN19 - 6:ROW_JAN19, 8] = np.nan     # 24 of 30 recent values left
    w = target_weights(make_view(aa_override=aa), P)
    assert w[ROW_JAN19, 8] == 0
    aa2 = aa.copy()
    aa2[ROW_JAN19 - 6:ROW_JAN19, 8] = 1000.0
    aa2[ROW_JAN19 - 5:ROW_JAN19, 8] = np.nan    # 25 left: eligible
    w2 = target_weights(make_view(aa_override=aa2), P)
    assert w2[ROW_JAN19, 8] > 0


def test_uses_addresses_up_to_t_minus_1_only():
    d = np.arange(T, dtype=float)[:, None]
    aa = 1000.0 * np.exp((np.arange(N) * 0.001)[None, :] * d)
    w0 = target_weights(make_view(aa_override=aa), P)
    aa[ROW_JAN19, 0] = 1e12        # the value stamped day t itself must not matter
    w1 = target_weights(make_view(aa_override=aa), P)
    assert np.array_equal(w0[ROW_JAN19], w1[ROW_JAN19])
    aa[ROW_JAN19 - 1, 0] = 1e12    # stamped t-1 does
    w2 = target_weights(make_view(aa_override=aa), P)
    assert w2[ROW_JAN19, 0] > 0


def test_universe_exit_and_reentry():
    uni = np.ones((T, N), dtype=bool)
    uni[ROW_JAN19 + 5:ROW_JAN19 + 10, 8] = False
    w = target_weights(make_view(universe=uni), P)
    assert w[ROW_JAN19 + 5, 8] == 0 and np.all(np.isnan(w[ROW_JAN19 + 5, :8]))
    assert np.isclose(w[ROW_FEB19, 8], 2.0 / 9.0)
    uni2 = np.ones((T, N), dtype=bool)
    uni2[ROW_FEB19, 8] = False          # non-member at a formation date: not held, not replaced
    w2 = target_weights(make_view(universe=uni2), P)
    assert w2[ROW_FEB19, 8] == 0 and np.allclose(w2[ROW_FEB19, 6:8], 2.0 / 9.0)


def test_params_change_output():
    v = make_view()
    base = target_weights(v, P)
    w_hold = target_weights(v, {"hold_days": 182, "lookback_days": 60})
    assert np.allclose(w_hold[ROW_JAN19, 6:], 1.0 / 18.0) and not np.allclose(np.nan_to_num(w_hold), np.nan_to_num(base))
    w_lb = target_weights(v, {"hold_days": 91, "lookback_days": 400})   # needs t >= 430
    assert np.all(w_lb[ROW_JAN19] == 0) and w_lb[ROW_APR19, 8] > 0
    assert PARAMS == {"hold_days": 182, "lookback_days": 180}


def test_missing_dataset_raises():
    v = make_view()
    v.series.clear()
    with pytest.raises(KeyError):
        target_weights(v, P)


def test_missing_listed_key_raises():
    with pytest.raises(KeyError):
        target_weights(make_view(missing_key="SRMUSDT.P"), P)      # not an instrument here
    with pytest.raises(KeyError):
        target_weights(make_view(missing_key="ETHUSDT.P"), P)


def test_card_key_list():
    assert len(AA_KEYS) == 57 and len(set(AA_KEYS)) == 57
    assert "BTCUSDT.P" in AA_KEYS and "ETHUSDT.P" in AA_KEYS
    assert not any(k.startswith(("BUSD", "DAI", "PAXG", "HBTC", "RENBTC")) for k in AA_KEYS)
    assert not any("_" in k for k in AA_KEYS)


def test_extra_dataset_keys_do_not_matter():
    v = make_view()
    base = target_weights(v, P)
    v.series["coinmetrics_adractcnt_1d.SDAI_ETHUSDT.P"] = np.full(T, np.nan)   # empty extra key
    v.series["coinmetrics_adractcnt_1d.XRPUSDT.P"] = np.ones(T)
    w = target_weights(v, P)
    assert np.array_equal(np.nan_to_num(base, nan=-1.0), np.nan_to_num(w, nan=-1.0))


def test_perp_style_names_map_too():
    v = make_view()
    v2 = dataclasses.replace(make_view(), instruments=tuple(n + ".P" for n in v.instruments))
    assert np.array_equal(np.nan_to_num(target_weights(v, P), nan=-1.0),
                          np.nan_to_num(target_weights(v2, P), nan=-1.0))


def test_short_view():
    v = make_view()
    w = target_weights(v, PARAMS)
    assert w.shape == (T, N)


def test_formation_groups_used_by_diagnostic():
    # diagnostic.py holds a verbatim copy of formation_signal (tests may import only strategy)
    recs, _ = formation_signal(make_view(), P)
    rec = [r for r in recs if r["t"] == ROW_JAN19][0]
    assert rec["formed"] and len(rec["elig"]) == 9
    assert sorted(rec["long"].tolist()) == [6, 7, 8]
    assert sorted(rec["short"].tolist()) == [0, 1, 2]
    earlier = [r for r in recs if r["t"] < ROW_JAN19]
    assert earlier and not any(r["formed"] for r in earlier)
    assert all(r["t"] in (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334) for r in earlier)
