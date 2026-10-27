import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import PARAMS, target_weights

INSTR = ("FXE", "FXY", "FXB", "FXA", "FXC", "FXF", "SHY", "SHV", "BIL",
         "SPY", "EZU", "EWJ", "EWU", "EWA", "EWC", "EWL")
CUR = ("FXE", "FXY", "FXB", "FXA", "FXC", "FXF")
NEVER = ("SHY", "SHV", "BIL", "SPY", "EZU", "EWJ", "EWU", "EWA", "EWC", "EWL")


def day(y, m, d):
    """Calendar day built by arithmetic (no date literals)."""
    month = np.array([(y - 1970) * 12 + (m - 1)]).astype("datetime64[M]")
    return (month.astype("datetime64[D]").astype("int64") + (d - 1)).astype("datetime64[D]")[0]


# NYSE holidays of the test months, written out by hand (independent of the strategy's rule-based calendar)
HOLIDAYS = [day(*x) for x in (
    (2018, 2, 19), (2018, 3, 30),                       # Washington's Birthday, Good Friday
    (2018, 12, 25), (2019, 1, 1), (2019, 1, 21),        # Christmas, New Year, MLK
    (2021, 4, 2), (2021, 5, 31),                        # Good Friday, Memorial Day
    (2020, 11, 26), (2020, 12, 25),                     # Thanksgiving, Christmas
    (2022, 1, 17), (2022, 2, 21),                       # MLK, Washington's Birthday
)]


def sessions(start, end):
    span = np.arange(start.astype("int64"), end.astype("int64") + 1)
    span = span.astype("datetime64[D]")
    return span[np.is_busday(span, holidays=np.array(HOLIDAYS))]


def month_end(y, m):
    ny, nm = (y, m + 1) if m < 12 else (y + 1, 1)
    return (day(ny, nm, 1).astype("int64") - 1).astype("datetime64[D]")


def month_view(y, m):
    """Previous month + month (y, m), one signal shock on the first session of the month."""
    py, pm = (y, m - 1) if m > 1 else (y - 1, 12)
    return make_view(day(py, pm, 1), month_end(y, m), day(y, m, 1))


def traded_and_exit(y, m, lead=4):
    v = month_view(y, m)
    w = target_weights(v, {"entry_lead": lead})
    nz = np.nonzero(np.nan_to_num(np.abs(w)).sum(axis=1) > 0)[0]
    first_nan_after = [i for i in range(len(v.dates)) if nz.size and i > nz[-1] and not np.isnan(w[i]).all()]
    return v.dates[nz].tolist(), v.dates[first_nan_after[0]].item() if first_nan_after else None


# ---------------------------------------------------------------- calendar


@pytest.mark.parametrize("ym, d_expected, t_expected", [
    ((2019, 1), (2019, 1, 24), (2019, 1, 31)),     # plain month
    ((2018, 3), (2018, 3, 22), (2018, 3, 29)),     # Good Friday is the last weekday
    ((2021, 5), (2021, 5, 21), (2021, 5, 28)),     # Memorial Day on the 31st
    ((2020, 11), (2020, 11, 20), (2020, 11, 30)),  # Thanksgiving inside the window
    ((2020, 12), (2020, 12, 23), (2020, 12, 31)),  # Christmas inside the window
    ((2022, 2), (2022, 2, 18), (2022, 2, 28)),     # Washington's Birthday inside the window
])
def test_decision_is_t_minus_5_sessions_and_exit_at_t(ym, d_expected, t_expected):
    traded, exit_row = traded_and_exit(*ym)
    assert traded == [day(*d_expected).item()]     # the only row with a non-zero target
    assert exit_row == day(*t_expected).item()      # first decided (zero) row after d is T


def test_entry_lead_moves_decision():
    assert traded_and_exit(2019, 1, 3)[0] == [day(2019, 1, 25).item()]
    assert traded_and_exit(2019, 1, 5)[0] == [day(2019, 1, 23).item()]


# ---------------------------------------------------------------- hand-made market

F_RET = {"FXE": 0.01, "FXY": -0.01, "FXB": 0.0, "FXA": 0.0, "FXC": 0.0, "FXF": 0.0}
CTY_RET = {"EZU": 0.03, "EWJ": 0.0, "EWU": 0.01, "EWA": -0.01, "EWC": 0.05, "EWL": 0.0}
SPY_RET = 0.02
PAIR = dict(zip(CUR, ("EZU", "EWJ", "EWU", "EWA", "EWC", "EWL")))


def expected_D():
    out = {}
    for c in CUR:
        r_loc = (1 + CTY_RET[PAIR[c]]) / (1 + F_RET[c]) - 1
        out[c] = SPY_RET - r_loc
    return out


def make_view(start=None, end=None, shock=None):
    if start is None:
        start, end, shock = day(2018, 12, 3), day(2019, 2, 28), day(2019, 1, 3)
    dates = sessions(start, end)
    T, N = len(dates), len(INSTR)
    ret_co = np.zeros((T, N))
    ret_oc = np.zeros((T, N))
    r = int(np.nonzero(dates >= shock)[0][0])
    col = {n: i for i, n in enumerate(INSTR)}
    for c, v in F_RET.items():
        ret_oc[r, col[c]] = v
    for c, v in CTY_RET.items():
        ret_oc[r, col[c]] = v
    ret_oc[r, col["SPY"]] = SPY_RET
    return DataView(
        dates=dates, instruments=INSTR, asset_class=tuple("us_etf" for _ in INSTR),
        ret_co=ret_co, ret_oc=ret_oc, tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool), delisting=np.zeros((T, N), dtype=bool),
        close=np.full((T, N), 10.0), dollar_volume=np.full((T, N), 1e7),
        universe=None, extras={}, series={}, cash_ret=np.zeros(T))


def row(view, y, m, d):
    return int(np.nonzero(view.dates == day(y, m, d))[0][0])


def test_weights_on_known_rows():
    v = make_view()
    w = target_weights(v, PARAMS)
    D = expected_D()
    gross = sum(abs(x) for x in D.values())
    d = row(v, 2019, 1, 24)
    for c in CUR:
        assert w[d, v.col(c)] == pytest.approx(D[c] / gross)
    assert w[d, v.col("FXA")] > 0 and w[d, v.col("FXC")] < 0
    assert np.nansum(np.abs(w[d])) == pytest.approx(1.0)
    for c in NEVER:
        assert np.all(np.nan_to_num(w[:, v.col(c)]) == 0.0)
    # held unchanged (NaN = keep) from d + 1 to T - 1, zero at the close of T (exit at the open of T + 1)
    for dd in (25, 28, 29, 30):
        assert np.all(np.isnan(w[row(v, 2019, 1, dd)]))
    assert np.all(w[row(v, 2019, 1, 31)] == 0.0)
    assert np.all(w[row(v, 2019, 1, 23)] == 0.0)
    assert np.all(w[row(v, 2019, 2, 1)] == 0.0)
    # December: first month in the data (no previous month-end) -> flat; February: all D = 0 -> flat
    assert np.all(np.nan_to_num(w[:row(v, 2019, 1, 2)]) == 0.0)
    assert np.all(np.nan_to_num(w[row(v, 2019, 2, 1):]) == 0.0)
    fin = np.where(np.isnan(w), 0.0, w)
    assert np.all(np.abs(fin).sum(axis=1) <= 1.0 + 1e-12)


def test_missing_country_price_skips_currency():
    v = make_view()
    v.close[row(v, 2019, 1, 24), v.col("EZU")] = np.nan
    w = target_weights(v, PARAMS)
    D = expected_D()
    D.pop("FXE")
    gross = sum(abs(x) for x in D.values())
    d = row(v, 2019, 1, 24)
    assert w[d, v.col("FXE")] == 0.0
    for c, x in D.items():
        assert w[d, v.col(c)] == pytest.approx(x / gross)


def test_late_listing_at_previous_month_end_skips_currency():
    v = make_view()
    pm = row(v, 2018, 12, 31)
    v.close[:pm + 1, v.col("FXF")] = np.nan
    v.listed[:pm + 1, v.col("FXF")] = False
    v.tradable[:pm + 2, v.col("FXF")] = False
    w = target_weights(v, PARAMS)
    d = row(v, 2019, 1, 24)
    assert w[d, v.col("FXF")] == 0.0
    assert np.abs(w[d]).sum() == pytest.approx(1.0)


def test_not_tradable_at_d_skips_currency():
    v = make_view()
    v.tradable[row(v, 2019, 1, 24), v.col("FXB")] = False
    w = target_weights(v, PARAMS)
    assert w[row(v, 2019, 1, 24), v.col("FXB")] == 0.0


def test_fewer_than_three_currencies_is_flat():
    v = make_view()
    d = row(v, 2019, 1, 24)
    for e in ("EZU", "EWJ", "EWU", "EWA"):
        v.close[d, v.col(e)] = np.nan
    w = target_weights(v, PARAMS)
    assert np.all(w[d] == 0.0)
    v2 = make_view()
    for e in ("EZU", "EWJ", "EWU"):
        v2.close[d, v2.col(e)] = np.nan
    w2 = target_weights(v2, PARAMS)
    assert np.abs(w2[d]).sum() == pytest.approx(1.0)
    assert np.count_nonzero(w2[d]) == 3


def test_missing_spy_is_flat():
    v = make_view()
    v.close[row(v, 2018, 12, 31), v.col("SPY")] = np.nan
    w = target_weights(v, PARAMS)
    assert np.all(w[row(v, 2019, 1, 24)] == 0.0)


def test_entry_lead_param_changes_output():
    v = make_view()
    w3 = target_weights(v, {"entry_lead": 3})
    w5 = target_weights(v, {"entry_lead": 5})
    assert np.abs(w3[row(v, 2019, 1, 25)]).sum() == pytest.approx(1.0)
    assert np.all(w3[row(v, 2019, 1, 24)] == 0.0)
    assert np.abs(w5[row(v, 2019, 1, 23)]).sum() == pytest.approx(1.0)
    assert np.all(np.isnan(w5[row(v, 2019, 1, 24)]))


def test_point_in_time_truncation():
    v = make_view()
    full = target_weights(v, PARAMS)
    for cut in (row(v, 2019, 1, 24), row(v, 2019, 1, 28), row(v, 2019, 1, 31)):
        n = cut + 1
        tv = DataView(
            dates=v.dates[:n], instruments=v.instruments, asset_class=v.asset_class,
            ret_co=v.ret_co[:n], ret_oc=v.ret_oc[:n], tradable=v.tradable[:n], listed=v.listed[:n],
            delisting=v.delisting[:n], close=v.close[:n], dollar_volume=v.dollar_volume[:n],
            universe=None, extras={}, series={}, cash_ret=v.cash_ret[:n])
        part = target_weights(tv, PARAMS)
        assert np.array_equal(np.isnan(part), np.isnan(full[:n]))
        assert np.allclose(np.nan_to_num(part), np.nan_to_num(full[:n]))


def test_missing_instrument_raises():
    v = make_view()
    keep = [i for i, n in enumerate(INSTR) if n != "EWL"]
    names = tuple(INSTR[i] for i in keep)
    bad = DataView(
        dates=v.dates, instruments=names, asset_class=tuple("us_etf" for _ in names),
        ret_co=v.ret_co[:, keep], ret_oc=v.ret_oc[:, keep], tradable=v.tradable[:, keep],
        listed=v.listed[:, keep], delisting=v.delisting[:, keep], close=v.close[:, keep],
        dollar_volume=v.dollar_volume[:, keep], universe=None, extras={}, series={}, cash_ret=v.cash_ret)
    with pytest.raises(Exception):
        target_weights(bad, PARAMS)

