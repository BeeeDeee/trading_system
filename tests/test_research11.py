"""Research 11: labels, train/test split, point-in-time insider and 13F features."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from qlab.data.panel import Panel
from qlab.research11 import features as F
from qlab.research11.model import forward_returns, rank_ic, split


def make_panel(T=60, N=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.arange(np.datetime64("2015-01-01"), np.datetime64("2015-01-01") + np.timedelta64(T, "D"))
    co, oc = rng.normal(0, 0.01, (T, N)), rng.normal(0, 0.02, (T, N))
    true = np.ones((T, N), dtype=bool)
    return Panel(dates, np.arange(N, dtype=np.int64), co, oc, true, true, ~true, np.ones((T, N)), np.ones((T, N)))


def test_forward_returns_match_brute_force():
    p = make_panel()
    p.ret_co[30, 1] = -1.0                                   # bankruptcy payout inside a window
    days = np.array([5, 20, 40, 55])
    f = forward_returns(p, days, chunk=2)
    for k in range(3):
        a, b = days[k] + 1, days[k + 1] + 1
        for j in range(3):
            g = (1 + p.ret_oc[a, j]) * np.prod((1 + p.ret_co[a + 1:b, j]) * (1 + p.ret_oc[a + 1:b, j])) \
                * (1 + p.ret_co[b, j])
            assert f[k, j] == pytest.approx(g - 1, abs=1e-12)
    assert f[1, 1] == pytest.approx(-1.0) and np.isnan(f[3]).all()


def test_split_uses_only_realized_labels():
    dates = np.arange(np.datetime64("2002-01-01"), np.datetime64("2012-01-01"))
    days = np.array([i for i in range(1, len(dates)) if dates[i].astype("datetime64[M]") != dates[i - 1].astype("datetime64[M]")])
    train, test = split(dates, days, 2008)
    assert len(test) == 12 and (np.asarray(dates[days[test]], dtype="datetime64[Y]") == np.datetime64("2008", "Y")).all()
    assert days[train.max() + 1] + 1 <= days[test[0]] + 1     # last train label realized by the first test fill
    assert dates[days[train.min()]] >= np.datetime64("2003-01-01")
    assert train.max() == test[0] - 1


def test_rank_ic_perfect_and_reversed():
    rng = np.random.default_rng(0)
    fwd = rng.normal(size=(3, 100))
    uni = np.ones((3, 100), dtype=bool)
    ic = rank_ic(np.vstack([fwd[0], -fwd[1], np.full(100, np.nan)]), fwd, uni)
    assert ic[0] == pytest.approx(1) and ic[1] == pytest.approx(-1) and np.isnan(ic[2])


@pytest.fixture
def src(tmp_path):
    pl.DataFrame({"ticker": ["AAA"], "table": ["SEP"], "permaticker": [7]}).write_parquet(tmp_path / "tickers.parquet")
    pl.DataFrame({"date": [date(2014, 12, 31), date(2015, 3, 31)], "ticker": ["AAA", "AAA"],
                  "shrholders": [100, 110], "shrvalue": [50.0, 60.0], "cllholders": [5, 5], "putholders": [10, 15]}
                 ).write_parquet(tmp_path / "holdings_ticker.parquet")
    pl.DataFrame({"date": [date(2014, 12, 31), date(2015, 3, 31)], "investorid": ["X", "Y"]}
                 ).write_parquet(tmp_path / "holdings_investor.parquet")
    pl.DataFrame({"ticker": ["AAA"] * 3, "date": [date(2014, 12, 31), date(2015, 3, 31), date(2015, 5, 15)],
                  "marketcap": [100.0, 120.0, 120.0]}).write_parquet(tmp_path / "daily.parquet")
    pl.DataFrame({"ticker": ["AAA"] * 4, "date": [date(2015, 2, 13), date(2015, 5, 14), date(2015, 5, 15), date(2015, 1, 1)],
                  "ownername": ["A", "B", "C", "D"], "transactioncode": ["P", "S", "P", "P"],
                  "transactionvalue": [10.0, 20.0, 30.0, 40.0], "transactionshares": [1, 1, 1, 1],
                  "transactionpricepershare": [1.0, 1.0, 1.0, 1.0]}).write_parquet(tmp_path / "insiders.parquet")
    return tmp_path


def test_13f_quarter_usable_only_after_46_days(src):
    tick = F.ticker_cols(src, np.array([7]))
    dates = [date(2015, 5, 15), date(2015, 5, 16)]                # 2015-03-31 + 45 / + 46 days
    out = F.flow_features(src, tick, dates, {d: k for k, d in enumerate(dates)}, (2, 1))
    assert out["io"][0, 0] == pytest.approx(50 / 100)            # still the December quarter
    assert out["io"][1, 0] == pytest.approx(60 / 120)
    assert out["d_io"][1, 0] == pytest.approx(0.5 - 0.5)
    assert out["d_holders"][1, 0] == pytest.approx(np.log(110 / 100))
    assert out["putcall"][1, 0] == pytest.approx((15 - 5) / 110)
    assert out["d_breadth"][1, 0] == pytest.approx(110 / 1 - 100 / 1)


def test_insider_window_excludes_filing_day_and_old_filings(src):
    tick = F.ticker_cols(src, np.array([7]))
    dates = [date(2015, 5, 15)]                                    # window [2015-02-13, 2015-05-15)
    mcap = np.array([[1000.0]])
    out = F.insider_features(src, tick, dates, {dates[0]: 0}, (1, 1), mcap)
    assert out["ins_n_buyers"][0, 0] == 1                          # A (2015-02-13); C filed on t, D too old
    assert out["ins_n_sellers"][0, 0] == 1 and out["ins_sell"][0, 0] == pytest.approx(20 / 1000)
    assert out["ins_net_n"][0, 0] == 0
