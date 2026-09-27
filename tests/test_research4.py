from datetime import date

import numpy as np
import polars as pl
import pytest

from qlab.data.normalize import normalize_prices
from qlab.data.panel import panel_from_bars
from qlab.data.synthetic import generate_market
from qlab.features.basic import sma_ratio, trailing_return, trailing_volatility
from qlab.research4.fundamentals import asof_filings, with_year_ago
from qlab.research4.sleeves import (composite, price_scores, sleeve_library, stock_decisions)
from qlab.schedule import period_starts
from qlab.validation.leakage import perturb_after


@pytest.fixture(scope="module")
def panel():
    m = generate_market(n_assets=20, seed=4, start="2008-01-01", end="2012-12-31")
    return panel_from_bars(normalize_prices(m.prices, m.delistings, m.calendar), m.calendar)


def _days(panel):
    return np.flatnonzero(period_starts(panel.dates, "M"))


def test_library_counts():
    lib = sleeve_library()
    kinds = [s.kind for s in lib]
    assert (kinds.count("stock"), kinds.count("etf"), kinds.count("cash")) == (62, 24, 1)
    assert len({s.family for s in lib}) == 25
    assert len({s.name for s in lib}) == len(lib)


def test_price_scores_match_reference_features(panel):
    days = _days(panel)
    sc = price_scores(panel, days, 0)
    k = len(days) - 3
    t = days[k]
    np.testing.assert_allclose(sc["mom_12_1"][k], trailing_return(panel, 252, 21)[t], rtol=1e-5)
    np.testing.assert_allclose(sc["lowvol_63"][k], -trailing_volatility(panel, 63)[t], rtol=1e-4)
    np.testing.assert_allclose(sc["trend_200"][k], sma_ratio(panel, 200)[t], rtol=1e-4,
                               atol=1e-6)


def test_price_scores_point_in_time(panel):
    days = _days(panel)
    full = price_scores(panel, days, 0)
    rng = np.random.default_rng(1)
    for k in (15, 30, len(days) - 2):
        t = days[k]
        trunc = price_scores(panel.slice(t + 1), days[: k + 1], 0)
        pert = price_scores(perturb_after(panel, t, rng), days[: k + 1], 0)
        for name, m in full.items():
            np.testing.assert_array_equal(trunc[name][k], m[k], err_msg=name)
            np.testing.assert_array_equal(pert[name][: k + 1], m[: k + 1], err_msg=name)


def test_asof_filing_visible_only_after_filing_day():
    filings = pl.DataFrame({"col": [0, 0, 1], "filed": [date(2020, 1, 10), date(2020, 4, 10),
                                                         date(2018, 1, 1)],
                            "v": [1.0, 2.0, 9.0]})
    j = asof_filings(filings, [date(2020, 1, 10), date(2020, 1, 11), date(2020, 4, 10),
                               date(2020, 4, 13)], max_age_days=400)
    got = {(r["date"], r["col"]): r["v"] for r in j.iter_rows(named=True)}
    assert (date(2020, 1, 10), 0) not in got           # filed that day -> not yet usable
    assert got[(date(2020, 1, 11), 0)] == 1.0
    assert got[(date(2020, 4, 10), 0)] == 1.0
    assert got[(date(2020, 4, 13), 0)] == 2.0
    assert not any(c == 1 for _, c in got)              # stale filing dropped


def test_year_ago_join():
    art = pl.DataFrame({"ticker": ["A"] * 3,
                        "calendardate": [date(2019, 3, 31), date(2019, 12, 31), date(2020, 3, 31)],
                        "assets": [100.0, 110.0, 150.0]})
    j = with_year_ago(art, ["assets"]).sort("calendardate")
    assert j["assets_ya"].to_list() == [None, None, 100.0]


def test_stock_decisions_top_n():
    score = np.array([[3.0, 1.0, np.nan, 2.0, 5.0]])
    uni = np.array([[True, True, True, True, False]])
    d = stock_decisions(score, uni, np.array([7]), 2)
    np.testing.assert_allclose(d.weights[0], [0.1, 0, 0, 0.1, 0])  # 1/2 capped at 10 %


def test_composite_requires_parts():
    a = np.array([[1.0, 2.0, np.nan]])
    b = np.array([[2.0, np.nan, np.nan]])
    c = composite([a, b], np.ones((1, 3), bool), 2)
    assert np.isfinite(c[0, 0]) and np.isnan(c[0, 1]) and np.isnan(c[0, 2])
