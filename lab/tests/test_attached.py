"""SF1 / insiders / 13F attached to the SEP stocks: point-in-time by filing date, validated fields."""

import numpy as np
import pytest

from lab.framework import attached, catalog, data
from lab.framework.data import DATA_ROOT
from lab.framework.paths import LAB_DIR

needs_data = pytest.mark.skipif(not DATA_ROOT.exists(), reason="no data snapshot on this machine")
D = lambda s: np.datetime64(s, "D")  # noqa: E731
DATES = np.arange(D("2020-01-01"), D("2021-01-01"))
DATES = DATES[np.is_busday(DATES)]


def row(day):
    return int(np.searchsorted(DATES, D(day)))


def test_asof_visible_only_after_the_filing_date_and_newest_period_wins():
    cols = np.array([0, 0, 0, 1])
    filed = np.array([D("2020-02-14"), D("2020-05-15"), D("2020-06-01"), D("2020-03-02")])
    period = np.array([D("2019-12-31"), D("2020-03-31"), D("2019-12-31"), D("2019-12-31")])   # 3rd: late correction of Q4
    value = np.array([1.0, 2.0, 99.0, 7.0])
    m = attached.asof_matrix(cols, filed, period, value, DATES, 2, 400)
    assert np.isnan(m[row("2020-02-14"), 0]) and m[row("2020-02-17"), 0] == 1.0     # filed Friday 14th: visible Monday
    assert m[row("2020-05-18"), 0] == 2.0 and m[row("2020-07-01"), 0] == 2.0        # the correction does not override Q1
    assert np.isnan(m[row("2020-03-02"), 1]) and m[row("2020-03-03"), 1] == 7.0
    assert np.isnan(m[:, 1][: row("2020-03-03")]).all()


def test_asof_values_go_stale():
    m = attached.asof_matrix(np.array([0]), np.array([D("2020-01-10")]), np.array([D("2019-12-31")]), np.array([5.0]),
                             DATES, 1, 60)
    assert m[row("2020-03-06"), 0] == 5.0 and np.isnan(m[row("2020-03-20"), 0])


def test_window_sums_use_filing_date_and_start_unknown():
    cols = np.array([0, 0, 1])
    filed = np.array([D("2020-03-02"), D("2020-04-01"), D("2020-03-02")])
    m = attached.window_sum_matrix(cols, filed, np.array([10.0, 5.0, 1.0]), DATES, 3, 91, D("2020-02-01"))
    assert m[row("2020-03-02"), 0] == 0.0 and m[row("2020-03-03"), 0] == 10.0      # not usable on the filing day itself
    assert m[row("2020-04-02"), 0] == 15.0 and m[row("2020-06-15"), 0] == 5.0      # the first filing left the window
    assert m[row("2020-09-01"), 0] == 0.0 and m[row("2020-03-03"), 2] == 0.0       # nobody filed: 0, not unknown
    assert np.isnan(m[row("2020-01-15"), 0])                                         # before the data starts


def test_window_sum_is_point_in_time():
    rng = np.random.default_rng(0)
    cols = rng.integers(0, 4, 60)
    filed = DATES[rng.integers(0, len(DATES) - 1, 60)]
    val = rng.random(60)
    full = attached.window_sum_matrix(cols, filed, val, DATES, 4, 91, DATES[0])
    cut = 150
    keep = filed < DATES[cut]
    part = attached.window_sum_matrix(cols[keep], filed[keep], val[keep], DATES[: cut + 1], 4, 91, DATES[0])
    assert np.allclose(part[:cut], full[:cut], atol=1e-4)


def test_field_grammar_and_requirement_problems():
    cat = catalog.load(LAB_DIR / "data" / "catalog.yaml")
    ok = {"data_requirements": [{"dataset": "sharadar_sep", "frequency": "1d"},
                                {"dataset": "sharadar_sf1", "frequency": "1q", "fields": ["sf1_art_roe", "sf1_arq_revenue"]},
                                {"dataset": "sharadar_insiders", "frequency": "event", "fields": ["ins_buy_value_91d"]},
                                {"dataset": "sharadar_13f", "frequency": "1q", "fields": ["f13_io"]}]}
    assert attached.requirement_problems(ok, cat) == []
    bad = {"data_requirements": [{"dataset": "sharadar_sf1", "frequency": "1q", "fields": ["sf1_art_nonsense", "roe"]},
                                 {"dataset": "sharadar_13f", "frequency": "1q"}]}
    problems = " | ".join(attached.requirement_problems(bad, cat))
    assert "attach to sharadar_sep" in problems and "sf1_art_nonsense" in problems and "list the fields" in problems
    many = {"data_requirements": [{"dataset": "sharadar_sep", "frequency": "1d"}, {"dataset": "sharadar_sf1", "frequency": "1q",
            "fields": [f"sf1_art_{c}" for c in attached.SF1_COLUMNS[:7]]}]}
    assert any("at most 6" in p for p in attached.requirement_problems(many, cat))


@needs_data
def test_real_data_spot_checks():
    import polars as pl
    src = DATA_ROOT / "parquet" / data.SHARADAR
    v = data.sharadar_sep({"kind": "sp500"})
    v = attached.attach(v, "sharadar_sf1", ["sf1_art_netinc", "sf1_arq_revenue"], src)
    v = attached.attach(v, "sharadar_insiders", ["ins_buy_n_91d"], src)
    v = attached.attach(v, "sharadar_13f", ["f13_io"], src)
    tick = (pl.read_parquet(src / "tickers.parquet").filter((pl.col("table") == "SEP") & (pl.col("ticker") == "AAPL"))
            .select(pl.col("permaticker").cast(pl.Int64)).item())
    j = v.col(f"E{tick}")
    day = D("2015-06-01")
    t = int(np.searchsorted(v.dates, day))
    art = (pl.scan_parquet(src / "fundamentals.parquet").filter((pl.col("ticker") == "AAPL") & (pl.col("dimension") == "ART"))
           .select("calendardate", pl.col("date").alias("filed"), "netinc").collect()
           .filter(pl.col("filed") < pl.lit(day.astype(object))).sort("calendardate", "filed").tail(1))
    assert np.isclose(v.extras["sf1_art_netinc"][t, j], float(art["netinc"][0]), rtol=1e-6)
    assert np.isnan(v.extras["ins_buy_n_91d"][row_of(v, "2005-01-03"), j]) and v.extras["ins_buy_n_91d"][t, j] >= 0
    assert 0 < v.extras["f13_io"][t, j] < 5 and np.isnan(v.extras["f13_io"][row_of(v, "2010-01-04"), j])
    assert all(m.dtype == np.float32 for k, m in v.extras.items() if k.startswith(("sf1", "ins", "f13")))


def row_of(v, day):
    return int(np.searchsorted(v.dates, D(day)))
