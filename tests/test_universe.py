from datetime import date

import polars as pl
import pytest

from qlab.data.normalize import normalize_prices
from qlab.data.synthetic import generate_market
from qlab.universe import eligibility_features, rank_universe, sp500_intervals

PARAMS = dict(top_n=10, min_price=5.0, min_history=60)


@pytest.fixture(scope="module")
def bars():
    m = generate_market(n_assets=40, seed=17)
    return normalize_prices(m.prices, m.delistings, m.calendar)


def universe(bars, sic=None):
    return rank_universe(eligibility_features(bars, sic, dv_window=20), **PARAMS)


def test_universe_is_point_in_time(bars):
    full = universe(bars)
    dates = sorted(bars["date"].unique())
    for t in dates[100::150]:
        cut = universe(bars.filter(pl.col("date") <= t))
        assert cut.filter(pl.col("date") == t).equals(full.filter(pl.col("date") == t))


def test_universe_rules(bars):
    u = universe(bars)
    assert u.group_by("date").len()["len"].max() == 10
    joined = u.join(bars, on=["permaticker", "date"])
    assert (joined["close_u"] >= 5).all()
    assert (joined["status"] != "delisted").all()
    first = bars.group_by("permaticker").agg(pl.col("date").sort().get(59).alias("d60"))
    early = u.join(first, on="permaticker").filter(pl.col("date") < pl.col("d60"))
    assert early.is_empty()  # no member before its 60th bar


def test_spac_interval_excluded(bars):
    u = universe(bars)
    victim, day = u["permaticker"][0], u["date"][0]
    sic = pl.DataFrame({"permaticker": [victim], "valid_from": [date(1900, 1, 1)],
                        "valid_to": [day], "sic": [6770]})
    sic_after = sic.with_columns(valid_from=pl.lit(day), valid_to=pl.lit(date(9999, 12, 31)))
    before = universe(bars, sic)  # SPAC until `day`, operating company from `day`
    after = universe(bars, sic_after)
    assert before.filter((pl.col("permaticker") == victim) & (pl.col("date") == day)).height == 1
    assert after.filter(pl.col("permaticker") == victim).is_empty()


def test_sp500_intervals():
    sp = pl.DataFrame({
        "date": [date(2026, 1, 2)] * 2 + [date(2020, 5, 1), date(2020, 5, 1), date(2010, 3, 1)],
        "action": ["current", "current", "added", "removed", "added"],
        "ticker": ["AAA", "BBB", "BBB", "OLD", "AAA"],
    })
    iv = {r["ticker"]: (r["start"], r["end"]) for r in sp500_intervals(sp).iter_rows(named=True)}
    assert iv["AAA"] == (date(2010, 3, 1), None)
    assert iv["BBB"] == (date(2020, 5, 1), None)
    assert iv["OLD"] == (None, date(2020, 5, 1))
