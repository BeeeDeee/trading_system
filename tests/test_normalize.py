import numpy as np
import polars as pl
import pytest

from qlab.data.normalize import (ACQUISITION, BANKRUPTCY, PERFORMANCE, SPAC_LIQUIDATION, UNKNOWN,
                                 normalize_prices, terminal_return)
from qlab.data.schema import ACTIVE, DELISTED, HALTED, NO_OPEN
from qlab.data.synthetic import generate_market


@pytest.fixture(scope="module")
def market():
    return generate_market(n_assets=60, seed=7)


@pytest.fixture(scope="module")
def bars(market):
    return normalize_prices(market.prices, market.delistings, market.calendar)


def test_generator_covers_all_cases(market, bars):
    kinds = set(market.events.filter(pl.col("event").str.starts_with("delisted"))["event"])
    assert {"split", "dividend"} <= set(market.events["event"])
    assert len(kinds) >= 4
    assert {ACTIVE, NO_OPEN, HALTED, DELISTED} <= set(bars["status"])


def test_total_return_matches_truth(market, bars):
    growth = (bars.sort("permaticker", "date")
              .with_columns(idx=((1 + pl.col("ret_co")) * (1 + pl.col("ret_oc")))
                            .cum_prod().over("permaticker")))
    joined = market.truth.join(growth, on=["permaticker", "date"])
    assert joined.height == market.truth.height
    check = joined.with_columns(
        expected=pl.col("tr_index") / pl.col("tr_index").first().over("permaticker"),
        actual=pl.col("idx") / pl.col("idx").first().over("permaticker"))
    np.testing.assert_allclose(check["actual"], check["expected"], rtol=1e-9)


def test_unadjusted_values_recovered(market, bars):
    joined = market.truth.join(bars, on=["permaticker", "date"], suffix="_bars")
    np.testing.assert_allclose(joined["close_u_bars"], joined["close_u"], rtol=1e-12)
    np.testing.assert_allclose(joined["volume_u_bars"], joined["volume_u"], rtol=1e-9)


def test_dividend_split_between_overnight_and_intraday_is_close(market, bars):
    ex = market.events.filter(pl.col("event") == "dividend").select("permaticker", "date")
    rows = bars.join(ex, on=["permaticker", "date"]).filter(pl.col("status") == ACTIVE)
    assert rows.height > 0
    # The dividend is attributed proportionally; the overnight part is off by at most ~yield * move.
    truth = market.truth.sort("permaticker", "date").with_columns(
        prev_tr=pl.col("tr_index").shift(1).over("permaticker"))
    joined = rows.join(truth, on=["permaticker", "date"])
    c2c = (1 + joined["ret_co"]) * (1 + joined["ret_oc"])
    np.testing.assert_allclose(c2c, joined["tr_index"] / joined["prev_tr"], rtol=1e-9)
    assert (joined["ret_co"].abs() < 0.5).all()


def test_halted_and_no_open_rows(bars):
    halted = bars.filter(pl.col("status") == HALTED)
    assert not halted["tradable"].any()
    assert (halted["ret_co"] == 0).all() and (halted["ret_oc"] == 0).all()
    no_open = bars.filter(pl.col("status") == NO_OPEN)
    assert not no_open["tradable"].any() and (no_open["ret_oc"] == 0).all()


def test_delisting_rows(market, bars):
    delisted = bars.filter(pl.col("status") == DELISTED)
    ended = market.truth.group_by("permaticker").agg(pl.col("date").max().alias("last"))
    ended = ended.filter(pl.col("last") < market.calendar[-1])
    assert delisted.height == ended.height
    cal = pl.Series(market.calendar)
    for row in delisted.join(ended, on="permaticker").iter_rows(named=True):
        assert row["date"] == cal[cal.search_sorted(row["last"]) + 1]
        assert not row["tradable"] and row["ret_oc"] == 0
        assert row["ret_co"] == row["terminal_ret"]
    # Securities without a delisting record fall back to UNKNOWN.
    unknown = set(market.events.filter(pl.col("event") == f"delisted:{UNKNOWN}")["permaticker"])
    assert unknown and not unknown & set(market.delistings["permaticker"])
    assert unknown <= set(delisted["permaticker"])


def test_terminal_return_policy():
    assert terminal_return(ACQUISITION, 10.0, 12.0) == pytest.approx(0.2)
    assert terminal_return(ACQUISITION, 10.0, None) == 0.0
    assert terminal_return(BANKRUPTCY, 3.0) == -1.0
    assert terminal_return(SPAC_LIQUIDATION, 10.1) == 0.0
    assert terminal_return(PERFORMANCE, 4.0) == -0.30
    assert terminal_return(UNKNOWN, 0.5) == -0.30
    assert terminal_return(UNKNOWN, 5.0) == 0.0
    with pytest.raises(ValueError):
        terminal_return("meteor", 1.0)
