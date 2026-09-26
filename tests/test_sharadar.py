from datetime import date

import polars as pl
import pytest

from qlab.data.normalize import ACQUISITION, BANKRUPTCY, PERFORMANCE, SPAC_LIQUIDATION
from qlab.data.sharadar import Snapshot, build_bars, delistings, securities, sic_history

D = [date(2020, 1, d) for d in (2, 3, 6, 7, 8)]


def price_rows(ticker, days, close):
    return [{"ticker": ticker, "date": d, "open": close, "high": close, "low": close,
             "close": close, "volume": 1000.0, "closeadj": close, "closeunadj": close,
             "lastupdated": d} for d in days]


@pytest.fixture
def snap(tmp_path):
    tickers = pl.DataFrame({
        "table": ["SEP"] * 8 + ["SFP"],
        "permaticker": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "ticker": ["ACQ", "CASHT", "STOCKT", "SPAC", "BUST", "REG", "GONE", "PREF", "BIG"],
        "category": ["Domestic Common Stock"] * 7 + ["Domestic Preferred Stock", "ETF"],
        "siccode": ["1000", "1000", "1000", "2834", "1000", "1000", "1000", "1000", None],
    })
    stocks = pl.DataFrame(
        price_rows("ACQ", D, 50.0) + price_rows("CASHT", D[:3], 10.0)
        + price_rows("STOCKT", D[:3], 20.0) + price_rows("SPAC", D[:2], 10.0)
        + price_rows("BUST", D[:4], 2.0) + price_rows("REG", D[:4], 0.5)
        + price_rows("GONE", D[:2], 3.0) + price_rows("PREF", D[:2], 25.0))
    funds = pl.DataFrame(price_rows("BIG", D, 100.0))
    actions = pl.DataFrame([
        ("2020-01-06", "acquisitionby", "CASHT", None, "ACQ"),
        ("2020-01-06", "acquisitioncash", "CASHT", 12.0, "ACQ"),
        ("2020-01-06", "delisted", "CASHT", 1e9, None),
        ("2020-01-06", "acquisitionby", "STOCKT", None, "ACQ"),
        ("2020-01-06", "acquisitionstock", "STOCKT", 0.5, "ACQ"),   # 0.5 * 50 = 25 per share
        ("2020-01-06", "delisted", "STOCKT", 1e9, None),
        ("2019-06-01", "sicchangefrom", "SPAC", 6770, None),       # was a SPAC before 2019-06
        ("2019-06-01", "sicchangeto", "SPAC", 2834, None),
        ("2021-06-01", "sicchangefrom", "SPAC", 6770, None),       # ...and again until 2021-06
        ("2021-06-01", "sicchangeto", "SPAC", 2834, None),
        ("2020-01-03", "bankruptcyliquidation", "SPAC", 1e8, None),
        ("2020-01-03", "delisted", "SPAC", 1e8, None),
        ("2020-01-07", "bankruptcyliquidation", "BUST", 1e6, None),
        ("2020-01-07", "delisted", "BUST", 1e6, None),
        ("2020-01-07", "regulatorydelisting", "REG", 1e6, None),
        ("2020-01-07", "delisted", "REG", 1e6, None),
    ], schema=["date", "action", "ticker", "value", "contraticker"], orient="row").with_columns(
        pl.col("date").str.to_date(), pl.col("value").cast(pl.Float64))
    for name, df in {"tickers": tickers, "stocks": stocks, "funds": funds,
                     "actions": actions}.items():
        df.write_parquet(tmp_path / f"{name}.parquet")
    return Snapshot(tmp_path)


def test_securities_keep_only_universe_categories(snap):
    secs = securities(snap, snap.connect())
    assert secs["ticker"].to_list() == ["ACQ", "CASHT", "STOCKT", "SPAC", "BUST", "REG", "GONE"]


def test_sic_history_is_point_in_time(snap):
    con = snap.connect()
    sic = sic_history(snap, con, securities(snap, con)).filter(pl.col("permaticker") == 4)
    at = lambda d: sic.filter((pl.col("valid_from") <= d) & (pl.col("valid_to") > d))["sic"].item()  # noqa: E731
    assert at(date(2019, 1, 1)) == 6770
    assert at(date(2020, 1, 3)) == 6770
    assert at(date(2022, 1, 1)) == 2834


def test_delisting_classification(snap):
    con = snap.connect()
    secs = securities(snap, con)
    dl = delistings(snap, con, secs, sic_history(snap, con, secs), D[-1])
    by = {r["permaticker"]: r for r in dl.iter_rows(named=True)}
    assert by[2]["kind"] == ACQUISITION and by[2]["consideration_per_share"] == 12.0
    assert by[3]["kind"] == ACQUISITION and by[3]["consideration_per_share"] == 25.0
    assert by[4]["kind"] == SPAC_LIQUIDATION
    assert by[5]["kind"] == BANKRUPTCY
    assert by[6]["kind"] == PERFORMANCE
    assert 7 not in by  # no action -> left to normalization as UNKNOWN
    assert 1 not in by  # still trading


def test_implausible_consideration_is_dropped(snap, tmp_path):
    actions = pl.read_parquet(tmp_path / "actions.parquet").with_columns(
        value=pl.when((pl.col("ticker") == "STOCKT") & (pl.col("action") == "acquisitionstock"))
        .then(5.0).otherwise(pl.col("value")))  # 5 * 50 = 250 vs last close 20
    actions.write_parquet(tmp_path / "actions.parquet")
    con = snap.connect()
    secs = securities(snap, con)
    dl = delistings(snap, con, secs, sic_history(snap, con, secs), D[-1])
    row = dl.filter(pl.col("permaticker") == 3).row(0, named=True)
    assert row["raw_consideration"] == 250.0 and row["consideration_per_share"] is None


def test_build_bars_terminal_returns(snap, tmp_path):
    out = tmp_path / "derived" / "bars"
    summary = build_bars(snap, out, chunk_size=3)
    assert summary["securities"] == 7
    bars = pl.read_parquet(out / "*.parquet")
    term = dict(bars.filter(pl.col("status") == "delisted")
                .select("permaticker", "terminal_ret").iter_rows())
    assert term[2] == pytest.approx(0.2)    # 12 / 10
    assert term[3] == pytest.approx(0.25)   # 25 / 20
    assert term[4] == 0.0                   # SPAC trust payout
    assert term[5] == -1.0                  # bankruptcy
    assert term[6] == pytest.approx(-0.3)   # regulatory delisting
    assert term[7] == 0.0                   # UNKNOWN above 1 USD
