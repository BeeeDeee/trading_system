"""Universe candidate seed filters. 04-DATA_AND_UNIVERSE.md §7.1."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from scout.cli.__main__ import main
from scout.data.store import write_parquet_atomic
from scout.universe.candidates import (
    ALLOWED_CATEGORIES,
    ALLOWED_EXCHANGES,
    MIN_LIFETIME_DOLLAR_VOLUME_USD,
    append_candidates,
    lifetime_max_dollar_volume,
    parse_candidate_lines,
    seed_candidates,
    select_candidates,
)
from scout.utils.errors import ScoutDataError


def _tickers() -> pd.DataFrame:
    rows = [
        _ticker("1", "AAA", "NYSE", "Domestic Common Stock", delisted=None),
        _ticker("2", "BBB", "NASDAQ", "Domestic Common Stock Primary Class", delisted=None),
        _ticker("3", "CCC", "NYSE", "Domestic Common Stock Secondary Class", delisted=None),
        _ticker("4", "SPY", "NYSEARCA", "ETF", delisted=None),
        _ticker("5", "ADR", "NYSE", "ADR Common Stock", delisted=None),
        _ticker("6", "OTC", "OTC", "Domestic Common Stock", delisted=None),
        _ticker("7", "PREF", "NYSE", "Domestic Preferred Stock", delisted=None),
        _ticker("8", "DEAD", "NASDAQ", "Domestic Common Stock", delisted=date(2010, 1, 1)),
        _ticker("9", "THIN", "NYSE", "Domestic Common Stock", delisted=None),
        _ticker("10", "BATS", "BATS", "ETF", delisted=None),
    ]
    return pd.DataFrame(rows)


def _ticker(
    asset_id: str,
    symbol: str,
    exchange: str,
    category: str,
    *,
    delisted: date | None,
) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "symbol": symbol,
        "exchange": exchange,
        "category": category,
        "sector": "TECH",
        "is_etf": category == "ETF",
        "listed_date": date(2000, 1, 1),
        "delisted_date": delisted,
        "delist_reason": "",
    }


def _dv(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, dtype="float64")


def test_selects_common_stock_etf_and_dual_class() -> None:
    selected = select_candidates(
        _tickers(),
        _dv(
            {
                "1": 3_000_000.0,
                "2": 3_000_000.0,
                "3": 3_000_000.0,
                "4": 3_000_000.0,
                "5": 9_000_000.0,
                "6": 9_000_000.0,
                "7": 9_000_000.0,
                "8": 3_000_000.0,
                "9": 3_000_000.0,
                "10": 3_000_000.0,
            }
        ),
    )
    assert set(selected["symbol"]) == {"AAA", "BBB", "CCC", "SPY", "DEAD", "THIN", "BATS"}
    assert "ADR" not in set(selected["symbol"])
    assert "OTC" not in set(selected["symbol"])
    assert "PREF" not in set(selected["symbol"])


def test_dollar_volume_floor_is_two_million() -> None:
    selected = select_candidates(
        _tickers(),
        _dv({"1": MIN_LIFETIME_DOLLAR_VOLUME_USD, "9": MIN_LIFETIME_DOLLAR_VOLUME_USD - 1.0}),
    )
    assert set(selected["symbol"]) == {"AAA"}


def test_missing_ohlcv_is_excluded() -> None:
    selected = select_candidates(_tickers(), _dv({"1": 5_000_000.0}))
    assert set(selected["symbol"]) == {"AAA"}


def test_append_never_deletes_existing_even_if_now_ineligible(tmp_path: Path) -> None:
    dest = tmp_path / "universe_candidates.txt"
    dest.write_text("# header\n99,OLD\n", encoding="utf-8")
    added = append_candidates(
        dest,
        pd.DataFrame({"asset_id": ["1"], "symbol": ["AAA"]}),
    )
    assert added == 1
    text = dest.read_text(encoding="utf-8")
    pairs = parse_candidate_lines(text)
    assert ("99", "OLD") in pairs
    assert ("1", "AAA") in pairs
    again = append_candidates(
        dest,
        pd.DataFrame({"asset_id": ["1", "2"], "symbol": ["AAA", "BBB"]}),
    )
    assert again == 1
    assert dest.read_text(encoding="utf-8").count("1,AAA") == 1


def test_lifetime_max_dollar_volume_across_years(tmp_path: Path) -> None:
    ohlcv = tmp_path / "ohlcv" / "1d"
    ohlcv.mkdir(parents=True)
    y1 = pd.DataFrame(
        {
            "asset_id": ["1", "1", "2"],
            "close": [10.0, 10.0, 10.0],
            "volume": [100_000.0, 250_000.0, 10.0],
        }
    )
    y2 = pd.DataFrame(
        {
            "asset_id": ["1", "2"],
            "close": [10.0, 20.0],
            "volume": [100_000.0, 200_000.0],
        }
    )
    write_parquet_atomic(ohlcv / "2015.parquet", y1)
    write_parquet_atomic(ohlcv / "2016.parquet", y2)
    mx = lifetime_max_dollar_volume(ohlcv)
    assert mx["1"] == pytest.approx(2_500_000.0)
    assert mx["2"] == pytest.approx(4_000_000.0)


def test_seed_refuses_low_delisted_fraction(tmp_path: Path) -> None:
    ohlcv = tmp_path / "ohlcv" / "1d"
    ohlcv.mkdir(parents=True)
    write_parquet_atomic(
        ohlcv / "2015.parquet",
        pd.DataFrame(
            {
                "asset_id": ["1"],
                "close": [10.0],
                "volume": [300_000.0],
            }
        ),
    )
    dest = tmp_path / "candidates.txt"
    with pytest.raises(ScoutDataError, match="delisted fraction"):
        seed_candidates(_tickers().loc[_tickers()["asset_id"] == "1"], ohlcv, dest)


def test_cli_seed_universe_appends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("SCOUT_"):
            monkeypatch.delenv(key, raising=False)
    raw = tmp_path / "raw"
    ohlcv = raw / "ohlcv" / "1d"
    ohlcv.mkdir(parents=True)
    tickers = _tickers()
    write_parquet_atomic(raw / "tickers.parquet", tickers)
    ids = ["1", "2", "3", "4", "8", "10"]
    write_parquet_atomic(
        ohlcv / "2015.parquet",
        pd.DataFrame(
            {
                "asset_id": ids,
                "close": [10.0] * len(ids),
                "volume": [300_000.0] * len(ids),
            }
        ),
    )
    dest = tmp_path / "universe_candidates.txt"
    dest.write_text("# Append-only\n", encoding="utf-8")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"""
data:
  vendor: fixture
  raw_dir: {raw.as_posix()}
  snapshot_id_path: {(raw / "SNAPSHOT.json").as_posix()}
universe:
  candidates_file: {dest.as_posix()}
""",
        encoding="utf-8",
    )
    code = main(["seed-universe", "--config", str(overlay)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "selected=" in captured.out
    pairs = parse_candidate_lines(dest.read_text(encoding="utf-8"))
    symbols = {sym for _, sym in pairs}
    assert "AAA" in symbols
    assert "DEAD" in symbols
    assert "ADR" not in symbols
    assert ALLOWED_EXCHANGES
    assert "ETF" in ALLOWED_CATEGORIES
