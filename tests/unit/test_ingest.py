from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from scout.cli.__main__ import main
from scout.config.loader import load_config
from scout.config.schema import ScoutConfig
from scout.data.fixture_source import (
    _DIVIDEND_ASSET_ID,
    _GAP_ASSET_ID,
    _SPLIT_ASSET_ID,
    FixtureEquitySource,
)
from scout.data.ingest import (
    MIN_DELISTED_FRACTION,
    build_source,
    delisted_fraction,
    require_min_delisted_fraction,
    run_ingest,
)
from scout.data.norgate_source import (
    NorgateCandleSource,
    entitlement_to_ex_date,
    infer_split_ratio,
)
from scout.data.schemas import (
    ACTIONS_COLUMNS,
    EARNINGS_COLUMNS,
    OHLCV_COLUMNS,
    SNAPSHOT_KEYS,
    TICKERS_COLUMNS,
)
from scout.data.sharadar_source import SharadarCandleSource, impute_unadjusted_ohlcv
from scout.data.store import write_json_atomic, write_parquet_atomic
from scout.utils.clock import BarClock
from scout.utils.errors import ScoutConfigError, ScoutDataError

FIXED_TS = datetime(2016, 6, 15, 22, 14, 3, tzinfo=UTC)


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("SCOUT_"):
            monkeypatch.delenv(key, raising=False)


def _overlay(tmp_path: Path, raw_dir: Path, text: str = "") -> Path:
    path = tmp_path / "overlay.yaml"
    body = f"""
data:
  vendor: fixture
  raw_dir: {raw_dir.as_posix()}
  snapshot_id_path: {(raw_dir / "SNAPSHOT.json").as_posix()}
{text}
"""
    path.write_text(body, encoding="utf-8")
    return path


def _run(
    tmp_path: Path,
    *,
    start: datetime = datetime(2015, 1, 1, tzinfo=UTC),
    end: datetime = datetime(2016, 12, 31, tzinfo=UTC),
    source: FixtureEquitySource | None = None,
    symbols_file: Path | None = None,
) -> tuple[ScoutConfig, Path]:
    raw_dir = tmp_path / "raw"
    cfg = load_config(_overlay(tmp_path, raw_dir))
    run_ingest(
        cfg,
        start=start,
        end=end,
        raw_dir=raw_dir,
        snapshot_path=raw_dir / "SNAPSHOT.json",
        clock=BarClock(FIXED_TS),
        source=source or FixtureEquitySource(),
        symbols_file=symbols_file,
    )
    return cfg, raw_dir


def test_infer_split_ratio_two_for_one() -> None:
    # Unadjusted halves; capital-adjusted is continuous.
    assert infer_split_ratio(100.0, 50.0, 50.0, 50.0) == pytest.approx(2.0)


def test_entitlement_to_ex_date_skips_weekend() -> None:
    assert entitlement_to_ex_date(date(2016, 1, 8)) == date(2016, 1, 11)


def test_entitlement_to_ex_date_skips_mlk_day() -> None:
    assert entitlement_to_ex_date(date(2016, 1, 15)) == date(2016, 1, 19)


def test_ingest_fifty_symbols_five_delisted_2015_2016(
    tmp_path: Path, isolated_env: None
) -> None:
    _, raw_dir = _run(tmp_path)
    tickers = pd.read_parquet(raw_dir / "tickers.parquet")
    ohlcv_2015 = pd.read_parquet(raw_dir / "ohlcv" / "1d" / "2015.parquet")
    ohlcv_2016 = pd.read_parquet(raw_dir / "ohlcv" / "1d" / "2016.parquet")
    ohlcv = pd.concat([ohlcv_2015, ohlcv_2016], ignore_index=True)
    snapshot = json.loads((raw_dir / "SNAPSHOT.json").read_text(encoding="utf-8"))

    assert len(tickers) >= 50
    assert int(tickers["delisted_date"].notna().sum()) >= 5
    assert set(ohlcv["session"].map(lambda d: pd.Timestamp(d).year)) == {2015, 2016}
    assert ohlcv["asset_id"].nunique() >= 50
    assert list(ohlcv.columns) == list(OHLCV_COLUMNS)
    assert list(tickers.columns) == list(TICKERS_COLUMNS)
    assert list(snapshot) == list(SNAPSHOT_KEYS)
    assert snapshot["data_snapshot_id"] == "20160615-fixture"
    assert snapshot["vendor"] == "fixture"
    assert snapshot["created_utc"] == "2016-06-15T22:14:03Z"


def test_ingest_writes_actions_and_earnings(tmp_path: Path, isolated_env: None) -> None:
    _, raw_dir = _run(tmp_path)
    actions = pd.read_parquet(raw_dir / "actions.parquet")
    earnings = pd.read_parquet(raw_dir / "earnings.parquet")
    assert list(actions.columns) == list(ACTIONS_COLUMNS)
    assert list(earnings.columns) == list(EARNINGS_COLUMNS)
    assert not actions.empty
    assert not earnings.empty
    assert _SPLIT_ASSET_ID in set(actions["asset_id"])
    assert _DIVIDEND_ASSET_ID in set(actions["asset_id"])


def test_ingest_does_not_fill_gap(tmp_path: Path, isolated_env: None) -> None:
    _, raw_dir = _run(tmp_path)
    bars = pd.read_parquet(raw_dir / "ohlcv" / "1d" / "2015.parquet")
    symbol = bars.loc[bars["asset_id"] == _GAP_ASSET_ID]
    sessions = set(pd.to_datetime(symbol["session"]).dt.date)
    assert date(2015, 6, 1) not in sessions
    assert date(2015, 6, 5) not in sessions
    assert date(2015, 5, 29) in sessions
    assert date(2015, 6, 8) in sessions


def test_rerun_is_idempotent(tmp_path: Path, isolated_env: None) -> None:
    raw_dir = tmp_path / "raw"
    cfg = load_config(_overlay(tmp_path, raw_dir))
    kwargs = dict(
        start=datetime(2015, 1, 1, tzinfo=UTC),
        end=datetime(2016, 12, 31, tzinfo=UTC),
        raw_dir=raw_dir,
        snapshot_path=raw_dir / "SNAPSHOT.json",
        clock=BarClock(FIXED_TS),
        source=FixtureEquitySource(),
    )
    run_ingest(cfg, **kwargs)
    first = {
        p.relative_to(raw_dir): p.read_bytes()
        for p in raw_dir.rglob("*")
        if p.is_file()
    }
    run_ingest(cfg, **kwargs)
    second = {
        p.relative_to(raw_dir): p.read_bytes()
        for p in raw_dir.rglob("*")
        if p.is_file()
    }
    assert first == second


def test_snapshot_written_last(
    tmp_path: Path, isolated_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    orig_pq = write_parquet_atomic
    orig_js = write_json_atomic

    def spy_pq(path: Path, frame: pd.DataFrame) -> None:
        order.append(path.name)
        orig_pq(path, frame)

    def spy_js(path: Path, payload: dict[str, object]) -> None:
        order.append(path.name)
        orig_js(path, payload)

    monkeypatch.setattr("scout.data.ingest.write_parquet_atomic", spy_pq)
    monkeypatch.setattr("scout.data.ingest.write_json_atomic", spy_js)
    _run(tmp_path)
    assert order[-1] == "SNAPSHOT.json"
    assert "tickers.parquet" in order
    assert "actions.parquet" in order
    assert "earnings.parquet" in order
    assert "2015.parquet" in order


def test_failed_replace_leaves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bars.parquet"
    first = pd.DataFrame({"x": [1.0]})
    write_parquet_atomic(path, first)
    original = path.read_bytes()

    def boom(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("killed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="killed"):
        write_parquet_atomic(path, pd.DataFrame({"x": [2.0]}))
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []


def test_kill_on_first_write_leaves_no_dest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bars.parquet"

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("killed")

    monkeypatch.setattr(pq, "write_table", boom)
    with pytest.raises(OSError, match="killed"):
        write_parquet_atomic(path, pd.DataFrame({"x": [1.0]}))
    assert not path.exists()


def test_delisted_fraction_refuses_below_fifteen_percent() -> None:
    candidates = pd.DataFrame(
        {"asset_id": [f"A{i}" for i in range(20)], "symbol": [f"S{i}" for i in range(20)]}
    )
    tickers = pd.DataFrame(
        {
            "asset_id": [f"A{i}" for i in range(20)],
            "delisted_date": [date(2016, 1, 1)] * 2 + [pd.NaT] * 18,
        }
    )
    assert delisted_fraction(candidates, tickers) == pytest.approx(0.10)
    with pytest.raises(ScoutDataError, match="15%"):
        require_min_delisted_fraction(candidates, tickers)


def test_delisted_fraction_accepts_threshold() -> None:
    candidates = pd.DataFrame(
        {"asset_id": [f"A{i}" for i in range(20)], "symbol": [f"S{i}" for i in range(20)]}
    )
    tickers = pd.DataFrame(
        {
            "asset_id": [f"A{i}" for i in range(20)],
            "delisted_date": [date(2016, 1, 1)] * 3 + [pd.NaT] * 17,
        }
    )
    assert require_min_delisted_fraction(candidates, tickers) == pytest.approx(0.15)
    assert MIN_DELISTED_FRACTION == 0.15


def test_build_universe_refuses_empty_candidate_list(
    tmp_path: Path, isolated_env: None, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_dir = tmp_path / "raw"
    _run(tmp_path)
    candidates = tmp_path / "candidates.txt"
    candidates.write_text("# none\n", encoding="utf-8")
    overlay = _overlay(
        tmp_path,
        raw_dir,
        f"""
universe:
  candidates_file: {candidates.as_posix()}
""",
    )
    assert main(["build-universe", "--config", str(overlay)]) == 1
    assert "delisted fraction" in capsys.readouterr().err


def test_cli_ingest_smoke(tmp_path: Path, isolated_env: None) -> None:
    raw_dir = tmp_path / "raw"
    overlay = _overlay(tmp_path, raw_dir)
    code = main(
        ["ingest", "--config", str(overlay), "--start", "2015-01-01", "--end", "2015-03-31"]
    )
    assert code == 0
    assert (raw_dir / "SNAPSHOT.json").is_file()
    assert (raw_dir / "ohlcv" / "1d" / "2015.parquet").is_file()


def test_build_source_rejects_unimplemented_vendor() -> None:
    cfg = ScoutConfig.model_validate({"data": {"vendor": "polygon"}})
    with pytest.raises(ScoutConfigError, match="polygon"):
        build_source(cfg)


def test_build_source_sharadar_requires_api_key(isolated_env: None) -> None:
    cfg = ScoutConfig.model_validate({"data": {"vendor": "sharadar"}})
    with pytest.raises(ScoutConfigError, match="SCOUT_SHARADAR_API_KEY"):
        build_source(cfg)


def test_norgate_earnings_are_empty() -> None:
    source = NorgateCandleSource(client=_FakeNorgate())
    earnings = source.fetch_earnings(
        ["1"],
        datetime(2015, 1, 1, tzinfo=UTC),
        datetime(2016, 12, 31, tzinfo=UTC),
    )
    assert earnings.empty
    assert list(earnings.columns) == list(EARNINGS_COLUMNS)


def test_norgate_maps_unadjusted_ohlcv_and_shifted_dividend() -> None:
    source = NorgateCandleSource(client=_FakeNorgate())
    tickers = source.fetch_tickers()
    assert set(tickers["asset_id"]) == {"100", "200"}
    assert int(tickers["delisted_date"].notna().sum()) == 1
    ohlcv = source.fetch_ohlcv(
        ["100"],
        datetime(2016, 1, 7, tzinfo=UTC),
        datetime(2016, 1, 12, tzinfo=UTC),
    )
    assert not ohlcv.empty
    assert list(ohlcv.columns) == list(OHLCV_COLUMNS)
    actions = source.fetch_actions(
        ["100"],
        datetime(2016, 1, 7, tzinfo=UTC),
        datetime(2016, 1, 12, tzinfo=UTC),
    )
    divs = actions.loc[actions["action_type"] == "DIVIDEND"]
    assert list(divs["ex_date"]) == [date(2016, 1, 11)]
    assert list(divs["cash_amount"]) == [pytest.approx(0.25)]
    splits = actions.loc[actions["action_type"] == "SPLIT"]
    assert list(splits["ex_date"]) == [date(2016, 1, 11)]
    assert list(splits["split_ratio"]) == [pytest.approx(2.0)]


class _FakeNorgate:
    def status(self) -> bool:
        return True

    def database(self, databasename: str) -> list[dict[str, object]]:
        if databasename == "US Equities":
            return [{"symbol": "AAA", "assetid": 100}]
        if databasename == "US Equities Delisted":
            return [{"symbol": "BBB", "assetid": 200}]
        return []

    def price_timeseries(self, symbol: object, **kwargs: object) -> pd.DataFrame:
        adj = kwargs.get("stock_price_adjustment_setting")
        sessions = [date(2016, 1, 7), date(2016, 1, 8), date(2016, 1, 11)]
        close = [50.0, 50.0, 51.0] if adj == "CAPITAL" else [100.0, 100.0, 51.0]
        return pd.DataFrame(
            {
                "Date": sessions,
                "Open": close,
                "High": [c + 1 for c in close],
                "Low": [c - 1 for c in close],
                "Close": close,
                "Volume": [1000.0, 1100.0, 1200.0],
                "Dividend": [0.0, 0.25, 0.0],
            }
        )

    def capital_event_timeseries(self, symbol: object, **kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            {"Date": [date(2016, 1, 8)], "Capital Event": [1.0]}
        )

    def exchange_name(self, symbol: object) -> str:
        return "NYSE"

    def subtype1(self, symbol: object) -> str:
        return "Equity"

    def subtype2(self, symbol: object) -> str:
        return "Operating/Holding Company"

    def first_quoted_date(self, symbol: object, **kwargs: object) -> date:
        return date(2010, 1, 4)

    def last_quoted_date(self, symbol: object, **kwargs: object) -> date | None:
        return date(2016, 3, 1) if int(symbol) == 200 else None

    def classification_at_level(self, symbol: object, *args: object, **kwargs: object) -> str:
        return "Financials"


def test_impute_unadjusted_ohlcv_two_for_one() -> None:
    frame = pd.DataFrame(
        {
            "open": [49.0],
            "high": [51.0],
            "low": [48.0],
            "close": [50.0],
            "volume": [2000.0],
            "closeunadj": [100.0],
        }
    )
    out = impute_unadjusted_ohlcv(frame)
    assert out["close"].iloc[0] == pytest.approx(100.0)
    assert out["open"].iloc[0] == pytest.approx(98.0)
    assert out["volume"].iloc[0] == pytest.approx(1000.0)


def test_sharadar_maps_tickers_actions_earnings() -> None:
    source = SharadarCandleSource(client=_FakeSharadar())
    tickers = source.fetch_tickers()
    assert set(tickers["asset_id"]) == {"199059", "200001"}
    assert int(tickers["delisted_date"].notna().sum()) == 1
    spy = tickers.loc[tickers["symbol"] == "SPY"].iloc[0]
    assert bool(spy["is_etf"]) is True
    ohlcv = source.fetch_ohlcv(
        ["199059"],
        datetime(2020, 8, 28, tzinfo=UTC),
        datetime(2020, 9, 1, tzinfo=UTC),
    )
    assert ohlcv["close"].iloc[0] == pytest.approx(100.0)
    assert ohlcv["volume"].iloc[0] == pytest.approx(1000.0)
    actions = source.fetch_actions(
        ["199059"],
        datetime(2020, 8, 1, tzinfo=UTC),
        datetime(2020, 9, 30, tzinfo=UTC),
    )
    splits = actions.loc[actions["action_type"] == "SPLIT"]
    assert splits["split_ratio"].iloc[0] == pytest.approx(4.0)
    assert splits["ex_date"].iloc[0] == date(2020, 8, 31)
    earnings = source.fetch_earnings(
        ["199059"],
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 12, 31, tzinfo=UTC),
    )
    assert earnings["earnings_date"].iloc[0] == date(2020, 7, 30)
    assert pd.isna(earnings["available_ts"].iloc[0])


def test_sharadar_accepts_legacy_sep_sfp_table_codes() -> None:
    """Live tickers.table uses Nasdaq codes SEP/SFP, not stocks/funds."""

    class _LegacyTables:
        def load_table(self, name: str) -> pd.DataFrame:
            if name == "tickers":
                return pd.DataFrame(
                    {
                        "table": ["SEP", "SF1", "SFP"],
                        "permaticker": ["199059", "199059", "200001"],
                        "ticker": ["AAPL", "AAPL", "SPY"],
                        "exchange": ["NASDAQ", "NASDAQ", "NYSEARCA"],
                        "isdelisted": ["N", "N", "N"],
                        "category": [
                            "Domestic Common Stock",
                            "Domestic Common Stock",
                            "ETF",
                        ],
                        "sector": ["Technology", "Technology", "Other"],
                        "firstpricedate": ["1986-01-01", "1986-01-01", "1993-01-29"],
                        "lastpricedate": ["2026-08-28", "2026-08-28", "2026-08-28"],
                    }
                )
            raise KeyError(name)

    tickers = SharadarCandleSource(client=_LegacyTables()).fetch_tickers()
    assert set(tickers["asset_id"]) == {"199059", "200001"}
    assert set(tickers["symbol"]) == {"AAPL", "SPY"}


class _FakeSharadar:
    def load_table(self, name: str) -> pd.DataFrame:
        if name == "tickers":
            return pd.DataFrame(
                {
                    "table": ["stocks", "fundamentals", "funds"],
                    "permaticker": ["199059", "199059", "200001"],
                    "ticker": ["AAPL", "AAPL", "SPY"],
                    "exchange": ["NASDAQ", "NASDAQ", "NYSEARCA"],
                    "isdelisted": ["N", "N", "Y"],
                    "category": ["Domestic Common Stock", "Domestic Common Stock", "ETF"],
                    "sector": ["Technology", "Technology", "Other"],
                    "firstpricedate": ["1986-01-01", "1986-01-01", "1993-01-29"],
                    "lastpricedate": ["2026-08-28", "2026-08-28", "2016-03-01"],
                }
            )
        if name == "stocks":
            return pd.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "date": ["2020-08-31"],
                    "open": [49.0],
                    "high": [51.0],
                    "low": [48.0],
                    "close": [50.0],
                    "volume": [2000.0],
                    "closeunadj": [100.0],
                }
            )
        if name == "funds":
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "closeunadj",
                ]
            )
        if name == "actions":
            return pd.DataFrame(
                {
                    "date": ["2020-08-31", "2020-08-07"],
                    "action": ["split", "dividend"],
                    "ticker": ["AAPL", "AAPL"],
                    "value": [4.0, 0.82],
                    "contraticker": [None, None],
                }
            )
        if name == "events":
            return pd.DataFrame(
                {
                    "ticker": ["AAPL", "AAPL"],
                    "date": ["2020-07-30", "2020-04-20"],
                    "eventcodes": ["22|91", "34"],
                }
            )
        raise KeyError(name)
