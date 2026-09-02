"""Shared synthetic fixtures for engine tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from scout.config.hashing import config_hash
from scout.config.schema import (
    AuditConfig,
    EdgeConfig,
    GatesConfig,
    PeriodConfig,
    PortfolioConfig,
    ResearchConfig,
    ScoringConfig,
    ScoutConfig,
    UniverseConfig,
)
from scout.domain.edge import MIN_BIN_SAMPLES, BinKey, BinStats, EdgeTable
from scout.domain.enums import Direction, RejectionReason, VolBucket
from scout.domain.market import (
    BENCHMARK_COLUMNS,
    MARKET_COLUMNS,
    Asset,
    BenchmarkPanel,
    MarketPanel,
)
from scout.domain.universe import UniverseEntry, UniverseSnapshot
from scout.universe.build import SnapshotBook

START = datetime(2015, 1, 5, 21, 0, tzinfo=UTC)


class MemoryCandleSource:
    def __init__(self, panel: MarketPanel) -> None:
        self._panel = panel

    def load_panel(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> MarketPanel:
        del timeframe
        frame = self._panel.frame
        if frame.empty:
            return self._panel
        wanted = {str(s) for s in symbols}
        mask = frame["asset_id"].astype(str).isin(wanted)
        ts = pd.DatetimeIndex(pd.to_datetime(frame["ts"], utc=True))
        mask = mask & (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
        sliced = frame.loc[mask].reset_index(drop=True)
        if sliced.empty:
            return MarketPanel(pd.DataFrame(columns=list(MARKET_COLUMNS)))
        return MarketPanel(sliced)

    def available_range(
        self, symbol: str, timeframe: str
    ) -> tuple[datetime, datetime] | None:
        del timeframe
        frame = self._panel.frame
        hit = frame.loc[frame["asset_id"].astype(str) == symbol]
        if hit.empty:
            return None
        ts = pd.DatetimeIndex(pd.to_datetime(hit["ts"], utc=True))
        return ts.min().to_pydatetime(), ts.max().to_pydatetime()


def timestamps(n: int, *, start: datetime = START) -> list[datetime]:
    return [start + timedelta(days=i) for i in range(n)]


def market_panel(
    symbols: Sequence[str],
    stamps: Sequence[datetime],
    *,
    close_fn: Any | None = None,
) -> MarketPanel:
    records: list[dict[str, object]] = []
    for i, ts in enumerate(stamps):
        for j, symbol in enumerate(symbols):
            close = (
                20.0 + 0.5 * i + 5.0 * j
                if close_fn is None
                else float(close_fn(symbol, i, ts))
            )
            records.append(
                {
                    "asset_id": symbol,
                    "symbol": symbol,
                    "ts": ts,
                    "session_index": i,
                    "open": close,
                    "high": close + 0.4,
                    "low": close - 0.2,
                    "close": close,
                    "close_raw": close,
                    "volume": 1_000_000.0,
                    "dollar_volume": close * 1_000_000.0,
                    "is_suspect": False,
                }
            )
    return MarketPanel(pd.DataFrame(records, columns=list(MARKET_COLUMNS)))


def benchmark_panel(stamps: Sequence[datetime]) -> BenchmarkPanel:
    spy = [200.0 + 0.5 * i for i in range(len(stamps))]
    frame = pd.DataFrame(
        {
            "ts": list(stamps),
            "session_index": list(range(len(stamps))),
            "open": spy,
            "high": [s + 0.5 for s in spy],
            "low": [s - 0.5 for s in spy],
            "close": spy,
            "vix_close": [18.0] * len(stamps),
            "vix9d_close": [17.0] * len(stamps),
            "vix3m_close": [19.0] * len(stamps),
        },
        columns=list(BENCHMARK_COLUMNS),
    )
    return BenchmarkPanel(frame)


def snapshot_frame(
    symbols: Sequence[str],
    ts: datetime,
    *,
    eligible: bool = True,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [ts] * len(symbols),
            "asset_id": list(symbols),
            "symbol": list(symbols),
            "eligible": [eligible] * len(symbols),
            "reason": [""] * len(symbols),
            "adv_usd_60": [5.0e8] * len(symbols),
            "adv_rank": list(range(1, len(symbols) + 1)),
            "spread_bps_est": [1.0] * len(symbols),
            "close_raw": [50.0] * len(symbols),
            "bars_available": [500] * len(symbols),
            "sector": ["ETF_BROAD"] * len(symbols),
            "is_etf": [True] * len(symbols),
        }
    )


def snapshot_book(
    symbols: Sequence[str],
    ts: datetime,
    *,
    eligible: bool = True,
) -> SnapshotBook:
    return SnapshotBook(snapshot_frame(symbols, ts, eligible=eligible))


def assets(symbols: Sequence[str], *, is_etf: bool = True) -> dict[str, Asset]:
    out: dict[str, Asset] = {}
    for symbol in symbols:
        out[symbol] = Asset(
            asset_id=symbol,
            symbol=symbol,
            exchange="NYSEARCA",
            quote_currency="USD",
            is_etf=is_etf,
            cluster="ETF_BROAD",
            tick_size=Decimal("0.01"),
            step_size=Decimal("1"),
            min_notional_usd=Decimal("0"),
            listed_at=None,
            delisted_at=None,
            delist_reason=None,
        )
    return out


def usable_edge_table(cfg: ScoutConfig, as_of: datetime) -> EdgeTable:
    stats: list[BinStats] = []
    for sid in ("xsec_momentum_v1", "donchian_breakout_v1"):
        for direction in (Direction.LONG, Direction.SHORT):
            for vol in (VolBucket.LOW, VolBucket.MID, VolBucket.HIGH):
                stats.append(
                    BinStats(
                        key=BinKey(sid, direction, vol),
                        as_of=as_of,
                        n=MIN_BIN_SAMPLES,
                        mean_r=0.40,
                        std_r=1.0,
                        ev_r_lcb=0.30,
                        mean_bars_held=10.0,
                        win_rate=0.45,
                        avg_win_r=1.5,
                        avg_loss_r=-1.0,
                        target_rate=0.3,
                        stop_rate=0.4,
                        time_rate=0.3,
                        median_r=0.2,
                        p05_r=-1.2,
                        p95_r=2.0,
                    )
                )
    return EdgeTable(stats, config_hash=config_hash(cfg), data_snapshot_id="test")


def write_candidates(path: Path, symbols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{s},{s}" for s in symbols]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def engine_config(
    tmp: Path,
    stamps: Sequence[datetime],
    *,
    warmup_index: int = 400,
) -> ScoutConfig:
    cand = tmp / "universe_candidates.txt"
    write_candidates(cand, ["AAA", "BBB", "CCC"])
    start = stamps[0]
    warmup = stamps[min(warmup_index, len(stamps) - 2)]
    end = stamps[-1]
    return ScoutConfig(
        period=PeriodConfig(start=start, warmup_end=warmup, end=end),
        universe=UniverseConfig(
            candidates_file=str(cand),
            snapshots_path=str(tmp / "snapshots.parquet"),
            min_bars_since_gap=0,
            min_history_bars=20,
        ),
        gates=GatesConfig(min_xs_population=1, skip_hard_to_borrow=True),
        scoring=ScoringConfig(min_ev_net_r=0.05),
        portfolio=PortfolioConfig(
            initial_equity_usd=100_000,
            max_positions=6,
            top_n=3,
        ),
        edge=EdgeConfig(min_bin_samples=MIN_BIN_SAMPLES, table_path=str(tmp / "edge.parquet")),
        audit=AuditConfig(results_dir=str(tmp / "results"), flush_every_cycles=50),
        research=ResearchConfig(registry_path=str(tmp / "registry.csv"), plot_dpi=150),
    )


def ineligible_snapshot(symbols: Sequence[str], ts: datetime) -> UniverseSnapshot:
    entries = {
        s: UniverseEntry(
            symbol=s,
            eligible=False,
            reason=RejectionReason.LOW_LIQUIDITY,
            adv_usd_30=1.0,
            spread_bps_est=50.0,
            bars_available=10,
            listed_days=10.0,
        )
        for s in symbols
    }
    return UniverseSnapshot(ts=ts, entries=entries)
