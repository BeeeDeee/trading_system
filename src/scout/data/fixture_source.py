"""Deterministic equity ingest source. Used when `data.vendor` is `fixture`."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime

import exchange_calendars as xcals  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from scout.data.schemas import (
    ACTIONS_COLUMNS,
    EARNINGS_COLUMNS,
    OHLCV_COLUMNS,
    TICKERS_COLUMNS,
)
from scout.domain.enums import ActionType

_SECTORS = (
    "Information Technology",
    "Financials",
    "Health Care",
    "Energy",
    "Consumer Discretionary",
)
_GAP_ASSET_ID = "FIX0003"
_GAP_START = date(2015, 6, 1)
_GAP_END = date(2015, 6, 5)
_SPLIT_ASSET_ID = "FIX0005"
_SPLIT_EX = date(2015, 8, 3)
_DIVIDEND_ASSET_ID = "FIX0002"
_DIVIDEND_EX = date(2015, 11, 12)
_DIVIDEND_CASH = 0.42


def _sessions(start: date, end: date) -> list[date]:
    cal = xcals.get_calendar("XNYS")
    index = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [ts.date() for ts in index]


def _asset_id(index: int) -> str:
    return f"FIX{index:04d}"


class FixtureEquitySource:
    """Synthetic unadjusted bars, actions, earnings, and delisted names."""

    vendor_name = "fixture"

    def __init__(
        self,
        *,
        n_listed: int = 50,
        n_delisted: int = 8,
        seed: int = 20260827,
    ) -> None:
        if n_listed < 1:
            raise ValueError("n_listed must be >= 1")
        if n_delisted < 0:
            raise ValueError("n_delisted must be >= 0")
        self.n_listed = n_listed
        self.n_delisted = n_delisted
        self.seed = seed

    def fetch_tickers(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for i in range(self.n_listed):
            is_etf = i < 2
            rows.append(
                {
                    "asset_id": _asset_id(i),
                    "symbol": "SPY" if i == 0 else ("IWM" if i == 1 else f"L{i:02d}"),
                    "exchange": "NYSEARCA" if is_etf else ("NYSE" if i % 2 == 0 else "NASDAQ"),
                    "category": "ETF" if is_etf else "Domestic Common Stock",
                    "sector": "Other" if is_etf else _SECTORS[i % len(_SECTORS)],
                    "is_etf": is_etf,
                    "listed_date": date(2010, 1, 4),
                    "delisted_date": pd.NaT,
                    "delist_reason": "",
                }
            )
        for j in range(self.n_delisted):
            i = self.n_listed + j
            rows.append(
                {
                    "asset_id": _asset_id(i),
                    "symbol": f"D{j:02d}",
                    "exchange": "NYSE" if j % 2 == 0 else "NASDAQ",
                    "category": "Domestic Common Stock",
                    "sector": _SECTORS[j % len(_SECTORS)],
                    "is_etf": False,
                    "listed_date": date(2008, 3, 3),
                    "delisted_date": date(2016, 3, 15 + min(j, 10)),
                    "delist_reason": "BANKRUPTCY" if j == 0 else "OTHER",
                }
            )
        return pd.DataFrame(rows, columns=list(TICKERS_COLUMNS))

    def fetch_ohlcv(
        self,
        asset_ids: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        tickers = self.fetch_tickers()
        wanted = set(asset_ids)
        tickers = tickers.loc[tickers["asset_id"].isin(wanted)]
        sessions = _sessions(start.date(), end.date())
        frames: list[pd.DataFrame] = []
        for row in tickers.itertuples(index=False):
            last = row.delisted_date
            usable = [
                s
                for s in sessions
                if (pd.isna(last) or s <= last)
                and not (
                    row.asset_id == _GAP_ASSET_ID and _GAP_START <= s <= _GAP_END
                )
            ]
            if not usable:
                continue
            frames.append(
                self._bars(
                    asset_id=str(row.asset_id),
                    symbol=str(row.symbol),
                    sessions=usable,
                )
            )
        if not frames:
            return pd.DataFrame(columns=list(OHLCV_COLUMNS))
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["session", "asset_id"], kind="mergesort").reset_index(
            drop=True
        )

    def fetch_actions(
        self,
        asset_ids: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        start_d, end_d = start.date(), end.date()
        if _SPLIT_ASSET_ID in asset_ids and start_d <= _SPLIT_EX <= end_d:
            rows.append(
                {
                    "asset_id": _SPLIT_ASSET_ID,
                    "ex_date": _SPLIT_EX,
                    "action_type": ActionType.SPLIT.value,
                    "split_ratio": 2.0,
                    "cash_amount": 0.0,
                    "new_symbol": "",
                }
            )
        if _DIVIDEND_ASSET_ID in asset_ids and start_d <= _DIVIDEND_EX <= end_d:
            rows.append(
                {
                    "asset_id": _DIVIDEND_ASSET_ID,
                    "ex_date": _DIVIDEND_EX,
                    "action_type": ActionType.DIVIDEND.value,
                    "split_ratio": 1.0,
                    "cash_amount": _DIVIDEND_CASH,
                    "new_symbol": "",
                }
            )
        if not rows:
            return pd.DataFrame(columns=list(ACTIONS_COLUMNS))
        return pd.DataFrame(rows, columns=list(ACTIONS_COLUMNS))

    def fetch_earnings(
        self,
        asset_ids: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        tickers = self.fetch_tickers()
        wanted = set(asset_ids)
        stocks = tickers.loc[
            tickers["asset_id"].isin(wanted) & ~tickers["is_etf"].astype(bool)
        ]
        dates = [
            date(2015, 4, 23),
            date(2015, 7, 23),
            date(2015, 10, 22),
            date(2016, 1, 28),
            date(2016, 4, 21),
            date(2016, 7, 21),
            date(2016, 10, 20),
        ]
        start_d, end_d = start.date(), end.date()
        rows: list[dict[str, object]] = []
        for row in stocks.itertuples(index=False):
            last = row.delisted_date
            for earn in dates:
                if earn < start_d or earn > end_d:
                    continue
                if not pd.isna(last) and earn > last:
                    continue
                available = datetime(earn.year, earn.month, earn.day, 21, 0, tzinfo=UTC)
                rows.append(
                    {
                        "asset_id": row.asset_id,
                        "earnings_date": earn,
                        "is_confirmed": True,
                        "available_ts": available,
                        "timing": "AMC",
                    }
                )
        if not rows:
            return pd.DataFrame(columns=list(EARNINGS_COLUMNS))
        return pd.DataFrame(rows, columns=list(EARNINGS_COLUMNS))

    def _bars(self, asset_id: str, symbol: str, sessions: list[date]) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + int(asset_id.removeprefix("FIX")))
        n = len(sessions)
        rets = rng.normal(0.0003, 0.012, size=n)
        close = 40.0 * np.exp(np.cumsum(rets))
        open_ = close * (1.0 + rng.normal(0.0, 0.002, size=n))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, size=n)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, size=n)))
        volume = rng.uniform(200_000.0, 2_000_000.0, size=n)
        dividend = np.zeros(n)
        split_ratio = np.ones(n)
        for i, sess in enumerate(sessions):
            if asset_id == _DIVIDEND_ASSET_ID and sess == _DIVIDEND_EX:
                dividend[i] = _DIVIDEND_CASH
            if asset_id == _SPLIT_ASSET_ID and sess == _SPLIT_EX:
                split_ratio[i] = 2.0
        return pd.DataFrame(
            {
                "asset_id": asset_id,
                "symbol": symbol,
                "session": sessions,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "dividend": dividend,
                "split_ratio": split_ratio,
            }
        )
