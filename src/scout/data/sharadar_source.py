"""Sharadar ingest adapter. Offline only; never called inside the decision loop."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd  # type: ignore[import-untyped]

from scout.data.schemas import (
    ACTIONS_COLUMNS,
    EARNINGS_COLUMNS,
    OHLCV_COLUMNS,
    TICKERS_COLUMNS,
)
from scout.domain.enums import ActionType

# Bulk API names vs tickers.table values (Nasdaq codes SEP/SFP on the live table).
_PRICE_API_TABLES = ("stocks", "funds")
_PRICE_TICKER_TABLES = frozenset({"stocks", "funds", "sep", "sfp"})
_EARNINGS_EVENT_CODE = "22"  # SEC 8-K Item 2.02; Sharadar uses pipe-separated codes

_ACTION_MAP: dict[str, ActionType] = {
    "split": ActionType.SPLIT,
    "dividend": ActionType.DIVIDEND,
    "spinoff": ActionType.SPINOFF,
    "merger": ActionType.MERGER,
    "acquisition": ActionType.MERGER,
    "tickerchangefrom": ActionType.TICKER_CHANGE,
    "tickerchangeto": ActionType.TICKER_CHANGE,
}

_EXCHANGE_MAP = {
    "NYSE": "NYSE",
    "NASDAQ": "NASDAQ",
    "NYSEARCA": "NYSEARCA",
    "BATS": "BATS",
    "NYSEMKT": "NYSE",
    "NYSE MKT": "NYSE",
}


@runtime_checkable
class SharadarClient(Protocol):
    """Bulk table loader. Production uses httpx inside scout.data.ingest."""

    def load_table(self, name: str) -> pd.DataFrame: ...


def impute_unadjusted_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Sharadar open/high/low/close/volume are split-adjusted; closeunadj is not.

    Unadjusted OHLC = split-adjusted * (closeunadj / close).
    Unadjusted volume is the inverse, so share counts stay in raw units.
    Rows with close==0 or a missing closeunadj are dropped, not filled.
    """
    close = frame["close"].astype("float64")
    unadj = frame["closeunadj"].astype("float64")
    valid = (close != 0.0) & close.notna() & unadj.notna()
    out = frame.loc[valid].copy()
    factor = out["closeunadj"].astype("float64") / out["close"].astype("float64")
    return out.assign(
        open=out["open"].astype("float64") * factor,
        high=out["high"].astype("float64") * factor,
        low=out["low"].astype("float64") * factor,
        close=out["closeunadj"].astype("float64"),
        volume=out["volume"].astype("float64") / factor,
    )


class SharadarCandleSource:
    vendor_name = "sharadar"

    def __init__(self, client: SharadarClient) -> None:
        self._client = client
        self._tables: dict[str, pd.DataFrame] = {}

    def _table(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            self._tables[name] = self._client.load_table(name)
        return self._tables[name]

    def fetch_tickers(self) -> pd.DataFrame:
        raw = self._table("tickers")
        if raw.empty:
            return pd.DataFrame(columns=list(TICKERS_COLUMNS))
        frame = raw.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        if "table" in frame.columns:
            frame = frame.loc[
                frame["table"].astype(str).str.lower().isin(_PRICE_TICKER_TABLES)
            ]
        if frame.empty:
            return pd.DataFrame(columns=list(TICKERS_COLUMNS))
        frame = frame.drop_duplicates(subset=["permaticker"], keep="first")
        is_delisted = frame["isdelisted"].astype(str).str.upper().eq("Y")
        last = pd.to_datetime(frame["lastpricedate"], errors="coerce")
        category = frame["category"].fillna("").astype(str)
        is_etf = category.str.upper().eq("ETF") | category.str.contains(
            "Exchange Traded Fund", case=False, regex=False
        )
        return pd.DataFrame(
            {
                "asset_id": frame["permaticker"].astype(str),
                "symbol": frame["ticker"].astype(str),
                "exchange": frame["exchange"].map(_map_exchange),
                "category": category,
                "sector": frame["sector"].fillna("").astype(str),
                "is_etf": is_etf,
                "listed_date": pd.to_datetime(frame["firstpricedate"], errors="coerce").dt.date,
                "delisted_date": last.where(is_delisted).dt.date,
                "delist_reason": "",
            }
        ).reset_index(drop=True)

    def fetch_ohlcv(
        self,
        asset_ids: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        wanted = set(asset_ids)
        id_map = self._ticker_to_asset()
        frames: list[pd.DataFrame] = []
        for table in _PRICE_API_TABLES:
            raw = self._table(table)
            if raw.empty:
                continue
            part = raw.copy()
            part.columns = [str(c).lower() for c in part.columns]
            part = part.assign(asset_id=part["ticker"].astype(str).map(id_map))
            part = part.loc[part["asset_id"].isin(wanted)]
            if part.empty:
                continue
            part = impute_unadjusted_ohlcv(part)
            part = part.assign(session=pd.to_datetime(part["date"], errors="coerce").dt.date)
            part = part.loc[
                (part["session"] >= start.date()) & (part["session"] <= end.date())
            ]
            part = part.assign(
                symbol=part["ticker"].astype(str),
                dividend=0.0,
                split_ratio=1.0,
            )
            frames.append(
                part.loc[
                    :,
                    [
                        "asset_id",
                        "symbol",
                        "session",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "dividend",
                        "split_ratio",
                    ],
                ]
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
        raw = self._table("actions")
        if raw.empty:
            return pd.DataFrame(columns=list(ACTIONS_COLUMNS))
        frame = raw.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        id_map = self._ticker_to_asset()
        frame = frame.assign(asset_id=frame["ticker"].astype(str).map(id_map))
        wanted = set(asset_ids)
        frame = frame.loc[frame["asset_id"].isin(wanted)]
        mapped = frame["action"].astype(str).str.lower().map(_ACTION_MAP)
        frame = frame.assign(action_type=mapped)
        frame = frame.loc[frame["action_type"].notna()]
        if frame.empty:
            return pd.DataFrame(columns=list(ACTIONS_COLUMNS))
        frame = frame.assign(ex_date=pd.to_datetime(frame["date"], errors="coerce").dt.date)
        frame = frame.loc[
            (frame["ex_date"] >= start.date()) & (frame["ex_date"] <= end.date())
        ]
        records: list[dict[str, object]] = []
        for row in frame.itertuples(index=False):
            atype: ActionType = row.action_type
            val = 0.0 if pd.isna(row.value) else float(row.value)
            contra_raw = getattr(row, "contraticker", "")
            contra = "" if pd.isna(contra_raw) else str(contra_raw)
            split_ratio = 1.0
            cash = 0.0
            new_symbol = ""
            if atype is ActionType.SPLIT:
                split_ratio = val if val != 0.0 else 1.0
            elif atype is ActionType.DIVIDEND:
                cash = val
            elif atype is ActionType.SPINOFF:
                split_ratio = val if val != 0.0 else 1.0
                new_symbol = contra
            elif atype is ActionType.TICKER_CHANGE:
                new_symbol = contra
            else:
                new_symbol = contra
            records.append(
                {
                    "asset_id": str(row.asset_id),
                    "ex_date": row.ex_date,
                    "action_type": atype.value,
                    "split_ratio": split_ratio,
                    "cash_amount": cash,
                    "new_symbol": new_symbol,
                }
            )
        if not records:
            return pd.DataFrame(columns=list(ACTIONS_COLUMNS))
        return pd.DataFrame(records, columns=list(ACTIONS_COLUMNS))

    def fetch_earnings(
        self,
        asset_ids: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        raw = self._table("events")
        if raw.empty:
            return pd.DataFrame(columns=list(EARNINGS_COLUMNS))
        frame = raw.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        id_map = self._ticker_to_asset()
        frame = frame.assign(asset_id=frame["ticker"].astype(str).map(id_map))
        wanted = set(asset_ids)
        frame = frame.loc[frame["asset_id"].isin(wanted)]
        codes = frame["eventcodes"].fillna("").astype(str)
        frame = frame.loc[codes.map(_has_earnings_code)]
        if frame.empty:
            return pd.DataFrame(columns=list(EARNINGS_COLUMNS))
        frame = frame.assign(
            earnings_date=pd.to_datetime(frame["date"], errors="coerce").dt.date
        )
        frame = frame.loc[
            (frame["earnings_date"] >= start.date())
            & (frame["earnings_date"] <= end.date())
        ]
        # Sharadar does not supply an as-of. Leave available_ts null so the
        # earnings gate uses the conservative uncertainty window (ADR-020).
        if frame.empty:
            return pd.DataFrame(columns=list(EARNINGS_COLUMNS))
        return pd.DataFrame(
            {
                "asset_id": frame["asset_id"].astype(str).to_numpy(),
                "earnings_date": frame["earnings_date"].to_numpy(),
                "is_confirmed": True,
                "available_ts": pd.NaT,
                "timing": "UNKNOWN",
            }
        )

    def _ticker_to_asset(self) -> dict[str, str]:
        tickers = self.fetch_tickers()
        if tickers.empty:
            return {}
        return dict(
            zip(
                tickers["symbol"].astype(str),
                tickers["asset_id"].astype(str),
                strict=True,
            )
        )


def _map_exchange(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    return _EXCHANGE_MAP.get(text, text)


def _has_earnings_code(raw: str) -> bool:
    tokens = [part.strip() for part in raw.replace(",", "|").split("|") if part.strip()]
    return _EARNINGS_EVENT_CODE in tokens
