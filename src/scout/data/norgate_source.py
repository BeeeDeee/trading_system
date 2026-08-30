"""Norgate Data ingest adapter. Offline only; never called inside the decision loop."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

import exchange_calendars as xcals  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from scout.data.schemas import (
    ACTIONS_COLUMNS,
    EARNINGS_COLUMNS,
    OHLCV_COLUMNS,
    TICKERS_COLUMNS,
)
from scout.domain.enums import ActionType
from scout.utils.errors import ScoutDataError

_LOG = logging.getLogger("scout.data.norgate")

_LISTED_DB = "US Equities"
_DELISTED_DB = "US Equities Delisted"

_EXCHANGE_MAP = {
    "NYSE": "NYSE",
    "Nasdaq": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "NYSE Arca": "NYSEARCA",
    "NYSE ARCA": "NYSEARCA",
    "Cboe BZX": "BATS",
    "BATS": "BATS",
    "Cboe BZX Exchange": "BATS",
}


@runtime_checkable
class NorgateClient(Protocol):
    """Narrow surface over the `norgatedata` package, injectable for tests."""

    def status(self) -> bool: ...

    def database(self, databasename: str) -> Any: ...

    def price_timeseries(self, symbol: Any, **kwargs: Any) -> pd.DataFrame: ...

    def capital_event_timeseries(self, symbol: Any, **kwargs: Any) -> pd.DataFrame: ...

    def exchange_name(self, symbol: Any) -> str | None: ...

    def subtype1(self, symbol: Any) -> str | None: ...

    def subtype2(self, symbol: Any) -> str | None: ...

    def first_quoted_date(self, symbol: Any, **kwargs: Any) -> Any: ...

    def last_quoted_date(self, symbol: Any, **kwargs: Any) -> Any: ...

    def classification_at_level(self, symbol: Any, *args: Any, **kwargs: Any) -> str | None: ...


class _NorgatePackageClient:
    """Forwards to the optional `norgatedata` package. Not added to pyproject.toml."""

    def __init__(self) -> None:
        try:
            import norgatedata  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ScoutDataError(
                "norgatedata is not installed. Install the Norgate Data Updater "
                "and its Python package; it is not a project dependency."
            ) from exc
        self._nd = norgatedata

    def status(self) -> bool:
        return bool(self._nd.status())

    def database(self, databasename: str) -> Any:
        return self._nd.database(databasename)

    def price_timeseries(self, symbol: Any, **kwargs: Any) -> pd.DataFrame:
        return self._nd.price_timeseries(symbol, **kwargs)

    def capital_event_timeseries(self, symbol: Any, **kwargs: Any) -> pd.DataFrame:
        return self._nd.capital_event_timeseries(symbol, **kwargs)

    def exchange_name(self, symbol: Any) -> str | None:
        value = self._nd.exchange_name(symbol)
        return None if value is None else str(value)

    def subtype1(self, symbol: Any) -> str | None:
        value = self._nd.subtype1(symbol)
        return None if value is None else str(value)

    def subtype2(self, symbol: Any) -> str | None:
        value = self._nd.subtype2(symbol)
        return None if value is None else str(value)

    def first_quoted_date(self, symbol: Any, **kwargs: Any) -> Any:
        return self._nd.first_quoted_date(symbol, **kwargs)

    def last_quoted_date(self, symbol: Any, **kwargs: Any) -> Any:
        return self._nd.last_quoted_date(symbol, **kwargs)

    def classification_at_level(self, symbol: Any, *args: Any, **kwargs: Any) -> str | None:
        value = self._nd.classification_at_level(symbol, *args, **kwargs)
        return None if value is None else str(value)

    @property
    def none_adjustment(self) -> Any:
        return self._nd.StockPriceAdjustmentType.NONE

    @property
    def capital_adjustment(self) -> Any:
        return self._nd.StockPriceAdjustmentType.CAPITAL

    @property
    def padding_none(self) -> Any:
        return self._nd.PaddingType.NONE


def infer_split_ratio(
    unadj_ent: float,
    unadj_ex: float,
    cap_ent: float,
    cap_ex: float,
) -> float:
    """Isolate the split factor from the ordinary return.

    Norgate's capital_event flag has no ratio. Comparing the unadjusted jump
    to the capital-adjusted jump removes the ordinary session return.
    """
    if unadj_ex == 0.0 or cap_ent == 0.0 or unadj_ent == 0.0:
        return 1.0
    return (cap_ex / cap_ent) * (unadj_ent / unadj_ex)


def entitlement_to_ex_date(entitlement: date, calendar_name: str = "XNYS") -> date:
    """Norgate action dates are the session *before* ex-date."""
    cal = xcals.get_calendar(calendar_name)
    ts = pd.Timestamp(entitlement)
    if cal.is_session(ts):
        nxt = cal.next_session(ts)
        return date(nxt.year, nxt.month, nxt.day)
    sess = cal.date_to_session(ts, direction="next")
    return date(sess.year, sess.month, sess.day)


def _as_date(value: object) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return date(int(ts.year), int(ts.month), int(ts.day))


def _normalize_exchange(name: str | None) -> str:
    if not name:
        return ""
    return _EXCHANGE_MAP.get(name, name)


def _category(subtype1: str | None, subtype2: str | None) -> tuple[str, bool]:
    s2 = subtype2 or ""
    s1 = subtype1 or ""
    is_etf = "Exchange Traded Fund" in s2 or s1 == "Exchange Traded Product"
    if is_etf:
        return "ETF", True
    if s1 == "Equity" and s2 == "Operating/Holding Company":
        return "Domestic Common Stock", False
    return (s2 or s1 or "UNKNOWN"), False


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OHLCV_COLUMNS))


def _empty_actions() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ACTIONS_COLUMNS))


def _empty_earnings() -> pd.DataFrame:
    return pd.DataFrame(columns=list(EARNINGS_COLUMNS))


class NorgateCandleSource:
    """Fetches unadjusted daily bars and related tables from a local NDU database."""

    vendor_name = "norgate"

    def __init__(self, client: NorgateClient | None = None) -> None:
        self._client = client

    def _client_or_raise(self) -> NorgateClient:
        if self._client is None:
            self._client = _NorgatePackageClient()
        client = self._client
        if not client.status():
            raise ScoutDataError("Norgate Data Updater is not running")
        return client

    def fetch_tickers(self) -> pd.DataFrame:
        client = self._client_or_raise()
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        for db_name, is_delisted_db in ((_LISTED_DB, False), (_DELISTED_DB, True)):
            for rec in _iter_database(client.database(db_name)):
                asset_id = str(rec["assetid"])
                if asset_id in seen:
                    continue
                seen.add(asset_id)
                symbol = str(rec.get("symbol") or "")
                key: Any = int(rec["assetid"])
                subtype1 = client.subtype1(key)
                subtype2 = client.subtype2(key)
                category, is_etf = _category(subtype1, subtype2)
                listed = _as_date(client.first_quoted_date(key, datetimeformat="datetime"))
                last = _as_date(client.last_quoted_date(key, datetimeformat="datetime"))
                rows.append(
                    {
                        "asset_id": asset_id,
                        "symbol": symbol,
                        "exchange": _normalize_exchange(client.exchange_name(key)),
                        "category": category,
                        "sector": client.classification_at_level(key, "GICS", "Name", 1)
                        or "",
                        "is_etf": is_etf,
                        "listed_date": listed,
                        "delisted_date": last if is_delisted_db else pd.NaT,
                        "delist_reason": "OTHER" if is_delisted_db else "",
                    }
                )
        if not rows:
            return pd.DataFrame(columns=list(TICKERS_COLUMNS))
        return pd.DataFrame(rows, columns=list(TICKERS_COLUMNS))

    def fetch_ohlcv(
        self,
        asset_ids: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        client = self._client_or_raise()
        frames: list[pd.DataFrame] = []
        start_s = start.date().isoformat()
        end_s = end.date().isoformat()
        adj_none, pad_none = _adjustment_settings(client)
        for asset_id in asset_ids:
            raw = client.price_timeseries(
                int(asset_id),
                stock_price_adjustment_setting=adj_none,
                padding_setting=pad_none,
                start_date=start_s,
                end_date=end_s,
                timeseriesformat="pandas-dataframe",
            )
            if raw is None or raw.empty:
                continue
            frames.append(_ohlcv_from_norgate(asset_id, raw))
        if not frames:
            return _empty_ohlcv()
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
        client = self._client_or_raise()
        rows: list[dict[str, object]] = []
        start_d, end_d = start.date(), end.date()
        start_s, end_s = start_d.isoformat(), end_d.isoformat()
        adj_none, pad_none = _adjustment_settings(client)
        adj_cap = _capital_adjustment(client)
        for asset_id in asset_ids:
            key = int(asset_id)
            unadj = client.price_timeseries(
                key,
                stock_price_adjustment_setting=adj_none,
                padding_setting=pad_none,
                start_date=start_s,
                end_date=end_s,
                timeseriesformat="pandas-dataframe",
            )
            if unadj is None or unadj.empty:
                continue
            unadj = _with_session(unadj)
            rows.extend(_dividends_from_unadj(asset_id, unadj, start_d, end_d))
            cap_events = client.capital_event_timeseries(
                key, timeseriesformat="pandas-dataframe"
            )
            if cap_events is None or cap_events.empty:
                continue
            cap = client.price_timeseries(
                key,
                stock_price_adjustment_setting=adj_cap,
                padding_setting=pad_none,
                start_date=start_s,
                end_date=end_s,
                timeseriesformat="pandas-dataframe",
            )
            cap_frame = _with_session(cap) if cap is not None and not cap.empty else None
            rows.extend(
                _splits_from_events(asset_id, cap_events, unadj, cap_frame, start_d, end_d)
            )
        if not rows:
            return _empty_actions()
        return pd.DataFrame(rows, columns=list(ACTIONS_COLUMNS))

    def fetch_earnings(
        self,
        asset_ids: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        # Norgate fundamentals are current-only. A current "next earnings" date
        # applied to history would be lookahead. Leave the table empty; the
        # earnings gate then blocks every non-ETF (the conservative fallback).
        _LOG.warning(
            "norgate provides no historical earnings calendar; writing an empty "
            "earnings table. Non-ETF names will be blocked at the earnings gate "
            "until a calendar is supplied."
        )
        _ = (asset_ids, start, end)
        return _empty_earnings()


def _iter_database(contents: Any) -> list[dict[str, Any]]:
    if contents is None:
        return []
    if isinstance(contents, pd.DataFrame):
        return [dict(row) for row in contents.to_dict(orient="records")]
    if isinstance(contents, list):
        out: list[dict[str, Any]] = []
        for item in contents:
            if isinstance(item, dict):
                out.append(item)
            else:
                symbol = getattr(item, "symbol", None)
                assetid = getattr(item, "assetid", None)
                if symbol is None and isinstance(item, list | tuple) and len(item) >= 2:
                    symbol, assetid = item[0], item[1]
                if symbol is None or assetid is None:
                    continue
                out.append({"symbol": symbol, "assetid": assetid})
        return out
    return []


def _adjustment_settings(client: NorgateClient) -> tuple[Any, Any]:
    if isinstance(client, _NorgatePackageClient):
        return client.none_adjustment, client.padding_none
    return "NONE", "NONE"


def _capital_adjustment(client: NorgateClient) -> Any:
    if isinstance(client, _NorgatePackageClient):
        return client.capital_adjustment
    return "CAPITAL"


def _with_session(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "Date" in out.columns:
        out = out.rename(columns={"Date": "session"})
    elif "date" in out.columns:
        out = out.rename(columns={"date": "session"})
    out["session"] = [_as_date(v) for v in out["session"]]
    return out


def _ohlcv_from_norgate(asset_id: str, raw: pd.DataFrame) -> pd.DataFrame:
    frame = _with_session(raw)
    rename = {c: c.lower() for c in frame.columns}
    frame = frame.rename(columns=rename)
    if "unadjusted close" in frame.columns and "close" not in frame.columns:
        frame = frame.rename(columns={"unadjusted close": "close"})
    symbol = str(frame["symbol"].iloc[0]) if "symbol" in frame.columns else ""
    dividend = (
        frame["dividend"].fillna(0.0).astype("float64")
        if "dividend" in frame.columns
        else 0.0
    )
    out = pd.DataFrame(
        {
            "asset_id": asset_id,
            "symbol": symbol,
            "session": frame["session"],
            "open": frame["open"].astype("float64"),
            "high": frame["high"].astype("float64"),
            "low": frame["low"].astype("float64"),
            "close": frame["close"].astype("float64"),
            "volume": frame["volume"].astype("float64")
            if "volume" in frame.columns
            else 0.0,
            "dividend": dividend,
            "split_ratio": 1.0,
        }
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def _dividends_from_unadj(
    asset_id: str,
    unadj: pd.DataFrame,
    start: date,
    end: date,
) -> list[dict[str, object]]:
    if "Dividend" in unadj.columns:
        series = unadj["Dividend"]
    elif "dividend" in unadj.columns:
        series = unadj["dividend"]
    else:
        return []
    rows: list[dict[str, object]] = []
    for sess, amount in zip(unadj["session"], series, strict=True):
        if sess is None or pd.isna(amount) or float(amount) == 0.0:
            continue
        ex_date = entitlement_to_ex_date(sess)
        if ex_date < start or ex_date > end:
            continue
        rows.append(
            {
                "asset_id": asset_id,
                "ex_date": ex_date,
                "action_type": ActionType.DIVIDEND.value,
                "split_ratio": 1.0,
                "cash_amount": float(amount),
                "new_symbol": "",
            }
        )
    return rows


def _splits_from_events(
    asset_id: str,
    events: pd.DataFrame,
    unadj: pd.DataFrame,
    cap: pd.DataFrame | None,
    start: date,
    end: date,
) -> list[dict[str, object]]:
    ev = _with_session(events)
    flag_col = None
    for name in ("Capital Event", "capitalevent", "value", "Value"):
        if name in ev.columns:
            flag_col = name
            break
    if flag_col is None:
        numeric = [c for c in ev.columns if c != "session"]
        if not numeric:
            return []
        flag_col = numeric[0]
    unadj_by = unadj.set_index("session")["close"] if "close" in unadj.columns else None
    if unadj_by is None and "Close" in unadj.columns:
        unadj_by = unadj.set_index("session")["Close"]
    cap_by = None
    if cap is not None:
        close_name = "close" if "close" in cap.columns else "Close"
        if close_name in cap.columns:
            cap_by = cap.set_index("session")[close_name]
    rows: list[dict[str, object]] = []
    sessions = list(unadj["session"])
    session_set = {s: i for i, s in enumerate(sessions)}
    for sess, flag in zip(ev["session"], ev[flag_col], strict=True):
        if sess is None or pd.isna(flag) or float(flag) == 0.0:
            continue
        if sess not in session_set:
            continue
        idx = session_set[sess]
        if idx + 1 >= len(sessions):
            continue
        ex_date = entitlement_to_ex_date(sess)
        if ex_date < start or ex_date > end:
            continue
        ratio = 1.0
        if unadj_by is not None and cap_by is not None:
            ent_u = unadj_by.get(sess)
            ex_u = unadj_by.get(sessions[idx + 1])
            ent_c = cap_by.get(sess)
            ex_c = cap_by.get(sessions[idx + 1])
            if (
                ent_u is not None
                and ex_u is not None
                and ent_c is not None
                and ex_c is not None
                and not any(pd.isna(x) for x in (ent_u, ex_u, ent_c, ex_c))
            ):
                ratio = infer_split_ratio(
                    float(ent_u), float(ex_u), float(ent_c), float(ex_c)
                )
        rows.append(
            {
                "asset_id": asset_id,
                "ex_date": ex_date,
                "action_type": ActionType.SPLIT.value,
                "split_ratio": float(ratio),
                "cash_amount": 0.0,
                "new_symbol": "",
            }
        )
    return rows
