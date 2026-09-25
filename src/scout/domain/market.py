from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

import pandas as pd  # type: ignore[import-untyped]

from scout.domain._checks import require_aware
from scout.domain.enums import ActionType
from scout.utils.errors import ScoutLookaheadError

MARKET_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "symbol",
    "ts",
    "session_index",
    "open",
    "high",
    "low",
    "close",
    "close_raw",
    "volume",
    "dollar_volume",
    "is_suspect",
)

_PRICE_FLOAT_COLS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "close_raw",
    "volume",
    "dollar_volume",
)

BENCHMARK_COLUMNS: tuple[str, ...] = (
    "ts",
    "session_index",
    "open",
    "high",
    "low",
    "close",
    "vix_close",
    "vix9d_close",
    "vix3m_close",
)


@dataclass(frozen=True, slots=True)
class Asset:
    """Static and slowly-changing symbol metadata. Loaded from the vendor's
    ticker table; never derived inside the loop.
    """

    asset_id: str  # vendor PERMANENT id. The join key everywhere
    symbol: str  # display ticker as of the pinned data snapshot
    exchange: str  # "NYSE" | "NASDAQ" | "NYSEARCA" | "BATS"
    quote_currency: str  # "USD" for all equities
    is_etf: bool
    cluster: str  # GICS sector, or an ETF cluster. See ADR-007
    tick_size: Decimal  # "0.01" above $1.00
    step_size: Decimal  # "1" for whole shares; "0.0001" if fractional
    min_notional_usd: Decimal  # "0" for US equities
    listed_at: datetime | None  # UTC; None if unknown
    delisted_at: datetime | None  # UTC; None if still listed
    delist_reason: str | None  # "MERGER" | "BANKRUPTCY" | "OTHER" | None
    is_perpetual: bool = False  # crypto (M7)

    def __post_init__(self) -> None:
        require_aware(self.listed_at, "listed_at")
        require_aware(self.delisted_at, "delisted_at")


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """One row of data/raw/equity/actions.parquet."""

    asset_id: str
    ex_date: date
    action_type: ActionType
    split_ratio: float  # new shares per old; 4.0 for 4-for-1. 1.0 otherwise
    cash_amount: float  # dividend per share USD. 0.0 otherwise
    new_symbol: str | None  # TICKER_CHANGE only


@dataclass(frozen=True, slots=True)
class EarningsEvent:
    """One announcement. `available_ts` is what makes the earnings gate causal."""

    asset_id: str
    earnings_date: date
    is_confirmed: bool  # False means a vendor estimate
    available_ts: datetime | None  # when WE could first have known. None => fallback
    timing: str  # "BMO" | "AMC" | "UNKNOWN"

    def __post_init__(self) -> None:
        require_aware(self.available_ts, "available_ts")


@dataclass(frozen=True, slots=True)
class Session:
    """One row of the materialised trading calendar."""

    session: date
    open_utc: datetime  # DST-correct; never a fixed offset from `session`
    close_utc: datetime  # this is the panel `ts`
    is_half_day: bool
    session_index: int

    def __post_init__(self) -> None:
        require_aware(self.open_utc, "open_utc")
        require_aware(self.close_utc, "close_utc")


@dataclass(frozen=True, slots=True)
class Bar:
    """A closed session bar. `ts` is the CLOSE time. Used for single-bar work
    and tests; the panel is a DataFrame, not a list of these.
    """

    asset_id: str
    symbol: str
    ts: datetime  # UTC, session close
    session_index: int
    open: float  # ADJUSTED
    high: float
    low: float
    close: float
    close_raw: float  # UNADJUSTED. Dollar-price gates and share counts only
    volume: float  # shares, adjusted
    dollar_volume: float  # close_raw * volume_raw; used for ADV
    is_suspect: bool  # failed a data-quality check; see 04-DATA SS6.7

    def __post_init__(self) -> None:
        require_aware(self.ts, "ts")


def _require_aware_ts_column(frame: pd.DataFrame, name: str) -> pd.Series:
    series = frame[name]
    dtype = series.dtype
    if not isinstance(dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name} must be timezone-aware UTC")
    converted: pd.Series = series.dt.tz_convert("UTC")
    return converted


def _normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = tuple(str(c) for c in frame.columns)
    if cols != MARKET_COLUMNS:
        raise ValueError(
            f"MarketPanel columns must be {MARKET_COLUMNS} in that order; got {cols}"
        )
    out = frame.copy()
    out["ts"] = _require_aware_ts_column(out, "ts")
    out["asset_id"] = out["asset_id"].astype("category")
    out["symbol"] = out["symbol"].astype("category")
    out["session_index"] = out["session_index"].astype("int32")
    for col in _PRICE_FLOAT_COLS:
        out[col] = out[col].astype("float64")
    out["is_suspect"] = out["is_suspect"].astype(bool)
    out = out.sort_values(["ts", "asset_id"], kind="mergesort").reset_index(drop=True)
    return out


def _normalize_benchmark_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = tuple(str(c) for c in frame.columns)
    if cols != BENCHMARK_COLUMNS:
        raise ValueError(
            f"BenchmarkPanel columns must be {BENCHMARK_COLUMNS} in that order; got {cols}"
        )
    out = frame.copy()
    out["ts"] = _require_aware_ts_column(out, "ts")
    out["session_index"] = out["session_index"].astype("int32")
    for col in ("open", "high", "low", "close", "vix_close", "vix9d_close", "vix3m_close"):
        out[col] = out[col].astype("float64")
    out = out.sort_values(["ts"], kind="mergesort").reset_index(drop=True)
    return out


def _build_ts_index(frame: pd.DataFrame) -> tuple[pd.DatetimeIndex, list[int]]:
    if frame.empty:
        empty = pd.DatetimeIndex([], tz="UTC")
        return empty, []
    timestamps = pd.DatetimeIndex(frame["ts"].drop_duplicates())
    end_idx = [int(x) for x in frame["ts"].searchsorted(timestamps, side="right")]
    return timestamps, end_idx


def exact_ts_index(timestamps: pd.DatetimeIndex, ts: datetime) -> int | None:
    """Index of `ts` in `timestamps`, or None. Compares UTC ns, not Timestamp identity."""
    require_aware(ts, "ts")
    req = pd.Timestamp(ts).tz_convert("UTC")
    i = int(timestamps.searchsorted(req, side="left"))
    if i >= len(timestamps):
        return None
    hit = pd.Timestamp(timestamps[i])
    hit = hit.tz_localize("UTC") if hit.tzinfo is None else hit.tz_convert("UTC")
    if int(hit.value) != int(req.value):
        return None
    return i


def _as_of_end(timestamps: pd.DatetimeIndex, end_idx: list[int], ts: datetime) -> int:
    require_aware(ts, "ts")
    req = pd.Timestamp(ts).tz_convert("UTC")
    i = int(timestamps.searchsorted(req, side="right"))
    if i == 0:
        return 0
    return end_idx[i - 1]


def _assert_causal(sliced: pd.DataFrame, ts: datetime) -> None:
    if sliced.empty:
        return
    max_ts = sliced["ts"].iloc[-1]
    req = pd.Timestamp(ts).tz_convert("UTC")
    if max_ts > req:
        raise ScoutLookaheadError(
            f"as_of({req.isoformat()}) returned max ts {max_ts}, which is after the request"
        )


def _readonly(frame: pd.DataFrame) -> pd.DataFrame:
    # pandas 2.2 has no writes_enabled flag. Callers must not mutate the view.
    return frame


class MarketPanel:
    """Immutable long-format panel of closed session bars.

    Index: RangeIndex. Sorted by (ts, asset_id). Never mutated in place.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = _normalize_market_frame(frame)
        self._timestamps, self._end_idx = _build_ts_index(self._frame)

    @classmethod
    def _from_validated_view(cls, frame: pd.DataFrame) -> "MarketPanel":
        obj = cls.__new__(cls)
        obj._frame = frame
        obj._timestamps, obj._end_idx = _build_ts_index(frame)
        return obj

    @property
    def frame(self) -> pd.DataFrame:
        return _readonly(self._frame)

    @property
    def asset_ids(self) -> tuple[str, ...]:
        if self._frame.empty:
            return ()
        ids = self._frame["asset_id"].astype(str).drop_duplicates()
        return tuple(sorted(ids.tolist()))

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return self._timestamps

    def as_of(self, ts: datetime) -> "MarketPanel":
        """All rows with self.frame.ts <= ts. THE causality primitive."""
        end = _as_of_end(self._timestamps, self._end_idx, ts)
        sliced = self._frame.iloc[:end]
        _assert_causal(sliced, ts)
        return MarketPanel._from_validated_view(sliced)

    def slice_symbol(self, asset_id: str) -> pd.DataFrame:
        """Single asset, ts-indexed, sorted ascending."""
        mask = self._frame["asset_id"].astype(str) == asset_id
        out = self._frame.loc[mask]
        return out.set_index("ts").sort_index()

    def latest(self, ts: datetime) -> pd.DataFrame:
        """One row per asset: the row with the greatest ts <= given ts."""
        view = self.as_of(ts).frame
        if view.empty:
            return view
        return view.drop_duplicates(subset=["asset_id"], keep="last").reset_index(drop=True)

    def rows_at(self, ts: datetime) -> pd.DataFrame:
        """Rows whose ts equals `ts` exactly. Searchsorted slice, not a mask."""
        i = exact_ts_index(self._timestamps, ts)
        if i is None:
            return self._frame.iloc[0:0]
        start = 0 if i == 0 else self._end_idx[i - 1]
        return self._frame.iloc[start : self._end_idx[i]]


class BenchmarkPanel:
    """The benchmark series, loaded from data/reference/benchmark_1d.parquet and
    NOT from the universe panel.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = _normalize_benchmark_frame(frame)
        self._timestamps, self._end_idx = _build_ts_index(self._frame)

    @classmethod
    def _from_validated_view(cls, frame: pd.DataFrame) -> "BenchmarkPanel":
        obj = cls.__new__(cls)
        obj._frame = frame
        obj._timestamps, obj._end_idx = _build_ts_index(frame)
        return obj

    @property
    def frame(self) -> pd.DataFrame:
        return _readonly(self._frame)

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return self._timestamps

    def as_of(self, ts: datetime) -> "BenchmarkPanel":
        end = _as_of_end(self._timestamps, self._end_idx, ts)
        sliced = self._frame.iloc[:end]
        _assert_causal(sliced, ts)
        return BenchmarkPanel._from_validated_view(sliced)
