from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from scout.domain._checks import require_aware
from scout.domain.enums import MarketRegime, Regime, VolBucket
from scout.domain.market import exact_ts_index
from scout.utils.errors import ScoutLookaheadError


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """All features for one (ts, symbol). Fixed schema — adding a feature means
    adding a field here and a column in FeaturePanel. No dict-of-floats: a
    typo in a dict key is a silent zero, and silent zeros in trading code cost
    money.
    """

    symbol: str
    ts: datetime

    # --- price / reference ---
    close: float
    atr_14: float  # Wilder ATR, absolute price units
    atr_pct: float  # atr_14 / close
    vol_20: float  # annualised log-return vol, 20 sessions
    vol_60: float

    # --- trend ---
    ema_fast: float  # length from config, default 20
    ema_slow: float  # default 50
    ema_spread_atr: float  # (ema_fast - ema_slow) / atr_14, signed
    slope_atr_20: float  # (close - close[-20]) / (atr_14 * sqrt(20))

    # --- regime backbone ---
    efficiency_ratio_20: float  # Kaufman ER, 20 sessions, [0, 1]
    efficiency_ratio_60: float  # same, 60 sessions (longer-horizon agreement)
    atr_percentile_1y: float  # [0, 1], rank of atr_pct in trailing ~2y
    regime: Regime
    vol_bucket: VolBucket

    # --- channels / levels ---
    donchian_high_20: float
    donchian_low_20: float
    donchian_high_55: float  # Donchian v1 uses 55; 20 is recorded for M6
    donchian_low_55: float
    keltner_upper: float  # ema_slow + k * atr_14
    keltner_lower: float
    dist_to_high_atr: float  # (donchian_high_55 - close) / atr_14
    dist_to_low_atr: float  # (close - donchian_low_55) / atr_14

    # --- momentum (primary strategy inputs) ---
    mom_252_skip21: float  # 12-1 total return; NOT a tunable
    mom_126_skip21: float  # recorded, not used in v1 decisions
    mom_21: float  # recorded, not used in v1 decisions
    mom_252_xs_pct: float  # rank of mom_252_skip21 in the eligible set at t
    vol_xs_pct: float  # cross-sectional vol rank
    xs_population: int  # eligible-set size used for the ranks above

    # --- gaps (recorded; not used in v1 decisions) ---
    gap_atr: float
    gap_abs_mean_20: float
    overnight_var_share_60: float

    # --- market (broadcast from BenchmarkPanel) ---
    market_regime: MarketRegime
    spy_dd_252: float
    spy_above_ma: bool
    vix_close: float  # nullable; NOT in the regime definition

    # --- beta / correlation vs SPY ---
    beta_bench_90: float  # rolling 90-session beta vs SPY
    corr_bench_90: float  # rolling 90-session correlation vs SPY

    # --- data quality ---
    bars_available: int  # bars with close_time <= ts for this symbol
    bars_since_gap: int  # bars since the last detected gap; large is good
    is_warm: bool  # bars_available >= required warm-up

    def __post_init__(self) -> None:
        require_aware(self.ts, "ts")

    def to_dict(self) -> dict[str, float | str | int | bool]:
        result: dict[str, float | str | int | bool] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, datetime):
                result[field.name] = value.isoformat()
            elif isinstance(value, Enum):
                result[field.name] = str(value.value)
            elif isinstance(value, str):
                result[field.name] = value
            elif isinstance(value, bool):
                result[field.name] = bool(value)
            elif isinstance(value, int):
                result[field.name] = int(value)
            else:
                result[field.name] = float(value)
        return result


FEATURE_COLUMNS = tuple(f.name for f in fields(FeatureRow))

_ENUM_FIELDS: dict[str, type[Regime] | type[VolBucket] | type[MarketRegime]] = {
    "regime": Regime,
    "vol_bucket": VolBucket,
    "market_regime": MarketRegime,
}

_BOOL_FIELDS = frozenset({"spy_above_ma", "is_warm"})
_INT_FIELDS = frozenset({"xs_population", "bars_available", "bars_since_gap"})


def _normalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = tuple(str(c) for c in frame.columns)
    if cols != FEATURE_COLUMNS:
        raise ValueError(
            f"FeaturePanel columns must equal FeatureRow fields in order; got {cols}"
        )
    dtype = frame["ts"].dtype
    if not isinstance(dtype, pd.DatetimeTZDtype):
        raise ValueError("ts must be timezone-aware UTC")
    # A full deep copy of ~46M object columns OOMs on the holdout panel.
    # Categorise symbol first; sort_values allocates new blocks anyway.
    symbol = frame["symbol"]
    if str(symbol.dtype) != "category":
        symbol = symbol.astype("category")
    ts = frame["ts"]
    if str(ts.dtype) != "datetime64[ns, UTC]":
        ts = ts.dt.tz_convert("UTC")
    out = frame.assign(ts=ts, symbol=symbol)
    return out.sort_values(["ts", "symbol"], kind="mergesort").reset_index(drop=True)


def _build_ts_index(frame: pd.DataFrame) -> tuple[pd.DatetimeIndex, list[int]]:
    if frame.empty:
        return pd.DatetimeIndex([], tz="UTC"), []
    timestamps = pd.DatetimeIndex(frame["ts"].drop_duplicates())
    end_idx = [int(x) for x in frame["ts"].searchsorted(timestamps, side="right")]
    return timestamps, end_idx


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"cannot convert {type(value).__name__} to int")


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, np.floating | np.integer):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"cannot convert {type(value).__name__} to float")


def _coerce_bool(value: object) -> bool:
    if isinstance(value, np.bool_):
        return bool(value)
    return bool(value)


def _row_to_feature_row(record: dict[str, object]) -> FeatureRow:
    kwargs: dict[str, object] = {}
    for name in FEATURE_COLUMNS:
        value = record[name]
        if name == "ts":
            kwargs[name] = pd.Timestamp(value).to_pydatetime()
        elif name in _ENUM_FIELDS:
            enum_cls = _ENUM_FIELDS[name]
            kwargs[name] = value if isinstance(value, enum_cls) else enum_cls(str(value))
        elif name in _BOOL_FIELDS:
            kwargs[name] = bool(value)
        elif name in _INT_FIELDS:
            kwargs[name] = _coerce_int(value)
        elif name == "symbol":
            kwargs[name] = str(value)
        else:
            kwargs[name] = _coerce_float(value)
    return FeatureRow(**kwargs)  # type: ignore[arg-type]


def feature_row_from_record(record: dict[str, object]) -> FeatureRow:
    """Build a FeatureRow from a FeaturePanel record. Used by the engine loop."""
    return _row_to_feature_row(record)


def feature_row_from_tuple(
    values: Sequence[object], *, ts: datetime | None = None
) -> FeatureRow:
    """FeatureRow from a FEATURE_COLUMNS-aligned tuple. No intermediate dict."""
    if len(values) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"expected {len(FEATURE_COLUMNS)} feature values, got {len(values)}"
        )
    kwargs: dict[str, object] = {}
    for name, value in zip(FEATURE_COLUMNS, values, strict=True):
        if name == "ts":
            if ts is not None:
                kwargs[name] = ts
            elif isinstance(value, datetime) and value.tzinfo is not None:
                kwargs[name] = value
            else:
                kwargs[name] = pd.Timestamp(value).to_pydatetime()
        elif name in _ENUM_FIELDS:
            enum_cls = _ENUM_FIELDS[name]
            kwargs[name] = value if isinstance(value, enum_cls) else enum_cls(str(value))
        elif name in _BOOL_FIELDS:
            kwargs[name] = _coerce_bool(value)
        elif name in _INT_FIELDS:
            kwargs[name] = _coerce_int(value)
        elif name == "symbol":
            kwargs[name] = str(value)
        else:
            kwargs[name] = _coerce_float(value)
    return FeatureRow(**kwargs)  # type: ignore[arg-type]


class FeaturePanel:
    """DataFrame wrapper whose columns are exactly FeatureRow's fields."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = _normalize_feature_frame(frame)
        self._timestamps, self._end_idx = _build_ts_index(self._frame)

    @classmethod
    def _from_validated_view(cls, frame: pd.DataFrame) -> "FeaturePanel":
        obj = cls.__new__(cls)
        obj._frame = frame
        obj._timestamps, obj._end_idx = _build_ts_index(frame)
        return obj

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    @property
    def symbols(self) -> tuple[str, ...]:
        if self._frame.empty:
            return ()
        return tuple(sorted(self._frame["symbol"].astype(str).drop_duplicates().tolist()))

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return self._timestamps

    def as_of(self, ts: datetime) -> "FeaturePanel":
        require_aware(ts, "ts")
        req = pd.Timestamp(ts).tz_convert("UTC")
        i = int(self._timestamps.searchsorted(req, side="right"))
        end = 0 if i == 0 else self._end_idx[i - 1]
        sliced = self._frame.iloc[:end]
        if not sliced.empty:
            max_ts = sliced["ts"].iloc[-1]
            if max_ts > req:
                raise ScoutLookaheadError(
                    f"as_of({req.isoformat()}) returned max ts {max_ts}, which is after the request"
                )
        return FeaturePanel._from_validated_view(sliced)

    def slice_symbol(self, symbol: str) -> pd.DataFrame:
        mask = self._frame["symbol"].astype(str) == symbol
        return self._frame.loc[mask].set_index("ts").sort_index()

    def latest(self, ts: datetime) -> pd.DataFrame:
        view = self.as_of(ts).frame
        if view.empty:
            return view
        return view.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)

    def rows_at(self, ts: datetime) -> pd.DataFrame:
        """Rows whose ts equals `ts` exactly. Searchsorted slice, not a mask."""
        i = exact_ts_index(self._timestamps, ts)
        if i is None:
            return self._frame.iloc[0:0]
        start = 0 if i == 0 else self._end_idx[i - 1]
        return self._frame.iloc[start : self._end_idx[i]]

    def row(self, ts: datetime, symbol: str) -> FeatureRow:
        require_aware(ts, "ts")
        req = pd.Timestamp(ts).tz_convert("UTC")
        matched = self._frame.loc[
            (self._frame["ts"] == req) & (self._frame["symbol"].astype(str) == symbol)
        ]
        if len(matched) != 1:
            raise KeyError(f"no unique FeatureRow for ts={req.isoformat()} symbol={symbol!r}")
        record = matched.iloc[0].to_dict()
        return _row_to_feature_row(record)
