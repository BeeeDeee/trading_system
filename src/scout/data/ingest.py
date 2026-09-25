"""Offline equity ingest: vendor fetch → raw parquet → SNAPSHOT.json last."""

from __future__ import annotations

import io
import logging
import os
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
import pandas as pd  # type: ignore[import-untyped]

from scout.config.schema import DataVendor, ScoutConfig
from scout.data.fixture_source import FixtureEquitySource
from scout.data.norgate_source import NorgateCandleSource
from scout.data.schemas import (
    ACTIONS_COLUMNS,
    EARNINGS_COLUMNS,
    MIN_DELISTED_FRACTION,
    OHLCV_COLUMNS,
    SNAPSHOT_KEYS,
    TICKERS_COLUMNS,
)
from scout.data.sharadar_source import SharadarCandleSource
from scout.data.store import read_parquet, write_json_atomic, write_parquet_atomic
from scout.utils.clock import Clock, WallClock
from scout.utils.errors import ScoutConfigError, ScoutDataError

_LOG = logging.getLogger("scout.data.ingest")


@runtime_checkable
class EquityIngestSource(Protocol):
    vendor_name: str

    def fetch_tickers(self) -> pd.DataFrame: ...

    def fetch_ohlcv(
        self, asset_ids: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame: ...

    def fetch_actions(
        self, asset_ids: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame: ...

    def fetch_earnings(
        self, asset_ids: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    data_snapshot_id: str
    vendor: str
    created_utc: str
    symbols: int
    rows: int
    date_min: str
    date_max: str
    actions_rows: int
    earnings_rows: int

    def as_dict(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in SNAPSHOT_KEYS}


def build_source(cfg: ScoutConfig) -> EquityIngestSource:
    vendor = cfg.data.vendor
    if vendor is DataVendor.FIXTURE:
        return FixtureEquitySource()
    if vendor is DataVendor.NORGATE:
        return NorgateCandleSource()
    if vendor is DataVendor.SHARADAR:
        key = os.environ.get("SCOUT_SHARADAR_API_KEY", "").strip()
        if not key:
            raise ScoutConfigError(
                "data.vendor: sharadar requires SCOUT_SHARADAR_API_KEY"
            )
        return SharadarCandleSource(SharadarHttpxClient(key))
    raise ScoutConfigError(
        f"data.vendor: ingest is not implemented for {vendor.value!r}; "
        f"known ingest vendors: ['sharadar', 'norgate', 'fixture']"
    )


def read_candidate_pairs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ScoutDataError(f"candidate list not found: {path}")
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) != 2:
            raise ScoutDataError(f"{path}: expected asset_id,symbol; got {stripped!r}")
        rows.append({"asset_id": parts[0], "symbol": parts[1]})
    return pd.DataFrame(rows, columns=["asset_id", "symbol"])


def delisted_fraction(candidates: pd.DataFrame, tickers: pd.DataFrame) -> float:
    if candidates.empty:
        return 0.0
    before = len(candidates)
    joined = candidates.merge(
        tickers[["asset_id", "delisted_date"]],
        on="asset_id",
        how="left",
        validate="many_to_one",
    )
    if len(joined) != before:
        raise ScoutDataError("candidate/ticker merge duplicated rows")
    n_delisted = int(joined["delisted_date"].notna().sum())
    return n_delisted / before


def require_min_delisted_fraction(
    candidates: pd.DataFrame,
    tickers: pd.DataFrame,
    *,
    minimum: float = MIN_DELISTED_FRACTION,
) -> float:
    fraction = delisted_fraction(candidates, tickers)
    if fraction < minimum:
        raise ScoutDataError(
            f"candidate list delisted fraction {fraction:.1%} is below "
            f"{minimum:.0%}; refusing to continue (survivorship bias)"
        )
    return fraction


def run_ingest(
    cfg: ScoutConfig,
    *,
    start: datetime,
    end: datetime,
    raw_dir: Path,
    snapshot_path: Path,
    clock: Clock | None = None,
    source: EquityIngestSource | None = None,
    symbols_file: Path | None = None,
) -> SnapshotManifest:
    if end < start:
        raise ScoutDataError("ingest end must be on or after start")
    used_clock = clock if clock is not None else WallClock()
    used_source = source if source is not None else build_source(cfg)
    tickers = _normalize_tickers(used_source.fetch_tickers())
    if symbols_file is not None:
        wanted = read_candidate_pairs(symbols_file)
        before = len(wanted)
        filtered = tickers.loc[tickers["asset_id"].isin(set(wanted["asset_id"]))]
        tickers = filtered.reset_index(drop=True)
        _LOG.info("symbols-file kept %s of %s requested ids", len(tickers), before)
    if tickers.empty:
        raise ScoutDataError("ingest produced no ticker rows")
    asset_ids = [str(x) for x in tickers["asset_id"].tolist()]
    ohlcv = _normalize_ohlcv(used_source.fetch_ohlcv(asset_ids, start, end), tickers)
    if ohlcv.empty:
        raise ScoutDataError("ingest produced no OHLCV rows")
    actions = _normalize_actions(used_source.fetch_actions(asset_ids, start, end))
    earnings = _normalize_earnings(used_source.fetch_earnings(asset_ids, start, end))

    raw_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(raw_dir / "tickers.parquet", tickers)
    write_parquet_atomic(raw_dir / "actions.parquet", actions)
    write_parquet_atomic(raw_dir / "earnings.parquet", earnings)
    _write_ohlcv_by_year(raw_dir / "ohlcv" / "1d", ohlcv)

    manifest = _manifest(used_source.vendor_name, used_clock, ohlcv, actions, earnings)
    # SNAPSHOT.json is the completeness marker. Write it last so a kill mid-write
    # cannot leave a snapshot that claims files we never finished.
    write_json_atomic(snapshot_path, manifest.as_dict())
    return manifest


def _write_ohlcv_by_year(out_dir: Path, ohlcv: pd.DataFrame) -> None:
    years = pd.to_datetime(ohlcv["session"]).dt.year
    for year in sorted(int(y) for y in years.unique()):
        part = ohlcv.loc[years == year].reset_index(drop=True)
        write_parquet_atomic(out_dir / f"{year}.parquet", part)


def _manifest(
    vendor: str,
    clock: Clock,
    ohlcv: pd.DataFrame,
    actions: pd.DataFrame,
    earnings: pd.DataFrame,
) -> SnapshotManifest:
    created = clock.now()
    sessions = pd.to_datetime(ohlcv["session"])
    return SnapshotManifest(
        data_snapshot_id=f"{created.strftime('%Y%m%d')}-{vendor}",
        vendor=vendor,
        created_utc=_utc_z(created),
        symbols=int(ohlcv["asset_id"].nunique()),
        rows=int(len(ohlcv)),
        date_min=sessions.min().date().isoformat(),
        date_max=sessions.max().date().isoformat(),
        actions_rows=int(len(actions)),
        earnings_rows=int(len(earnings)),
    )


def _utc_z(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_tickers(frame: pd.DataFrame) -> pd.DataFrame:
    out = _require_columns(frame, TICKERS_COLUMNS, "tickers")
    out = out.assign(
        asset_id=out["asset_id"].astype(str),
        symbol=out["symbol"].astype(str),
        exchange=out["exchange"].astype(str),
        category=out["category"].astype(str),
        sector=out["sector"].astype(str),
        is_etf=out["is_etf"].astype(bool),
        listed_date=_as_optional_date(out["listed_date"]),
        delisted_date=_as_optional_date(out["delisted_date"]),
        delist_reason=out["delist_reason"].fillna("").astype(str),
    )
    return out.sort_values(["asset_id"], kind="mergesort").reset_index(drop=True)


def _normalize_ohlcv(frame: pd.DataFrame, tickers: pd.DataFrame) -> pd.DataFrame:
    out = _require_columns(frame, OHLCV_COLUMNS, "ohlcv")
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    if out.empty:
        return out
    before = len(out)
    if out["symbol"].eq("").any() or out["symbol"].isna().any():
        sym = tickers[["asset_id", "symbol"]].rename(columns={"symbol": "symbol_meta"})
        merged = out.merge(sym, on="asset_id", how="left", validate="many_to_one")
        if len(merged) != before:
            raise ScoutDataError("ohlcv/ticker merge duplicated rows")
        filled = merged["symbol"].where(
            merged["symbol"].astype(str).str.len() > 0, merged["symbol_meta"]
        )
        out = merged.assign(symbol=filled.astype(str)).drop(columns=["symbol_meta"])
    out = out.assign(
        asset_id=out["asset_id"].astype(str),
        symbol=out["symbol"].astype(str),
        session=_as_required_date(out["session"], "ohlcv.session"),
        open=out["open"].astype("float64"),
        high=out["high"].astype("float64"),
        low=out["low"].astype("float64"),
        close=out["close"].astype("float64"),
        volume=out["volume"].astype("float64"),
        dividend=out["dividend"].fillna(0.0).astype("float64"),
        split_ratio=out["split_ratio"].fillna(1.0).astype("float64"),
    )
    out = out.loc[:, list(OHLCV_COLUMNS)]
    return out.sort_values(["session", "asset_id"], kind="mergesort").reset_index(
        drop=True
    )


def _normalize_actions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(ACTIONS_COLUMNS))
    out = _require_columns(frame, ACTIONS_COLUMNS, "actions")
    out = out.assign(
        asset_id=out["asset_id"].astype(str),
        ex_date=_as_required_date(out["ex_date"], "actions.ex_date"),
        action_type=out["action_type"].astype(str),
        split_ratio=out["split_ratio"].astype("float64"),
        cash_amount=out["cash_amount"].astype("float64"),
        new_symbol=out["new_symbol"].fillna("").astype(str),
    )
    return out.sort_values(
        ["ex_date", "asset_id", "action_type"], kind="mergesort"
    ).reset_index(drop=True)


def _normalize_earnings(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(EARNINGS_COLUMNS))
    out = _require_columns(frame, EARNINGS_COLUMNS, "earnings")
    available = pd.to_datetime(out["available_ts"], utc=True)
    out = out.assign(
        asset_id=out["asset_id"].astype(str),
        earnings_date=_as_required_date(out["earnings_date"], "earnings.earnings_date"),
        is_confirmed=out["is_confirmed"].astype(bool),
        available_ts=available,
        timing=out["timing"].astype(str),
    )
    return out.sort_values(
        ["earnings_date", "asset_id"], kind="mergesort"
    ).reset_index(drop=True)


def _require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], name: str
) -> pd.DataFrame:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ScoutDataError(f"{name} missing columns: {missing}")
    return frame.loc[:, list(required)].copy()


def _as_required_date(series: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ScoutDataError(f"{name}: unparseable or null dates")
    return parsed.dt.date


def _as_optional_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def load_tickers(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "tickers.parquet"
    if not path.is_file():
        raise ScoutDataError(f"tickers parquet not found: {path}")
    return read_parquet(path)


class SharadarHttpxClient:
    """Bulk CSV zip downloads from api.sharadar.com. httpx lives only here."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.sharadar.com/v1.0/data",
        timeout_s: float = 1200.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def load_table(self, name: str) -> pd.DataFrame:
        url = f"{self._base_url}/{name}"
        _LOG.info("sharadar bulk download table=%s", name)
        try:
            with httpx.Client(follow_redirects=True, timeout=self._timeout_s) as client:
                response = client.get(
                    url, params={"api_key": self._api_key, "years": "full"}
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ScoutDataError(
                f"sharadar download failed for table {name!r}: {exc}"
            ) from exc
        return _read_sharadar_payload(name, response.content)


def _read_sharadar_payload(name: str, content: bytes) -> pd.DataFrame:
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.namelist()
            if not members:
                raise ScoutDataError(f"sharadar zip for {name!r} is empty")
            with archive.open(members[0]) as handle:
                return pd.read_csv(handle)
    text = content[:200].decode("utf-8", errors="replace")
    if text.lstrip().startswith("{") and "error" in text.lower():
        raise ScoutDataError(f"sharadar rejected table {name!r}: {text}")
    return pd.read_csv(io.BytesIO(content))
