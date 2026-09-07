from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from scout.config.hashing import config_hash
from scout.config.loader import load_config
from scout.config.schema import ScoutConfig
from scout.data.calendar import reference_calendar_path
from scout.data.ingest import load_tickers, read_candidate_pairs
from scout.data.parquet_source import ParquetCandleSource
from scout.data.store import read_json, read_parquet, write_json_atomic, write_parquet_atomic
from scout.domain.market import BENCHMARK_COLUMNS, BenchmarkPanel, EarningsEvent
from scout.features.engine import compute_features
from scout.scoring.labeling import (
    LABEL_COLUMNS,
    AssetTick,
    _join_adv,
    empty_label_frame,
    label_setups,
)
from scout.strategies.registry import build_strategies
from scout.utils.clock import WallClock
from scout.utils.errors import ScoutConfigError, ScoutDataError
from scout.utils.logging import configure_logging

DEFAULT_LABELS_DIR = Path("data") / "labels"
PROGRESS_NAME = "progress.json"
CKPT_DIR_NAME = "_ckpt"
MANIFEST_NAME = "manifest.json"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--strategy", default=None, help="Restrict to one strategy_id")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Output directory (default: labeling.labels_dir from config)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Checkpoint after this many assets (default 25)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints in --labels-dir and start from scratch",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Skip detect/resolve; concat checkpoints, join ADV, write setups_*.parquet",
    )


def run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    digest = config_hash(cfg)
    configure_logging(f"label-{digest[:12]}", level=cfg.logging.level)
    log = logging.getLogger("scout.cli.label_setups")
    clock = WallClock()
    started = clock.now()

    if args.labels_dir is not None:
        labels_dir = Path(args.labels_dir)
    else:
        labels_dir = Path(cfg.labeling.labels_dir)
    ckpt_dir = labels_dir / CKPT_DIR_NAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log.info("resolved config hash=%s labels_dir=%s", digest, labels_dir)
    print(f"config_hash={digest}")
    print(f"labels_dir={labels_dir.as_posix()}")

    strategies = list(build_strategies(cfg.strategies))
    if args.strategy is not None:
        strategies = [s for s in strategies if s.strategy_id == args.strategy]
        if not strategies:
            known = sorted(sc.strategy_id for sc in cfg.strategies if sc.enabled)
            raise ScoutConfigError(
                f"unknown or disabled strategy_id {args.strategy!r}; known: {known}"
            )
    strategy_ids = [s.strategy_id for s in strategies]
    if not strategy_ids:
        raise ScoutConfigError("no enabled strategies to label")

    flush_every = int(args.flush_every)
    if flush_every < 1:
        raise ScoutConfigError("--flush-every must be >= 1")

    manifest_path = ckpt_dir / MANIFEST_NAME
    skip: set[str] = set()
    next_batch = 1
    batch_files: list[str] = []
    if not args.no_resume and manifest_path.is_file():
        skip, next_batch = _load_resume(manifest_path, digest, strategy_ids)
        print(f"resume_assets_done={len(skip)}")
        log.info("resuming skip=%s next_batch=%s", len(skip), next_batch)

    if args.finalize_only:
        if args.no_resume:
            raise ScoutConfigError("--finalize-only cannot be combined with --no-resume")
        if not manifest_path.is_file():
            raise ScoutDataError(f"no checkpoint manifest at {manifest_path}")
        batch_files = _existing_batch_files(manifest_path)
        print("stage=finalize_only", flush=True)
        snapshots = _load_snapshots(cfg)
        labeled = _concat_checkpoints(ckpt_dir, batch_files)
        labeled = _join_adv(labeled, snapshots) if not labeled.empty else labeled
        _write_by_strategy(labeled, labels_dir, strategy_ids)
        write_json_atomic(
            labels_dir / PROGRESS_NAME,
            {
                "command": "label",
                "status": "done",
                "config_hash": digest,
                "pid": os.getpid(),
                "labels_dir": labels_dir.as_posix(),
                "started_utc": started.isoformat(),
                "updated_utc": clock.now().isoformat(),
                "note": "finalize-only",
                "setups": 0 if labeled.empty else int(len(labeled)),
                "elapsed_s": (clock.now() - started).total_seconds(),
            },
        )
        _print_strategy_summary(labeled, labels_dir, strategy_ids)
        print(f"progress_path={(labels_dir / PROGRESS_NAME).as_posix()}")
        return 0

    pid = os.getpid()
    progress_path = labels_dir / PROGRESS_NAME

    def _boot_progress(*, status: str, note: str) -> None:
        now = clock.now()
        write_json_atomic(
            progress_path,
            {
                "command": "label",
                "status": status,
                "config_hash": digest,
                "pid": pid,
                "labels_dir": labels_dir.as_posix(),
                "started_utc": started.isoformat(),
                "updated_utc": now.isoformat(),
                "note": note,
                "assets_total": 0,
                "assets_done": 0,
                "assets_skipped_resume": len(skip),
                "setups": 0,
                "warm_bars_seen": 0,
                "elapsed_s": (now - started).total_seconds(),
            },
        )

    _boot_progress(status="loading_panel", note="reading processed parquet")
    print("stage=loading_panel", flush=True)
    candidates = read_candidate_pairs(Path(cfg.universe.candidates_file))
    asset_ids = [str(x) for x in candidates["asset_id"].tolist()]
    source = ParquetCandleSource(cfg.data.processed_dir)
    panel = source.load_panel(
        asset_ids,
        cfg.data.decision_timeframe,
        cfg.period.start,
        cfg.period.end,
    )
    snapshots = _load_snapshots(cfg)
    benchmark = _load_benchmark(Path(cfg.data.processed_dir))
    _boot_progress(status="compute_features", note="cross-sectional features; no asset counter yet")
    print("stage=compute_features", flush=True)
    features = compute_features(panel, benchmark, snapshots, cfg.features)
    calendar = _load_calendar(cfg)
    tickers = load_tickers(Path(cfg.data.raw_dir))
    is_etf = _etf_map(tickers)
    earnings_by_asset = _load_earnings(Path(cfg.data.raw_dir))
    _boot_progress(status="labeling", note="starting per-asset detect/resolve")
    print("stage=labeling", flush=True)

    warm_bars_seen = 0
    last_tick: AssetTick | None = None
    batch_no = next_batch
    done: set[str] = set(skip)
    if skip:
        batch_files = _existing_batch_files(manifest_path)

    def _write_progress(*, status: str) -> None:
        now = clock.now()
        elapsed = (now - started).total_seconds()
        tick = last_tick
        assets_done = len(done)
        total = tick.total if tick is not None else 0
        remaining = max(0, total - assets_done)
        rate = assets_done / elapsed * 3600 if elapsed > 0 and assets_done else 0.0
        eta_s = remaining / (assets_done / elapsed) if elapsed > 0 and assets_done else None
        payload: dict[str, Any] = {
            "command": "label",
            "status": status,
            "config_hash": digest,
            "pid": pid,
            "labels_dir": labels_dir.as_posix(),
            "started_utc": started.isoformat(),
            "updated_utc": now.isoformat(),
            "assets_total": total,
            "assets_done": assets_done,
            "assets_skipped_resume": len(skip),
            "last_asset_id": None if tick is None else tick.asset_id,
            "warm_bars_seen": warm_bars_seen,
            "setups": 0 if tick is None else tick.setups_total,
            "elapsed_s": elapsed,
            "assets_per_hour": rate,
            "eta_s": eta_s,
            "flush_batches": batch_no - 1,
        }
        write_json_atomic(labels_dir / PROGRESS_NAME, payload)

    def on_asset(tick: AssetTick) -> None:
        nonlocal last_tick, warm_bars_seen
        last_tick = tick
        if not tick.skipped:
            warm_bars_seen += tick.warm_bars
            done.add(tick.asset_id)
        elapsed = (clock.now() - started).total_seconds()
        processed = tick.index + 1
        rate = processed / elapsed * 3600 if elapsed > 0 else 0.0
        remaining = max(0, tick.total - processed)
        eta_h = (remaining / rate) if rate > 0 else float("nan")
        if tick.skipped:
            return
        if processed % 10 == 0 or tick.index == 0:
            line = (
                f"progress assets={processed}/{tick.total} "
                f"setups={tick.setups_total} warm_bars={warm_bars_seen} "
                f"rate_per_h={rate:.1f} eta_h={eta_h:.2f} last={tick.asset_id}"
            )
            print(line, flush=True)
            log.info(line)
            _write_progress(status="running")

    def on_batch(frame: pd.DataFrame, batch_ids: tuple[str, ...]) -> None:
        nonlocal batch_no, batch_files
        del batch_ids
        name = f"batch_{batch_no:05d}.parquet"
        write_parquet_atomic(ckpt_dir / name, frame)
        batch_files.append(name)
        batch_no += 1
        write_json_atomic(
            manifest_path,
            {
                "config_hash": digest,
                "strategy_ids": strategy_ids,
                "done_asset_ids": sorted(done),
                "batch_files": batch_files,
                "next_batch": batch_no,
            },
        )
        _write_progress(status="running")

    try:
        label_setups(
            features,
            panel,
            strategies,
            cfg.labeling,
            snapshots=snapshots,
            earnings_by_asset=earnings_by_asset,
            is_etf=is_etf,
            calendar=calendar,
            skip_asset_ids=skip,
            flush_every=flush_every,
            on_asset=on_asset,
            on_batch=on_batch,
        )
    except KeyboardInterrupt:
        _write_progress(status="interrupted")
        print("interrupted=1")
        print(f"progress_path={(labels_dir / PROGRESS_NAME).as_posix()}")
        raise

    labeled = _concat_checkpoints(ckpt_dir, batch_files)
    labeled = _join_adv(labeled, snapshots) if not labeled.empty else labeled
    _write_by_strategy(labeled, labels_dir, strategy_ids)
    _write_progress(status="done")
    _print_strategy_summary(labeled, labels_dir, strategy_ids)
    print(f"progress_path={(labels_dir / PROGRESS_NAME).as_posix()}")
    return 0


def _print_strategy_summary(
    labeled: pd.DataFrame,
    labels_dir: Path,
    strategy_ids: Sequence[str],
) -> None:
    for sid in strategy_ids:
        part = labeled.loc[labeled["strategy_id"] == sid] if not labeled.empty else labeled
        n = 0 if part.empty else len(part)
        n_open = 0 if part.empty else int((part["outcome"] == "OPEN").sum())
        path = labels_dir / f"setups_{sid}.parquet"
        print(f"strategy_id={sid}")
        print(f"setups={n}")
        print(f"open={n_open}")
        print(f"resolved={n - n_open}")
        print(f"labels_path={path.as_posix()}")


def _load_resume(
    manifest_path: Path,
    digest: str,
    strategy_ids: Sequence[str],
) -> tuple[set[str], int]:
    payload = read_json(manifest_path)
    stored_hash = payload.get("config_hash")
    if stored_hash != digest:
        raise ScoutConfigError(
            f"checkpoint config_hash {stored_hash!r} != current {digest!r}; "
            "use --no-resume or a different --labels-dir"
        )
    stored_ids = payload.get("strategy_ids")
    if stored_ids != list(strategy_ids):
        raise ScoutConfigError(
            f"checkpoint strategy_ids {stored_ids!r} != current {list(strategy_ids)!r}"
        )
    done_raw = payload.get("done_asset_ids", [])
    if not isinstance(done_raw, list):
        raise ScoutDataError(f"{manifest_path}: done_asset_ids must be a list")
    next_batch = int(payload.get("next_batch", 1))
    return {str(x) for x in done_raw}, next_batch


def _existing_batch_files(manifest_path: Path) -> list[str]:
    if not manifest_path.is_file():
        return []
    payload = read_json(manifest_path)
    raw = payload.get("batch_files", [])
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw]


def _concat_checkpoints(ckpt_dir: Path, batch_files: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name in batch_files:
        path = ckpt_dir / name
        if not path.is_file():
            raise ScoutDataError(f"checkpoint batch missing: {path}")
        frames.append(read_parquet(path))
    if not frames:
        return empty_label_frame()
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return empty_label_frame()
    return combined.loc[:, list(LABEL_COLUMNS)]


def _load_snapshots(cfg: ScoutConfig) -> pd.DataFrame:
    path = Path(cfg.universe.snapshots_path)
    if not path.is_file():
        raise ScoutDataError(f"universe snapshots not found: {path}; run build-universe first")
    return read_parquet(path)


def _load_benchmark(processed_dir: Path) -> BenchmarkPanel:
    path = processed_dir.parent / "reference" / "benchmark_1d.parquet"
    if not path.is_file():
        raise ScoutDataError(
            f"benchmark not found: {path}; expected data/reference/benchmark_1d.parquet"
        )
    frame = read_parquet(path)
    missing = [c for c in BENCHMARK_COLUMNS if c not in frame.columns]
    if missing:
        raise ScoutDataError(f"benchmark missing columns: {missing}")
    return BenchmarkPanel(frame.loc[:, list(BENCHMARK_COLUMNS)])


def _load_calendar(cfg: ScoutConfig) -> pd.DataFrame:
    path = reference_calendar_path(Path(cfg.data.processed_dir), cfg.data.calendar)
    if not path.is_file():
        raise ScoutDataError(f"calendar not found: {path}; run adjust first")
    return read_parquet(path)


def _etf_map(tickers: pd.DataFrame) -> dict[str, bool]:
    if tickers.empty or "asset_id" not in tickers.columns:
        return {}
    has_etf = "is_etf" in tickers.columns
    out: dict[str, bool] = {}
    for rec in tickers.to_dict("records"):
        aid = str(rec["asset_id"])
        out[aid] = bool(rec["is_etf"]) if has_etf else False
    return out


def _load_earnings(raw_dir: Path) -> dict[str, tuple[EarningsEvent, ...]]:
    path = raw_dir / "earnings.parquet"
    if not path.is_file():
        return {}
    frame = read_parquet(path)
    if frame.empty:
        return {}
    grouped: dict[str, list[EarningsEvent]] = {}
    for rec in frame.to_dict("records"):
        aid = str(rec["asset_id"])
        available = rec.get("available_ts")
        available_ts: datetime | None
        if available is None or pd.isna(available):
            available_ts = None
        else:
            ts = pd.Timestamp(available)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            available_ts = ts.to_pydatetime()
        event = EarningsEvent(
            asset_id=aid,
            earnings_date=_as_date(rec["earnings_date"]),
            is_confirmed=bool(rec.get("is_confirmed", False)),
            available_ts=available_ts,
            timing=str(rec.get("timing", "UNKNOWN")),
        )
        grouped.setdefault(aid, []).append(event)
    return {aid: tuple(events) for aid, events in grouped.items()}


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed: date = pd.Timestamp(value).date()
    return parsed


def _write_by_strategy(
    frame: pd.DataFrame,
    labels_dir: Path,
    strategy_ids: Sequence[str],
) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for sid in strategy_ids:
        if frame.empty:
            part = empty_label_frame()
        else:
            part = frame.loc[frame["strategy_id"] == sid].reset_index(drop=True)
            part = empty_label_frame() if part.empty else part.loc[:, list(LABEL_COLUMNS)]
        write_parquet_atomic(labels_dir / f"setups_{sid}.parquet", part)
