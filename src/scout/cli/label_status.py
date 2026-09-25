"""Print labeling progress.json. Safe to run while labeling is in flight."""

from __future__ import annotations

import argparse
from pathlib import Path

from scout.cli.label_setups import DEFAULT_LABELS_DIR, PROGRESS_NAME
from scout.data.store import read_json
from scout.utils.errors import ScoutDataError


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help=f"Directory with {PROGRESS_NAME} (default: {DEFAULT_LABELS_DIR.as_posix()})",
    )


def run(args: argparse.Namespace) -> int:
    path = Path(args.labels_dir) / PROGRESS_NAME
    if not path.is_file():
        raise ScoutDataError(f"progress file not found: {path}")
    payload = read_json(path)
    keys = (
        "status",
        "note",
        "config_hash",
        "pid",
        "assets_done",
        "assets_total",
        "assets_skipped_resume",
        "setups",
        "warm_bars_seen",
        "assets_per_hour",
        "eta_s",
        "elapsed_s",
        "last_asset_id",
        "updated_utc",
        "labels_dir",
    )
    for key in keys:
        if key in payload:
            print(f"{key}={payload[key]}")
    print(f"progress_path={path.as_posix()}")
    done = payload.get("assets_done")
    total = payload.get("assets_total")
    if isinstance(done, int) and isinstance(total, int) and total > 0:
        pct = 100.0 * done / total
        print(f"pct={pct:.2f}")
    return 0
