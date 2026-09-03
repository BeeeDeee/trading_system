"""Trial registry. One row per backtest run; the trial count feeds DSR."""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime
from pathlib import Path

from scout.config.loader import git_tree_is_clean
from scout.utils.errors import ScoutError

REGISTRY_COLUMNS: tuple[str, ...] = (
    "trial_id",
    "run_id",
    "timestamp_utc",
    "git_sha",
    "config_hash",
    "split",
    "period_start",
    "period_end",
    "strategies",
    "n_trades",
    "mean_r",
    "sharpe",
    "max_dd_pct",
    "total_return_pct",
    "notes",
)


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    sha = result.stdout.strip() if result.returncode == 0 else "unknown"
    if not git_tree_is_clean():
        return f"{sha}--dirty"
    return sha


def read_registry(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def development_trial_count(path: Path | None) -> int:
    """DEVELOPMENT rows already in the registry. Conservative: all of them.

    12-RESEARCH_PROTOCOL.md §4 asks for runs since the last strategy-code
    change. The registry does not record that event, so undercounting is the
    dangerous direction (it inflates deflated Sharpe). Counting every
    DEVELOPMENT row never undercounts.
    """
    if path is None:
        return 0
    return sum(1 for row in read_registry(path) if row.get("split") == "DEVELOPMENT")


def append_trial(
    path: Path,
    *,
    run_id: str,
    git_sha: str,
    config_hash: str,
    split: str,
    period_start: str,
    period_end: str,
    strategies: str,
    n_trades: int,
    mean_r: float,
    sharpe: float,
    max_dd_pct: float,
    total_return_pct: float,
    notes: str,
    timestamp_utc: datetime,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_registry(path)
    trial_id = 1 if not existing else max(int(r["trial_id"]) for r in existing) + 1
    if timestamp_utc.tzinfo is None:
        raise ScoutError("registry timestamp_utc must be timezone-aware")
    row = {
        "trial_id": str(trial_id),
        "run_id": run_id,
        "timestamp_utc": timestamp_utc.isoformat(),
        "git_sha": git_sha,
        "config_hash": config_hash,
        "split": split,
        "period_start": period_start,
        "period_end": period_end,
        "strategies": strategies,
        "n_trades": str(n_trades),
        "mean_r": _fmt(mean_r),
        "sharpe": _fmt(sharpe),
        "max_dd_pct": _fmt(max_dd_pct),
        "total_return_pct": _fmt(total_return_pct),
        "notes": notes,
    }
    write_header = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REGISTRY_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return trial_id


def _fmt(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):
        return ""
    return f"{value:.10g}"
