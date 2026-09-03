"""Holdout lockbox. Budget of 3; dirty trees never produce a holdout result."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from scout.research.splits import period_overlaps_holdout
from scout.utils.clock import Clock, WallClock
from scout.utils.errors import ScoutConfigError

LOCKBOX_PATH = Path("experiments") / "holdout_lockbox.json"

FORCE_HOLDOUT_WARNING = (
    "\n"
    "************************************************************************\n"
    "***  WARNING: --force-holdout is consuming lockbox budget anyway.    ***\n"
    "***  This is an auditable peek, recorded with forced: true.          ***\n"
    "***  The holdout is converted into training data by this act.        ***\n"
    "************************************************************************\n"
)


def request_holdout_evaluation(
    reason: str,
    git_sha: str,
    *,
    path: Path | None = None,
    run_id: str = "",
    forced: bool = False,
    clock: Clock | None = None,
) -> None:
    """Spend one lockbox evaluation. Raises ScoutConfigError if refused."""
    if git_sha.endswith("--dirty"):
        raise ScoutConfigError("holdout requires a clean git tree")
    lock_path = path if path is not None else LOCKBOX_PATH
    state = _load(lock_path)
    used = int(state["used"])
    budget = int(state["budget"])
    if used >= budget and not forced:
        raise ScoutConfigError(
            f"holdout lockbox budget is exhausted ({used}/{budget})"
        )
    now = (clock or WallClock()).now()
    evaluations = list(state.get("evaluations") or [])
    evaluations.append(
        {
            "ts": now.isoformat(),
            "git_sha": git_sha,
            "reason": reason,
            "run_id": run_id,
            "forced": bool(forced),
        }
    )
    state["used"] = used + 1
    state["evaluations"] = evaluations
    _save(lock_path, state)


def period_requires_lockbox(split: str, start: datetime, end: datetime) -> bool:
    if split == "HOLDOUT":
        return True
    return period_overlaps_holdout(start, end)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ScoutConfigError(f"holdout lockbox file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScoutConfigError(f"{path}: invalid lockbox file") from exc
    if not isinstance(raw, dict):
        raise ScoutConfigError(f"{path}: invalid lockbox file")
    try:
        int(raw["used"])
        int(raw["budget"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoutConfigError(f"{path}: invalid lockbox file") from exc
    return raw


def _save(path: Path, state: dict[str, Any]) -> None:
    text = json.dumps(state, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
