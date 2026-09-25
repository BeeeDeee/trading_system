"""Holdout lockbox: 3 evaluations, ever. Dirty trees are refused."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from scout.research.lockbox import FORCE_HOLDOUT_WARNING, request_holdout_evaluation
from scout.research.splits import HOLDOUT_START, period_overlaps_holdout
from scout.utils.clock import BarClock
from scout.utils.errors import ScoutConfigError


def _lock(tmp_path: Path, *, used: int = 0, budget: int = 3) -> Path:
    path = tmp_path / "holdout_lockbox.json"
    path.write_text(
        f'{{"budget": {budget}, "used": {used}, "evaluations": []}}\n',
        encoding="utf-8",
    )
    return path


def test_refuses_fourth_holdout_evaluation(tmp_path: Path) -> None:
    path = _lock(tmp_path)
    clock = BarClock(datetime(2024, 6, 1, tzinfo=UTC))
    for i in range(3):
        request_holdout_evaluation(
            f"eval {i}",
            "abc123",
            path=path,
            run_id=f"r{i}",
            clock=clock,
        )
    with pytest.raises(ScoutConfigError, match="exhausted"):
        request_holdout_evaluation(
            "eval 3",
            "abc123",
            path=path,
            run_id="r3",
            clock=clock,
        )


def test_force_consumes_budget_after_exhaustion(tmp_path: Path) -> None:
    path = _lock(tmp_path, used=3)
    clock = BarClock(datetime(2024, 6, 1, tzinfo=UTC))
    request_holdout_evaluation(
        "forced peek",
        "abc123",
        path=path,
        run_id="forced",
        forced=True,
        clock=clock,
    )
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["used"] == 4
    assert raw["evaluations"][-1]["forced"] is True


def test_refuses_dirty_git_tree(tmp_path: Path) -> None:
    path = _lock(tmp_path)
    with pytest.raises(ScoutConfigError, match="clean git"):
        request_holdout_evaluation(
            "peek",
            "abc123--dirty",
            path=path,
            run_id="r0",
            clock=BarClock(datetime(2024, 6, 1, tzinfo=UTC)),
        )


def test_force_does_not_bypass_dirty_git(tmp_path: Path) -> None:
    path = _lock(tmp_path)
    with pytest.raises(ScoutConfigError, match="clean git"):
        request_holdout_evaluation(
            "peek",
            "abc123--dirty",
            path=path,
            run_id="r0",
            forced=True,
            clock=BarClock(datetime(2024, 6, 1, tzinfo=UTC)),
        )


def test_period_overlaps_holdout() -> None:
    assert not period_overlaps_holdout(
        datetime(1998, 1, 1, tzinfo=UTC),
        datetime(2017, 12, 31, 23, 59, 59, tzinfo=UTC),
    )
    assert period_overlaps_holdout(
        datetime(1998, 1, 1, tzinfo=UTC),
        datetime(2018, 1, 1, tzinfo=UTC),
    )
    assert period_overlaps_holdout(HOLDOUT_START, datetime(2026, 8, 1, tzinfo=UTC))


def test_force_warning_is_loud() -> None:
    assert "WARNING" in FORCE_HOLDOUT_WARNING
    assert "--force-holdout" in FORCE_HOLDOUT_WARNING
    assert len(FORCE_HOLDOUT_WARNING) > 80
