"""Bin statistics, LCB, monthly as_of grid, causal sample construction."""

from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from scout.cli.__main__ import main
from scout.config.hashing import config_hash
from scout.config.schema import EdgeConfig, LcbMethod
from scout.data.store import write_parquet_atomic
from scout.domain.edge import MIN_BIN_SAMPLES, BinKey, BinStats, EdgeTable
from scout.domain.enums import Direction, SetupOutcome, VolBucket
from scout.scoring.edge import (
    REQUIRED_EDGE_COLUMNS,
    build_edge_table,
    compute_bin_stats,
    monthly_as_of_grid,
    normal_lcb,
)
from scout.scoring.labeling import LABEL_COLUMNS
from scout.utils.errors import ScoutLookaheadError

REL = 1e-12
APRIL_FIRST_SESSION = datetime(2023, 4, 3, 20, 0, tzinfo=UTC)
APRIL_17 = datetime(2023, 4, 17, 20, 0, tzinfo=UTC)
MARCH_1 = datetime(2023, 3, 1, 21, 0, tzinfo=UTC)
MAY_1 = datetime(2023, 5, 1, 20, 0, tzinfo=UTC)

# Hand-computed: r = [1.0, -1.0, 0.5, -0.5]
# mean = 0
# sample std ddof=1 = sqrt(2.5 / 3) = sqrt(5/6)
_HAND_R = (1.0, -1.0, 0.5, -0.5)
_HAND_STD = math.sqrt(2.5 / 3.0)
_HAND_N = 4
_HAND_Z = 1.28
_HAND_LCB = 0.0 - _HAND_Z * _HAND_STD / math.sqrt(_HAND_N)


def _cfg(**overrides: Any) -> EdgeConfig:
    kwargs: dict[str, Any] = {}
    kwargs.update(overrides)
    return EdgeConfig(**kwargs)


def _key(**overrides: Any) -> BinKey:
    kwargs: dict[str, Any] = {
        "strategy_id": "donchian_breakout_v1",
        "direction": Direction.LONG,
        "vol_bucket": VolBucket.MID,
    }
    kwargs.update(overrides)
    return BinKey(**kwargs)


def _row(
    *,
    resolution_ts: datetime | None,
    realised_r_gross: float,
    outcome: str = SetupOutcome.STOP.value,
    bars_held: int = 5,
    strategy_id: str = "donchian_breakout_v1",
    direction: str = Direction.LONG.value,
    vol_bucket: str = VolBucket.MID.value,
    setup_ts: datetime | None = None,
) -> dict[str, object]:
    if setup_ts is None:
        setup_ts = MARCH_1 if resolution_ts is None else resolution_ts - timedelta(days=7)
    return {
        "strategy_id": strategy_id,
        "direction": direction,
        "vol_bucket": vol_bucket,
        "setup_ts": setup_ts,
        "resolution_ts": resolution_ts,
        "realised_r_gross": realised_r_gross,
        "outcome": outcome,
        "bars_held": bars_held,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _sessions_mar_apr_may() -> list[datetime]:
    """XNYS-like session closes: first of March, every weekday in April, May 1."""
    out: list[datetime] = [MARCH_1]
    day = datetime(2023, 4, 3, 20, 0, tzinfo=UTC)
    while day.month == 4:
        if day.weekday() < 5:
            out.append(day)
        day = day + timedelta(days=1)
    out.append(MAY_1)
    return out


def test_normal_lcb_hand_computed() -> None:
    got = normal_lcb(0.0, _HAND_STD, _HAND_N, _HAND_Z)
    assert got == pytest.approx(_HAND_LCB, rel=REL)


def test_only_resolved_before_as_of_included() -> None:
    """A setup resolving after as_of is excluded. The causality test."""
    as_of = APRIL_FIRST_SESSION
    rows = [
        _row(
            resolution_ts=datetime(2023, 3, 15, 20, 0, tzinfo=UTC),
            realised_r_gross=1.0,
            outcome=SetupOutcome.TARGET.value,
        ),
        _row(
            resolution_ts=datetime(2023, 4, 10, 20, 0, tzinfo=UTC),
            realised_r_gross=2.0,
            outcome=SetupOutcome.TARGET.value,
        ),
        _row(
            resolution_ts=as_of,
            realised_r_gross=3.0,
            outcome=SetupOutcome.TARGET.value,
        ),
    ]
    table = build_edge_table(_frame(rows), _cfg(), sessions=_sessions_mar_apr_may())
    stats = table.lookup(_key(), as_of)
    assert stats is not None
    assert stats.n == 1
    assert stats.mean_r == pytest.approx(1.0, rel=REL)
    may = table.lookup(_key(), MAY_1)
    assert may is not None
    assert may.n == 3


def test_open_setups_excluded() -> None:
    rows = [
        _row(
            resolution_ts=datetime(2023, 3, 15, 20, 0, tzinfo=UTC),
            realised_r_gross=0.5,
        ),
        _row(resolution_ts=None, realised_r_gross=9.0, outcome=SetupOutcome.OPEN.value),
    ]
    table = build_edge_table(_frame(rows), _cfg(), sessions=_sessions_mar_apr_may())
    stats = table.lookup(_key(), APRIL_FIRST_SESSION)
    assert stats is not None
    assert stats.n == 1
    assert stats.mean_r == pytest.approx(0.5, rel=REL)


def test_lcb_below_mean() -> None:
    sample = _frame(
        [
            _row(
                resolution_ts=datetime(2023, 3, 10, 20, 0, tzinfo=UTC),
                realised_r_gross=r,
            )
            for r in _HAND_R
        ]
    )
    stats = compute_bin_stats(sample, _key(), APRIL_FIRST_SESSION, _cfg(z=_HAND_Z))
    assert stats.n == _HAND_N
    assert stats.ev_r_lcb < stats.mean_r
    assert stats.ev_r_lcb == pytest.approx(_HAND_LCB, rel=1e-9)
    assert stats.mean_r == pytest.approx(0.0, abs=1e-12)


def test_lcb_tightens_with_n() -> None:
    mean_r, std_r, z = 0.2, 1.0, 1.28
    small = normal_lcb(mean_r, std_r, 100, z)
    large = normal_lcb(mean_r, std_r, 1000, z)
    assert abs(large - mean_r) < abs(small - mean_r)


def test_lcb_widens_with_std() -> None:
    mean_r, n, z = 0.2, 100, 1.28
    narrow = normal_lcb(mean_r, 1.0, n, z)
    wide = normal_lcb(mean_r, 2.0, n, z)
    assert wide < narrow


def test_insufficient_samples_unusable() -> None:
    rows = [
        _row(
            resolution_ts=datetime(2023, 3, 1, 21, 0, tzinfo=UTC) + timedelta(minutes=i),
            realised_r_gross=float((-1) ** i) * 0.4,
        )
        for i in range(MIN_BIN_SAMPLES - 1)
    ]
    stats = compute_bin_stats(_frame(rows), _key(), MAY_1, _cfg())
    assert stats.n == MIN_BIN_SAMPLES - 1
    assert stats.is_usable is False


def test_bootstrap_matches_normal_large_n() -> None:
    rng = np.random.default_rng(20260827)
    values = rng.normal(loc=0.15, scale=1.0, size=5000)
    as_of = APRIL_FIRST_SESSION
    sample = _frame(
        [
            _row(resolution_ts=datetime(2023, 3, 2, 21, 0, tzinfo=UTC), realised_r_gross=float(v))
            for v in values
        ]
    )
    key = _key()
    normal = compute_bin_stats(sample, key, as_of, _cfg(lcb_method=LcbMethod.NORMAL))
    boot = compute_bin_stats(sample, key, as_of, _cfg(lcb_method=LcbMethod.BOOTSTRAP))
    assert boot.ev_r_lcb == pytest.approx(normal.ev_r_lcb, rel=0.05)


def test_bootstrap_deterministic() -> None:
    sample = _frame(
        [
            _row(
                resolution_ts=datetime(2023, 3, 10, 20, 0, tzinfo=UTC),
                realised_r_gross=r,
            )
            for r in _HAND_R
        ]
    )
    cfg = _cfg(lcb_method=LcbMethod.BOOTSTRAP, bootstrap_seed=20260827)
    a = compute_bin_stats(sample, _key(), APRIL_FIRST_SESSION, cfg)
    b = compute_bin_stats(sample, _key(), APRIL_FIRST_SESSION, cfg)
    assert a.ev_r_lcb == b.ev_r_lcb


def test_as_of_lookup_rounds_backward() -> None:
    """2023-04-17 uses the first trading session of April (2023-04-03), not the 1st.

    07-EDGE_AND_SCORING.md §5: the grid is sessions, not calendar dates.
    2023-04-01 was a Saturday.
    """
    rows = [
        _row(
            resolution_ts=datetime(2023, 3, 15, 20, 0, tzinfo=UTC),
            realised_r_gross=0.25,
        )
    ]
    table = build_edge_table(_frame(rows), _cfg(), sessions=_sessions_mar_apr_may())
    hit = table.lookup(_key(), APRIL_17)
    assert hit is not None
    assert hit.as_of == APRIL_FIRST_SESSION
    grid = monthly_as_of_grid(_sessions_mar_apr_may())
    assert APRIL_FIRST_SESSION in grid
    assert datetime(2023, 4, 1, 20, 0, tzinfo=UTC) not in grid


def test_lookup_is_log_n() -> None:
    key = _key()
    start = datetime(2000, 1, 3, 21, 0, tzinfo=UTC)
    rows = []
    for i in range(400):
        month = start.month - 1 + i
        year = start.year + month // 12
        month = month % 12 + 1
        as_of = datetime(year, month, 1, 21, 0, tzinfo=UTC)
        rows.append(
            BinStats(
                key=key,
                as_of=as_of,
                n=MIN_BIN_SAMPLES,
                mean_r=0.1,
                std_r=1.0,
                ev_r_lcb=0.0,
                mean_bars_held=8.0,
                win_rate=0.4,
                avg_win_r=1.0,
                avg_loss_r=-1.0,
                target_rate=0.3,
                stop_rate=0.5,
                time_rate=0.2,
                median_r=0.0,
                p05_r=-1.0,
                p95_r=2.0,
            )
        )
    table = EdgeTable(rows)
    query = datetime(2010, 6, 15, 21, 0, tzinfo=UTC)
    t0 = time.perf_counter()
    for _ in range(100_000):
        table.lookup(key, query)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"100k lookups took {elapsed:.3f}s"


def test_save_load_config_hash(tmp_path: Path) -> None:
    rows = [
        _row(
            resolution_ts=datetime(2023, 3, 15, 20, 0, tzinfo=UTC),
            realised_r_gross=0.1,
            outcome=SetupOutcome.TIME.value,
        ),
        _row(
            resolution_ts=datetime(2023, 3, 16, 20, 0, tzinfo=UTC),
            realised_r_gross=-0.2,
        ),
    ]
    digest = "abc123confighash"
    snapshot = "20260830-fixture"
    table = build_edge_table(
        _frame(rows),
        _cfg(),
        sessions=_sessions_mar_apr_may(),
        config_hash=digest,
        data_snapshot_id=snapshot,
    )
    path = tmp_path / "edge_table.parquet"
    table.save(path)
    loaded = EdgeTable.load(path)
    assert loaded.config_hash == digest
    assert loaded.data_snapshot_id == snapshot
    got = loaded.lookup(_key(), APRIL_17)
    assert got is not None
    assert got.n == 2


def test_causal_tripwire_raises() -> None:
    dirty = _frame(
        [
            _row(
                resolution_ts=datetime(2023, 4, 10, 20, 0, tzinfo=UTC),
                realised_r_gross=1.0,
            )
        ]
    )
    with pytest.raises(ScoutLookaheadError, match="resolution_ts"):
        compute_bin_stats(dirty, _key(), APRIL_FIRST_SESSION, _cfg())


@given(
    values=st.lists(
        st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=40,
    ),
    z=st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
)
def test_lcb_never_above_mean(values: list[float], z: float) -> None:
    std = float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
    assume(math.isfinite(std) and std > 0.0)
    mean = float(np.mean(values))
    lcb = normal_lcb(mean, std, len(values), z)
    assert lcb <= mean + 1e-12


def test_diagnostics_on_hand_sample() -> None:
    sample = _frame(
        [
            _row(
                resolution_ts=datetime(2023, 3, 10, 20, 0, tzinfo=UTC),
                realised_r_gross=1.0,
                outcome=SetupOutcome.TARGET.value,
                bars_held=3,
            ),
            _row(
                resolution_ts=datetime(2023, 3, 11, 20, 0, tzinfo=UTC),
                realised_r_gross=-1.0,
                outcome=SetupOutcome.STOP.value,
                bars_held=2,
            ),
            _row(
                resolution_ts=datetime(2023, 3, 12, 20, 0, tzinfo=UTC),
                realised_r_gross=0.5,
                outcome=SetupOutcome.TARGET.value,
                bars_held=4,
            ),
            _row(
                resolution_ts=datetime(2023, 3, 13, 20, 0, tzinfo=UTC),
                realised_r_gross=-0.5,
                outcome=SetupOutcome.TIME.value,
                bars_held=5,
            ),
        ]
    )
    stats = compute_bin_stats(sample, _key(), APRIL_FIRST_SESSION, _cfg())
    assert stats.win_rate == pytest.approx(0.5, rel=REL)
    assert stats.avg_win_r == pytest.approx(0.75, rel=REL)
    assert stats.avg_loss_r == pytest.approx(-0.75, rel=REL)
    assert stats.target_rate == pytest.approx(0.5, rel=REL)
    assert stats.stop_rate == pytest.approx(0.25, rel=REL)
    assert stats.time_rate == pytest.approx(0.25, rel=REL)
    assert stats.mean_bars_held == pytest.approx(3.5, rel=REL)
    assert stats.median_r == pytest.approx(0.0, abs=1e-12)


def _label_row(
    *,
    resolution_ts: datetime,
    realised_r_gross: float,
    outcome: str = "STOP",
) -> dict[str, object]:
    rec: dict[str, object] = {col: None for col in LABEL_COLUMNS}
    rec.update(
        {
            "asset_id": "A1",
            "symbol": "AAA",
            "setup_ts": resolution_ts - timedelta(days=5),
            "entry_ts": resolution_ts - timedelta(days=4),
            "resolution_ts": resolution_ts,
            "strategy_id": "donchian_breakout_v1",
            "direction": "LONG",
            "regime": "TREND_UP",
            "market_regime": "RISK_ON",
            "vol_bucket": "MID",
            "reference_price": 100.0,
            "entry_price": 100.0,
            "stop_price": 95.0,
            "exit_price": 95.0,
            "target_price": 110.0,
            "risk_per_unit": 5.0,
            "reward_risk_ratio": 2.0,
            "entry_gap_atr": 0.0,
            "outcome": outcome,
            "bars_held": 5,
            "realised_r_gross": realised_r_gross,
            "mae_r": -0.4,
            "mfe_r": 0.3,
            "atr_pct": 0.02,
            "efficiency_ratio_20": 0.5,
            "beta_bench_90": 1.0,
            "adv_usd_60": 1e8,
            "mom_252_xs_pct": 0.8,
            "overnight_var_share_60": 0.4,
            "xs_population": 200.0,
            "had_earnings_in_window": False,
        }
    )
    return rec


def test_cli_build_edge_writes_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for key in list(os.environ):
        if key.startswith("SCOUT_"):
            monkeypatch.delenv(key, raising=False)

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    labels = pd.DataFrame(
        [
            _label_row(
                resolution_ts=datetime(2023, 3, 15, 20, 0, tzinfo=UTC),
                realised_r_gross=0.2,
                outcome="TARGET",
            ),
            _label_row(
                resolution_ts=datetime(2023, 3, 16, 20, 0, tzinfo=UTC),
                realised_r_gross=-0.3,
            ),
        ],
        columns=list(LABEL_COLUMNS),
    )
    write_parquet_atomic(labels_dir / "setups_donchian_breakout_v1.parquet", labels)

    sessions = _sessions_mar_apr_may()
    calendar = pd.DataFrame(
        {
            "session": [ts.date() for ts in sessions],
            "open_utc": [ts - timedelta(hours=6, minutes=30) for ts in sessions],
            "close_utc": sessions,
            "is_half_day": [False] * len(sessions),
            "session_index": list(range(len(sessions))),
        }
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "SNAPSHOT.json").write_text(
        '{"data_snapshot_id": "20260830-fixture"}\n', encoding="utf-8"
    )
    reference = tmp_path / "reference"
    write_parquet_atomic(reference / "calendar_xnys.parquet", calendar)
    table_path = tmp_path / "edge" / "edge_table.parquet"
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"""
period:
  start: 2022-09-01T20:00:00Z
  warmup_end: 2023-03-15T20:00:00Z
  end: 2023-05-01T20:00:00Z
data:
  vendor: fixture
  raw_dir: {raw_dir.as_posix()}
  processed_dir: {(tmp_path / "processed").as_posix()}
  snapshot_id_path: {(raw_dir / "SNAPSHOT.json").as_posix()}
labeling:
  labels_dir: {labels_dir.as_posix()}
edge:
  table_path: {table_path.as_posix()}
""",
        encoding="utf-8",
    )
    code = main(["build-edge", "--config", str(overlay)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert table_path.is_file()
    loaded = EdgeTable.load(table_path)
    from scout.config.loader import load_config

    digest = config_hash(load_config(overlay))
    assert loaded.config_hash == digest
    assert loaded.data_snapshot_id == "20260830-fixture"
    assert "config_hash=" in captured.out


def test_required_columns_match_schema() -> None:
    assert "resolution_ts" in REQUIRED_EDGE_COLUMNS
    assert "realised_r_gross" in REQUIRED_EDGE_COLUMNS
