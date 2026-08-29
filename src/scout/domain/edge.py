import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from scout.domain._checks import require_aware
from scout.domain.enums import Direction, VolBucket

# Documented default in 07-EDGE_AND_SCORING.md §5. Config arrives in M0.3;
# this is the value `is_usable` uses until then.
MIN_BIN_SAMPLES = 100

_EDGE_COLUMNS: tuple[str, ...] = (
    "strategy_id",
    "direction",
    "vol_bucket",
    "as_of",
    "n",
    "mean_r",
    "std_r",
    "ev_r_lcb",
    "mean_bars_held",
    "win_rate",
    "avg_win_r",
    "avg_loss_r",
    "target_rate",
    "stop_rate",
    "time_rate",
    "median_r",
    "p05_r",
    "p95_r",
)


@dataclass(frozen=True, slots=True)
class BinKey:
    """The conditioning cell whose history is used to estimate this setup's edge."""

    strategy_id: str
    direction: Direction
    vol_bucket: VolBucket

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.strategy_id, self.direction.value, self.vol_bucket.value)


@dataclass(frozen=True, slots=True)
class BinStats:
    """Outcome statistics for one bin, computed from setups RESOLVED strictly
    before `as_of`. All *_r fields are in units of one trade's risk.
    """

    key: BinKey
    as_of: datetime

    n: int  # resolved setups in the sample
    mean_r: float  # mean realised R
    std_r: float  # sample standard deviation of realised R
    ev_r_lcb: float  # one-sided lower confidence bound on mean_r
    mean_bars_held: float

    # diagnostics only; never enter the ranking statistic
    win_rate: float  # fraction with realised_r > 0
    avg_win_r: float
    avg_loss_r: float  # negative
    target_rate: float  # fraction resolved as TARGET
    stop_rate: float
    time_rate: float
    median_r: float
    p05_r: float
    p95_r: float

    def __post_init__(self) -> None:
        require_aware(self.as_of, "as_of")

    @property
    def is_usable(self) -> bool:
        """n >= min_bin_samples AND std_r > 0 AND all values finite."""
        if self.n < MIN_BIN_SAMPLES:
            return False
        if not (self.std_r > 0.0):
            return False
        numeric = (
            self.mean_r,
            self.std_r,
            self.ev_r_lcb,
            self.mean_bars_held,
            self.win_rate,
            self.avg_win_r,
            self.avg_loss_r,
            self.target_rate,
            self.stop_rate,
            self.time_rate,
            self.median_r,
            self.p05_r,
            self.p95_r,
        )
        return all(math.isfinite(x) for x in numeric)


class EdgeTable:
    """Bin statistics indexed by (BinKey, as_of). Built once per run by the
    labeling pass; queried per decision. Lookup must be O(log n), not a scan.

    Precomputed on a coarse grid (default: monthly `as_of` points) and looked up
    with backward search, so a decision at 2023-04-17 uses the 2023-04-01 table.
    """

    def __init__(self, stats: Sequence[BinStats] = ()) -> None:
        grouped: dict[BinKey, list[BinStats]] = {}
        for row in stats:
            grouped.setdefault(row.key, []).append(row)
        self._by_key: dict[BinKey, tuple[list[datetime], list[BinStats]]] = {}
        for key, rows in grouped.items():
            ordered = sorted(rows, key=lambda s: s.as_of)
            as_ofs = [s.as_of for s in ordered]
            if len(as_ofs) != len(set(as_ofs)):
                raise ValueError(f"duplicate as_of for bin {key.as_tuple()}")
            self._by_key[key] = (as_ofs, ordered)

    def lookup(self, key: BinKey, as_of: datetime) -> BinStats | None:
        require_aware(as_of, "as_of")
        slot = self._by_key.get(key)
        if slot is None:
            return None
        as_ofs, rows = slot
        i = bisect.bisect_right(as_ofs, as_of) - 1
        if i < 0:
            return None
        return rows[i]

    def save(self, path: Path) -> None:
        records: list[dict[str, object]] = []
        for _, rows in self._by_key.items():
            for stats in rows[1]:
                records.append(
                    {
                        "strategy_id": stats.key.strategy_id,
                        "direction": stats.key.direction.value,
                        "vol_bucket": stats.key.vol_bucket.value,
                        "as_of": stats.as_of,
                        "n": stats.n,
                        "mean_r": stats.mean_r,
                        "std_r": stats.std_r,
                        "ev_r_lcb": stats.ev_r_lcb,
                        "mean_bars_held": stats.mean_bars_held,
                        "win_rate": stats.win_rate,
                        "avg_win_r": stats.avg_win_r,
                        "avg_loss_r": stats.avg_loss_r,
                        "target_rate": stats.target_rate,
                        "stop_rate": stats.stop_rate,
                        "time_rate": stats.time_rate,
                        "median_r": stats.median_r,
                        "p05_r": stats.p05_r,
                        "p95_r": stats.p95_r,
                    }
                )
        frame = pd.DataFrame(records, columns=list(_EDGE_COLUMNS))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        frame.to_parquet(tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "EdgeTable":
        frame = pd.read_parquet(path)
        stats: list[BinStats] = []
        for rec in frame.to_dict(orient="records"):
            as_of = rec["as_of"]
            if isinstance(as_of, pd.Timestamp):
                as_of = as_of.to_pydatetime()
            stats.append(
                BinStats(
                    key=BinKey(
                        strategy_id=str(rec["strategy_id"]),
                        direction=Direction(str(rec["direction"])),
                        vol_bucket=VolBucket(str(rec["vol_bucket"])),
                    ),
                    as_of=as_of,
                    n=int(rec["n"]),
                    mean_r=float(rec["mean_r"]),
                    std_r=float(rec["std_r"]),
                    ev_r_lcb=float(rec["ev_r_lcb"]),
                    mean_bars_held=float(rec["mean_bars_held"]),
                    win_rate=float(rec["win_rate"]),
                    avg_win_r=float(rec["avg_win_r"]),
                    avg_loss_r=float(rec["avg_loss_r"]),
                    target_rate=float(rec["target_rate"]),
                    stop_rate=float(rec["stop_rate"]),
                    time_rate=float(rec["time_rate"]),
                    median_r=float(rec["median_r"]),
                    p05_r=float(rec["p05_r"]),
                    p95_r=float(rec["p95_r"]),
                )
            )
        return cls(stats)
