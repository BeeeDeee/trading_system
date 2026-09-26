"""Walk-forward folds and the stitched out-of-sample ensemble portfolio (spec §9.3).

Fold y: training window = [train_start, last day of y-1 minus the embargo], test = calendar year y.
The selected members' own decisions (computed point in time over the whole history) are combined
position by position: on any member's decision day the ensemble target is the sum over members of
weight_each * that member's latest target. A new fold's members take over with a forced decision
after the close of the last day before the test year, so switching pays full costs on the net
difference of positions.
"""

from dataclasses import dataclass

import numpy as np

from qlab.engine.vector import Decisions


@dataclass(frozen=True)
class Fold:
    year: int
    train_start: int   # row index (inclusive)
    train_end: int     # row index (exclusive), after the embargo
    test_start: int    # first row of the test year
    test_end: int      # row after the last test day


def make_folds(dates: np.ndarray, first_test_year: int, last_test_year: int, train_start: int,
               embargo: int) -> list[Fold]:
    years = np.asarray(dates, dtype="datetime64[Y]").astype(int) + 1970
    folds = []
    for y in range(first_test_year, last_test_year + 1):
        rows = np.flatnonzero(years == y)
        if len(rows) == 0:
            continue
        test_start, test_end = int(rows[0]), int(rows[-1]) + 1
        folds.append(Fold(y, train_start, test_start - embargo, test_start, test_end))
    return folds


class SparseDecisions:
    """Memory-light copy of a candidate's Decisions (nonzero weights only)."""

    def __init__(self, d: Decisions):
        self.days = np.asarray(d.days)
        self.rows = [(np.flatnonzero(w), w[np.flatnonzero(w)]) for w in d.weights]

    def latest(self, day: int) -> tuple[np.ndarray, np.ndarray] | None:
        k = int(np.searchsorted(self.days, day, side="right")) - 1
        return self.rows[k] if k >= 0 else None


def stitch_decisions(folds: list[Fold], members: list[list[int]], weight_each: float,
                     decisions: dict[int, SparseDecisions], n_assets: int) -> Decisions:
    """Ensemble decisions over all test years (cash before the first fold)."""
    days_out, rows_out = [], []
    for fold, mem in zip(folds, members):
        start = fold.test_start - 1  # decide after the close of the last pre-test day
        days = {start}
        for c in mem:
            d = decisions[c].days
            days.update(d[(d >= start) & (d < fold.test_end - 1)].tolist())
        for day in sorted(days):
            w = np.zeros(n_assets)
            for c in mem:
                latest = decisions[c].latest(day)
                if latest is not None:
                    w[latest[0]] += weight_each * latest[1]
            days_out.append(day)
            rows_out.append(w)
    return Decisions(np.array(days_out, dtype=int), np.array(rows_out))
