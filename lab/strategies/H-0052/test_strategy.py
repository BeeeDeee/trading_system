import numpy as np
import pytest

from strategy import target_weights, PARAMS

INSTR = ("VIXY", "SHY", "IEF", "TLT", "AGG", "GLD")
NAN = np.nan


class View:
    """Minimal stand-in exposing only the fields the strategy reads."""

    def __init__(self, cpi, emp, fomc, close_vixy=None, tradable=None):
        T = len(cpi)
        self.instruments = INSTR
        self.series = {
            "macro_release_calendar_1d.next_cpi_sessions": np.asarray(cpi, dtype=float),
            "macro_release_calendar_1d.next_empsit_sessions": np.asarray(emp, dtype=float),
            "macro_release_calendar_1d.next_fomc_sessions": np.asarray(fomc, dtype=float),
        }
        self.close = np.full((T, len(INSTR)), 10.0)
        if close_vixy is not None:
            self.close[:, 0] = close_vixy
        self.tradable = np.ones((T, len(INSTR)), dtype=bool) if tradable is None else tradable
        self.universe = None

    def col(self, name):
        return self.instruments.index(name)


def countdown(T, release_rows):
    """next_X_sessions for rows 0..T-1 given release session rows (strictly after s)."""
    out = np.full(T, NAN)
    for s in range(T):
        later = [r for r in release_rows if r > s]
        if later:
            out[s] = later[0] - s
    return out


def base(T=20):
    cpi = countdown(T, [6, 40])        # CPI on row 6 -> release step decided at close of row 4
    emp = countdown(T, [14, 40])       # jobs on row 14 -> decided at row 12
    fomc = countdown(T, [9, 40])       # FOMC on row 9 -> decided at row 8
    return cpi, emp, fomc


def test_release_steps_and_post_steps():
    cpi, emp, fomc = base()
    w = target_weights(View(cpi, emp, fomc), {"post_steps": 1})
    short = np.flatnonzero(w[:, 0] == -1.0)
    assert list(short) == [4, 5, 8, 9, 12, 13]
    assert np.all(w[:, 1:] == 0.0)
    w0 = target_weights(View(cpi, emp, fomc), {"post_steps": 0})
    assert list(np.flatnonzero(w0[:, 0] == -1.0)) == [4, 8, 12]
    w2 = target_weights(View(cpi, emp, fomc), {"post_steps": 2})
    assert list(np.flatnonzero(w2[:, 0] == -1.0)) == [4, 5, 6, 8, 9, 10, 12, 13, 14]


def test_overlap_merges_not_doubled():
    T = 12
    cpi = countdown(T, [5, 40])        # step at row 3
    emp = countdown(T, [6, 40])        # step at row 4
    fomc = countdown(T, [40])
    w = target_weights(View(cpi, emp, fomc), {"post_steps": 1})
    assert list(np.flatnonzero(w[:, 0] == -1.0)) == [3, 4, 5]
    assert np.nanmin(w[:, 0]) == -1.0
    assert np.all(np.nansum(np.abs(w), axis=1) <= 1.0 + 1e-12)


def test_missing_calendar_flat_and_trailing_ok():
    cpi, emp, fomc = base()
    cpi[13:] = NAN                     # trailing rows missing: no position, no error
    w = target_weights(View(cpi, emp, fomc), {"post_steps": 1})
    assert np.all(w[13:, 0] == 0.0)
    assert w[12, 0] == -1.0


def test_interior_gap_raises():
    cpi, emp, fomc = base()
    fomc[2:8] = NAN                    # 6 consecutive missing sessions
    with pytest.raises(ValueError):
        target_weights(View(cpi, emp, fomc), PARAMS)
    fomc2 = base()[2]
    fomc2[2:7] = NAN                   # 5 is allowed
    w = target_weights(View(cpi, emp, fomc2), PARAMS)
    assert np.all(w[2:7, 0] == 0.0)


def test_no_vixy_bar_keeps_position():
    cpi, emp, fomc = base()
    close = np.full(20, 10.0)
    close[:2] = NAN
    close[5] = NAN
    w = target_weights(View(cpi, emp, fomc, close_vixy=close), PARAMS)
    assert np.all(np.isnan(w[:2, 0]))
    assert np.isnan(w[5, 0])
    assert w[4, 0] == -1.0


def test_point_in_time_truncation():
    cpi, emp, fomc = base(30)
    full = target_weights(View(cpi, emp, fomc), PARAMS)
    for t in range(1, 30):
        part = target_weights(View(cpi[: t + 1], emp[: t + 1], fomc[: t + 1]), PARAMS)
        np.testing.assert_array_equal(part, full[: t + 1])
