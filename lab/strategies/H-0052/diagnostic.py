"""H-0052 mechanism events: VIXY marked with side -1 after the close of session s when a scheduled release
falls in the next open-to-open step (next_cpi_sessions(s) == 2 or next_empsit_sessions(s) == 2 or
next_fomc_sessions(s) == 1). SHY, IEF, TLT, AGG, GLD are never marked (same-date peers)."""
import numpy as np

CAL = "macro_release_calendar_1d."
KEYS = ("next_cpi_sessions", "next_empsit_sessions", "next_fomc_sessions")
TRADED = "VIXY"
PEERS = ("SHY", "IEF", "TLT", "AGG", "GLD")
MAX_GAP = 5


def events(data, params):
    cols = [np.asarray(data.series[CAL + k], dtype=float).reshape(-1) for k in KEYS]
    missing = ~(np.isfinite(cols[0]) & np.isfinite(cols[1]) & np.isfinite(cols[2]))
    valid = np.flatnonzero(~missing)
    if valid.size >= 2 and (np.diff(valid) - 1).max() > MAX_GAP:
        raise ValueError("macro_release_calendar_1d: more than %d consecutive sessions without data" % MAX_GAP)
    with np.errstate(invalid="ignore"):
        rel = (np.rint(cols[0]) == 2.0) | (np.rint(cols[1]) == 2.0) | (np.rint(cols[2]) == 1.0)

    j = data.col(TRADED)
    for p in PEERS:
        data.col(p)
    T = rel.shape[0]
    N = len(data.instruments)
    ok = np.asarray(data.tradable, dtype=bool)[:, j]
    if data.universe is not None:
        ok = ok & np.asarray(data.universe, dtype=bool)[:, j]
    mask = np.zeros((T, N), dtype=bool)
    mask[:, j] = rel & ok
    side = np.zeros((T, N), dtype=float)
    side[:, j] = -1.0
    return mask, side
