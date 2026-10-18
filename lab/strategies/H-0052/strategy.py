"""H-0052 Macro-release volatility crush: short VIXY over the open-to-open step containing each scheduled
CPI, Employment Situation or FOMC release, plus `post_steps` following steps. SHY, IEF, TLT, AGG, GLD are
never traded (unmarked peers for the event study).

Decision after the close of session s (row s), executed at the open of s+1:
  release_step(s) = next_cpi_sessions(s) == 2 or next_empsit_sessions(s) == 2 or next_fomc_sessions(s) == 1
  weight(VIXY, s) = -1.0 if release_step(s - k) for any k in 0..post_steps, else 0.0
Missing calendar on row s -> no position (0). VIXY without a bar on row s -> NaN (position unchanged).
"""
import numpy as np

PARAMS = {"post_steps": 1}

CAL = "macro_release_calendar_1d."
KEYS = ("next_cpi_sessions", "next_empsit_sessions", "next_fomc_sessions")
TRADED = "VIXY"
PEERS = ("SHY", "IEF", "TLT", "AGG", "GLD")
MAX_GAP = 5


def calendar_inputs(data):
    """Return (cpi, emp, fomc, missing) as (T,) arrays; raise on interior gaps longer than MAX_GAP sessions."""
    cols = [np.asarray(data.series[CAL + k], dtype=float).reshape(-1) for k in KEYS]
    missing = ~(np.isfinite(cols[0]) & np.isfinite(cols[1]) & np.isfinite(cols[2]))
    valid = np.flatnonzero(~missing)
    if valid.size >= 2:
        gaps = np.diff(valid) - 1
        if gaps.max() > MAX_GAP:
            raise ValueError("macro_release_calendar_1d: more than %d consecutive sessions without data" % MAX_GAP)
    return cols[0], cols[1], cols[2], missing


def release_steps(cpi, emp, fomc):
    """(T,) bool: the open-to-open step starting at the next open contains a scheduled release."""
    with np.errstate(invalid="ignore"):
        # counts are integers; rint only guards against float storage, it never changes an integer value
        return (np.rint(cpi) == 2.0) | (np.rint(emp) == 2.0) | (np.rint(fomc) == 1.0)


def target_weights(data, params):
    post_steps = int(params["post_steps"])
    if post_steps < 0:
        raise ValueError("post_steps must be >= 0")
    cpi, emp, fomc, missing = calendar_inputs(data)
    rel = release_steps(cpi, emp, fomc)
    T = rel.shape[0]
    N = len(data.instruments)

    # window active on row s if a release step started on any of rows s-post_steps..s (past rows only)
    active = rel.copy()
    for k in range(1, post_steps + 1):
        if k < T:
            active[k:] |= rel[:-k]

    j = data.col(TRADED)
    for p in PEERS:
        data.col(p)  # fail loudly if a peer is missing from the universe

    w = np.zeros((T, N), dtype=float)
    vixy = np.where(active, -1.0, 0.0)
    vixy[missing] = 0.0                      # calendar unknown on this session: hold no position
    no_bar = ~np.isfinite(np.asarray(data.close, dtype=float)[:, j])
    vixy[no_bar] = np.nan                    # VIXY has no bar: position unchanged
    w[:, j] = vixy
    return w
