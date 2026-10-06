"""Mechanism test: an event study with a placebo, run by the judge before any strategy is simulated.

A card may declare `mechanism_test` (the event its mechanism is about and one primary horizon). The Builder
implements `diagnostic.py: events(data, params) -> (mask, side)`: `mask[t, j]` is True when the event happens to
instrument j at the close of day t (decided from rows <= t, like a strategy), `side[t, j]` is +1 where the
mechanism predicts the instrument outperforms afterwards and -1 where it predicts underperformance (None = +1).

The judge measures, on dev rows only, the forward return from the next open to the close of day t+h minus the
return of the control (the mean of the other eligible instruments on the same date, so market moves cancel),
sign-adjusted by `side`. Then the same statistic is computed for 200 placebos: the whole event set shifted in
time by a random number of rows (the cross-section of events, their clustering and their count are kept, only
the link to the following returns is cut). The mechanism passes if the observed mean abnormal return beats the
95th percentile of the placebos, keeps its sign in at least two of three sub-periods, there are enough events,
and the gross abnormal return per event covers the round-trip cost of trading it.

It is the cheap, informative death: "the mechanism is absent" instead of "the strategy lost to the benchmark".
It counts as one trial of the card's family (horizons other than the primary are descriptive only).
"""

import numpy as np

from lab.framework.data import DataView

MAX_HORIZON = 60


def forward_returns(view: DataView, h: int) -> tuple[np.ndarray, np.ndarray]:
    """F[t, j]: return of buying at the open of day t+1 and selling at the close of day t+h; valid[t, j]: the
    order can fill (tradable at t+1), the window exists, and the value is finite."""
    T, N = view.shape
    g = np.log(np.maximum((1.0 + view.ret_co.astype(np.float64)) * (1.0 + view.ret_oc.astype(np.float64)), 1e-6))
    L = np.cumsum(g, axis=0)
    co = np.log(np.maximum(1.0 + view.ret_co.astype(np.float64), 1e-6))
    F = np.full((T, N), np.nan)
    if T > h + 1:
        F[: T - h] = np.expm1(L[h:] - L[: T - h] - co[1: T - h + 1])
    valid = np.zeros((T, N), dtype=bool)
    valid[: T - h] = view.tradable[1: T - h + 1] & np.isfinite(F[: T - h])
    return F, valid


def eligible(view: DataView) -> np.ndarray:
    return (view.listed if view.universe is None else view.universe) & view.tradable


def _stats(Fz: np.ndarray, valid: np.ndarray, S_all: np.ndarray, n_all: np.ndarray, mask: np.ndarray,
           side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-date sum of side-adjusted abnormal returns and number of events (date needs >= 3 non-event controls)."""
    E = mask & valid
    nE = E.sum(axis=1)
    sE = (Fz * E).sum(axis=1)
    n_c = n_all - nE
    ok = (nE > 0) & (n_c >= 3)
    control = np.where(ok, (S_all - sE) / np.maximum(n_c, 1), 0.0)
    sg = np.where(E, side, 0.0)
    A = (Fz * sg).sum(axis=1)
    B = sg.sum(axis=1)
    return np.where(ok, A - control * B, 0.0), np.where(ok, nE, 0)


def event_study(view: DataView, mask: np.ndarray, side: np.ndarray | None, horizon: int, *,
                horizons: tuple[int, ...] = (), placebo_runs: int = 200, n_blocks: int = 3,
                cost_rate: np.ndarray | None = None, seed: int = 0) -> dict:
    """The mechanism statistics for one event set on `view` (dev rows only). Pure numpy, no state."""
    T, N = view.shape
    side = np.ones((T, N)) if side is None else np.where(np.asarray(side) < 0, -1.0, 1.0)
    mask = np.asarray(mask, dtype=bool) & eligible(view)
    F, valid = forward_returns(view, horizon)
    valid &= eligible(view)
    Fz = np.where(valid, F, 0.0)
    S_all, n_all = Fz.sum(axis=1), valid.sum(axis=1)
    sums, counts = _stats(Fz, valid, S_all, n_all, mask, side)
    n_events = int(counts.sum())
    out = {"horizon_days": horizon, "n_events": n_events, "n_event_dates": int((counts > 0).sum()),
           "events_per_year": round(n_events / max(T / 252.0, 1e-9), 1)}
    if n_events == 0:
        return out | {"mean_abnormal": None}
    obs = float(sums.sum() / n_events)
    rng = np.random.default_rng(seed)
    placebo = []
    lo, hi = horizon + 22, T - horizon - 22
    for _ in range(placebo_runs if hi > lo else 0):
        k = int(rng.integers(lo, hi))
        s, c = _stats(Fz, valid, S_all, n_all, np.roll(mask, k, axis=0), np.roll(side, k, axis=0))
        if c.sum() > 0:
            placebo.append(float(s.sum() / c.sum()))
    placebo = np.array(placebo)
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = []
    for a, b in zip(edges[:-1], edges[1:]):
        n = int(counts[a:b].sum())
        blocks.append({"events": n, "mean_abnormal": float(sums[a:b].sum() / n) if n else None})
    out |= {"mean_abnormal": obs, "blocks": blocks,
            "placebo": {"runs": len(placebo), "mean": float(placebo.mean()) if len(placebo) else None,
                        "sd": float(placebo.std(ddof=1)) if len(placebo) > 1 else None,
                        "p95": float(np.quantile(placebo, 0.95)) if len(placebo) else None},
            "placebo_percentile": float((placebo < obs).mean()) if len(placebo) else None}
    if len(placebo) > 1 and placebo.std(ddof=1) > 0:
        out["z_vs_placebo"] = float((obs - placebo.mean()) / placebo.std(ddof=1))
    if cost_rate is not None:
        e = mask & valid
        nxt = np.vstack([cost_rate[1:], cost_rate[-1:]])
        out["round_trip_cost"] = float(2.0 * nxt[e].mean()) if e.any() else None
        out["cost_ratio"] = obs / out["round_trip_cost"] if out["round_trip_cost"] else None
    by_h = {}
    for h in horizons:
        if h == horizon or not 1 <= h <= MAX_HORIZON:
            continue
        Fh, vh = forward_returns(view, h)
        vh &= eligible(view)
        Fhz = np.where(vh, Fh, 0.0)
        s, c = _stats(Fhz, vh, Fhz.sum(axis=1), vh.sum(axis=1), mask, side)
        by_h[str(h)] = float(s.sum() / c.sum()) if c.sum() else None
    if by_h:
        out["other_horizons_mean_abnormal"] = by_h        # descriptive only, never a gate
    return out


def evaluate(stats: dict, th: dict) -> dict:
    """Checks (same shape as the gate checks) from `event_study` output and the thresholds of G1.mechanism."""
    from lab.framework.evaluator import _check
    obs = stats.get("mean_abnormal")
    pct = stats.get("placebo_percentile")
    blocks = [b["mean_abnormal"] for b in stats.get("blocks", []) if b["mean_abnormal"] is not None]
    cr = stats.get("cost_ratio")
    return {
        "mechanism_events": _check(stats["n_events"] if stats["n_event_dates"] >= th["min_event_dates"] else 0,
                                   ">=", th["min_events"]),
        "mechanism_abnormal_return": _check(obs if obs is not None else float("nan"), ">", 0.0),
        "mechanism_placebo_percentile": _check(pct if pct is not None else float("nan"), ">=",
                                               th["min_placebo_percentile"]),
        "mechanism_blocks": _check(sum(b > 0 for b in blocks), ">=", th["min_blocks_same_sign"]),
        "mechanism_costs": _check(cr if cr is not None else float("nan"), ">=", th["min_cost_ratio"]),
    }
