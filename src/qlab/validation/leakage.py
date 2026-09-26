"""Look-ahead detection (spec §14.2).

A point-in-time function maps a panel to an array whose row t may only depend on panel rows 0..t
(data up to and including the close of day t). Two checks, both on a sample of cut days t:

1. truncation: fn(panel.slice(t + 1))[t] equals fn(panel)[t],
2. perturbation: replacing all data after t with noise leaves rows 0..t of the output unchanged.

The first catches functions that use future rows, the second additionally catches functions whose
output depends on the length of the input (e.g. full-sample normalization).
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from qlab.data.panel import Panel

PanelFn = Callable[[Panel], np.ndarray]


@dataclass(frozen=True)
class LeakReport:
    name: str
    cut_days: list[int]
    truncation_failures: list[int]
    perturbation_failures: list[int]

    @property
    def ok(self) -> bool:
        return not self.truncation_failures and not self.perturbation_failures


def perturb_after(panel: Panel, t: int, rng: np.random.Generator) -> Panel:
    """Copy of `panel` with every row after `t` replaced by noise (row t itself kept)."""
    out = {}
    for field in ("ret_co", "ret_oc", "close_u", "dollar_volume"):
        m = getattr(panel, field).copy()
        tail = m[t + 1:]
        m[t + 1:] = np.where(np.isnan(tail), tail, tail * rng.uniform(0.5, 1.5, tail.shape)
                             + rng.normal(0, 0.01, tail.shape))
        out[field] = m
    for field in ("tradable", "listed", "delisting"):
        m = getattr(panel, field).copy()
        m[t + 1:] = rng.random(m[t + 1:].shape) < 0.5
        out[field] = m
    return replace(panel, **out)


def _same(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.allclose(a, b, rtol=tol, atol=tol, equal_nan=True)


def check_point_in_time(fn: PanelFn, panel: Panel, name: str = "", n_cuts: int = 12,
                        min_day: int = 1, seed: int = 0, tol: float = 1e-12) -> LeakReport:
    rng = np.random.default_rng(seed)
    n_days = panel.shape[0]
    cuts = sorted(set(rng.integers(min_day, n_days - 1, n_cuts).tolist()) | {n_days - 2})
    full = np.asarray(fn(panel))
    if full.shape[0] != n_days:
        raise ValueError(f"{name}: output has {full.shape[0]} rows, panel has {n_days}")

    trunc_fail, pert_fail = [], []
    for t in cuts:
        if not _same(np.asarray(fn(panel.slice(t + 1)))[t], full[t], tol):
            trunc_fail.append(t)
        if not _same(np.asarray(fn(perturb_after(panel, t, rng)))[: t + 1], full[: t + 1], tol):
            pert_fail.append(t)
    return LeakReport(name, cuts, trunc_fail, pert_fail)


def assert_point_in_time(fn: PanelFn, panel: Panel, name: str = "", **kwargs) -> None:
    report = check_point_in_time(fn, panel, name, **kwargs)
    if not report.ok:
        raise AssertionError(
            f"look-ahead in {name or fn}: truncation fails at {report.truncation_failures}, "
            f"perturbation fails at {report.perturbation_failures}")
