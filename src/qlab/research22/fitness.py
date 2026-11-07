"""Fold-quantile fitness of research 22 (spec: robust fitness at 2x costs, jittered copies).

f1 = Q25 over the dev folds of the excess information ratio (net of costs, versus the equal-weight eligible
universe with the same rebalance and costs); each fold's IR is averaged over K jittered copies of the tree
(lookback +-1 step in its family, constants x U(0.8, 1.2)). f2 = annual turnover, f3 = node count (raw size:
simplification to "effective" nodes is added with the search driver). Everything is a pure function of
(tree, seed) so a generation shares the same jitter draws.
"""

from dataclasses import dataclass

import numpy as np

from qlab.research22.compact import LEAF_NAMES, RET_LAGS, VOL_LAGS, Compact
from qlab.research22.grammar import Evaluator, Tree, tree_size
from qlab.research22.setup import (ANN_WEEKS, FITNESS_COST_MULTIPLIER, FITNESS_QUANTILE, MIN_DISPERSION_SHARE,
                                   TOPK)

N_JITTER = 4
_FAMILIES = {"ret": RET_LAGS, "vol": VOL_LAGS}


@dataclass(frozen=True)
class Fitness:
    q25: float
    turnover_year: float
    nodes: int
    dispersion: float
    feasible: bool
    fold_irs: tuple[float, ...]


def _shift(name: str, step: int) -> str:
    """Move the lookback of a terminal by `step` positions inside its family (clipped at the ends)."""
    z = name[0] == "z" and name[1:3].isdigit()
    if z:                                    # z{w}_{base}: shift the base's lookback, keep the window
        window, base = name[1:].split("_", 1)
        return f"z{window}_{_shift(base, step)}"
    family, _, lag = name.partition("_")
    if family not in _FAMILIES or not lag.isdigit():
        return name
    lags = _FAMILIES[family]
    k = min(max(lags.index(int(lag)) + step, 0), len(lags) - 1)
    return f"{family}_{lags[k]}"


def jitter_tree(tree: Tree, rng: np.random.Generator) -> Tree:
    """Perturbed copy for the robustness average. Z-score terminals only exist for the bases of `Z_BASES`:
    if a shifted name has no terminal, the original is kept."""

    def go(t: Tree) -> Tree:
        if t[0] == "leaf":
            name = _shift(t[1], int(rng.integers(-1, 2)))
            return ("leaf", name if name in LEAF_NAMES else t[1])
        if t[0] == "const":
            return ("const", float(t[1] * rng.uniform(0.8, 1.2)))
        if t[0] == "state":
            return t
        return (t[0],) + tuple(go(c) for c in t[1:])

    return go(tree)


def fold_indices(n_weeks: int, n_folds: int) -> list[np.ndarray]:
    return np.array_split(np.arange(n_weeks), n_folds)


def excess_ir(excess: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(len(folds))
    for k, f in enumerate(folds):
        sd = excess[f].std(ddof=1)
        out[k] = excess[f].mean() / sd * np.sqrt(ANN_WEEKS) if sd > 1e-12 else 0.0
    return out


def dispersion(score: np.ndarray) -> float:
    """Share of decision weeks in which the score separates names (>= 2 distinct finite values)."""
    finite = np.isfinite(score)
    hi = np.where(finite, score, -np.inf).max(axis=1)
    lo = np.where(finite, score, np.inf).min(axis=1)
    return float((hi > lo).mean())


def evaluate_tree(ev: Evaluator, tree: Tree, seed: int, n_jitter: int = N_JITTER, topk: int = TOPK,
                  cost_scale: float = FITNESS_COST_MULTIPLIER, data: Compact | None = None) -> Fitness:
    """Fitness of `tree`. Scores come from `ev` (terminals only); returns come from `data`, which defaults to
    `ev.c`. A permuted-return null dataset has the same terminals, so one evaluator and its cache serve them all."""
    c = ev.c if data is None else data
    folds = fold_indices(c.shape[0], c.n_folds)
    bench = c.benchmark(cost_scale).net
    irs, turns, disps = [], [], []
    for k in range(n_jitter):
        copy = jitter_tree(tree, np.random.default_rng([seed, k])) if n_jitter > 1 else tree
        score = ev.score(copy)
        w = c.weekly(score, topk, cost_scale)
        irs.append(excess_ir(w.net - bench, folds))
        turns.append(w.turnover.mean() * ANN_WEEKS)
        disps.append(dispersion(score))
    fold_irs = np.mean(irs, axis=0)
    disp = float(np.mean(disps))
    q25 = float(np.quantile(fold_irs, FITNESS_QUANTILE))
    ok = bool(np.isfinite(q25) and disp >= MIN_DISPERSION_SHARE)
    return Fitness(q25, float(np.mean(turns)), tree_size(tree), disp, ok, tuple(float(x) for x in fold_irs))
