"""Late-stage evaluation of a frozen ensemble (gate G6) and the frozen-selection file of research 22.

`prepare_frozen` runs on dev data only and fixes everything the late stage needs: the ensemble E, the random-search
ensemble E_rand, the validation champion C, the canonical strategies, 1000 random strategies matched to E in turnover
and size, an optional hand-written simple strategy D, and the worst dev-fold relative drawdown of E.
`late_report` then evaluates those frozen trees on the late stage, once. The late stage is "late validation", not a
clean holdout (README: 2020-2026 is consumed), so a pass is a forward candidate, nothing more.

G6: excess IR of E > 0 at 1x costs and >= 0 at 2x costs, above E_rand at 1x, and the relative drawdown of E versus the
equal-weight universe (maximum drawdown of the wealth ratio E / benchmark) at most 1.5 times its worst dev fold.
"""

import hashlib
from math import sqrt
from pathlib import Path

import numpy as np

from qlab.research22.compact import Compact
from qlab.research22.fitness import excess_ir, fold_indices
from qlab.research22.grammar import Evaluator, Tree, validate
from qlab.research22.search import Ledger
from qlab.research22.selection import ir_per_column, streams
from qlab.research22.setup import ANN_WEEKS, MAX_DEPTH, MAX_NODES, TOPK
from qlab.research22.store import tree_from_repr
from qlab.validation.stats import bootstrap_ci

CANONICAL: dict[str, Tree] = {
    "mom_12_1": ("sub", ("leaf", "ret_250"), ("leaf", "ret_20")),   # 12-month minus last month (log returns)
    "reversal_1w": ("neg", ("leaf", "ret_5")),
    "lowvol_60d": ("neg", ("leaf", "vol_60")),
}
DRAWDOWN_FACTOR = 1.5


def rel_drawdown(net: np.ndarray, bench: np.ndarray) -> float:
    """Maximum drawdown of the wealth ratio of the strategy to the benchmark (0 = never below its peak)."""
    ratio = np.cumprod((1.0 + net) / (1.0 + bench))
    peak = np.maximum.accumulate(np.concatenate(([1.0], ratio)))[1:]
    return float((1.0 - ratio / peak).max())


def perf(net: np.ndarray) -> dict:
    wealth = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    sd = net.std(ddof=1)
    return {"cagr": float(wealth[-1] ** (ANN_WEEKS / len(net)) - 1.0),
            "sharpe": float(net.mean() / sd * sqrt(ANN_WEEKS)) if sd > 0 else 0.0,
            "max_dd": float((1.0 - wealth / peak).max())}


def ensemble_series(ev: Evaluator, compact: Compact, trees: list[Tree], topk: int = TOPK,
                    scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(net returns of the equal-weight ensemble, benchmark net returns); no netting of trades between members."""
    net, bench = streams(ev, compact, trees, topk, scale)
    return net.mean(axis=1), bench


def single_ir(excess: np.ndarray) -> float:
    return float(excess_ir(excess, [np.arange(len(excess))])[0])


def worst_fold_rel_dd(ev: Evaluator, dev: Compact, trees: list[Tree], topk: int = TOPK) -> float:
    net, bench = ensemble_series(ev, dev, trees, topk, 1.0)
    return max(rel_drawdown(net[f], bench[f]) for f in fold_indices(len(net), dev.n_folds))


def matched_random(ledger: Ledger, turnover_year: float, nodes: float, n: int = 1000) -> list[Tree]:
    """The `n` feasible random trees closest to (turnover, nodes) in relative distance; ties by repr."""
    rows = [(abs(f.turnover_year - turnover_year) / max(turnover_year, 1e-9)
             + abs(f.nodes - nodes) / max(nodes, 1e-9), repr(t), t) for t, f in ledger.rows.items() if f.feasible]
    rows.sort(key=lambda r: (r[0], r[1]))
    return [t for _, _, t in rows[:n]]


def g6_decision(ir_1x: float, ir_2x: float, ir_rand_1x: float, rel_dd: float, worst_dev_dd: float,
                factor: float = DRAWDOWN_FACTOR) -> bool:
    return bool(ir_1x > 0 and ir_2x >= 0 and ir_1x > ir_rand_1x and rel_dd <= factor * worst_dev_dd)


def prepare_frozen(selection: dict, random_ledger: Ledger, ev_dev: Evaluator, dev: Compact, run: str,
                   d_tree: Tree | None = None, n_matched: int = 1000, topk: int = TOPK) -> dict:
    """Everything the late stage needs, computed from dev/validation results only (trees as Python objects)."""
    for t in list(selection["ensemble_trees"]) + [t for t in CANONICAL.values()] + ([d_tree] if d_tree else []):
        validate(t, MAX_DEPTH, MAX_NODES)
    e = selection["ensemble_trees"]
    profile = selection.get("ensemble_profile") or {}
    return {
        "run": run, "gates": {k: selection.get(k) for k in ("G1", "G2", "G3", "G4")},
        "ensemble_trees": e, "random_ensemble_trees": selection.get("random_ensemble_trees", []),
        "random_ensemble_fallback": bool(selection.get("random_ensemble_fallback", False)),
        "champion_tree": selection.get("champion_tree"), "canonical": dict(CANONICAL), "D": d_tree,
        "ensemble_profile": profile,
        "matched_random_trees": (matched_random(random_ledger, profile["turnover_year"], profile["nodes"], n_matched)
                                 if e and profile else []),
        "worst_dev_fold_rel_dd": worst_fold_rel_dd(ev_dev, dev, e, topk) if e else None,
    }


def late_report(frozen: dict, ev: Evaluator, late: Compact, *, topk: int = TOPK, n_boot: int = 1000,
                seed: int = 0) -> dict:
    """Evaluate the frozen trees on `late` (the only place a late-stage number is produced)."""
    e = frozen["ensemble_trees"]
    if not e:
        return {"passed": False, "reason": "empty ensemble (G3 failed)"}
    out: dict = {"n_members": len(e), "n_weeks": int(late.shape[0])}
    series = {}
    for scale in (1.0, 2.0, 3.0):
        net, bench = ensemble_series(ev, late, e, topk, scale)
        series[scale] = (net, bench)
        out[f"ir_{int(scale)}x"] = single_ir(net - bench)
    net1, bench1 = series[1.0]
    ex = net1 - bench1
    point, lo, hi = bootstrap_ci(ex, lambda x: float(x.mean() / x.std(ddof=1) * sqrt(ANN_WEEKS)), n_boot=n_boot,
                                 mean_block=4.0, alpha=0.10, seed=seed)
    out["ir_1x_ci90"] = [lo, hi]
    out["ensemble_perf_1x"], out["benchmark_perf_1x"] = perf(net1), perf(bench1)
    out["rel_drawdown"] = rel_drawdown(net1, bench1)
    out["worst_dev_fold_rel_drawdown"] = frozen["worst_dev_fold_rel_dd"]

    rand = frozen["random_ensemble_trees"]
    if rand:
        rn, rb = ensemble_series(ev, late, rand, topk, 1.0)
        out["ir_random_ensemble_1x"] = single_ir(rn - rb)
    else:
        out["ir_random_ensemble_1x"] = float("-inf")
    out["random_ensemble_fallback"] = frozen["random_ensemble_fallback"]
    if frozen.get("champion_tree"):
        cn, cb = streams(ev, late, [frozen["champion_tree"]], topk, 1.0)
        out["ir_champion_1x"] = single_ir(cn[:, 0] - cb)
    out["ir_canonical_1x"] = {}
    for name, tree in frozen["canonical"].items():
        cn, cb = streams(ev, late, [tree], topk, 1.0)
        out["ir_canonical_1x"][name] = single_ir(cn[:, 0] - cb)
    matched = frozen["matched_random_trees"]
    if matched:
        mn, mb = streams(ev, late, matched, topk, 1.0)
        m_ir = ir_per_column(mn - mb[:, None])
        out["matched_random"] = {"n": len(matched), "median_ir_1x": float(np.median(m_ir)),
                                 "share_above_ensemble": float((m_ir >= out["ir_1x"]).mean())}
    if frozen.get("D"):
        for scale in (1.0, 2.0):
            dn, db = streams(ev, late, [frozen["D"]], topk, scale)
            out[f"ir_D_{int(scale)}x"] = single_ir(dn[:, 0] - db)
    out["passed"] = g6_decision(out["ir_1x"], out["ir_2x"], out["ir_random_ensemble_1x"], out["rel_drawdown"],
                                frozen["worst_dev_fold_rel_dd"])
    return out


_TREE_LISTS = ("ensemble_trees", "random_ensemble_trees", "matched_random_trees")


def frozen_to_json(frozen: dict) -> dict:
    """Frozen selection with every tree as its `repr` (JSON-friendly)."""
    out = dict(frozen)
    for key in _TREE_LISTS:
        out[key] = [repr(t) for t in frozen[key]]
    out["champion_tree"] = repr(frozen["champion_tree"]) if frozen["champion_tree"] else None
    out["canonical"] = {k: repr(v) for k, v in frozen["canonical"].items()}
    out["D"] = repr(frozen["D"]) if frozen["D"] else None
    return out


def frozen_from_json(data: dict) -> dict:
    out = dict(data)
    for key in _TREE_LISTS:
        out[key] = [tree_from_repr(t) for t in data[key]]
    out["champion_tree"] = tree_from_repr(data["champion_tree"]) if data["champion_tree"] else None
    out["canonical"] = {k: tree_from_repr(v) for k, v in data["canonical"].items()}
    out["D"] = tree_from_repr(data["D"]) if data["D"] else None
    return out


def hash_files(files: list[str]) -> str:
    """Hash of file names and contents in the given order (the methodology freeze)."""
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode() + b"\0" + Path(f).read_bytes() + b"\0")
    return h.hexdigest()
