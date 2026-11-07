"""Selection and diagnostics of research 22 (gates G1-G4, ensemble E; G5 is in `dna`).

Pipeline for one run (A or B), all on weekly excess returns versus the equal-weight eligible universe:

1. candidates = final populations and Pareto fronts of the GA seeds (a tree belongs to the lowest seed that has it);
2. dev streams -> drop exact duplicates -> correlation clusters -> one representative per cluster (best dev f1);
3. validation streams of the representatives at 1x costs -> StepM (FWER) and BY-FDR against H0: mean excess <= 0;
4. E = equal-weight of the representatives that pass either test (ordered by dev f1, at most `max_ensemble`);
   no netting of trades across members, so E's costs are the average of its members' costs (conservative);
5. G1 (GA vs the permutation null), G2 (GA vs random search with the same pipeline), G3, G4 and the diagnostics
   DSR (effective trials from the cluster count), MinTRL, PBO and the break-even cost.

Nothing here reads the late stage; the caller decides which compacts to pass in.
"""

from dataclasses import dataclass

import numpy as np

from qlab.research22.compact import Compact
from qlab.research22.dna import COMPONENTS, design, enrichment, frequency_table, replicated
from qlab.research22.fitness import Fitness, excess_ir
from qlab.research22.grammar import Evaluator, Tree
from qlab.research22.search import Ledger, SearchResult
from qlab.research22.setup import ANN_WEEKS, TOPK
from qlab.research22.stats22 import cluster_returns, min_track_record_length, multiple_testing, unique_columns
from qlab.validation.stats import deflated_sharpe, pbo_cscv, sharpe


@dataclass(frozen=True)
class Cand:
    tree: Tree
    fit: Fitness
    seed: int


def ga_candidates(results: list[SearchResult]) -> list[Cand]:
    """Feasible members of the final populations and fronts; duplicates go to the lowest seed."""
    seen: dict[Tree, Cand] = {}
    for res in sorted(results, key=lambda r: r.seed):
        for tree, fit in list(res.population) + list(res.front):
            if fit.feasible and tree not in seen:
                seen[tree] = Cand(tree, fit, res.seed)
    return list(seen.values())


def ledger_candidates(ledger: Ledger, top: int | None = None, seed: int = 0) -> list[Cand]:
    """Feasible ledger rows, best dev f1 first (optionally only the `top`)."""
    rows = [Cand(t, f, seed) for t, f in ledger.rows.items() if f.feasible]
    rows.sort(key=lambda c: (-c.fit.q25, repr(c.tree)))
    return rows[:top] if top else rows


def streams(ev: Evaluator, compact: Compact, trees: list[Tree], topk: int = TOPK,
            scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(net returns T x m, benchmark net returns T) of the top-k portfolios, unjittered, at `scale` x costs."""
    bench = compact.benchmark(scale).net
    net = np.empty((compact.shape[0], len(trees)))
    for j, t in enumerate(trees):
        net[:, j] = compact.weekly(ev.score(t), topk, scale).net
    return net, bench


def ir_per_column(excess: np.ndarray) -> np.ndarray:
    """Annualised information ratio of every column of a weekly excess-return matrix."""
    return np.array([excess_ir(excess[:, j], [np.arange(excess.shape[0])])[0] for j in range(excess.shape[1])])


def representatives(dev_excess: np.ndarray, cands: list[Cand]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(unique column indices, cluster label of each unique column, representative candidate indices).

    Representatives are the best dev-f1 member of each cluster, ordered by dev f1 (ties by index).
    """
    keep = unique_columns(dev_excess)
    labels = cluster_returns(dev_excess[:, keep])
    reps = []
    for c in np.unique(labels):
        members = keep[labels == c]
        reps.append(max(members, key=lambda j: (cands[j].fit.q25, -int(j))))
    reps.sort(key=lambda j: (-cands[j].fit.q25, int(j)))
    return keep, labels, reps


def break_even_cost(compact: Compact, score: np.ndarray, topk: int = TOPK, hi: float = 50.0) -> float:
    """Cost multiplier at which the mean weekly excess return of the rule becomes zero (0 if never positive)."""
    def mean_excess(m: float) -> float:
        return float((compact.weekly(score, topk, m).net - compact.benchmark(m).net).mean())

    if mean_excess(0.0) <= 0.0:
        return 0.0
    if mean_excess(hi) > 0.0:
        return float("inf")
    lo, up = 0.0, hi
    for _ in range(40):
        mid = 0.5 * (lo + up)
        lo, up = (mid, up) if mean_excess(mid) > 0 else (lo, mid)
    return 0.5 * (lo + up)


# ------------------------------------------------------------------ gates
def gate_g1(best_real: list[float], null_best: list[float]) -> dict:
    """G1: median over the real seeds of the best dev f1 must exceed the 95th percentile of the null runs."""
    real, null = np.asarray(best_real, dtype=float), np.asarray(null_best, dtype=float)
    p95 = float(np.nanquantile(null, 0.95))
    med = float(np.nanmedian(real))
    return {"median_real_best_f1": med, "null_p95": p95, "n_null": int(np.isfinite(null).sum()),
            "null_rank": float(np.nanmean(null < med)), "passed": bool(med > p95)}


def pipeline_on_pool(ev_dev: Evaluator, dev: Compact, ev_val: Evaluator, val: Compact, cands: list[Cand],
                     top: int = 20, topk: int = TOPK) -> dict:
    """Dedupe -> clusters -> representatives -> validation IR (1x and 2x costs) of the top-`top` representatives."""
    net, bench = streams(ev_dev, dev, [c.tree for c in cands], topk, 1.0)
    dev_excess = net - bench[:, None]
    keep, labels, reps = representatives(dev_excess, cands)
    chosen = reps[:top]
    trees = [cands[j].tree for j in chosen]
    n1, b1 = streams(ev_val, val, trees, topk, 1.0)
    n2, b2 = streams(ev_val, val, trees, topk, 2.0)
    return {"n_candidates": len(cands), "n_unique": int(len(keep)), "n_clusters": int(len(np.unique(labels))),
            "reps": chosen, "reps_all": reps, "n_reps_total": len(reps), "val_ir_1x": ir_per_column(n1 - b1[:, None]),
            "val_ir_2x": ir_per_column(n2 - b2[:, None]), "dev_excess": dev_excess, "keep": keep}


def select_run(ga: list[SearchResult], random_ledger: Ledger, null_best: list[float], ev_dev: Evaluator,
               dev: Compact, ev_val: Evaluator, val: Compact, *, top: int = 20, max_ensemble: int = 20,
               alpha: float = 0.10, n_boot: int = 2000, mean_block: float = 4.0, pbo_sample: int = 100_000,
               pbo_blocks: int = 16, seed: int = 0, topk: int = TOPK) -> dict:
    """Everything except the DNA gate. Returns a JSON-friendly dict plus `ensemble_trees` for the late stage."""
    out: dict = {}
    out["G1"] = gate_g1([res.best_q25() for res in ga], null_best)

    cands = ga_candidates(ga)
    if not cands:
        out.update({"G2": None, "G3": {"passed": False}, "G4": None, "ensemble_trees": []})
        return out
    pool = pipeline_on_pool(ev_dev, dev, ev_val, val, cands, top, topk)
    rand = ledger_candidates(random_ledger, top=len(cands))
    pool_r = pipeline_on_pool(ev_dev, dev, ev_val, val, rand, top, topk) if rand else None
    med_ga = float(np.median(pool["val_ir_1x"]))
    med_r = float(np.median(pool_r["val_ir_1x"])) if pool_r else float("nan")
    out["G2"] = {"median_val_ir_ga": med_ga, "median_val_ir_random": med_r, "passed": bool(med_ga > med_r)}
    out["pools"] = {"ga": {k: pool[k] for k in ("n_candidates", "n_unique", "n_clusters", "n_reps_total")},
                    "random": ({k: pool_r[k] for k in ("n_candidates", "n_unique", "n_clusters", "n_reps_total")}
                               if pool_r else None)}

    # G3: all representatives, not only the top ones
    reps_all = pool["reps_all"]
    trees_all = [cands[j].tree for j in reps_all]
    net_v, bench_v = streams(ev_val, val, trees_all, topk, 1.0)
    excess_v = net_v - bench_v[:, None]
    mt = multiple_testing(excess_v, alpha, n_boot, mean_block, seed)
    passers = [i for i in np.flatnonzero(mt.passed)]
    members = passers[:max_ensemble]
    out["G3"] = {"n_reps": len(reps_all), "n_stepm": int(mt.stepm.sum()), "n_by": int(mt.by.sum()),
                 "n_passed": len(passers), "passed": bool(len(passers) > 0)}
    out["ensemble_trees"] = [trees_all[i] for i in members]
    out["champion_tree"] = None
    out["ensemble_profile"] = None
    out["random_ensemble_trees"] = []
    out["random_ensemble_fallback"] = False

    # diagnostics
    n_hat = pool["n_clusters"]
    diag: dict = {"n_effective_trials": n_hat}
    if members:
        e_excess = net_v[:, members].mean(axis=1) - bench_v
        sr_var = float(np.var([sharpe(excess_v[:, j]) for j in range(excess_v.shape[1])], ddof=1)) \
            if excess_v.shape[1] > 1 else 0.0
        diag["dsr"] = {str(m): float(deflated_sharpe(e_excess, max(2, int(round(n_hat * m))), sr_var))
                       for m in (0.3, 1.0, 3.0)}
        # conservative row: every distinct return stream counts as an independent trial (an upper bound on N)
        diag["n_unique_series"] = pool["n_unique"]
        diag["dsr_unique"] = float(deflated_sharpe(e_excess, max(2, pool["n_unique"]), sr_var))
        diag["min_trl_weeks"] = float(min_track_record_length(e_excess))
        diag["ensemble_val_ir_1x"] = float(excess_ir(e_excess, [np.arange(len(e_excess))])[0])
        champion = trees_all[int(np.argmax(ir_per_column(excess_v)))]
        diag["champion_break_even_cost_multiplier"] = break_even_cost(val, ev_val.score(champion), topk)
        out["champion_tree"] = champion
        out["ensemble_profile"] = {
            "turnover_year": float(np.mean([cands[reps_all[i]].fit.turnover_year for i in members])),
            "nodes": float(np.mean([cands[reps_all[i]].fit.nodes for i in members]))}
        if pool_r:                                    # E_rand: same rule on the random pool, else the best by dev f1
            trees_r = [rand[j].tree for j in pool_r["reps_all"]]
            net_r, bench_r = streams(ev_val, val, trees_r, topk, 1.0)
            mt_r = multiple_testing(net_r - bench_r[:, None], alpha, n_boot, mean_block, seed + 1)
            idx_r = [int(i) for i in np.flatnonzero(mt_r.passed)][:max_ensemble]
            out["random_ensemble_fallback"] = not idx_r
            if not idx_r:
                idx_r = list(range(min(len(members), len(trees_r))))
            out["random_ensemble_trees"] = [trees_r[i] for i in idx_r]
    pbo, n_pbo = pbo_for_pool(ev_dev, dev, ev_val, val, ga, pbo_sample, pbo_blocks, seed, topk)
    out["G4"] = {"pbo": pbo, "n_pbo_candidates": n_pbo, "passed": bool(pbo < 0.5)}
    out["diagnostics"] = diag
    return out


def _pbo_pool(ga: list[SearchResult]) -> list[Tree]:
    seen: dict[Tree, None] = {}
    for res in sorted(ga, key=lambda r: r.seed):
        for t, f in res.ledger.rows.items():
            if f.feasible:
                seen.setdefault(t, None)
    return list(seen)


def pbo_for_pool(ev_dev: Evaluator, dev: Compact, ev_val: Evaluator, val: Compact, ga: list[SearchResult],
                 n: int, n_blocks: int, seed: int, topk: int) -> tuple[float, int]:
    """PBO over a deduplicated random sample of everything the GA evaluated, on dev + validation weeks."""
    pool = _pbo_pool(ga)
    rng = np.random.default_rng([22, seed, 4])
    if len(pool) > n:
        pool = [pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))]
    nd, bd = streams(ev_dev, dev, pool, topk, 1.0)
    nv, bv = streams(ev_val, val, pool, topk, 1.0)
    both = np.vstack([nd - bd[:, None], nv - bv[:, None]])
    both = both[:, unique_columns(both)]
    pbo, _ = pbo_cscv(both, n_blocks)
    return float(pbo), int(both.shape[1])


# ------------------------------------------------------------------ G5: strategy DNA
def dna_half(ev_dev: Evaluator, dev: Compact, ev_val: Evaluator, val: Compact, cands: list[Cand],
             topk: int = TOPK) -> dict:
    """Cluster-level rows of one seed half: trees, robust label, controls and seed strata.

    robust = validation excess IR > 0 at 1x costs and >= 0 at 2x costs (the representative survived the later
    period); controls are annual turnover, effective nodes and the dev f1.
    """
    net, bench = streams(ev_dev, dev, [c.tree for c in cands], topk, 1.0)
    _, _, reps = representatives(net - bench[:, None], cands)
    chosen = [cands[j] for j in reps]
    trees = [c.tree for c in chosen]
    n1, b1 = streams(ev_val, val, trees, topk, 1.0)
    n2, b2 = streams(ev_val, val, trees, topk, 2.0)
    ir1, ir2 = ir_per_column(n1 - b1[:, None]), ir_per_column(n2 - b2[:, None])
    return {"trees": trees, "y": ((ir1 > 0) & (ir2 >= 0)).astype(int),
            "controls": np.array([[c.fit.turnover_year, c.fit.nodes, c.fit.q25] for c in chosen]),
            "strata": np.array([c.seed for c in chosen]), "val_ir_1x": ir1, "val_ir_2x": ir2}


def gate_g5(ga: list[SearchResult], random_ledger: Ledger, ev_dev: Evaluator, dev: Compact, ev_val: Evaluator,
            val: Compact, *, split_seed: int = 5, n_perm: int = 5000, seed: int = 0, topk: int = TOPK,
            prior_sample: int = 2000) -> dict:
    """G5: a component enriched among robust representatives, replicated across the two seed halves."""
    cands = ga_candidates(ga)
    first = [c for c in cands if c.seed <= split_seed]
    second = [c for c in cands if c.seed > split_seed]
    if len(first) < 4 or len(second) < 4:
        return {"passed": False, "replicated": [], "reason": "too few candidates in a seed half"}
    rows = [dna_half(ev_dev, dev, ev_val, val, half, topk) for half in (first, second)]
    enr = [enrichment(design(r["trees"]), r["y"], r["controls"], r["strata"], n_perm, seed + k)
           for k, r in enumerate(rows)]
    repl = replicated(enr[0], enr[1])
    surv = rows[0]["trees"] + rows[1]["trees"]
    robust = [t for r in rows for t, y in zip(r["trees"], r["y"]) if y]
    prior = [t for t in list(random_ledger.rows)[:prior_sample]]
    table = frequency_table({"prior": prior, "survivors": surv, "robust": robust})
    return {"passed": bool(repl), "replicated": repl,
            "n_rows": [len(r["trees"]) for r in rows], "n_robust": [int(r["y"].sum()) for r in rows],
            "min_possible_q": [e["min_possible_q"] for e in enr],
            "enrichment": {name: {"or": [float(enr[k]["odds_ratio"][i]) for k in (0, 1)],
                                  "q": [float(enr[k]["q"][i]) for k in (0, 1)]}
                           for i, name in enumerate(COMPONENTS)},
            "frequency": {name: {g: float(v[i]) for g, v in table.items()} for i, name in enumerate(COMPONENTS)}}
