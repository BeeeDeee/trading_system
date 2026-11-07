"""NSGA-II search (DEAP selection), random search and the ledger of research 22.

Objectives: maximise Q25 of the fold-wise excess IR at 2x costs, minimise annual turnover, minimise effective
nodes. Fitness is a pure function of the normal form of a tree and of the run's jitter seed, so it is cached in
the ledger and equivalent genomes are counted once. DEAP draws from Python's global `random` in its tournament,
so every run seeds it (`random.seed`); everything else uses a numpy Generator derived from (SEED, seed).
"""

import random
from dataclasses import asdict, dataclass, field

import numpy as np
from deap import base, tools

from qlab.research22.compact import Compact
from qlab.research22.fitness import N_JITTER, Fitness, evaluate_tree
from qlab.research22.grammar import Evaluator, Tree, tree_size
from qlab.research22.setup import (FITNESS_COST_MULTIPLIER, MAX_DEPTH, MAX_NODES, P_CX, P_MUT, SEED, TOPK)
from qlab.research22.trees import crossover, mutate, normal_form, random_tree

INFEASIBLE = (-10.0, 200.0, float(MAX_NODES))   # objective vector of a degenerate score (sorts last)


class _Fit(base.Fitness):
    weights = (1.0, -1.0, -1.0)


class Ind:
    __slots__ = ("tree", "fitness")

    def __init__(self, tree: Tree):
        self.tree = tree
        self.fitness = _Fit()


def objectives(f: Fitness) -> tuple[float, float, float]:
    return (f.q25, f.turnover_year, float(f.nodes)) if f.feasible else INFEASIBLE


@dataclass
class Ledger:
    """Every evaluated normal form with its fitness and the generation it first appeared in."""

    rows: dict[Tree, Fitness] = field(default_factory=dict)
    first_gen: dict[Tree, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)

    def add(self, tree: Tree, fit: Fitness, gen: int) -> None:
        self.rows[tree] = fit
        self.first_gen[tree] = gen

    def to_records(self) -> list[dict]:
        return [{"tree": repr(t), "gen": self.first_gen[t], **asdict(f)} for t, f in self.rows.items()]


@dataclass
class SearchResult:
    seed: int
    population: list[tuple[Tree, Fitness]]
    front: list[tuple[Tree, Fitness]]
    history: list[dict]
    ledger: Ledger
    n_backtests: int

    def best_q25(self) -> float:
        feasible = [f.q25 for _, f in self.ledger.rows.items() if f.feasible]
        return max(feasible) if feasible else float("nan")


class Searcher:
    """Evaluation context shared by the GA, the random search and the nulls."""

    def __init__(self, ev: Evaluator, data: Compact | None = None, jitter_seed: int = SEED,
                 n_jitter: int = N_JITTER, topk: int = TOPK, cost_scale: float = FITNESS_COST_MULTIPLIER):
        self.ev, self.data = ev, data
        self.jitter_seed, self.n_jitter, self.topk, self.cost_scale = jitter_seed, n_jitter, topk, cost_scale
        self.ledger = Ledger()
        self.n_backtests = 0

    def fitness(self, tree: Tree, gen: int) -> tuple[Tree, Fitness]:
        nf = normal_form(tree)
        hit = self.ledger.rows.get(nf)
        if hit is None:
            hit = evaluate_tree(self.ev, nf, self.jitter_seed, self.n_jitter, self.topk, self.cost_scale, self.data)
            self.ledger.add(nf, hit, gen)
            self.n_backtests += self.n_jitter
        return nf, hit


def _ramped(rng: np.random.Generator) -> Tree:
    return random_tree(rng, "float", int(rng.integers(2, MAX_DEPTH + 1)), MAX_NODES)


def _assign(ind: Ind, fit: Fitness) -> None:
    ind.fitness.values = objectives(fit)


def run_ga(searcher: Searcher, seed: int, pop_size: int, n_gen: int, p_cx: float = P_CX,
           p_mut: float = P_MUT, progress=None) -> SearchResult:
    """NSGA-II (mu + lambda) with typed-tree crossover and mutation. Duplicates of normal forms already in the
    population are replaced by random immigrants, so the population never collapses onto one genome."""
    if pop_size % 4:
        raise ValueError("pop_size must be divisible by 4 (DEAP tournament)")
    random.seed(seed)
    rng = np.random.default_rng([SEED, seed])
    history: list[dict] = []

    def make(tree: Tree, gen: int) -> Ind:
        nf, fit = searcher.fitness(tree, gen)
        ind = Ind(nf)
        _assign(ind, fit)
        return ind

    pop: list[Ind] = []
    seen: set[Tree] = set()
    while len(pop) < pop_size:
        ind = make(_ramped(rng), 0)
        if ind.tree not in seen:
            seen.add(ind.tree)
            pop.append(ind)
    pop = tools.selNSGA2(pop, len(pop))              # also assigns crowding distances

    def log(gen: int) -> None:
        feasible = [ind.fitness.values[0] for ind in pop if tuple(ind.fitness.values) != INFEASIBLE]
        history.append({"gen": gen, "best_q25": max(feasible) if feasible else None,
                        "median_q25": float(np.median(feasible)) if feasible else None,
                        "feasible_share": len(feasible) / len(pop), "ledger": len(searcher.ledger),
                        "backtests": searcher.n_backtests, "cache_hits": searcher.ev.hits})
        if progress is not None:
            progress(history[-1])

    log(0)
    for gen in range(1, n_gen + 1):
        parents = tools.selTournamentDCD(pop, len(pop))
        children: list[Tree] = []
        for i in range(0, len(parents), 2):
            a, b = parents[i].tree, parents[(i + 1) % len(parents)].tree
            if rng.random() < p_cx:
                a, b = crossover(rng, a, b), crossover(rng, b, a)
            children += [a, b]
        offspring: list[Ind] = []
        taken = {ind.tree for ind in pop}
        for tree in children[: len(pop)]:
            if rng.random() < p_mut:
                tree = mutate(rng, tree)
            nf = normal_form(tree)
            tries = 0
            while nf in taken and tries < 10:        # duplicate genome -> immigrant
                nf = normal_form(_ramped(rng))
                tries += 1
            taken.add(nf)
            offspring.append(make(nf, gen))
        pop = tools.selNSGA2(pop + offspring, pop_size)
        log(gen)

    ranked = [(ind.tree, searcher.ledger.rows[ind.tree]) for ind in pop]
    front = [(ind.tree, searcher.ledger.rows[ind.tree])
             for ind in tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
             if searcher.ledger.rows[ind.tree].feasible]
    return SearchResult(seed, ranked, front, history, searcher.ledger, searcher.n_backtests)


def run_random(searcher: Searcher, seed: int, n: int, max_attempts_factor: int = 5) -> Ledger:
    """N2: `n` distinct random normal forms from the same grammar, evaluated with the same fitness."""
    rng = np.random.default_rng([SEED, 2, seed])
    start = len(searcher.ledger)
    attempts = 0
    while len(searcher.ledger) - start < n and attempts < n * max_attempts_factor:
        attempts += 1
        nf = normal_form(_ramped(rng))
        if nf not in searcher.ledger.rows:
            searcher.fitness(nf, 0)
    return searcher.ledger


def effective_size(tree: Tree) -> int:
    return tree_size(normal_form(tree))
