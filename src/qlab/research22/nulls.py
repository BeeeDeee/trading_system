"""Null datasets for research 22 (N1): the whole search run on data without selection skill.

`permute_forward` shuffles the forward (holding-week) returns among the eligible names of every decision week (positions held outside
the eligible set, e.g. stuck ones, keep their own returns).
It keeps the market return of the week, the cross-sectional distribution, the eligible set, the terminals, the
costs and therefore the turnover of any rule, and destroys any link between a terminal and the return that
follows. A terminal (delisting) return moves to a random name together with its payout flag; a name that really
delists is still paid out (at a random return), otherwise it would stay in the portfolio forever as a ghost position. The terminals are unchanged, so the evaluator and
its subtree cache are shared by all null datasets.
"""

from dataclasses import replace

import numpy as np

from qlab.research22.compact import Compact
from qlab.research22.grammar import Evaluator
from qlab.research22.search import Searcher, SearchResult, run_ga
from qlab.research22.setup import N_GEN, POP_SIZE, SEED, TOPK


def permute_forward(c: Compact, rng: np.random.Generator) -> Compact:
    """Copy of `c` with the holding return and the payout flag permuted among the eligible names of each week."""
    wk, dead = c.wk_all.copy(), c.dead_all.copy()
    for i in range(c.shape[0]):
        n = int(c.valid[i].sum())               # valid columns are the leading ones (ascending asset order)
        ids = c.idx[i, :n]
        perm = rng.permutation(n)
        wk[i, ids] = c.wk_all[i, ids][perm]
        dead[i, ids] = c.dead_all[i, ids][perm] | c.dead_all[i, ids]   # a delisted name is always paid out
    return replace(c, wk_all=wk, dead_all=dead, _bench={})


def null_dataset(c: Compact, index: int) -> Compact:
    return permute_forward(c, np.random.default_rng([SEED, 9000, index]))


def run_null_ga(ev: Evaluator, c: Compact, index: int, pop_size: int = POP_SIZE, n_gen: int = N_GEN,
                jitter_seed: int = SEED, topk: int = TOPK) -> SearchResult:
    """One N1 replicate: the same GA (same budget, same fitness) on the permuted dataset number `index`."""
    searcher = Searcher(ev, null_dataset(c, index), jitter_seed=jitter_seed, topk=topk)
    return run_ga(searcher, seed=10_000 + index, pop_size=pop_size, n_gen=n_gen)
