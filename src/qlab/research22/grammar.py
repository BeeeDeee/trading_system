"""Typed expression grammar and cached evaluator of research 22.

A tree is a nested tuple:  ("leaf", name) | ("state", name) | ("const", value) | (op, child, ...).
Types: float (F, K) float32 arrays and bool arrays (F, K) or (F, 1) market states.

    float:  add sub mul div(x, y)   neg rank   where(bool, float, float)   leaf const
    bool:   gt(float, float)   state

`div` is protected (|y| <= DIV_EPS gives 0), `rank` is the cross-sectional average rank in [0, 1) among the
eligible names, constants come from CONSTANTS. Limits: depth <= MAX_DEPTH, nodes <= MAX_NODES.
"""

from collections import OrderedDict

import numpy as np

from qlab.research22.compact import LEAF_NAMES, STATE_NAMES, Compact
from qlab.research22.kernels import rank_rows

CONSTANTS = (-1.0, 0.5, 1.0, 2.0)
DIV_EPS = 1e-6
FLOAT_BINARY = ("add", "sub", "mul", "div")
FLOAT_UNARY = ("neg", "rank")
ARITY = {**{op: 2 for op in FLOAT_BINARY}, **{op: 1 for op in FLOAT_UNARY}, "gt": 2, "where": 3}
BOOL_OPS = ("gt",)

Tree = tuple


class GrammarError(ValueError):
    pass


def node_type(tree: Tree) -> str:
    """'float' or 'bool'; raises GrammarError for ill-typed or unknown nodes."""
    kind = tree[0]
    if kind == "leaf":
        if len(tree) != 2 or tree[1] not in LEAF_NAMES:
            raise GrammarError(f"unknown terminal {tree!r}")
        return "float"
    if kind == "state":
        if len(tree) != 2 or tree[1] not in STATE_NAMES:
            raise GrammarError(f"unknown market state {tree!r}")
        return "bool"
    if kind == "const":
        if len(tree) != 2 or tree[1] not in CONSTANTS:
            raise GrammarError(f"constant {tree!r} not in {CONSTANTS}")
        return "float"
    if kind not in ARITY:
        raise GrammarError(f"unknown operator {kind!r}")
    if len(tree) != ARITY[kind] + 1:
        raise GrammarError(f"{kind} takes {ARITY[kind]} arguments")
    types = [node_type(c) for c in tree[1:]]
    want = ["bool", "float", "float"] if kind == "where" else ["float"] * ARITY[kind]
    if types != want:
        raise GrammarError(f"{kind} expects {want}, got {types}")
    return "bool" if kind in BOOL_OPS else "float"


def tree_size(tree: Tree) -> int:
    return 1 + (sum(tree_size(c) for c in tree[1:]) if tree[0] in ARITY else 0)


def tree_depth(tree: Tree) -> int:
    return 1 + (max(tree_depth(c) for c in tree[1:]) if tree[0] in ARITY else 0)


def validate(tree: Tree, max_depth: int, max_nodes: int) -> None:
    if node_type(tree) != "float":
        raise GrammarError("a score must be float")
    if tree_depth(tree) > max_depth or tree_size(tree) > max_nodes:
        raise GrammarError(f"tree exceeds depth {max_depth} / nodes {max_nodes}")


def _compute(op: str, args: list[np.ndarray]) -> np.ndarray:
    if op == "add":
        return args[0] + args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "div":
        y = args[1]
        ok = np.abs(y) > DIV_EPS
        return np.where(ok, args[0] / np.where(ok, y, np.float32(1.0)), np.float32(0.0)).astype(np.float32)
    if op == "neg":
        return -args[0]
    if op == "rank":
        return rank_rows(args[0])
    if op == "gt":
        return args[0] > args[1]
    if op == "where":
        return np.where(args[0], args[1], args[2]).astype(np.float32)
    raise GrammarError(op)


class Evaluator:
    """Evaluates trees on one `Compact`, memoising subtrees (LRU by bytes).

    Offspring of crossover/mutation share most subtrees with their parents, so the cache removes most of the
    `rank` work. The cache never changes a result.
    """

    def __init__(self, compact: Compact, cache_bytes: int = 600_000_000):
        self.c = compact
        self.limit = cache_bytes
        self.used = 0
        self.cache: OrderedDict[Tree, np.ndarray] = OrderedDict()
        self.hits = self.misses = 0

    def _eval(self, tree: Tree) -> np.ndarray:
        kind = tree[0]
        if kind == "leaf":
            return self.c.leaves[tree[1]]
        if kind == "state":
            return self.c.states[tree[1]]
        if kind == "const":
            return np.float32(tree[1])
        hit = self.cache.get(tree)
        if hit is not None:
            self.cache.move_to_end(tree)
            self.hits += 1
            return hit
        self.misses += 1
        args = [self._eval(child) for child in tree[1:]]
        if kind == "rank":   # rank only the eligible columns, also when the argument is a constant
            full = np.broadcast_to(args[0], self.c.shape).astype(np.float32)
            args = [np.where(self.c.valid, full, np.float32(np.nan))]
        out = _compute(kind, args)
        if out.nbytes <= self.limit:
            self.cache[tree] = out
            self.used += out.nbytes
            while self.used > self.limit:
                _, old = self.cache.popitem(last=False)
                self.used -= old.nbytes
        return out

    def score(self, tree: Tree) -> np.ndarray:
        """(F, K) float32 score; padding columns are NaN (not selectable)."""
        out = np.broadcast_to(self._eval(tree), self.c.shape).astype(np.float32, copy=False)
        return np.where(self.c.valid, out, np.float32(np.nan))
