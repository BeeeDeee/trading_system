"""Typed tree generation, variation operators and normal form for the research 22 search.

DEAP's generators cannot cap the node count, so trees are generated and varied here (nested tuples, see
`grammar`); DEAP provides the NSGA-II selection. Every operator keeps depth <= MAX_DEPTH and nodes <= MAX_NODES.

`normal_form` (simplify + sorted commutative arguments) is what gets evaluated, so that fitness is a pure
function of the semantics, the ledger counts equivalent genomes once, and the objective "nodes" is the number
of *effective* nodes.
"""

import numpy as np

from qlab.research22.compact import LEAF_NAMES, STATE_NAMES
from qlab.research22.grammar import ARITY, BOOL_OPS, CONSTANTS, Tree, tree_depth, tree_size
from qlab.research22.setup import MAX_DEPTH, MAX_NODES

FLOAT_TERMINALS: tuple[Tree, ...] = (tuple(("leaf", n) for n in LEAF_NAMES)
                                     + tuple(("const", float(c)) for c in CONSTANTS))
BOOL_TERMINALS: tuple[Tree, ...] = tuple(("state", n) for n in STATE_NAMES)
PRIMITIVES = {
    "float": (("add", ("float", "float")), ("sub", ("float", "float")), ("mul", ("float", "float")),
              ("div", ("float", "float")), ("neg", ("float",)), ("rank", ("float",)),
              ("where", ("bool", "float", "float"))),
    "bool": (("gt", ("float", "float")),),
}
MIN_NODES = {"float": 1, "bool": 1}
TERMINAL_PROB = 0.3          # "grow" mode: chance of stopping early at each node
COMMUTATIVE = ("add", "mul")


def kind_of(tree: Tree) -> str:
    return "bool" if tree[0] == "state" or tree[0] in BOOL_OPS else "float"


def _min_nodes(args: tuple[str, ...]) -> int:
    return 1 + sum(MIN_NODES[a] for a in args)


def random_tree(rng: np.random.Generator, kind: str = "float", max_depth: int = MAX_DEPTH,
                max_nodes: int = MAX_NODES, mode: str | None = None) -> Tree:
    """Random well-typed tree ("full" or "grow", chosen at random if `mode` is None) within the limits."""
    if mode is None:
        mode = "full" if rng.random() < 0.5 else "grow"
    return _gen(rng, kind, max_depth, max_nodes, mode)


def _gen(rng: np.random.Generator, kind: str, depth_left: int, nodes_left: int, mode: str) -> Tree:
    prims = [p for p in PRIMITIVES[kind] if depth_left > 1 and _min_nodes(p[1]) <= nodes_left]
    if not prims or (mode == "grow" and rng.random() < TERMINAL_PROB):
        terms = FLOAT_TERMINALS if kind == "float" else BOOL_TERMINALS
        return terms[int(rng.integers(len(terms)))]
    op, args = prims[int(rng.integers(len(prims)))]
    children: list[Tree] = []
    left = nodes_left - 1
    for j, arg in enumerate(args):
        reserve = sum(MIN_NODES[a] for a in args[j + 1:])
        child = _gen(rng, arg, depth_left - 1, left - reserve, mode)
        children.append(child)
        left -= tree_size(child)
    return (op, *children)


def _nodes(tree: Tree, path: tuple[int, ...] = (), depth: int = 1):
    yield path, tree, depth
    if tree[0] in ARITY:
        for i, child in enumerate(tree[1:], 1):
            yield from _nodes(child, path + (i,), depth + 1)


def _replace(tree: Tree, path: tuple[int, ...], new: Tree) -> Tree:
    if not path:
        return new
    i = path[0]
    return tree[:i] + (_replace(tree[i], path[1:], new),) + tree[i + 1:]


def crossover(rng: np.random.Generator, a: Tree, b: Tree, max_depth: int = MAX_DEPTH,
              max_nodes: int = MAX_NODES) -> Tree:
    """Child of `a` with one subtree replaced by a type-compatible subtree of `b`; `a` itself if none fits."""
    nodes_a, nodes_b = list(_nodes(a)), list(_nodes(b))
    size_a = tree_size(a)
    for _ in range(20):
        path, sub, depth = nodes_a[int(rng.integers(len(nodes_a)))]
        kind = kind_of(sub)
        rest = size_a - tree_size(sub)
        fits = [nb for nb in nodes_b if kind_of(nb[1]) == kind
                and depth - 1 + tree_depth(nb[1]) <= max_depth and rest + tree_size(nb[1]) <= max_nodes]
        if fits:
            return _replace(a, path, fits[int(rng.integers(len(fits)))][1])
    return a


def mutate(rng: np.random.Generator, a: Tree, max_depth: int = MAX_DEPTH, max_nodes: int = MAX_NODES) -> Tree:
    """Replace a random subtree by a new random one of the same type that fits the remaining budget."""
    path, sub, depth = list(_nodes(a))[int(rng.integers(tree_size(a)))]
    budget = max_nodes - (tree_size(a) - tree_size(sub))
    return _replace(a, path, random_tree(rng, kind_of(sub), max_depth - depth + 1, budget))


# ------------------------------------------------------------------ simplification and normal form
def _simplify_node(t: Tree) -> Tree:
    op = t[0]
    if op == "neg" and t[1][0] == "neg":
        return t[1][1]
    if op == "rank" and t[1][0] == "rank":
        return t[1]
    if op == "where" and t[2] == t[3]:
        return t[2]
    one = ("const", 1.0)
    if op == "mul":
        if t[2] == one:
            return t[1]
        if t[1] == one:
            return t[2]
    if op == "div" and t[2] == one:
        return t[1]
    return t


def _bottom_up(t: Tree, f) -> Tree:
    if t[0] not in ARITY:
        return t
    return f((t[0],) + tuple(_bottom_up(c, f) for c in t[1:]))


def _sort_commutative(t: Tree) -> Tree:
    if t[0] in COMMUTATIVE:
        a, b = sorted(t[1:], key=repr)
        return (t[0], a, b)
    return t


def simplify(tree: Tree) -> Tree:
    prev = None
    while tree != prev:
        prev, tree = tree, _bottom_up(tree, _simplify_node)
    return tree


def normal_form(tree: Tree) -> Tree:
    """Simplified tree with sorted commutative arguments (fixed point of both)."""
    prev = None
    while tree != prev:
        prev = tree
        tree = _bottom_up(simplify(tree), _sort_commutative)
    return tree
