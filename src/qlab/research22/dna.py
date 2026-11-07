"""'Strategy DNA' analysis for research 22 (gate G5, hypothesis H4).

Rows are cluster representatives. A component is *present* in a tree if the tree contains that terminal, state or
operator. Three groups are compared: random trees of the same grammar (prior), in-sample survivors, and
validation-robust representatives. Enrichment of robust versus non-robust survivors is estimated by an L2-penalised
logistic regression with the controls (turnover, size, in-sample f1); the p-value comes from permuting the labels
within seeds (the GA oversamples whatever it finds early, so a label shuffle must keep the seed structure), and
the p-values are adjusted with Benjamini-Yekutieli. G5 asks for the same sign, q < 0.10 and an odds ratio above
1.5 in both the discovery half (seeds 1-5) and the replication half (seeds 6-10).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from qlab.research22.compact import LEAF_NAMES, STATE_NAMES
from qlab.research22.grammar import Tree
from qlab.research22.stats22 import by_qvalues

OPS = ("add", "sub", "mul", "div", "neg", "rank", "where", "gt")
COMPONENTS = tuple(LEAF_NAMES) + tuple(STATE_NAMES) + OPS + ("const",)
_INDEX = {name: i for i, name in enumerate(COMPONENTS)}


def components_of(tree: Tree) -> set[str]:
    kind = tree[0]
    if kind in ("leaf", "state"):
        return {tree[1]}
    if kind == "const":
        return {"const"}
    out = {kind}
    for child in tree[1:]:
        out |= components_of(child)
    return out


def design(trees: list[Tree]) -> np.ndarray:
    """(n, n_components) boolean presence matrix."""
    x = np.zeros((len(trees), len(COMPONENTS)), dtype=bool)
    for i, t in enumerate(trees):
        for name in components_of(t):
            x[i, _INDEX[name]] = True
    return x


def frequency_table(groups: dict[str, list[Tree]]) -> dict[str, np.ndarray]:
    """Share of trees containing each component, per group (the three-way comparison)."""
    return {g: design(trees).mean(axis=0) if trees else np.full(len(COMPONENTS), np.nan)
            for g, trees in groups.items()}


def _standardise(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    sd = c.std(axis=0)
    return (c - c.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def _coef(x_j: np.ndarray, z: np.ndarray, y: np.ndarray, c_reg: float) -> float:
    model = LogisticRegression(C=c_reg, max_iter=200)
    model.fit(np.column_stack([x_j.astype(float), z]), y)
    return float(model.coef_[0, 0])


def enrichment(x: np.ndarray, y: np.ndarray, controls: np.ndarray, strata: np.ndarray, n_perm: int = 500,
               seed: int = 0, c_reg: float = 1.0) -> dict[str, np.ndarray]:
    """Per-component log-odds, odds ratio, permutation p and BY-adjusted q for y ~ component + controls.

    Components absent or present in every row, or a constant y, give NaN. `strata` (seed ids) are kept fixed
    when the labels are permuted. `min_possible_q` is the smallest BY-adjusted p this many permutations can
    reach; if it is above the q threshold the test cannot pass whatever the data say.
    """
    x = np.asarray(x, dtype=bool)
    y = np.asarray(y, dtype=int)
    n, m = x.shape
    coef = np.full(m, np.nan)
    p = np.full(m, np.nan)
    if len(np.unique(y)) < 2:
        return {"coef": coef, "odds_ratio": np.exp(coef), "p": p, "q": by_qvalues(p), "min_possible_q": float("nan")}
    z = _standardise(controls)
    rng = np.random.default_rng(seed)
    perms = np.empty((n_perm, n), dtype=int)
    for b in range(n_perm):
        yp = y.copy()
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            yp[idx] = y[idx][rng.permutation(len(idx))]
        perms[b] = yp
    for j in range(m):
        if x[:, j].all() or not x[:, j].any():
            continue
        coef[j] = _coef(x[:, j], z, y, c_reg)
        null = np.array([_coef(x[:, j], z, perms[b], c_reg) if len(np.unique(perms[b])) > 1 else 0.0
                         for b in range(n_perm)])
        p[j] = (1.0 + (np.abs(null) >= abs(coef[j])).sum()) / (n_perm + 1.0)
    m_tested = int(np.isfinite(p).sum())
    # the smallest p a permutation test can give is 1 / (n_perm + 1); BY needs p <= q / (m * c_m)
    min_q = (m_tested * (1.0 / np.arange(1, m_tested + 1)).sum() / (n_perm + 1.0)) if m_tested else float("nan")
    return {"coef": coef, "odds_ratio": np.exp(coef), "p": p, "q": by_qvalues(p), "min_possible_q": float(min_q)}


def replicated(discovery: dict[str, np.ndarray], replication: dict[str, np.ndarray], q: float = 0.10,
               min_odds_ratio: float = 1.5) -> list[str]:
    """Components enriched (OR above the bound, q below `q`) with the same sign in both halves."""
    out = []
    for j, name in enumerate(COMPONENTS):
        if not (np.isfinite(discovery["q"][j]) and np.isfinite(replication["q"][j])):
            continue
        if (discovery["q"][j] < q and replication["q"][j] < q
                and discovery["odds_ratio"][j] > min_odds_ratio and replication["odds_ratio"][j] > min_odds_ratio):
            out.append(name)
    return out
