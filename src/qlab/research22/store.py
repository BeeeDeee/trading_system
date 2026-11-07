"""Reading and writing the outputs of `scripts/r22_search.py` (JSON summaries and gzip JSONL ledgers).

Trees are stored with `repr` and restored with `ast.literal_eval`; they contain only tuples, strings and floats.
"""

import ast
import gzip
import json
from dataclasses import asdict
from pathlib import Path

from qlab.research22.fitness import Fitness
from qlab.research22.grammar import Tree
from qlab.research22.search import Ledger, SearchResult


def tree_from_repr(text: str) -> Tree:
    return ast.literal_eval(text)


def fitness_from_dict(d: dict) -> Fitness:
    return Fitness(q25=float(d["q25"]), turnover_year=float(d["turnover_year"]), nodes=int(d["nodes"]),
                   dispersion=float(d["dispersion"]), feasible=bool(d["feasible"]),
                   fold_irs=tuple(float(x) for x in d["fold_irs"]))


def pairs_to_rows(pairs: list[tuple[Tree, Fitness]]) -> list[dict]:
    return [{"tree": repr(t), **asdict(f)} for t, f in pairs]


def rows_to_pairs(rows: list[dict]) -> list[tuple[Tree, Fitness]]:
    return [(tree_from_repr(r["tree"]), fitness_from_dict(r)) for r in rows]


def save_result(out: Path, summary: dict, result_ledger: Ledger) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    with gzip.open(out.with_suffix(".ledger.jsonl.gz"), "wt") as f:
        for row in result_ledger.to_records():
            f.write(json.dumps(row) + "\n")


def load_ledger(path: Path) -> Ledger:
    ledger = Ledger()
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            ledger.add(tree_from_repr(row["tree"]), fitness_from_dict(row), int(row["gen"]))
    return ledger


def load_ga_result(summary_path: Path) -> SearchResult:
    summary = json.loads(summary_path.read_text())
    ledger = load_ledger(summary_path.with_suffix("").with_suffix(".ledger.jsonl.gz"))
    return SearchResult(int(summary["seed"]), rows_to_pairs(summary["population"]), rows_to_pairs(summary["front"]),
                        summary["history"], ledger, int(summary["n_backtests"]))
