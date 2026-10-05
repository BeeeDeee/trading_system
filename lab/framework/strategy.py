"""The strategy contract, its static scan (part of G0) and loading.

A strategy is one file `lab/strategies/<id>/strategy.py`:

    PARAMS = {"lookback": 180}                       # primary configuration (same as the card)

    def target_weights(data: DataView, params: dict) -> np.ndarray:
        ...                                          # (T, N) weights over data.instruments

Row t is decided after the close of day t from rows 0..t only and executed at the next open. A NaN row means
"no decision, keep positions"; a NaN element in a decision row keeps that position. Weights are signed,
gross <= 1. The framework handles execution, costs, calendars and cash.

Static rules (G0): imports only from ALLOWED_IMPORTS; no I/O, eval/exec, dunder tricks; no date literals and
no instrument names outside the card's universe (an LLM must not hard-code remembered history).
"""

import ast
import importlib.util
import re
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

ALLOWED_IMPORTS = {"numpy", "math", "statistics", "lab.framework.api", "lab.framework.data", "__future__", "typing",
                   "dataclasses", "functools", "itertools"}
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "input", "breakpoint", "memoryview"}
FORBIDDEN_ATTRS = {"datetime64", "today", "now", "load", "save", "fromfile", "tofile", "memmap", "loadtxt",
                   "genfromtxt"}
RESERVED_KEYS = {"alt_universe", "funding_paid"}   # G2's alternative universe; the engine's funding charge
DATE_RE = re.compile(r"(19|20)\d\d-[01]\d(-[0-3]\d)?")


def scan(source: str, instruments_allowed: set[str], instruments_all: set[str],
         imports: set[str] = ALLOWED_IMPORTS, check_dates: bool = True) -> list[str]:
    """Static problems of a strategy file (empty = clean). Ingest fetchers use their own import list and may
    contain dates (API start dates); strategies may not."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in imports and a.name.split(".")[0] not in {"numpy"}:
                    problems.append(f"line {node.lineno}: import {a.name} not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in imports and (node.module or "").split(".")[0] != "numpy":
                problems.append(f"line {node.lineno}: from {node.module} import not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {node.lineno}: {node.id} not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                problems.append(f"line {node.lineno}: .{node.attr} not allowed (no I/O or clocks)")
            if (isinstance(node.value, ast.Attribute) and node.value.attr == "random"
                    and node.attr != "default_rng"):
                problems.append(f"line {node.lineno}: random.{node.attr} not allowed; use "
                                "numpy.random.default_rng(seed)")
            if node.attr.startswith("__") and node.attr != "__init__":
                problems.append(f"line {node.lineno}: dunder attribute {node.attr} not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if check_dates and DATE_RE.search(node.value):
                problems.append(f"line {node.lineno}: date literal {node.value!r} (hard-coded history)")
            if node.value in RESERVED_KEYS:
                problems.append(f"line {node.lineno}: {node.value!r} is reserved for the gate runner")
            if node.value in instruments_all - instruments_allowed:
                problems.append(f"line {node.lineno}: instrument {node.value!r} is not in the card's universe")
    return problems


def load(path: Path, entry: str = "target_weights") -> ModuleType:
    name = f"lab_strategy_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules.pop(name, None)
    if not callable(getattr(module, entry, None)):
        raise TypeError(f"{path.name} must define {entry}(...)")
    return module


class StrategyTimeout(RuntimeError):
    pass


@contextmanager
def time_limit(seconds: float):
    """Wall-clock limit for one strategy call (main thread). The bwrap sandbox of step 4 adds RAM limits."""
    def handler(signum, frame):
        raise StrategyTimeout(f"strategy exceeded {seconds:.0f} s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
