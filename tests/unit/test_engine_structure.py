"""AST check: BacktestEngine.run's outermost for iterates timestamps."""

from __future__ import annotations

import ast
import inspect
import textwrap

from scout.backtest.engine import BacktestEngine


def test_run_outer_loop_iterates_timestamps() -> None:
    src = textwrap.dedent(inspect.getsource(BacktestEngine.run))
    tree = ast.parse(src)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    for_nodes = [node for node in func.body if isinstance(node, ast.For)]
    assert for_nodes, "BacktestEngine.run must have a top-level for loop"
    outer = for_nodes[0]
    iter_src = ast.unparse(outer.iter)
    assert "timestamps" in iter_src, f"outer loop iterates {iter_src!r}, not timestamps"
    assert "symbol" not in iter_src.lower(), f"outer loop must not iterate symbols: {iter_src}"
