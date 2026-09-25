"""Walks src/scout/ and asserts the dependency direction from 01-ARCHITECTURE.md §4."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "scout"

FORBIDDEN: dict[str, set[str]] = {
    "scout.domain": {
        "scout.data",
        "scout.features",
        "scout.strategies",
        "scout.scoring",
        "scout.portfolio",
        "scout.execution",
        "scout.backtest",
        "scout.research",
        "scout.config",
    },
    "scout.strategies": {
        "scout.costs",
        "scout.portfolio",
        "scout.execution",
        "scout.sentiment",
        "scout.data",
        "scout.backtest",
    },
    "scout.features": {"scout.strategies", "scout.scoring", "scout.portfolio"},
    "scout.scoring": {"scout.portfolio", "scout.execution"},
}

EXCHANGE_AND_HTTP = {
    "httpx",
    "ib_insync",
    "ibapi",
    "alpaca",
    "alpaca_trade_api",
    "ccxt",
    "binance",
}

HTTP_ALLOWED_PREFIXES = ("scout.execution", "scout.data.ingest")


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_from(path: Path, node: ast.ImportFrom) -> list[str]:
    names: list[str] = []
    if node.level == 0:
        if node.module is not None:
            names.append(node.module)
            if node.module == "scout":
                names.extend(f"scout.{alias.name}" for alias in node.names)
        return names
    pkg_parts = _module_name(path).split(".")
    parent = pkg_parts[: len(pkg_parts) - node.level]
    if node.module:
        names.append(".".join([*parent, *node.module.split(".")]))
    else:
        names.append(".".join(parent))
        names.extend(".".join([*parent, alias.name]) for alias in node.names)
    return names


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.extend(_resolve_from(path, node))
    return found


def _under(imported: str, package: str) -> bool:
    return imported == package or imported.startswith(package + ".")


def test_forbidden_imports() -> None:
    py_files = sorted(SRC.rglob("*.py"))
    assert py_files, f"no python files under {SRC}"
    violations: list[str] = []
    for path in py_files:
        module = _module_name(path)
        imported = _imports(path)
        for owner, banned in FORBIDDEN.items():
            if not _under(module, owner):
                continue
            for item in imported:
                for ban in banned:
                    if _under(item, ban):
                        violations.append(f"{module} imports {item} (forbidden for {owner})")
    assert not violations, "\n".join(violations)


def test_domain_imports_only_utils_from_scout() -> None:
    domain = SRC / "domain"
    violations: list[str] = []
    for path in sorted(domain.rglob("*.py")):
        module = _module_name(path)
        for item in _imports(path):
            if not item.startswith("scout."):
                continue
            if _under(item, "scout.domain") or _under(item, "scout.utils"):
                continue
            violations.append(f"{module} imports {item}")
    assert not violations, "\n".join(violations)


def test_httpx_and_sdks_only_in_execution_and_ingest() -> None:
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        module = _module_name(path)
        if any(module == p or module.startswith(p + ".") for p in HTTP_ALLOWED_PREFIXES):
            continue
        for item in _imports(path):
            root = item.split(".")[0]
            if root in EXCHANGE_AND_HTTP:
                violations.append(f"{module} imports {item}")
    assert not violations, "\n".join(violations)
