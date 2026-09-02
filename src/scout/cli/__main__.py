from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from scout.cli import adjust, build_edge_table, build_universe, ingest, label_setups, run_backtest
from scout.utils.errors import ScoutError

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE from .env without overriding variables already set."""
    import os

    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main(argv: Sequence[str] | None = None) -> int:
    _load_dotenv(_REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(prog="scout")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Download unadjusted equity data")
    ingest.add_arguments(ingest_p)

    adjust_p = sub.add_parser("adjust", help="Build XNYS calendar and causally adjust the panel")
    adjust.add_arguments(adjust_p)

    uni_p = sub.add_parser("build-universe", help="Build point-in-time universe snapshots")
    build_universe.add_arguments(uni_p)

    label_p = sub.add_parser("label", help="Detect and triple-barrier-label historical setups")
    label_setups.add_arguments(label_p)

    edge_p = sub.add_parser("build-edge", help="Build the walk-forward edge table")
    build_edge_table.add_arguments(edge_p)

    bt_p = sub.add_parser("backtest", help="Run the timestamp-outer backtest")
    run_backtest.add_arguments(bt_p)

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "ingest":
            return ingest.run(args)
        if args.command == "adjust":
            return adjust.run(args)
        if args.command == "build-universe":
            return build_universe.run(args)
        if args.command == "label":
            return label_setups.run(args)
        if args.command == "build-edge":
            return build_edge_table.run(args)
        if args.command == "backtest":
            return run_backtest.run(args)
    except ScoutError as exc:
        print(exc, file=sys.stderr)
        return 1
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
