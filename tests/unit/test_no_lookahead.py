"""Greps features/, scoring/, and strategies/ for lookahead-bias patterns."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "scout"
SCAN_DIRS = ("features", "scoring", "strategies")
NOQA = "# noqa: lookahead"

FORBIDDEN_PATTERNS = [
    r"\.shift\(\s*-",
    r"\.bfill\(",
    r'fillna\([^)]*method\s*=\s*["\']bfill',
    r"\.interpolate\(",
    r"center\s*=\s*True",
    r"\.iloc\[::-1\]",
    r"limit_direction\s*=\s*['\"]backward",
]


def test_no_lookahead_patterns() -> None:
    compiled = [(pat, re.compile(pat)) for pat in FORBIDDEN_PATTERNS]
    violations: list[str] = []
    for folder in SCAN_DIRS:
        root = SRC / folder
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            lines = path.read_text(encoding="utf-8").splitlines()
            rel = path.relative_to(SRC.parent.parent)
            for lineno, line in enumerate(lines, start=1):
                if NOQA in line:
                    continue
                for pat, rx in compiled:
                    if rx.search(line):
                        violations.append(f"{rel}:{lineno}: {pat}: {line.strip()}")
    assert not violations, "lookahead patterns:\n" + "\n".join(violations)
