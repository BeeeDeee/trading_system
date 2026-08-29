"""Greps src/scout/ for wall-clock calls. Only utils/clock.py may use them."""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "scout"
ALLOWED = (SRC / "utils" / "clock.py").resolve()

FORBIDDEN = (
    "datetime.now",
    "datetime.utcnow",
    "time.time",
    "pd.Timestamp.now",
    "date.today",
)


def test_no_wallclock_outside_clock_module() -> None:
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if path.resolve() == ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        for pat in FORBIDDEN:
            if pat in text:
                rel = path.relative_to(SRC.parent.parent)
                violations.append(f"{rel}: {pat}")
    assert not violations, "wall-clock calls outside utils/clock.py:\n" + "\n".join(violations)
