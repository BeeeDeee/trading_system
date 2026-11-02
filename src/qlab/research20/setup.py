"""Constants of research 20 (prereg §3.3, §6, §7)."""

from datetime import date
from pathlib import Path

RAW = Path("data/raw/yahoo_forward20")
RESULTS = Path("docs/research20/results")
REGISTRY = Path("runs/registry/research20.sqlite")
LOCK = Path("docs/research20/forward_lock.json")

# quality gate (prereg §3.3): Yahoo-built daily close-to-close TR vs Sharadar over the overlap year
GATE_WINDOW = (date(2025, 9, 26), date(2026, 9, 25))
GATE_MEDIAN_BPS, GATE_P99_BPS, GATE_MAX_BPS, GATE_CUM_PCT = 1.0, 10.0, 50.0, 0.5
NASDAQ_TOL_PCT = 0.3
