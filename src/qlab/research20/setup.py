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

T0 = date(2026, 11, 3)                 # first US trading day after the commit of pre-registration v1.0 (decision log)
SHARADAR_END = date(2026, 9, 25)       # last day of panel_r19_etf; later days come from the daily snapshots
LOOKS_MONTHS = (12, 24, 36)
ALPHA_TOTAL = 0.05
ALPHA_LOOK = ALPHA_TOTAL / len(LOOKS_MONTHS)
MIN_DAYS_F1 = 240
MIN_TRADES_F2 = (50, 100, 150)
BAR_COST_MULT = 1.5
BOOT_REPS, BOOT_BLOCK, SEED = 10_000, 21, 0
