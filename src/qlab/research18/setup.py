"""Constants of research 18 (docs/research18/PREREGISTRATION.md §4-7). Nothing here changes after the freeze."""

from datetime import date
from pathlib import Path

T0 = date(2026, 10, 20)                # first UTC day after the commit of the pre-registration v1.0 (§6)
END = {1: date(2027, 10, 19), 2: date(2028, 10, 19)}      # last day of the first (12 months) and second (24) decision
MIN_EVENTS = {"H1": 30, "H2": 60}
ALPHA = 0.025                          # one-sided, Holm over the living hypotheses, per decision (§7)
N_BOOT = 10_000                        # event bootstrap for the p-value
N_BOOT_DESC = 1000                     # paired bootstrap of the descriptive H2 - V01 difference
MA_DAYS = 200
MA_CANDLES = MA_DAYS * 96              # BTC spot close vs the mean of its last 19 200 closes (§4)
COST_MARGIN = 1.5e-3                   # criterion 3: 15 bps per side (§7)

LOCK = Path("docs/research18/forward_lock.json")
RESULTS = Path("docs/research18/results")
REGISTRY = Path("runs/registry/research18.sqlite")
