"""Constants and paths of research 21 (pre-registration v1.1; every value is fixed there).

Research 21 has no clean holdout: it declares the whole Sharadar history (to 2026-09-25) already seen, like
`research19.setup`. It reads `panel_r5` and `panel_r19_etf` without any vault, so do not use it to select strategies.
"""

from datetime import date
from pathlib import Path

import numpy as np

from qlab.data.panel import Panel, load_panel

SHARADAR = "sharadar_2026-09-25"
SEED = 21
REGISTRY = Path("runs/registry/research21.sqlite")
SCRATCH_REGISTRY = Path("runs/scratch/research21_smoke.sqlite")
RESULTS = Path("docs/research21/results")
LOCK = Path("docs/research21/run_lock.json")

# --- S0 (prereg §3) ---
ADV_MIN = 50e6
RSI_N, RSI_ENTRY = 2, 10.0
SMA_REGIME, SMA_EXIT = 200, 5
MAX_POSITIONS, MAX_HOLD = 10, 10
PERIOD_START = date(1999, 1, 4)
E_START = date(2005, 1, 1)               # v1.1: E and C only from here (8-K Item 2.02 coverage)

# --- proxy filters (prereg §4, v1.1 operationalisation) ---
VOL_RATIO, VOL_WINDOW, VOL_MIN_OBS = 3.0, 20, 15
GAP_Z, GAP_WINDOW, GAP_MIN_OBS = -2.0, 60, 40
IDIO_Z = -2.0
E_BEFORE, E_AFTER = 2, 2                 # E1 window [t-2, t+2]; r2 uses [t-2, t]
FILTERS = ("E", "V", "G", "I", "C")

# --- sectors (Sharadar `sector` -> ETF column of panel_r19_etf); Communication Services has no ETF before 2018 ---
SECTOR_ETF = {"Technology": "XLK", "Healthcare": "XLV", "Financial Services": "XLF", "Energy": "XLE",
              "Industrials": "XLI", "Basic Materials": "XLB", "Consumer Cyclical": "XLY",
              "Consumer Defensive": "XLP", "Utilities": "XLU", "Real Estate": "VNQ"}

# --- evaluation (prereg §6, v1.1) ---
THIRDS = ((1999, 2007), (2008, 2016), (2017, 2026))
THIRDS_E = ((2005, 2011), (2012, 2018), (2019, 2026))
MIN_SHARE = 0.15
BOOT_REPS, BOOT_BLOCK_MONTHS, CI_LEVEL = 1000, 3.0, 0.90
C2_SIMS = 1000
PRIOR_STOCK_TRIALS = 993                 # research 5 (cumulative count on stocks, reported next to N = 6)
ANN = 252

# features are computed for every security that ever had ADV20 >= FEATURE_ADV (the lowest threshold of r4)
FEATURE_ADV = 20e6


def der_dir(snapshot: str = SHARADAR) -> Path:
    return Path("data/derived") / snapshot


def features_dir(snapshot: str = SHARADAR) -> Path:
    return der_dir(snapshot) / "r21"


def load_stock_panel(snapshot: str = SHARADAR) -> tuple[Panel, dict]:
    return load_panel(der_dir(snapshot) / "panel_r5")


def load_etf_panel(snapshot: str = SHARADAR) -> tuple[Panel, dict, list[str]]:
    import json
    d = der_dir(snapshot) / "panel_r19_etf"
    p, extra = load_panel(d)
    return p, extra, json.loads((d / "tickers.json").read_text())["columns"]


def index_of(dates: np.ndarray, d: date) -> int:
    return int(np.searchsorted(dates, np.datetime64(d, "D")))
