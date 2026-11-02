"""Constants, paths and data loading of research 19 (PLAN §3-§7; all values fixed by the pre-registration).

Research 19 has no clean holdout: it replicates literature rules with fixed parameters and declares the whole
history (Sharadar to 2026-09-25, Binance to 2026-08-31) as already seen, like `research5.setup.load_full`.
`load_etf` and `load_crypto` therefore bypass every vault on purpose; do not use them for anything that
selects among strategies.
"""

from datetime import date
from pathlib import Path

import numpy as np

from qlab.data.panel import Panel, load_panel
from qlab.validation.registry import TrialRegistry

SHARADAR = "sharadar_2026-09-25"
BINANCE = "binance_2026-10-03"
REGISTRY = Path("runs/registry/research19.sqlite")
SCRATCH_REGISTRY = Path("runs/scratch/research19_smoke.sqlite")
RESULTS = Path("docs/research19/results")
LOCK = Path("docs/research19/run_lock.json")
SEED = 19

# --- tickers (§3.2) ---
ETF_TREND = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "TIP", "LQD", "HYG", "GLD", "SLV", "DBC", "USO", "UUP"]
RSI_UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
ALL_ETF = sorted(set(ETF_TREND) | set(RSI_UNIVERSE))      # 26 unique
ETF_TREND_9 = ["SPY", "EFA", "EEM", "IEF", "TLT", "TIP", "GLD", "DBC", "VNQ"]   # robustness: research 2 basket
RSI_INDEX_4 = ["SPY", "QQQ", "IWM", "DIA"]                                      # robustness r6
EXPECTED_START = {"SPY": "1997-12-31", "DIA": "1998-01-20", "QQQ": "1999-03-10", "IWM": "2000-05-26",
                  "EFA": "2001-08-27", "TLT": "2002-07-30", "IEF": "2002-07-30", "LQD": "2002-07-30",
                  "EEM": "2003-04-14", "TIP": "2003-12-05", "VNQ": "2004-09-29", "GLD": "2004-11-18",
                  "DBC": "2006-02-06", "USO": "2006-04-10", "SLV": "2006-04-28", "UUP": "2007-03-01",
                  "HYG": "2007-04-11", **{f"XL{s}": "1998-12-22" for s in "BEFIKPUVY"}}

# non-crypto assets removed from every crypto universe before the top-N cut (prereg v1.1, §4.1)
EXCLUDED_NONCRYPTO = frozenset({"PAXGUSDT", "XAUTUSDT", "BULLUSDT", "BEARUSDT", "UUSDT"})

# --- lookbacks, sizing, bands (§4) ---
LOOKBACKS_CRYPTO = (5, 10, 20, 30, 60, 90, 150, 250, 360)
LOOKBACKS_ETF = (50, 100, 150, 200, 250)
VOL_WINDOW = 90
CRYPTO_YEAR, ETF_YEAR = 365, 252
CRYPTO_VOL_TARGET, ETF_VOL_TARGET, LEVERAGE_CAP = 0.25, 0.10, 2.0
CRYPTO_TOP_N, CRYPTO_MIN_AGE = 20, 365
BAND = 0.20
ETF_COST_BPS, PRE_DECIMAL_MULT = 3.0, 2.5
RSI_ENTRY, RSI_N, MAX_POSITIONS, MAX_HOLD = 10.0, 2, 4, 10
TOM_BEFORE, TOM_AFTER = 1, 3
VOL_TARGET_PORTFOLIO, VOL_TARGET_WINDOW = 0.10, 20
CRASH_VOL_MULT, CRASH_MEDIAN_WINDOW, CRASH_SCALE = 1.5, 252, 0.5

# --- periods (§4, §7) ---
PERIOD_H1A = (date(2019, 1, 1), date(2026, 8, 31))
PERIOD_H1A_RECENT = (date(2022, 1, 1), date(2026, 8, 31))
PERIOD_H1A_POST_PUB = (date(2025, 4, 1), date(2026, 8, 31))
PERIOD_H1B = (date(2007, 7, 2), date(2026, 9, 25))
PERIOD_H1B_POST = (date(2010, 1, 1), date(2026, 9, 25))
PERIOD_H2 = (date(1999, 11, 1), date(2026, 9, 25))
PERIOD_H2_PRE = (date(1999, 11, 1), date(2008, 12, 31))
PERIOD_H2_POST = (date(2009, 1, 1), date(2026, 9, 25))
PERIOD_TOM = (date(2002, 1, 2), date(2026, 9, 25))
PERIOD_H4 = (date(2019, 1, 2), date(2026, 8, 31))

# --- evaluation (§7) ---
BOOT_REPS, BOOT_BLOCK_ETF, BOOT_BLOCK_CRYPTO, CI_LEVEL = 10_000, 21, 30, 0.90
NULL_REPS_TREND, NULL_REPS_H2 = 500, 1000
CUM_CRYPTO_N_PRIOR = 89     # cumulative crypto trials, research 8-18 (§6)


def etf_panel_dir(snapshot: str = SHARADAR) -> Path:
    return Path("data/derived") / snapshot / "panel_r19_etf"


def crypto_panel_dir(snapshot: str = BINANCE) -> Path:
    return Path("data/derived") / snapshot / "panel_r9"


def load_etf(snapshot: str = SHARADAR) -> tuple[Panel, dict, list[str]]:
    """Full-history ETF panel (memory-mapped), extras (`cash_ret`) and the ticker of each column."""
    import json
    p, extra = load_panel(etf_panel_dir(snapshot))
    return p, extra, json.loads((etf_panel_dir(snapshot) / "tickers.json").read_text())["columns"]


def load_crypto(snapshot: str = BINANCE) -> tuple[Panel, dict, list[str]]:
    """Full-history crypto panel (`extra`: open, qv) and symbols. No vault: research 19 declares all history seen."""
    p, extra = load_panel(crypto_panel_dir(snapshot))
    return p, extra, (crypto_panel_dir(snapshot) / "symbols.txt").read_text().split()


def index_of(p: Panel, d: date) -> int:
    return int(np.searchsorted(p.dates, np.datetime64(d, "D")))


def registry(path: Path | None = None) -> TrialRegistry:
    return TrialRegistry(path or REGISTRY)
