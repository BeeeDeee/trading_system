"""Periods, vaults, registry and data loading of research 10 (prereg §4-5).

Stocks/ETF (A, B): only data <= 2019-12-31 (the research 1 holdout boundary); no final stage.
Crypto (C): development <= 2021-12-31, holdout 2022-01-01 .. 2026-08-31 once through `runs/vault_r10`.
"""

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from qlab.data.panel import Panel, load_panel
from qlab.validation.registry import TrialRegistry, current_commit
from qlab.validation.vault import Vault

STOCK_END = date(2019, 12, 31)
A_SELECT = (date(2001, 1, 2), date(2012, 12, 31))
A_CHECK = (date(2013, 1, 2), date(2019, 12, 31))
B_SELECT = (date(2004, 1, 2), date(2012, 12, 31))
CHECK_SUBPERIODS = {"2013..2016": (date(2013, 1, 2), date(2016, 12, 31)),
                    "2017..2019": (date(2017, 1, 1), date(2019, 12, 31))}
C_DEV = (date(2018, 4, 1), date(2021, 12, 31))
C_HOLDOUT = (date(2022, 1, 1), date(2026, 8, 31))
C_SUBPERIODS = {"2022-01..2023-12": (date(2022, 1, 1), date(2023, 12, 31)),
                "2024-01..2026-08": (date(2024, 1, 1), date(2026, 8, 31))}
FINAL_ENV = "QLAB_R10_METHODOLOGY"
REGISTRY = Path("runs/registry/research10.sqlite")


def stock_vault() -> Vault:
    return Vault(STOCK_END, Path("runs/vault/frozen.lock"), Path("runs/vault/vault.log"))


def crypto_vault() -> Vault:
    return Vault(C_DEV[1], Path("runs/vault_r10/frozen.lock"), Path("runs/vault_r10/vault.log"))


def registry() -> TrialRegistry:
    return TrialRegistry(REGISTRY)


def index_of(dates: np.ndarray, d: date) -> int:
    return int(np.searchsorted(np.asarray(dates, dtype="datetime64[D]"), np.datetime64(d, "D")))


def window(dates: np.ndarray, a: date, b: date) -> slice:
    return slice(index_of(dates, a), int(np.searchsorted(np.asarray(dates, dtype="datetime64[D]"),
                                                         np.datetime64(b, "D"), side="right")))


def _cut(p: Panel, extra: dict, end: int) -> tuple[Panel, dict]:
    return p.slice(end), {k: np.asarray(m[:end]) for k, m in extra.items()}


def load_etf(snapshot: str) -> tuple[Panel, dict, list[str]]:
    d = Path("data/derived") / snapshot / "panel_etf"
    p, extra = load_panel(d)
    v = stock_vault()
    p, extra = _cut(p, extra, v.last_visible_index(p.dates))
    v.check_dev_only(p.dates)
    return p, extra, json.loads((d / "tickers.json").read_text())["columns"]


def load_sleeves(snapshot: str) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Daily net returns of the 87 research 4 sleeves up to STOCK_END: (returns T x S, dates, sleeve table)."""
    der = Path("data/derived") / snapshot
    dates = np.load(der / "panel_r4" / "dates.npy").astype("datetime64[D]")
    v = stock_vault()
    end = v.last_visible_index(dates)
    dates = dates[:end]
    v.check_dev_only(dates)
    R = np.load(der / "research4" / "sleeves" / "returns.npy", mmap_mode="r")[:end]
    return np.array(R), dates, pl.read_parquet(der / "research4" / "sleeves" / "sleeves.parquet")


def _crypto(snapshot: str, end: int | None) -> tuple[Panel, dict, list[str], np.ndarray]:
    d = Path("data/derived") / snapshot
    p, extra = load_panel(d / "panel_r9")
    syms = (d / "panel_r9" / "symbols.txt").read_text().split()
    end = len(p.dates) if end is None else end
    p, extra = _cut(p, extra, end)
    return p, extra, syms, funding(p.dates, syms, d / "r8_panel.npz")


def funding(dates: np.ndarray, syms: list[str], r8_path: Path) -> np.ndarray:
    """T x N funding earned by a perp short held over each day (research 8 `fund_hold`); 0 where missing."""
    z = np.load(r8_path, allow_pickle=True)
    r8_dates, r8_syms, fh = z["dates"].astype("datetime64[D]"), list(z["symbols"]), z["fund_hold"]
    out = np.zeros((len(dates), len(syms)))
    pos = {d: i for i, d in enumerate(r8_dates)}
    rows = np.array([pos.get(d, -1) for d in np.asarray(dates, dtype="datetime64[D]")])
    ok = rows >= 0
    for j, s in enumerate(syms):
        if s in r8_syms:
            out[ok, j] = np.nan_to_num(fh[rows[ok], r8_syms.index(s)])
    return out


def load_crypto_dev(snapshot: str) -> tuple[Panel, dict, list[str], np.ndarray]:
    d = Path("data/derived") / snapshot / "panel_r9"
    dates = np.load(d / "dates.npy").astype("datetime64[D]")
    v = crypto_vault()
    end = v.last_visible_index(dates)
    out = _crypto(snapshot, end)
    v.check_dev_only(out[0].dates)
    return out


def load_crypto_final(snapshot: str) -> tuple[Panel, dict, list[str], np.ndarray]:
    crypto_vault().check_final_session(os.environ.get(FINAL_ENV, ""), snapshot, current_commit())
    p, extra, syms, fund = _crypto(snapshot, None)
    end = index_of(p.dates, C_HOLDOUT[1]) + 1
    return p.slice(end), {k: v[:end] for k, v in extra.items()}, syms, fund[:end]
