"""15-minute data, stages, vault and registry of research 15 (prereg §3, §7).

Panel: one `.npz` per symbol and market (`spot`/`perp`) with `t0` (first 15m index, a multiple of 16 so 1h/4h
bars align), `o,h,l,c` (carried close on missing candles), `real` (a candle exists) and `funding` (perp rate
settled at the start of that candle, NaN where no data). A 15m index is `epoch_ms // 900_000`, so candle t
covers [t, t+1) and its close is known at t+1.
"""

import os
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from qlab.validation.registry import TrialRegistry, current_commit
from qlab.validation.vault import Vault

PRIMARY = ("BTCUSDT", "ETHUSDT")
GRID_START = date(2017, 8, 17)
GRID_END = date(2026, 9, 30)          # last candle 23:45 UTC
DEV = (date(2018, 2, 1), date(2021, 12, 31))
HOLDOUT = (date(2022, 1, 1), date(2026, 9, 30))
SUBPERIODS = {"2022-01..2023-12": (date(2022, 1, 1), date(2023, 12, 31)),
              "2024-01..2026-09": (date(2024, 1, 1), date(2026, 9, 30))}
SECONDARY_START = date(2020, 5, 1)
BASKET_WEIGHT = 0.1                   # capital share of each basket sleeve (idle sleeves hold cash)
PER_DAY = 96
FINAL_ENV = "QLAB_R15_METHODOLOGY"
REGISTRY = Path("runs/registry/research15.sqlite")


def epoch_15m(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()) // 900


def grid_range() -> tuple[int, int]:
    """[first, last+1) 15m index of the common grid (starts and ends on a UTC day boundary)."""
    return epoch_15m(GRID_START), epoch_15m(GRID_END) + PER_DAY


def to_dates(t: np.ndarray) -> np.ndarray:
    return (np.asarray(t) * 900).astype("datetime64[s]").astype("datetime64[D]")


def vault() -> Vault:
    return Vault(DEV[1], Path("runs/vault_r15/frozen.lock"), Path("runs/vault_r15/vault.log"))


def panel_dir(snapshot: str, market: str = "spot") -> Path:
    return Path("data/derived") / snapshot / "r15_15m" / market


def raw_dir(snapshot: str, market: str, sym: str) -> Path:
    return Path("data/raw") / snapshot / f"{market}_15m" / sym


def _load(snapshot: str, market: str, sym: str, end_date: date | None) -> dict[str, np.ndarray]:
    with np.load(panel_dir(snapshot, market) / f"{sym}.npz") as z:
        d = {k: z[k] for k in ("o", "h", "l", "c", "real", "funding")}
        t0 = int(z["t0"])
    d["t"] = t0 + np.arange(len(d["c"]), dtype=np.int64)
    if end_date is not None:                                    # dev stage never sees later candles
        n = int((to_dates(d["t"]) <= np.datetime64(end_date)).sum())
        d = {k: v[:n].copy() for k, v in d.items()}
    return d


def load_dev(snapshot: str, syms=PRIMARY, market: str = "spot") -> dict[str, dict[str, np.ndarray]]:
    v = vault()
    out = {s: _load(snapshot, market, s, DEV[1]) for s in syms}
    for d in out.values():
        v.check_dev_only(to_dates(d["t"]))
    return out


def load_final(snapshot: str, syms=PRIMARY, market: str = "spot") -> dict[str, dict[str, np.ndarray]]:
    vault().check_final_session(os.environ.get(FINAL_ENV, ""), snapshot, current_commit())
    return {s: _load(snapshot, market, s, None) for s in syms}


def registry() -> TrialRegistry:
    return TrialRegistry(REGISTRY)
