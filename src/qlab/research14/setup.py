"""Hourly data, stages, vault and registry of research 14 (prereg §3, §7).

The panel is one `.npz` per symbol on a common gap-free UTC hour grid: `close` (forward-filled, NaN before
the first candle), `real` (a candle exists in that hour) and `funding` (perp rate settled at that hour,
NaN where no data). Hours are epoch hours (int64), so hour h covers [h, h+1) and its close is known at h+1.
"""

import os
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from qlab.validation.registry import TrialRegistry, current_commit
from qlab.validation.vault import Vault

PRIMARY = ("BTCUSDT", "ETHUSDT")
SECONDARY = ("BNBUSDT", "LTCUSDT", "ADAUSDT", "XRPUSDT", "ETCUSDT", "TRXUSDT", "XLMUSDT", "NEOUSDT", "EOSUSDT")
RAW_1H = Path("../market_relations_2026_oct/data/raw/binance_1h/spot_1h")
GRID_START = date(2017, 8, 17)
GRID_END = date(2026, 8, 31)          # last hour 23:00 UTC
DEV = (date(2018, 2, 1), date(2021, 12, 31))
HOLDOUT = (date(2022, 1, 1), date(2026, 8, 31))
SUBPERIODS = {"2022-01..2023-12": (date(2022, 1, 1), date(2023, 12, 31)),
              "2024-01..2026-08": (date(2024, 1, 1), date(2026, 8, 31))}
FINAL_ENV = "QLAB_R14_METHODOLOGY"
REGISTRY = Path("runs/registry/research14.sqlite")


def epoch_hour(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()) // 3600


def hour_grid() -> np.ndarray:
    return np.arange(epoch_hour(GRID_START), epoch_hour(GRID_END) + 24, dtype=np.int64)


def hours_to_dates(hours: np.ndarray) -> np.ndarray:
    return (np.asarray(hours) * 3600).astype("datetime64[s]").astype("datetime64[D]")


def vault() -> Vault:
    return Vault(DEV[1], Path("runs/vault_r14/frozen.lock"), Path("runs/vault_r14/vault.log"))


def panel_dir(snapshot: str) -> Path:
    return Path("data/derived") / snapshot / "r14_1h"


def _load(snapshot: str, sym: str, end: int) -> dict[str, np.ndarray]:
    with np.load(panel_dir(snapshot) / f"{sym}.npz") as z:
        return {k: z[k][:end].copy() for k in ("hours", "close", "real", "funding")}


def load_dev(snapshot: str, syms=PRIMARY) -> dict[str, dict[str, np.ndarray]]:
    v = vault()
    hours = hour_grid()
    end = v.last_visible_index(hours_to_dates(hours))
    out = {s: _load(snapshot, s, end) for s in syms}
    for d in out.values():
        v.check_dev_only(hours_to_dates(d["hours"]))
    return out


def load_final(snapshot: str, syms=PRIMARY) -> dict[str, dict[str, np.ndarray]]:
    vault().check_final_session(os.environ.get(FINAL_ENV, ""), snapshot, current_commit())
    return {s: _load(snapshot, s, len(hour_grid())) for s in syms}


def registry() -> TrialRegistry:
    return TrialRegistry(REGISTRY)
