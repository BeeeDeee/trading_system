"""Stage, vault and registry of research 17 (prereg §3, §5, §7). Data: the 15m panel of research 15."""

import os
from datetime import date
from pathlib import Path

import numpy as np

from qlab.research15.setup import _load, epoch_15m
from qlab.validation.registry import TrialRegistry, current_commit
from qlab.validation.vault import Vault

START = date(2020, 5, 1)
END = date(2026, 9, 30)
SUBPERIODS = {"2020-05..2022-12": (date(2020, 5, 1), date(2022, 12, 31)),
              "2023-01..2026-09": (date(2023, 1, 1), date(2026, 9, 30))}
COST = 1e-3                           # 10 bps per side (prereg §5)
WEIGHT = 0.1                          # capital share of each member sleeve
K, HOLD = 3.0, 16                     # fixed rule of research 16 (LEAD_K3_LONG), 4 h
N_BOOT, N_RND = 2000, 1000
FINAL_ENV = "QLAB_R17_METHODOLOGY"
REGISTRY = Path("runs/registry/research17.sqlite")


def day0() -> int:
    """First UTC day (days since epoch) of the test and number of days of the fixed daily grid."""
    return epoch_15m(START) // 96


def n_days() -> int:
    return epoch_15m(END) // 96 - day0() + 1


def vault() -> Vault:
    return Vault(date(2020, 4, 30), Path("runs/vault_r17/frozen.lock"), Path("runs/vault_r17/vault.log"))


def load(snapshot: str, sym: str, market: str = "perp") -> dict[str, np.ndarray]:
    """Panel of one symbol; only inside an open final session (the whole test period is guarded)."""
    vault().check_final_session(os.environ.get(FINAL_ENV, ""), snapshot, current_commit())
    return _load(snapshot, market, sym, None)


def registry() -> TrialRegistry:
    return TrialRegistry(REGISTRY)
