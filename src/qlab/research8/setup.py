"""Stages, vault and registry of research 8 (prereg §5)."""

import os
from datetime import date
from pathlib import Path

import numpy as np

from qlab.validation.registry import TrialRegistry, current_commit
from qlab.validation.vault import Vault

from . import panel as P

DEV = (date(2020, 1, 6), date(2022, 12, 31))
HOLDOUT = (date(2023, 1, 1), date(2026, 9, 30))
SUBPERIODS = {"2023-01..2024-06": (date(2023, 1, 1), date(2024, 6, 30)),
              "2024-07..2026-09": (date(2024, 7, 1), date(2026, 9, 30))}
WARMUP_START = date(2019, 12, 1)         # 30-day universe window and trailing funding before DEV[0]
FINAL_ENV = "QLAB_R8_METHODOLOGY"
REGISTRY = Path("runs/registry/research8.sqlite")


def vault() -> Vault:
    return Vault(DEV[1], Path("runs/vault_r8/frozen.lock"), Path("runs/vault_r8/vault.log"))


def panel_path(snapshot: str) -> Path:
    return Path("data/derived") / snapshot / "r8_panel.npz"


def load_dev(snapshot: str) -> P.Panel:
    p = P.load(panel_path(snapshot), DEV[1])
    vault().check_dev_only(p.dates)
    return p


def load_final(snapshot: str) -> P.Panel:
    """Whole panel; only inside the one logged final evaluation (vault.open_final first)."""
    vault().check_final_session(os.environ.get(FINAL_ENV, ""), snapshot, current_commit())
    return P.load(panel_path(snapshot), HOLDOUT[1])


def index_of(p: P.Panel, d: date) -> int:
    return int(np.searchsorted(p.dates, np.datetime64(d, "D")))


def registry() -> TrialRegistry:
    return TrialRegistry(REGISTRY)
