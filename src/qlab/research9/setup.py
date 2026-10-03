"""Stages, vault and registry of research 9 (prereg §5)."""

import os
from datetime import date
from pathlib import Path

import numpy as np

from qlab.data.panel import Panel, load_panel, save_panel
from qlab.validation.registry import TrialRegistry, current_commit
from qlab.validation.vault import Vault

PANEL_START = date(2017, 8, 1)
DEV = (date(2018, 4, 1), date(2021, 12, 31))
HOLDOUT = (date(2022, 1, 1), date(2026, 8, 31))
SUBPERIODS = {"2022-01..2023-12": (date(2022, 1, 1), date(2023, 12, 31)),
              "2024-01..2026-08": (date(2024, 1, 1), date(2026, 8, 31))}
FINAL_ENV = "QLAB_R9_METHODOLOGY"
REGISTRY = Path("runs/registry/research9.sqlite")


def vault() -> Vault:
    return Vault(DEV[1], Path("runs/vault_r9/frozen.lock"), Path("runs/vault_r9/vault.log"))


def panel_dir(snapshot: str) -> Path:
    return Path("data/derived") / snapshot / "panel_r9"


def symbols(snapshot: str) -> list[str]:
    return (panel_dir(snapshot) / "symbols.txt").read_text().split()


def save(snapshot: str, p: Panel, extra: dict[str, np.ndarray], syms: list[str]) -> None:
    save_panel(p, panel_dir(snapshot), extra)
    (panel_dir(snapshot) / "symbols.txt").write_text("\n".join(syms) + "\n")


def _cut(p: Panel, extra: dict, end: int) -> tuple[Panel, dict]:
    return p.slice(end), {k: np.asarray(m[:end]) for k, m in extra.items()}


def load_dev(snapshot: str) -> tuple[Panel, dict]:
    p, extra = load_panel(panel_dir(snapshot))
    v = vault()
    p, extra = _cut(p, extra, v.last_visible_index(p.dates))
    v.check_dev_only(p.dates)
    return p, extra


def load_final(snapshot: str) -> tuple[Panel, dict]:
    vault().check_final_session(os.environ.get(FINAL_ENV, ""), snapshot, current_commit())
    p, extra = load_panel(panel_dir(snapshot))
    return _cut(p, extra, len(p.dates))


def index_of(p: Panel, d: date) -> int:
    return int(np.searchsorted(p.dates, np.datetime64(d, "D")))


def registry() -> TrialRegistry:
    return TrialRegistry(REGISTRY)
