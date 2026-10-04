"""Stages, vault and registry of research 13 (prereg §7). Data and periods of research 9."""

import os
from pathlib import Path

import numpy as np

from qlab.data.panel import Panel, load_panel
from qlab.research9.setup import DEV, HOLDOUT, PANEL_START, SUBPERIODS, index_of  # noqa: F401
from qlab.validation.registry import TrialRegistry, current_commit
from qlab.validation.vault import Vault

FINAL_ENV = "QLAB_R13_METHODOLOGY"
REGISTRY = Path("runs/registry/research13.sqlite")


def vault() -> Vault:
    return Vault(DEV[1], Path("runs/vault_r13/frozen.lock"), Path("runs/vault_r13/vault.log"))


def panel_dir(snapshot: str) -> Path:
    return Path("data/derived") / snapshot / "panel_r13"


def symbols(snapshot: str) -> list[str]:
    return (panel_dir(snapshot) / "symbols.txt").read_text().split()


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


def registry() -> TrialRegistry:
    return TrialRegistry(REGISTRY)
