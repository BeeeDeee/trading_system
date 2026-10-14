"""Stages, vault and registry of research 16 (prereg §3, §7). Data: the 15m panel of research 15."""

import os
from datetime import date
from pathlib import Path

from qlab.research15.setup import DEV, HOLDOUT, PRIMARY, SUBPERIODS, _load, to_dates  # noqa: F401
from qlab.validation.registry import TrialRegistry, current_commit
from qlab.validation.vault import Vault

FUND_DEV_START = date(2020, 4, 1)     # funding data from 2020-01 + 90 days of history
FINAL_ENV = "QLAB_R16_METHODOLOGY"
REGISTRY = Path("runs/registry/research16.sqlite")


def vault() -> Vault:
    return Vault(DEV[1], Path("runs/vault_r16/frozen.lock"), Path("runs/vault_r16/vault.log"))


def load_dev(snapshot: str, syms=PRIMARY, market: str = "spot") -> dict:
    v = vault()
    out = {s: _load(snapshot, market, s, DEV[1]) for s in syms}
    for d in out.values():
        v.check_dev_only(to_dates(d["t"]))
    return out


def load_final(snapshot: str, syms=PRIMARY, market: str = "spot") -> dict:
    vault().check_final_session(os.environ.get(FINAL_ENV, ""), snapshot, current_commit())
    return {s: _load(snapshot, market, s, None) for s in syms}


def registry() -> TrialRegistry:
    return TrialRegistry(REGISTRY)
