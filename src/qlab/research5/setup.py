"""Research 5 stages and their data boundaries (pre-registration §7).

dev          simulate 1999-01-04 .. 2014-12-31, panel rows up to 2014-12-31
validation   simulate 2015-01-01 .. 2019-12-31, panel rows up to 2019-12-31 (run once)
late         simulate 2020-01-01 .. snapshot end, only inside the logged research 5 vault session
"""

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from qlab.data.panel import load_panel
from qlab.research5.strategy import Context
from qlab.validation.registry import TrialRegistry, config_hash, current_commit
from qlab.validation.vault import Vault, VaultError

STAGES = {
    "dev": (date(1999, 1, 4), date(2014, 12, 31)),
    "validation": (date(2015, 1, 1), date(2019, 12, 31)),
    "late": (date(2020, 1, 1), None),
}
FINAL_ENV = "QLAB_R5_METHODOLOGY"
REGISTRY = Path("runs/registry/research5.sqlite")
EXPERIMENTS_LOG = Path("runs/research5/experiments.log")


def vault() -> Vault:
    return Vault(STAGES["validation"][1], Path("runs/vault_r5/frozen.lock"),
                 Path("runs/vault_r5/vault.log"))


def load_stage(snapshot: str, stage: str) -> Context:
    first, last = STAGES[stage]
    der = Path("data/derived") / snapshot
    full, extra = load_panel(der / "panel_r5")
    if stage == "late":
        vault().check_final_session(os.environ.get(FINAL_ENV, ""), snapshot, current_commit())
        end = full.shape[0]
        panel = full
    else:
        v = Vault(last, vault().lock_path, vault().log_path)
        end = v.last_visible_index(full.dates)
        panel = full.slice(end)
        v.check_dev_only(panel.dates)
    extra = {k: m[:end] for k, m in extra.items()}  # extras obey the same boundary
    start = int(np.searchsorted(panel.dates, np.datetime64(first)))
    spy_perm = json.loads((der / "panel_r5" / "special_assets.json").read_text())["SPY"]
    return Context(panel, extra, start, der / "r5_cache" / stage,
                   int(np.searchsorted(panel.assets, spy_perm)))


def registry() -> TrialRegistry:
    return TrialRegistry(REGISTRY)


def log_experiment(stage: str, config, result: dict, note: str = "") -> None:
    """Human-readable companion of the append-only registry (one JSON line per run)."""
    EXPERIMENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENTS_LOG.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "commit": current_commit(),
                            "stage": stage, "config_hash": config_hash(config), "config": config,
                            "result": result, "note": note}, default=str) + "\n")


def once(stage: str, config) -> None:
    """Refuse a second run of a run-once stage for the same configuration."""
    reg = registry()
    key = {"stage": stage, "config": config}
    if reg.has(key):
        raise VaultError(f"{stage} was already run for this configuration")
    reg.record("methodology_eval", key, 1, f"research5 {stage}")
