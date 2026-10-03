"""Stages, vault, registry and data of research 11 (prereg §6)."""

import json
import os
from datetime import date
from pathlib import Path

import numpy as np

from qlab.data.panel import Panel, load_panel
from qlab.research11 import features as F
from qlab.validation.registry import TrialRegistry, current_commit
from qlab.validation.vault import Vault

DEV_END = date(2019, 12, 31)
DEV_YEARS = list(range(2010, 2020))
FINAL_YEARS = list(range(2020, 2027))
FINAL_END = date(2026, 8, 31)        # last decision 2026-08 (its label ends at the 2026-09 fill)
SUBPERIODS = {"2020-01..2022-12": (date(2020, 1, 1), date(2022, 12, 31)),
              "2023-01..2026-08": (date(2023, 1, 1), date(2026, 9, 25))}
FINAL_ENV = "QLAB_R11_METHODOLOGY"
REGISTRY = Path("runs/registry/research11.sqlite")


def vault() -> Vault:
    return Vault(DEV_END, Path("runs/vault_r11/frozen.lock"), Path("runs/vault_r11/vault.log"))


def registry() -> TrialRegistry:
    return TrialRegistry(REGISTRY)


def _load(snapshot: str, end: int | None) -> tuple[Panel, dict, dict, np.ndarray, dict]:
    der = Path("data/derived") / snapshot
    p, extra = load_panel(der / "panel_r4")
    meta = json.loads((der / "panel_r4" / "columns.json").read_text())
    if end is not None:
        p, extra = p.slice(end), {k: v[:end] for k, v in extra.items()}
    days_all = np.load(der / "research4" / "sleeves" / "days.npy")
    days = days_all[days_all < p.shape[0]]
    return p, extra, meta, days, {"days_all": days_all}


def load_dev(snapshot: str):
    der = Path("data/derived") / snapshot
    dates = np.load(der / "panel_r4" / "dates.npy").astype("datetime64[D]")
    v = vault()
    out = _load(snapshot, v.last_visible_index(dates))
    v.check_dev_only(out[0].dates)
    return out


def load_final(snapshot: str):
    vault().check_final_session(os.environ.get(FINAL_ENV, ""), snapshot, current_commit())
    return _load(snapshot, None)


def scores(snapshot: str, stage: str, p: Panel, meta: dict, days: np.ndarray) -> dict[str, np.ndarray]:
    """Research 4 scores (rows = days) + research 11 features, cached per stage."""
    der = Path("data/derived") / snapshot
    r4 = np.load(der / "research4" / "sleeves" / "scores.npz", mmap_mode="r")
    out = {k: np.asarray(r4[k][:len(days)]) for k in r4.files}
    cache = der / "research11" / f"features_{stage}.npz"
    if cache.exists():
        new = dict(np.load(cache))
    else:
        dec = [d.item() for d in np.asarray(p.dates[days], dtype="datetime64[D]")]
        new = F.build(Path("data/parquet") / snapshot, np.asarray(p.assets[:meta["n_stock_cols"]]), dec)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **new)
    for k, v in new.items():
        full = np.full((len(days), p.shape[1]), np.nan, dtype=np.float32)
        full[:, :v.shape[1]] = v
        out[k] = full
    return out
