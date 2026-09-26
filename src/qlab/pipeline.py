"""Shared setup of the development-period context (vault-protected) for grid and WFO runs."""

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from qlab.data.panel import Panel, load_panel
from qlab.engine.costs import CostModel
from qlab.strategies.run import Context
from qlab.validation.registry import current_commit
from qlab.validation.vault import Vault

FINAL_ENV = "QLAB_FINAL_METHODOLOGY"  # set by scripts/final_evaluation.py

CONFIG = Path("configs/frozen_defaults.yaml")


@dataclass
class DevSetup:
    cfg: dict
    ctx: Context
    panel: Panel          # rows up to the vault boundary
    start: int            # first development day (row)
    end: int              # rows in the panel
    extra: dict


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def cost_model(costs_cfg: dict, multiplier: float = 1.0) -> CostModel:
    return CostModel(costs_cfg["commission_bps"], costs_cfg["floor_bps"],
                     tuple(tuple(t) for t in costs_cfg["tiers"]), costs_cfg["default_bps"],
                     costs_cfg["pre_decimal_multiplier"], multiplier)


UNIVERSES = {"liq1000": "in_liq1000", "sp500": "in_sp500"}
DELISTING_SCENARIOS = ("base", "optimistic", "pessimistic")


def apply_delisting_scenario(panel: Panel, delistings: pl.DataFrame, scenario: str) -> Panel:
    """Replace terminal returns on delisting rows (spec §4.5 sensitivity).

    optimistic: every delisting pays the last close (0 %); pessimistic: regulatory/performance
    delistings -100 %, delistings without any action (unknown) -50 %; everything else unchanged.
    """
    if scenario == "base":
        return panel
    ret_co = np.array(panel.ret_co)
    t_idx, a_idx = np.nonzero(np.asarray(panel.delisting))
    if scenario == "optimistic":
        ret_co[t_idx, a_idx] = 0.0
    elif scenario == "pessimistic":
        kind = dict(zip(delistings["permaticker"].to_list(), delistings["kind"].to_list()))
        for t, a in zip(t_idx, a_idx):
            k = kind.get(int(panel.assets[a]), "unknown")
            if k == "performance":
                ret_co[t, a] = -1.0
            elif k == "unknown":
                ret_co[t, a] = -0.5
    else:
        raise ValueError(f"unknown delisting scenario {scenario!r}")
    return replace(panel, ret_co=ret_co)


def vault_for(cfg: dict) -> Vault:
    return Vault(cfg["periods"]["development"][1], Path("runs/vault/frozen.lock"),
                 Path("runs/vault/vault.log"))


def dev_setup(snapshot: str, cost_multiplier: float = 1.0, universe: str = "liq1000",
              delisting: str = "base", final: bool = False) -> DevSetup:
    """Development-period setup; with `final=True` the whole history including the holdout,
    allowed only inside the logged final evaluation (spec §9.7)."""
    cfg = load_config()
    der = Path("data/derived") / snapshot
    dev_start = cfg["periods"]["development"][0]
    vault = vault_for(cfg)
    full, extra = load_panel(der / "panel_liq1000")
    if final:
        vault.check_final_session(os.environ.get(FINAL_ENV, ""), snapshot, current_commit())
        end = full.shape[0]
        panel = full
    else:
        end = vault.last_visible_index(full.dates)
        panel = full.slice(end)
        vault.check_dev_only(panel.dates)
    panel = apply_delisting_scenario(panel, pl.read_parquet(der / "delistings.parquet"), delisting)
    start = int(np.searchsorted(panel.dates, np.datetime64(dev_start)))
    spy_perm = json.loads((der / "panel_liq1000" / "special_assets.json").read_text())["SPY"]
    spy = int(np.searchsorted(panel.assets, spy_perm))
    ctx = Context(panel, np.asarray(extra[UNIVERSES[universe]][:end]), spy, first_decision=start - 1,
                  cost_rate=cost_model(cfg["costs"], cost_multiplier).rate(
                      np.asarray(extra["liq_rank"][:end]), panel.dates),
                  cash_ret=np.asarray(extra["cash_ret"][:end]))
    return DevSetup(cfg, ctx, panel, start, end, extra)


def spy_returns(snapshot: str, dates: np.ndarray) -> np.ndarray:
    funds = pl.read_parquet(Path("data/parquet") / snapshot / "funds.parquet",
                            columns=["ticker", "date", "closeadj"])
    spy = funds.filter(pl.col("ticker") == "SPY").sort("date").with_columns(
        r=pl.col("closeadj").pct_change())
    return (pl.DataFrame({"date": dates}).with_columns(pl.col("date").cast(pl.Date))
            .join(spy, on="date", how="left")["r"].fill_null(0.0).to_numpy())
