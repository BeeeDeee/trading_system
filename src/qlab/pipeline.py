"""Shared setup of the development-period context (vault-protected) for grid and WFO runs."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from qlab.data.panel import Panel, load_panel
from qlab.engine.costs import CostModel
from qlab.strategies.run import Context
from qlab.validation.vault import Vault

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


def dev_setup(snapshot: str, cost_multiplier: float = 1.0) -> DevSetup:
    cfg = load_config()
    der = Path("data/derived") / snapshot
    dev_start, dev_end = cfg["periods"]["development"]
    vault = Vault(dev_end, Path("runs/vault/frozen.lock"), Path("runs/vault/vault.log"))
    full, extra = load_panel(der / "panel_liq1000")
    end = vault.last_visible_index(full.dates)
    panel = full.slice(end)
    vault.check_dev_only(panel.dates)
    start = int(np.searchsorted(panel.dates, np.datetime64(dev_start)))
    spy_perm = json.loads((der / "panel_liq1000" / "special_assets.json").read_text())["SPY"]
    spy = int(np.searchsorted(panel.assets, spy_perm))
    ctx = Context(panel, np.asarray(extra["in_liq1000"][:end]), spy, first_decision=start - 1,
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
