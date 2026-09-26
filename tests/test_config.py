"""Keep code defaults in sync with configs/frozen_defaults.yaml."""

from datetime import date
from pathlib import Path

import numpy as np
import yaml

from qlab.data import normalize
from qlab.data.sharadar import MAX_CONSIDERATION_DEVIATION, SPAC_SIC, UNIVERSE_CATEGORIES
from qlab.engine.costs import DECIMALIZATION, CostModel

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "frozen_defaults.yaml").read_text())


def test_cost_defaults():
    c, cm = CFG["costs"], CostModel()
    assert (cm.commission_bps, cm.floor_bps, cm.default_bps, cm.pre_decimal_multiplier) == (
        c["commission_bps"], c["floor_bps"], c["default_bps"], c["pre_decimal_multiplier"])
    assert [list(t) for t in cm.tiers] == c["tiers"]
    assert DECIMALIZATION == np.datetime64(c["decimalization"])


def test_universe_and_delisting_defaults():
    u, d = CFG["universe"], CFG["delisting"]
    assert list(UNIVERSE_CATEGORIES) == u["categories"] and SPAC_SIC == u["exclude_sic"]
    assert MAX_CONSIDERATION_DEVIATION == d["max_consideration_deviation"]
    assert normalize.terminal_return(normalize.BANKRUPTCY, 1.0) == d["bankruptcy"]
    assert normalize.terminal_return(normalize.PERFORMANCE, 1.0) == d["performance"]
    assert normalize.terminal_return(normalize.SPAC_LIQUIDATION, 1.0) == d["spac_liquidation"]
    low = d["unknown_low_price"]
    assert normalize.terminal_return(normalize.UNKNOWN, low * 0.99) == d["performance"]
    assert normalize.terminal_return(normalize.UNKNOWN, low) == 0.0


def test_periods_are_contiguous():
    p = CFG["periods"]
    assert p["development"][1] == date(2019, 12, 31)
    assert (p["holdout"][0] - p["development"][1]).days == 1
    assert (p["development"][0] - p["warmup"][1]).days == 1
