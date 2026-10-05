"""`lab try`: the Builder's G0 on a synthetic market, without real data."""

import numpy as np
import pytest

from lab.framework import agents, dryrun, invocations

from .conftest import submit

MOMENTUM = '''
import numpy as np
from lab.framework.api import only_on, period_starts, total_return_index, trailing_return
PARAMS = {"lookback_days": 252, "top_n": 3}
def target_weights(data, params):
    mom = np.where(data.listed, trailing_return(total_return_index(data), params["lookback_days"], 21), np.nan)
    w = np.zeros(data.shape)
    for t in range(len(w)):
        ok = np.flatnonzero(np.isfinite(mom[t]))
        if len(ok) >= params["top_n"]:
            w[t, ok[np.argsort(-mom[t, ok])[: params["top_n"]]]] = 1.0 / params["top_n"]
    return only_on(period_starts(data.dates, "M"), w)
'''
PEEK = MOMENTUM.replace("total_return_index(data)", "np.roll(total_return_index(data), -30, axis=0)")
TEST = '''
from lab.framework.data import synthetic
from strategy import target_weights, PARAMS
def test_shape():
    v = synthetic().columns(["S00", "S01", "S02", "S03"])
    assert target_weights(v, dict(PARAMS, top_n=2)).shape == v.shape
'''


@pytest.fixture
def workspace(lab, card):
    hid = submit(lab, card)
    inv = invocations.start(lab, "builder", hid, task="implement")
    return inv.workspace


def write(ws, strategy, test=TEST):
    (ws / "strategy" / "strategy.py").write_text(strategy)
    if test is not None:
        (ws / "strategy" / "test_strategy.py").write_text(test)


def test_builder_workspace_has_gates(workspace):
    assert (workspace / "gates.yaml").exists() and (workspace / "card.yaml").exists()


def test_faithful_strategy_is_ready(workspace):
    write(workspace, MOMENTUM)
    ok, lines = dryrun.run(workspace)
    assert ok, "\n".join(lines)
    text = "\n".join(lines)
    assert "passed" in text and "grid neighbors: 8" in text and "one_way_turnover_per_year" in text
    assert "sharpe" not in text.lower() and "cagr" not in text.lower()   # no performance on any data


def test_lookahead_fails_on_synthetic_data(workspace):
    write(workspace, PEEK)
    ok, lines = dryrun.run(workspace)
    assert not ok and any("g0_lookahead" in x for x in lines)


def test_missing_tests_and_forbidden_test_imports(workspace):
    write(workspace, MOMENTUM, test=None)
    ok, lines = dryrun.run(workspace)
    assert not ok and any("none found" in x for x in lines)
    write(workspace, MOMENTUM, test="import os\ndef test_x():\n    assert os\n")
    ok, lines = dryrun.run(workspace)
    assert not ok and any("static rules" in x for x in lines)


def test_synthetic_view_follows_the_card(card):
    from lab.framework import catalog
    from lab.framework.paths import LAB_DIR
    cat = catalog.load(LAB_DIR / "data" / "catalog.yaml")
    v = dryrun.synthetic_view(card, cat, seed=1)
    assert list(v.instruments) == card["universe"]["instruments"]
    assert str(v.dates[-1]) == "2020-12-31" and np.is_busday(v.dates).all()
    assert (~v.listed).any()                       # a late listing


def test_crypto_top_n_view_has_a_universe(card):
    from lab.framework import catalog
    from lab.framework.paths import LAB_DIR
    cat = catalog.load(LAB_DIR / "data" / "catalog.yaml")
    card = dict(card, asset_classes=["crypto_spot"], universe={"kind": "crypto_top_n", "n": 10},
                data_requirements=[{"dataset": "binance_spot_1d", "frequency": "1d", "period": ["2019-01-01", "2022-12-31"]}])
    v = dryrun.synthetic_view(card, cat, seed=1)
    assert v.universe is not None and v.universe.sum(axis=1).max() == 10
    assert v.delisting.any() and not np.is_busday(v.dates).all()


def test_builder_definition():
    s = agents.spec("builder")
    assert "try" in s.lab_verbs and "check" not in s.lab_verbs and "WebSearch" not in s.tools


def test_synthetic_market_cards_get_the_generator_with_its_series():
    import yaml
    from lab.framework.data import SYNTHETIC_DATASET
    from lab.framework.paths import LAB_DIR
    card = yaml.safe_load((LAB_DIR / "canaries" / "positive" / "card.yaml").read_text())
    v = dryrun.synthetic_view(card, {"synthetic_market": SYNTHETIC_DATASET}, seed=0)
    assert "signal" in v.series and str(v.dates[-1]) < SYNTHETIC_DATASET["holdout_from"]


def test_synthetic_views_for_stocks_and_perps(card):
    from lab.framework import catalog
    from lab.framework.paths import LAB_DIR
    cat = catalog.load(LAB_DIR / "data" / "catalog.yaml")
    stocks = dict(card, asset_classes=["us_equity"], universe={"kind": "liq_n", "n": 200},
                  data_requirements=[{"dataset": "sharadar_sep", "frequency": "1d"}])
    v = dryrun.synthetic_view(stocks, cat, seed=2)
    assert all(i.startswith("E") for i in v.instruments) and v.universe.sum(axis=1).max() == 40
    assert {"liq_rank", "alt_universe"} <= set(v.extras) and v.delisting.sum() == 2
    perps = dict(card, asset_classes=["crypto_perp"], universe={"kind": "instruments", "instruments": ["BTCUSDT.P", "ETHUSDT.P", "SOLUSDT.P"]},
                 data_requirements=[{"dataset": "binance_perp_1d", "frequency": "1d"}])
    v = dryrun.synthetic_view(perps, cat, seed=2)
    assert {"funding", "funding_paid", "basis"} <= set(v.extras)
