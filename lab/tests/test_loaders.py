"""Loaders for data already on disk: single stocks (SEP), perpetuals with funding, FRED as signal series."""

import shutil

import numpy as np
import pytest

from lab.framework import catalog, data, engine
from lab.framework.data import DATA_ROOT, DataView
from lab.framework.paths import LAB_DIR

needs_data = pytest.mark.skipif(not DATA_ROOT.exists(), reason="no data snapshot on this machine")


def test_funding_is_paid_by_longs_and_earned_by_shorts():
    T = 5
    z = np.zeros((T, 2))
    targets = np.full((T, 2), np.nan)
    targets[0] = [0.5, -0.5]
    funding = np.zeros((T, 2))
    funding[2:] = 0.001
    r = engine.simulate(z, z, targets, 0.0, np.ones((T, 2), bool), np.zeros((T, 2), bool), funding=funding)
    assert np.allclose(r.returns, 0.0)                    # long pays, short receives the same rate
    targets[0] = [0.5, 0.0]
    r = engine.simulate(z, z, targets, 0.0, np.ones((T, 2), bool), np.zeros((T, 2), bool), funding=funding)
    assert r.nav[-1] == pytest.approx(1 - 3 * 0.0005)


def test_merge_keeps_float32_when_a_part_is_float32():
    d = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-11"))
    def v(name, dtype):
        a = np.zeros((10, 1), dtype=dtype)
        b = np.ones((10, 1), bool)
        return DataView(d, (name,), ("us_equity",), a, a, b, b, ~b, a + 1, a + 1, extras={"x": a})
    m = data.merge([v("A", np.float32), v("B", np.float64)])
    assert m.ret_co.dtype == np.float32 and m.extras["x"].dtype == np.float32


def test_enable_generic_edits_only_that_entry(tmp_path):
    cat = tmp_path / "catalog.yaml"
    shutil.copy(LAB_DIR / "data" / "catalog.yaml", cat)
    before = catalog.load(cat)
    victim = next(d for d in before.values() if d.get("loader") is False and d["id"] != "binance_spot_1h")
    catalog.enable_generic(cat, victim["id"], ["A", "B"], "LAB_HOME/data/x")
    after = catalog.load(cat)
    assert after[victim["id"]]["loader"] == "generic" and after[victim["id"]]["fields"] == ["A", "B"]
    assert after[victim["id"]]["holdout_from"] == victim["holdout_from"]
    assert {k: v for k, v in after.items() if k != victim["id"]} == {k: v for k, v in before.items() if k != victim["id"]}
    assert cat.read_text().startswith("# Data catalog")    # comments survive


@needs_data
def test_perp_loader():
    v = data.binance_perp(["BTCUSDT.P", "ETHUSDT.P"])
    assert v.asset_class == ("crypto_perp", "crypto_perp") and {"funding", "funding_paid", "basis"} <= set(v.extras)
    assert np.isfinite(v.extras["funding"][v.listed]).all()
    assert (np.abs(v.extras["basis"][v.listed & np.isfinite(v.extras["basis"])]) < 0.2).all()


@needs_data
def test_sep_loader_is_ticker_blind_and_point_in_time_masked():
    v = data.sharadar_sep({"kind": "liq_n", "n": 50})
    assert all(i.startswith("E") for i in v.instruments) and v.ret_co.dtype == np.float32
    assert v.universe.sum(axis=1).max() == 50
    assert not (v.universe & ~v.listed).any()
    with pytest.raises(NotImplementedError, match="1..500"):
        data.sharadar_sep({"kind": "liq_n", "n": 1000})
