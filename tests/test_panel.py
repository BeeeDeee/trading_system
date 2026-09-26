import numpy as np
import polars as pl

from qlab.data.normalize import normalize_prices
from qlab.data.panel import panel_from_bars
from qlab.data.schema import DELISTED
from qlab.data.synthetic import generate_market


def test_panel_matches_bars():
    m = generate_market(n_assets=20, seed=3)
    bars = normalize_prices(m.prices, m.delistings, m.calendar)
    p = panel_from_bars(bars, m.calendar)
    assert p.shape == (len(m.calendar), bars["permaticker"].n_unique())
    assert p.listed.sum() == bars.height
    assert p.delisting.sum() == (bars["status"] == DELISTED).sum()
    growth = np.prod((1 + p.ret_co) * (1 + p.ret_oc), axis=0)
    expected = (bars.group_by("permaticker", maintain_order=True)
                .agg(g=((1 + pl.col("ret_co")) * (1 + pl.col("ret_oc"))).product())
                .sort("permaticker")["g"].to_numpy())
    np.testing.assert_allclose(growth, expected, rtol=1e-12)
    # Not listed -> never tradable; a slice is a prefix.
    assert not (p.tradable & ~p.listed).any()
    s = p.slice(100)
    assert s.shape == (100, p.shape[1]) and np.array_equal(s.ret_co, p.ret_co[:100])
