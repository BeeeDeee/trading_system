import numpy as np
import pytest

from qlab.benchmarks import buy_and_hold_returns, equal_weight_targets
from qlab.data.panel import Panel, load_panel, save_panel


def make_panel(ret_co, ret_oc, delisting=None):
    ret_co, ret_oc = np.asarray(ret_co, float), np.asarray(ret_oc, float)
    shape = ret_co.shape
    return Panel(np.arange(shape[0]).astype("datetime64[D]"), np.arange(shape[1]), ret_co, ret_oc,
                 np.ones(shape, bool), np.ones(shape, bool),
                 np.zeros(shape, bool) if delisting is None else np.asarray(delisting),
                 np.full(shape, 10.0), np.full(shape, 1e6))


def test_equal_weight_targets():
    members = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], bool)
    t = equal_weight_targets(members, np.array([True, False, True]))
    assert t[0].tolist() == [0.5, 0.5, 0.0] and np.isnan(t[1]).all() and t[2].tolist() == [0, 0, 0]


def test_buy_and_hold_reinvests_delisting_pro_rata():
    # Two assets; asset 1 delists on day 2 with +50 % payout that goes into asset 0.
    ret_co = np.array([[0, 0], [0.0, 0.0], [0.0, 0.5], [0.1, 0.0]])
    ret_oc = np.array([[0.0, 0.0], [0.1, -0.1], [0.0, 0.0], [0.0, 0.0]])
    delisting = np.zeros((4, 2), bool)
    delisting[2, 1] = True
    r = buy_and_hold_returns(make_panel(ret_co, ret_oc, delisting), np.array([1, 1]), start=0)
    # day 1: 0.5*1.1 + 0.5*0.9 = 1.0 ; day 2: asset 1 pays 0.45*1.5=0.675 -> total 1.225
    # day 3: everything in asset 0, +10 %
    np.testing.assert_allclose(np.cumprod(1 + r), [1.0, 1.0, 1.225, 1.3475])


def test_panel_roundtrip(tmp_path):
    p = make_panel(np.random.default_rng(0).normal(size=(5, 3)), np.zeros((5, 3)))
    save_panel(p, tmp_path / "p", extra={"liq_rank": np.ones((5, 3), np.float32)})
    q, extra = load_panel(tmp_path / "p")
    assert np.array_equal(q.ret_co, p.ret_co) and np.array_equal(q.dates, p.dates)
    assert extra["liq_rank"].dtype == np.float32
    assert pytest.approx(float(q.ret_co.sum())) == float(p.ret_co.sum())


def test_delisting_scenarios():
    import polars as pl
    from qlab.pipeline import apply_delisting_scenario
    ret_co = np.zeros((3, 3))
    delisting = np.zeros((3, 3), bool)
    delisting[1] = True
    ret_co[1] = [0.2, -0.3, 0.0]  # acquisition, performance, unknown (no record)
    p = make_panel(ret_co, np.zeros((3, 3)), delisting)
    dl = pl.DataFrame({"permaticker": [0, 1], "kind": ["acquisition", "performance"]})
    assert apply_delisting_scenario(p, dl, "base") is p
    np.testing.assert_array_equal(apply_delisting_scenario(p, dl, "optimistic").ret_co[1], [0, 0, 0])
    np.testing.assert_array_equal(apply_delisting_scenario(p, dl, "pessimistic").ret_co[1],
                                  [0.2, -1.0, -0.5])
    assert p.ret_co[1, 1] == -0.3  # original untouched
