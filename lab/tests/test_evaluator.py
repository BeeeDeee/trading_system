"""The real evaluator on the synthetic market, gate by gate."""

import json
import shutil

import pytest

from lab.framework import canaries as C
from lab.framework import invocations
from lab.framework.invocations import stage
from lab.framework.tick import tick
from lab.framework.evaluator import QlabEvaluator
from lab.framework.gates import run_gate
from lab.framework.states import S


@pytest.fixture
def slab(tmp_path):
    return C._lab(tmp_path)


def gate_rows(lab, hid):
    return [(r["gate"], bool(r["passed"]), r["reason_code"], json.loads(r["metrics_json"]))
            for r in lab.con.execute("SELECT * FROM gate_results WHERE hypothesis_id = ? ORDER BY id", (hid,))]


def test_neighbors_are_one_step_combinations():
    ev = QlabEvaluator(require_canaries=False)
    card = {"signal": {"params": {"a": {"value": 2, "grid": [1, 2, 3]}, "b": {"value": 10, "grid": [10, 20]}}}}
    assert ev.neighbors(card) == [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 20}, {"a": 3, "b": 10},
                                  {"a": 3, "b": 20}]
    many = {"signal": {"params": {k: {"value": 1, "grid": [0, 1, 2]} for k in "abcd"}}}
    assert len(ev.neighbors(many)) == 8          # > 3 parameters: one at a time


def test_positive_control_metrics_and_trials(slab):
    ev = QlabEvaluator(C.QUICK, require_canaries=False)
    hid = C.submit_and_build(slab, C._card("positive"), C.CANARIES / "positive" / "strategy.py", ev)
    assert slab.hypothesis(hid)["status"] == S.SKEPTIC_REVIEW
    rows = gate_rows(slab, hid)
    assert [g for g, *_ in rows] == ["G0", "G1", "G2", "G3"]
    g1 = rows[1][3]
    assert g1["benchmark"] == "SYN_MKT" and g1["end"] == "2019-12-31"      # dev only: holdout_from 2020-01-01
    assert all(c["pass"] for c in g1["checks"].values())
    d = g1["decomposition"]                                                  # descriptive: gross vs net, not a gate
    assert d["sharpe_gross"] >= d["sharpe_net"] and d["cost_drag_per_year"] >= 0
    assert slab.family_trials("canary-positive") == 1 + 2                   # primary + two threshold neighbors
    thresholds = {r[0] for r in slab.con.execute("SELECT thresholds_sha256 FROM gate_results")}
    assert len(thresholds) == 1


def test_g3_stops_the_overfit_seed_even_when_its_search_is_hidden(slab):
    """Force the lucky seed past G2 and check that the deflated Sharpe alone rejects it."""
    seed, _ = C.pick_overfit_seed()
    card = C._card("overfit")
    card["signal"]["params"]["seed"] = {"value": seed, "grid": [seed - 1, seed, seed + 1]}
    ev = QlabEvaluator(C.QUICK, require_canaries=False)
    hid = C.submit_and_build(slab, card, C.CANARIES / "overfit" / "strategy.py", ev)
    assert slab.hypothesis(hid)["reject_stage"] == "G2"
    slab.con.execute("UPDATE hypotheses SET status = 'GATE_2', reject_stage = NULL WHERE id = ?", (hid,))
    out = run_gate(slab, hid, "G3", ev)
    assert not out.passed and out.reason_code == "g3_dsr"
    assert out.metrics["n_trials"] == 10                                    # only the floor, nothing recorded


def test_holdout_is_scored_only_after_the_boundary(slab):
    ev = QlabEvaluator(C.QUICK, require_canaries=False)
    hid = C.submit_and_build(slab, C._card("positive"), C.CANARIES / "positive" / "strategy.py", ev)
    C.clean_review(slab, hid, ev)
    g4 = [r for r in gate_rows(slab, hid) if r[0] == "G4"][0][3]
    assert g4["start"] >= "2020-01-01" and g4["end"] == "2025-12-31"
    assert g4["family_attempts_before"] == 0 and g4["ci_level"] == 0.80


def test_gates_refuse_to_run_without_passing_canaries(slab):
    ev = QlabEvaluator(C.QUICK)                                            # require_canaries=True
    slab.send("NEW_HYPOTHESIS", "human", "system", None, {"card": C._card("positive")})
    tick(slab, None)
    hid = slab.hypotheses()[-1]["id"]
    inv = invocations.start(slab, "builder", hid)
    shutil.copy(C.CANARIES / "positive" / "strategy.py", inv.workspace / "strategy" / "strategy.py")
    stage(inv.workspace, "IMPL_DONE", "gatekeeper", hid, {"files": ["strategy/strategy.py"], "summary": "x"})
    invocations.finish(slab, inv.id)
    with pytest.raises(RuntimeError, match="canaries have not passed"):
        tick(slab, ev)
    C.record(slab, [C.Result("fake", True, "recorded for the test")])
    tick(slab, ev)
    assert slab.hypothesis(hid)["status"] == S.SKEPTIC_REVIEW


def test_two_requirements_on_one_dataset_add_up(card):
    from lab.framework.evaluator import strategy_instruments
    card = dict(card, universe={"kind": "instruments", "instruments": ["SPY", "IEF"]}, data_requirements=[
        {"dataset": "sharadar_sfp", "frequency": "1d", "instruments": ["SPY"]},
        {"dataset": "sharadar_sfp", "frequency": "1d", "instruments": ["IEF"]},
        {"dataset": "sharadar_sfp", "frequency": "1d"}])
    assert strategy_instruments(card) == {"sharadar_sfp": ["SPY", "IEF"]}
    crypto = dict(card, universe={"kind": "crypto_top_n", "n": 20},
                  data_requirements=[{"dataset": "binance_spot_1d", "frequency": "1d"}])
    assert strategy_instruments(crypto) == {"binance_spot_1d": None}


def test_series_without_dev_values_fail_g0_instead_of_showing_zero_events():
    from dataclasses import replace

    import numpy as np

    from lab.framework import data
    from lab.framework.evaluator import empty_series
    v = data.synthetic()
    T = len(v.dates)
    late = np.full(T, np.nan)
    late[T - 5:] = 1.0                                  # starts in the last rows only (like ICE BofA on FRED)
    full = replace(v, series={"fred_macro.OAS": late, "fred_macro.DGS10": np.ones(T)})
    dev = full.slice(T - 100)
    assert empty_series(dev, full) == [f"fred_macro.OAS (first valid value {v.dates[T - 5]})"]
    assert empty_series(full, full) == []


def test_warm_up_rows_are_not_scored_and_the_benchmark_starts_with_the_scored_rows():
    import numpy as np

    from lab.framework import data
    from lab.framework.evaluator import Context, Run
    v = data.synthetic()
    T = len(v.dates)
    ev = QlabEvaluator(require_canaries=False)
    ctx = Context("x", {}, v, ["S00", "S01"], "B", {"S00": 0.6, "S01": 0.4}, start_row=120, dev_end=T, data_end=T)
    w = np.full(v.shape, np.nan)
    w[10:, 0] = 1.0                                       # a strategy that decides during the warm-up rows
    w[10:, 1] = 0.0
    run = Run(w, np.zeros(T), None)
    assert ev.window(ctx, run) == slice(120, T)           # first decision row 10 would give 11
    assert ev.window(replace_start(ctx, 0), run) == slice(11, T)
    bench = ev.benchmark(ctx, T)
    assert np.allclose(bench[:119], 0.0) and np.abs(bench[125:]).sum() > 0    # nothing held before the scored rows


def replace_start(ctx, n):
    from dataclasses import replace
    return replace(ctx, start_row=n)
