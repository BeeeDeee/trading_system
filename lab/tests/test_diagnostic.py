"""Mechanism test (event study with placebo): planted effects pass, noise rarely does, point-in-time."""

import numpy as np
import pytest
import yaml

from lab.framework import diagnostic
from lab.framework.data import perturb_after, synthetic
from lab.framework.evaluator import _integrity_events, load_events
from lab.framework.paths import LAB_DIR

TH = yaml.safe_load((LAB_DIR / "gates.yaml").read_text())["G1"]["mechanism"]
EDGE = [f"S0{i}" for i in range(5)]


@pytest.fixture(scope="module")
def view():
    return synthetic()


def planted(view):
    mask = np.zeros(view.shape, dtype=bool)
    side = np.ones(view.shape)
    on = view.series["signal"] > 0
    for a in EDGE:
        j = view.instruments.index(a)
        mask[:, j] = True
        side[:, j] = np.where(on, 1.0, -1.0)
    return mask, side


def test_forward_return_is_next_open_to_close_of_h(view):
    F, valid = diagnostic.forward_returns(view, 1)
    t, j = 100, 3
    assert np.isclose(F[t, j], view.ret_oc[t + 1, j]) and valid[t, j] and not valid[-1, j]
    F3, _ = diagnostic.forward_returns(view, 3)
    expect = (1 + view.ret_oc[t + 1, j]) * np.prod([(1 + view.ret_co[t + k, j]) * (1 + view.ret_oc[t + k, j])
                                                    for k in (2, 3)]) - 1
    assert np.isclose(F3[t, j], expect)


def test_planted_effect_is_found_and_the_side_matters(view):
    mask, side = planted(view)
    st = diagnostic.event_study(view, mask, side, 1, placebo_runs=100)
    assert st["mean_abnormal"] > 0.002 and st["placebo_percentile"] >= 0.99
    checks = diagnostic.evaluate(st, TH)
    assert all(c["pass"] for k, c in checks.items() if k != "mechanism_costs"), checks
    wrong = diagnostic.event_study(view, mask, -side, 1, placebo_runs=100)
    assert wrong["mean_abnormal"] < 0 and not diagnostic.evaluate(wrong, TH)["mechanism_abnormal_return"]["pass"]


def test_random_events_pass_at_most_a_few_times(view):
    passed = 0
    for seed in range(20):
        mask = np.random.default_rng(seed).random(view.shape) < 0.02
        st = diagnostic.event_study(view, mask, None, 1, placebo_runs=100, seed=seed)
        passed += all(c["pass"] for c in diagnostic.evaluate(st, TH).values())
    assert passed <= 3


def test_too_few_events_fail(view):
    mask = np.zeros(view.shape, dtype=bool)
    mask[1000:1010, 0] = True
    st = diagnostic.event_study(view, mask, None, 1, placebo_runs=20)
    assert not diagnostic.evaluate(st, TH)["mechanism_events"]["pass"]


def test_placebo_is_a_time_shift_not_a_reshuffle(view):
    """Events clustered in one sub-period must not look special just because that period was good."""
    mask = np.zeros(view.shape, dtype=bool)
    mask[2000:2600, 5] = True                     # one asset, one 600-day stretch: no link to the next-day return
    st = diagnostic.event_study(view, mask, None, 1, placebo_runs=150, seed=3)
    assert st["placebo_percentile"] < 0.99


def write(tmp_path, body):
    d = tmp_path / f"d{abs(hash(body))}"
    d.mkdir(exist_ok=True)
    p = d / "diagnostic.py"
    p.write_text(body)
    return p


def test_g0_for_diagnostics_catches_peeking(view, tmp_path):
    sv = view.slice(2500)
    good = write(tmp_path, "import numpy as np\n\ndef events(data, params):\n"
                           "    r = data.ret_oc\n    return r < np.roll(r, 1, axis=0) * 0.0 - 0.01, None\n")
    peek = write(tmp_path, "import numpy as np\n\ndef events(data, params):\n"
                           "    return np.roll(data.ret_oc, -1, axis=0) > 0.01, None\n")
    th = {"leakage_cut_days": 12}
    assert _integrity_events(good, sv, {}, th, {"problems": []}, 60).passed
    out = _integrity_events(peek, sv, {}, th, {"problems": []}, 60)
    assert not out.passed and out.reason_code == "g0_lookahead"


def test_load_events_validates_shape_and_side(view, tmp_path):
    sv = view.slice(300)
    bad_shape = write(tmp_path, "import numpy as np\n\ndef events(data, params):\n    return np.zeros((3, 3), bool), None\n")
    with pytest.raises(ValueError, match="shape"):
        load_events(bad_shape, sv, {})
    bad_side = write(tmp_path, "import numpy as np\n\ndef events(data, params):\n"
                               "    return np.ones(data.shape, bool), np.full(data.shape, 2.0)\n")
    with pytest.raises(ValueError, match="side"):
        load_events(bad_side, sv, {})
