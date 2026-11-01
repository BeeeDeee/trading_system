"""The ML layer: point-in-time walk-forward models, the `ml` card block and the Builder's `lab try` path."""

import copy

import numpy as np
import pytest

from lab import stubs
from lab.framework import briefs, dryrun, invocations, ml, validate
from lab.framework.api import total_return_index, trailing_return
from lab.framework.data import perturb_after, synthetic

from .conftest import submit


@pytest.fixture(scope="module")
def view():
    return synthetic().between(None, np.datetime64("2012-12-31")).columns([f"S{i:02d}" for i in range(20)])


def features(v):
    idx = total_return_index(v)
    return ml.stack(*[ml.cs_zscore(trailing_return(idx, k), v.universe) for k in (5, 21, 63)])


def scores(v, model="ridge", h=5, **kw):
    kw = {"retrain_every": 126, "min_train_days": 300} | kw
    return ml.walk_forward(features(v), ml.forward_return(v, h), horizon=h, model=model, universe=v.universe, **kw)


def test_forward_return_is_next_open_to_open_h_later(view):
    h, t = 3, 100
    y = ml.forward_return(view, h)
    j = 4
    want = 1.0
    for u in range(t + 1, t + h + 1):
        want *= (1 + view.ret_oc[u, j]) * (1 + view.ret_co[u + 1, j])
    assert y[t, j] == pytest.approx(want - 1) and np.isnan(y[-h - 1:]).all()


def test_scoring_starts_after_the_first_fit_and_is_deterministic(view):
    s = scores(view)
    first = np.flatnonzero(np.isfinite(s).any(axis=1))[0]
    assert first >= 300 and first % 126 == 0
    assert np.array_equal(s, scores(view), equal_nan=True)


@pytest.mark.parametrize("model", ["ridge", "gbm"])
def test_walk_forward_is_point_in_time(view, model):
    full = scores(view, model)
    rng = np.random.default_rng(1)
    for t in (700, 1500, 2600):
        trunc = scores(view.slice(t + 1), model)
        assert np.allclose(trunc[t], full[t], equal_nan=True, atol=1e-12)
        noisy = scores(perturb_after(view, t, rng), model)
        assert np.allclose(noisy[: t + 1], full[: t + 1], equal_nan=True, atol=1e-12)


def test_labels_not_yet_realized_are_never_trained_on(view):
    """Changing the labels of the last h rows before a refit must not change that refit's scores."""
    h, every = 10, 126
    X, y = features(view), ml.forward_return(view, h)
    base = ml.walk_forward(X, y, horizon=h, model="ridge", retrain_every=every, min_train_days=300, universe=view.universe)
    r = 126 * 8
    y2 = y.copy()
    y2[r - h:r] = np.random.default_rng(0).normal(size=y2[r - h:r].shape)       # realized only after the open of r
    alt = ml.walk_forward(X, y2, horizon=h, model="ridge", retrain_every=every, min_train_days=300, universe=view.universe)
    assert np.allclose(base[r:r + every], alt[r:r + every], equal_nan=True)
    y3 = y.copy()
    y3[r - h - 1] = -y3[r - h - 1]                                                          # the last usable label does matter
    alt = ml.walk_forward(X, y3, horizon=h, model="ridge", retrain_every=every, min_train_days=300, universe=view.universe)
    assert not np.array_equal(base[r:r + every], alt[r:r + every], equal_nan=True)


def test_unknown_hyperparameter_and_fit_budget_fail_loudly(view):
    with pytest.raises(ValueError, match="unknown hyperparameter"):
        scores(view, "gbm", params={"depth": 3})
    with pytest.raises(ValueError, match="scheduled refits"):
        scores(view, "ridge", retrain_every=20)
    with pytest.raises(ValueError, match="outside"):
        scores(view, "gbm", params={"n_estimators": 5000})


def test_memo_serves_unchanged_fits(view):
    ml._MEMO.clear()
    scores(view, "gbm")
    n = len(ml._MEMO)
    scores(view, "gbm")
    assert n > 5 and len(ml._MEMO) == n


def test_gbm_finds_an_interaction_that_ridge_cannot():
    rng = np.random.default_rng(0)
    T, N = 1400, 60
    a, b = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    fwd = 0.02 * np.sign(a * b) + rng.normal(0, 0.03, (T, N))      # the sign of the product pays
    view = synthetic().columns([f"S{i:02d}" for i in range(10)])    # only for the type; labels are passed directly
    X = ml.stack(a, b)
    y = np.full((T, N), np.nan)
    y[:-7] = fwd[:-7]

    def ic(model, **kw):
        s = ml.walk_forward(X, y, horizon=5, model=model, retrain_every=126, min_train_days=500, seed=3, **kw)
        ok = np.isfinite(s[800:1300]).all(axis=1)
        return np.mean([np.corrcoef(s[t], fwd[t])[0, 1] for t in range(800, 1300) if ok[t - 800]])
    assert ic("gbm", params={"n_estimators": 100}) > 0.15 > abs(ic("ridge"))
    assert view is not None


def test_score_weights_book():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(5, 30))
    w = ml.score_weights(s, None, top=0.2, bottom=0.2)
    assert np.allclose(np.abs(w).sum(axis=1), 1.0) and np.allclose(w.sum(axis=1), 0.0)
    assert (ml.score_weights(s, None, top=0.2) >= 0).all()
    assert (ml.score_weights(s[:, :6], None, min_names=10) == 0).all()      # too thin: flat
    s[2] = np.nan
    assert np.isnan(ml.score_weights(s, None)[2]).all()


def test_cs_helpers_ignore_non_members():
    x = np.arange(12.0).reshape(2, 6)
    uni = np.array([[1, 1, 1, 1, 0, 0]] * 2, dtype=bool)
    z, r = ml.cs_zscore(x, uni), ml.cs_rank(x, uni)
    assert np.isnan(z[:, 4:]).all() and abs(np.nansum(z[0])) < 1e-9 and r[0, 3] == 0.5 and r[0, 0] == -0.5


def test_torch_models_run_when_available(view):
    if not ml.available()["lstm"]:
        pytest.skip("no torch")
    small = view.slice(1500)
    for m in ("lstm", "tcn"):
        s = scores(small, m, retrain_every=252, params={"hidden": 4, "seq_len": 10, "epochs": 1})
        assert np.isfinite(s).sum() > 0


# ---------------------------------------------------------------- the card and `lab try`

@pytest.fixture
def ml_card():
    return copy.deepcopy(stubs.load_example("ml_stock_gbm.yaml"))


def test_example_card_is_valid(ml_card):
    assert validate.card_errors(ml_card) == [] and validate.grid_problems(ml_card) == []
    assert validate.ml_problems(ml_card) == []


def test_ml_block_survives_the_blackboard(lab, ml_card):
    hid = submit(lab, ml_card)
    assert lab.card(hid)["ml"] == ml_card["ml"]                       # CARD_ORDER once dropped it silently
    ws = invocations.start(lab, "builder", hid, task="implement").workspace
    assert "ml:" in (ws / "card.yaml").read_text()


def test_ml_card_problems(ml_card):
    bad = copy.deepcopy(ml_card)
    bad["ml"]["retrain_every_days"] = 10
    assert any("limit" in p for p in validate.ml_problems(bad))
    bad = copy.deepcopy(ml_card)
    bad["ml"]["horizon_days"] = 40
    assert any("holding_period" in p for p in validate.ml_problems(bad))
    bad = copy.deepcopy(ml_card)
    del bad["ml"]["why_not_linear"]
    assert validate.card_errors(bad)
    bad = copy.deepcopy(ml_card)
    bad["ml"]["model"] = "transformer"
    assert validate.card_errors(bad)


def test_brief_asks_for_a_learned_model_until_a_quarter_of_cards_have_one(lab, card, ml_card):
    submit(lab, dict(card, asset_classes=["crypto_spot"], holding_period={"typical_days": 0.2, "rebalance": "daily"}))
    b = briefs.choose(lab)
    assert b["horizon"] != "scalp" and b["ml"] and "LEARNED-MODEL" in briefs.text(b) and "why_not_linear" in briefs.text(b)


DIAGNOSTIC = '''
import numpy as np
from lab.framework.api import period_starts, total_return_index, trailing_return
from lab.framework.ml import cs_rank, cs_zscore, forward_return, stack, walk_forward

def events(data, params):
    uni = data.universe
    idx = total_return_index(data)
    X = stack(*[cs_zscore(trailing_return(idx, k), uni) for k in (5, 21, 63)])
    h = int(params["horizon"])
    score = walk_forward(X, forward_return(data, h), horizon=h, model="gbm",
                         params={"n_estimators": int(params["n_estimators"])}, retrain_every=126, universe=uni)
    top = cs_rank(score, uni) >= 0.5 - params["top"]
    weak = cs_rank(trailing_return(idx, 5), uni) < 0.0
    mask = top & weak & period_starts(data.dates, "W")[:, None] & data.tradable
    return mask, None
'''

STRATEGY = '''
import numpy as np
from lab.framework.api import only_on, period_starts, total_return_index, trailing_return
from lab.framework.ml import cs_zscore, forward_return, score_weights, stack, walk_forward

PARAMS = {"horizon": 5, "n_estimators": 100, "top": 0.2}

def target_weights(data, params):
    uni = data.universe
    idx = total_return_index(data)
    X = stack(*[cs_zscore(trailing_return(idx, k), uni) for k in (5, 21, 63)])
    h = int(params["horizon"])
    score = walk_forward(X, forward_return(data, h), horizon=h, model="gbm",
                         params={"n_estimators": int(params["n_estimators"])}, retrain_every=126, universe=uni)
    return only_on(period_starts(data.dates, "W"), score_weights(score, uni, top=params["top"]))
'''
TEST = '''
from lab.framework.data import synthetic
from strategy import target_weights, PARAMS
def test_shape_and_gross():
    v = synthetic().columns(["S00", "S01", "S02", "S03"])
    w = target_weights(v, PARAMS)
    assert w.shape == v.shape and (abs(w[~(w != w).all(axis=1)]).sum(axis=1) <= 1 + 1e-9).all()
'''


def test_lab_try_runs_an_ml_strategy_and_catches_a_peek(lab, ml_card):
    import time
    hid = submit(lab, ml_card)
    ws = invocations.start(lab, "builder", hid, task="implement").workspace
    (ws / "strategy" / "strategy.py").write_text(STRATEGY)
    (ws / "strategy" / "test_strategy.py").write_text(TEST)
    (ws / "strategy" / "diagnostic.py").write_text(DIAGNOSTIC)
    t0 = time.time()
    ok, lines = dryrun.run(ws)
    assert ok, "\n".join(lines)
    print("lab try seconds:", round(time.time() - t0))
    # a feature that looks one day ahead must fail G0 even though the model machinery is clean
    (ws / "strategy" / "strategy.py").write_text(STRATEGY.replace("total_return_index(data)", "np.roll(total_return_index(data), -30, axis=0)"))
    ok, lines = dryrun.run(ws)
    assert not ok and any("g0_lookahead" in x for x in lines)
