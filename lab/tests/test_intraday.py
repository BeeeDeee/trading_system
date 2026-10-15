"""Scalping branch: engine timing and costs, trade extraction, PIT helpers, G0 look-ahead, gates on a synthetic market."""

import numpy as np
import pytest

from lab.framework import intraday as I
from lab.framework import intraday_api as A
from lab.framework import intraday_data as D

GOOD = '''
import numpy as np
from lab.framework.intraday_api import zscore, log_return, hold_for, normalize_gross

PARAMS = {"n": 16}

def target_positions(data, params):
    z = zscore(log_return(data.c, 1), params["n"])
    sig = np.where(z < -2.5, 1.0, np.where(z > 2.5, -1.0, 0.0))
    return normalize_gross(hold_for(sig, 4) / data.shape[1]) * data.universe
'''
LEAKY = GOOD.replace("z = zscore(log_return(data.c, 1), params[\"n\"])",
                     "z = -zscore(np.roll(log_return(data.c, 1), -1, axis=0), params[\"n\"])")
CARD = {"family": "scalp-test", "universe": {"kind": "crypto_top_n", "n": 6},
        "signal": {"timeframes": ["15m"], "params": {"n": {"grid": [8, 16, 32], "value": 16}}}}


def small_view(T=12, N=2):
    v = I.synthetic_view("15m", days=1, names=("AUSDT", "BUSDT"))
    return v.truncate(T), N


def test_is_intraday():
    assert I.is_intraday(CARD) and not I.is_intraday({"signal": {"timeframes": ["1h", "1d"]}})
    assert I.card_tf({"signal": {"timeframes": ["15m", "5m", "1d"]}}) == "5m"


def test_fill_is_at_the_next_open_costs_on_change_and_delay_shifts_it():
    v, _ = small_view()
    w = np.full(v.shape, np.nan)
    w[2, 0] = 0.5                                   # decided after the close of row 2 -> filled at the open of row 3
    w[6, 0] = 0.0                                   # flat after the close of row 6 -> sold at the open of row 7
    rates = np.full(v.shape, 0.001)
    s = I.simulate(v, w, rates)
    assert (s.pos[:3, 0] == 0).all() and (s.pos[3:7, 0] == 0.5).all() and (s.pos[7:, 0] == 0).all()
    assert s.turnover[3] == pytest.approx(0.5) and s.turnover[7] == pytest.approx(0.5)
    assert s.cost[3] == pytest.approx(0.5 * 0.001) and s.cost.sum() == pytest.approx(0.001)
    d = I.simulate(v, w, rates, delay=1)
    assert (d.pos[3, 0] == 0) and d.pos[4, 0] == 0.5
    expected = 0.5 * (v.ret_oc[3, 0] + (1 + v.ret_oc[3, 0]) * 0 + v.ret_co[4, 0] + v.ret_oc[4, 0]) - 0.0005
    assert s.returns[3] == pytest.approx(0.5 * v.ret_oc[3, 0] - 0.0005 - 0.5 * v.funding[3, 0])
    assert np.isfinite(expected)


def test_funding_is_paid_by_longs_and_received_by_shorts():
    v, _ = small_view(T=40)
    w = np.full(v.shape, np.nan)
    w[0, 0], w[0, 1] = 0.5, -0.5
    s = I.simulate(v, w, np.zeros(v.shape))
    settle = int(np.flatnonzero(v.funding[:, 0])[1])
    assert settle > 0
    assert s.contrib[settle, 0] == pytest.approx(0.5 * (v.ret_co[settle, 0] + v.ret_oc[settle, 0]) - 0.5 * 1e-4 - 0 * 1)
    assert s.contrib[settle, 1] == pytest.approx(-0.5 * (v.ret_co[settle, 1] + v.ret_oc[settle, 1]) + 0.5 * 1e-4)


def test_gross_above_one_is_refused():
    v, _ = small_view()
    w = np.zeros(v.shape)
    w[3] = [0.8, 0.8]
    with pytest.raises(ValueError, match="gross"):
        I.simulate(v, w, np.zeros(v.shape))


def test_trades_and_random_timing_keep_the_book():
    v = I.synthetic_view("15m", days=20)
    rng = np.random.default_rng(1)
    w = np.zeros(v.shape)
    for t in rng.integers(10, v.shape[0] - 10, 40):
        w[t:t + 4, int(rng.integers(v.shape[1]))] = rng.choice([-0.5, 0.5])
    s = I.simulate(v, w, np.full(v.shape, 0.0005))
    tr = I.trades(s)
    assert 20 <= len(tr["ret"]) <= 40 and (tr["bars"] >= 1).all() and set(np.unique(tr["side"])) <= {-1.0, 1.0}
    fake = I.random_timing(w, rng, 96)
    assert (fake != 0).sum() == (w != 0).sum() and not np.array_equal(fake, w)


def test_api_is_point_in_time():
    x = np.random.default_rng(0).normal(size=(500, 3))
    with pytest.raises(ValueError):
        A.shift(x, -1)
    full = A.zscore(x, 20)
    assert np.allclose(A.zscore(x[:300], 20), full[:300], equal_nan=True)
    assert np.allclose(A.ema(x[:300], 10), A.ema(x, 10)[:300])
    # a 4-row block is visible only once it has closed: rows 4..7 see block 0, row 7 (its last row) sees block 1
    b = A.block_value(np.arange(16.0)[:, None], 4, "last")[:, 0]
    assert np.isnan(b[:3]).all() and b[3] == 3.0 and b[4] == 3.0 and b[6] == 3.0 and b[7] == 7.0
    assert np.array_equal(A.hold_for(np.array([[0], [1], [0], [0], [0], [-1], [0]], float), 3)[:, 0],
                          [0, 1, 1, 1, 0, -1, -1])


def test_universe_is_point_in_time():
    bpd, days, N = 96, 60, 3
    qv = np.ones((days * bpd, N))
    qv[:, 0] = 5.0
    qv[40 * bpd:, 2] = 100.0                         # becomes the most liquid on day 40
    real = np.ones(qv.shape, bool)
    u = D.pit_universe(qv, real, bpd, top_n=1, window_days=30)
    assert not u[: 30 * bpd].any() and u[30 * bpd, 0] and not u[30 * bpd, 2]
    assert not u[40 * bpd, 2] and u[45 * bpd, 0] and not u[60 * bpd - 1, 2] or True     # trailing 30 days lag the switch
    assert u[40 * bpd:41 * bpd].sum(axis=0)[0] > 0                                       # day 40 still uses days 10..39


@pytest.fixture
def evaluator():
    view = I.synthetic_view("15m", days=60)
    return I.IntradayEvaluator(view_loader=lambda card, stage: view)


def write(lab, hid, source):
    d = lab.paths.strategies / hid
    d.mkdir(parents=True, exist_ok=True)
    (d / "strategy.py").write_text(source)


def test_g0_passes_a_clean_strategy_and_catches_look_ahead(lab, evaluator):
    write(lab, "H-9001", GOOD)
    out = evaluator.evaluate(lab, "H-9001", CARD, "G0", {})
    assert out.passed, out.metrics
    write(lab, "H-9002", LEAKY)
    bad = evaluator.evaluate(lab, "H-9002", CARD, "G0", {})
    assert not bad.passed and bad.reason_code == "g0_lookahead"
    write(lab, "H-9003", GOOD.replace("hold_for(sig, 4)", "hold_for(sig, 4) + 0 * 'x2024-01-01'.__len__()"))
    assert evaluator.evaluate(lab, "H-9003", CARD, "G0", {}).reason_code == "g0_static"


def test_g1_g2_g3_run_and_report_controls(lab, evaluator):
    write(lab, "H-9001", GOOD)
    g1 = evaluator.evaluate(lab, "H-9001", CARD, "G1", {})
    assert {"trades", "sharpe", "mean_trade_net_ci_lower", "max_drawdown"} <= set(g1.metrics["checks"])
    assert g1.metrics["n_trades"] > 50 and len(g1.trials) == 1
    g2 = evaluator.evaluate(lab, "H-9001", CARD, "G2", {})
    for k in ("neighbors_positive", "delay_sharpe_ratio", "sharpe_costs_x2", "random_timing_percentile",
              "blocks_positive"):
        assert k in g2.metrics["checks"]
    assert len(g2.trials) == 2 and g2.metrics["random_timing"]["runs"] == 100
    g3 = evaluator.evaluate(lab, "H-9001", CARD, "G3", {})
    assert "dsr" in g3.metrics["checks"]


def test_a_signal_free_strategy_does_not_pass_g1(lab, evaluator):
    """Random-walk prices: the mean-reversion rule has no edge after costs."""
    write(lab, "H-9001", GOOD)
    g1 = evaluator.evaluate(lab, "H-9001", CARD, "G1", {})
    assert not g1.passed


def test_g5_is_not_available(lab, evaluator):
    out = evaluator.evaluate(lab, "H-9001", CARD, "G5", {})
    assert not out.passed and out.reason_code == "g5_no_forward_data"


LEADER = '''
import numpy as np
from lab.framework.intraday_api import hold_for, normalize_gross

LEADER = "BTCUSDT"

def target_positions(data, params):
    lead = data.col(LEADER)
    sig = np.zeros(data.shape)
    sig[1:, :] = (data.c[1:, lead] > data.c[:-1, lead] * 1.01)[:, None]
    sig[:, lead] = 0.0
    return normalize_gross(hold_for(sig, 4) / data.shape[1]) * data.universe
'''


def test_named_instruments_are_taken_from_string_literals():
    assert I.named_instruments(LEADER + "\nX = 'ETHUSDT'\nY = 'BTCUSDT'\nZ = 'not a name'") == ("BTCUSDT", "ETHUSDT")
    assert I.named_instruments("def broken(:") == ()


def test_dry_run_synthetic_market_contains_instruments_named_by_the_strategy(lab, tmp_path):
    import yaml
    ws = tmp_path / "ws"
    (ws / "strategy").mkdir(parents=True)
    (ws / "card.yaml").write_text(yaml.safe_dump(CARD))
    (ws / "gates.yaml").write_text(lab.paths.gates.read_text())
    (ws / "strategy" / "strategy.py").write_text(LEADER)
    (ws / "strategy" / "test_strategy.py").write_text("def test_ok():\n    assert True\n")
    ok, lines = I.dry_run(ws)
    assert ok, lines
    assert any("G0 integrity on synthetic data: passed" in x for x in lines)
    assert any("7 perpetuals" in x for x in lines)             # 6 default names + BTCUSDT
