"""Research 13 (Smart Zones): pivots, ranges, candidates, frozen-zone exits, random benchmark, point in time."""

import numpy as np
import pytest

from qlab.data.panel import Panel
from qlab.research5.sim import Candidates, SimSpec, simulate
from qlab.research13 import strategy as S
from qlab.research13.zones import MAX_AGE, pivots, ranges
from qlab.validation.leakage import assert_point_in_time


def make_panel(close: np.ndarray, qv: float | np.ndarray = 2e9) -> tuple[Panel, dict]:
    T, N = close.shape
    dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + np.timedelta64(T, "D"))
    opn = np.vstack([close[:1], close[:-1]])                  # open = previous close (continuous market)
    true = np.ones((T, N), dtype=bool)
    qv = np.broadcast_to(qv, (T, N)).astype(float)
    p = Panel(dates, np.arange(N, dtype=np.int64), np.zeros((T, N)), close / opn - 1, true, true, ~true,
              close, qv)
    return p, {"open": opn, "qv": qv, "high": close.copy(), "low": close.copy()}


def col(*v) -> np.ndarray:
    return np.array(v, dtype=float)[:, None]


def test_pivot_confirmed_after_k_days_and_ties():
    x = col(1, 2, 5, 3, 2, 1, 4, 4)
    ph = pivots(x, 2, "high")[:, 0]
    assert list(np.flatnonzero(ph)) == [4]                    # pivot of day 2, visible from the close of day 4
    tie = pivots(col(1, 1, 5, 5, 1, 1), 2, "high")[:, 0]       # left strict, right inclusive -> only day 2
    assert list(np.flatnonzero(tie)) == [4]
    assert list(np.flatnonzero(pivots(col(5, 4, 1, 2, 3), 2, "low")[:, 0])) == [4]


def test_gap_inside_window_means_no_pivot():
    assert not pivots(col(1, 2, 5, np.nan, 2), 2, "high").any()


PATH = [10, 11, 12, 15, 13, 12, 11, 10, 8, 9, 10, 11]


def test_range_extension_and_expiry():
    x = col(*PATH, 16, *[12] * 90)
    top, bot, valid = (a[:, 0] for a in ranges(x, x, 2))
    assert not valid[9] and valid[10]                          # pivot low of day 8 confirmed at day 10
    assert (top[10], bot[10]) == (15, 8)
    assert top[12] == 16 and bot[12] == 8                      # extension before the day-12 pivot is confirmed
    assert valid[8 + MAX_AGE] and not valid[8 + MAX_AGE + 1]   # pivot low of day 8 expires first


def test_range_min_width():
    x = col(100, 101, 102, 103, 102, 101, 100.5, 100, 99, 99.5, 100, 100.5)
    top, bot, valid = ranges(x, x, 2)
    assert (top[10, 0], bot[10, 0]) == (103, 99) and not valid[10, 0]   # 4 % < 5 %


def test_grid_24_unique():
    g = S.grid()
    assert len(g) == 24 and len({c.id for c in g}) == 24


# 60 flat days, pivot high 120 (day 51), pivot low 100 (day 55); discount zone (z = 0.25) = (100, 105]
BASE = [110] * 50 + [115, 120, 115, 110, 105, 100, 103, 106, 107, 104]


def _trade(tail: list[float], **kw) -> tuple:
    close = col(*BASE, *tail)
    p, extra = make_panel(close)
    rate = np.full(p.shape, 0.001)
    cfg = S.Config(2, 0.25, "EQ", "touch", pivot_src="close")
    univ = S.universe_mask(p, extra["qv"], cfg.universe_n)
    return S.run(p, extra, cfg, rate, 50, len(close), univ, **kw), close


def test_hand_trade_target():
    out, close = _trade([108, 111, 111, 111])
    tr = out.trades.row(0, named=True)
    assert (tr["signal_day"], tr["entry_day"], tr["exit_day"], tr["reason"]) == (59, 60, 62, "target")
    c = 0.001
    assert tr["net_ret"] == pytest.approx(111 / 104 * (1 - c) / (1 + c) - 1, rel=1e-12)
    assert out.trades.height == 1


def test_hand_trade_stop():
    out, _ = _trade([98, 98, 98])
    tr = out.trades.row(0, named=True)
    assert (tr["entry_day"], tr["exit_day"], tr["reason"]) == (60, 61, "stop")   # close 98 < 100 * 0.99
    assert tr["gross_ret"] == pytest.approx(98 / 104 - 1)


def test_stop_fill_at_level_upper_bound():
    out, _ = _trade([98, 98, 98], stop_at_level=True)
    tr = out.trades.row(0, named=True)
    assert (tr["exit_day"], tr["reason"]) == (60, "stop")                      # sold on the close of day 60
    assert tr["gross_ret"] == pytest.approx(99 / 104 - 1)                      # at the stop price 100 * 0.99
    assert out.n_positions[60 - 50] == 0


def test_ew_decisions_follow_the_mask():
    p, extra = make_panel(np.full((80, 4), 100.0), np.array([4e9, 3e9, 2e9, 1e9]))
    univ = S.universe_mask(p, extra["qv"], 2, exclude=[0])
    d = S.ew_decisions(p, univ, 0)
    assert (d.weights[d.days >= 59] == [0, 0.5, 0.5, 0]).all()


def test_no_rebuy_at_the_exit_open():
    close = col(100, 100, 100, 106, 106, 106)
    p, _ = make_panel(close)
    cands = Candidates(np.array([0, 3]), np.array([0, 0]), np.zeros(2), stop_px=np.zeros(2),
                       target_px=np.array([105.0, np.inf]))
    fee = lambda t, a: 0.0  # noqa: E731
    run = lambda re: simulate(p, cands, SimSpec(5, 60, reentry_same_open=re), np.full(p.shape, np.nan),  # noqa: E731
                              0, 6, fee, fee).trades
    assert run(False).height == 1                              # target exit at open 4, no rebuy
    assert run(True).height == 2


def _random_world(T=400, N=12, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.04, (T, N)), axis=0)
    p, extra = make_panel(close, rng.uniform(1e7, 1e9, (T, N)))
    u = rng.uniform(0, 0.03, (T, N))
    return p, extra, u


def test_random_benchmark_matches_entry_profile():
    p, extra, u = _random_world()
    extra = {**extra, "high": p.close_u * (1 + u), "low": p.close_u * (1 - u)}
    rate = np.full(p.shape, 0.001)
    cfg = S.Config(5, 0.5, "EQ", "touch")
    univ = S.universe_mask(p, extra["qv"], cfg.universe_n)
    out = S.run(p, extra, cfg, rate, 60, p.shape[0], univ)
    assert out.trades.height > 10
    m = np.bincount(out.trades["signal_day"].to_numpy(), minlength=p.shape[0])
    rnd = S.random_benchmark(p, univ, out, rate, 60, p.shape[0], seed=0)
    r = np.bincount(rnd.trades["signal_day"].to_numpy(), minlength=p.shape[0])
    assert (r <= m).all() and r.sum() > 0.5 * m.sum()           # random holds keep slots busy longer
    timed = rnd.trades.filter(rnd.trades["reason"] == "time_stop")["held"].to_numpy()
    assert set(timed) <= set(out.trades["held"].to_numpy())


def test_universe_exclusion_ranks_without_the_excluded_column():
    p, extra = make_panel(np.full((80, 4), 100.0), np.array([4e9, 3e9, 2e9, 1e9]))
    m = S.universe_mask(p, extra["qv"], 2, exclude=[0])
    assert list(np.flatnonzero(m[70])) == [1, 2]


def test_whole_pipeline_is_point_in_time():
    p, extra, u = _random_world(T=260, N=8, seed=1)
    cfg = S.Config(5, 0.5, "PREM", "bounce")

    def fn(panel: Panel) -> np.ndarray:
        T = panel.shape[0]
        c = np.asarray(panel.close_u)
        ex = {"qv": np.asarray(panel.dollar_volume), "high": c * (1 + u[:T]), "low": c * (1 - u[:T])}
        univ = S.universe_mask(panel, ex["qv"], cfg.universe_n)
        depth, bot, top = S.entry_signal(c, ex["high"], ex["low"], univ, cfg)
        nav = S.run(panel, ex, cfg, np.full(panel.shape, 0.001), 0, T, univ).nav
        return np.column_stack([np.nan_to_num(depth, nan=-1), np.nan_to_num(bot), np.nan_to_num(top), nav])

    assert_point_in_time(fn, p, "research13 pipeline", n_cuts=6, min_day=70)
