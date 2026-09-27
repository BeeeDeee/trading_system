"""Research 5 (STR-TF): engine correctness tests required by the brief §12, plus unit tests."""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qlab.data.normalize import BANKRUPTCY, PERFORMANCE, ACQUISITION, normalize_prices
from qlab.data.panel import Panel, load_panel, panel_from_bars
from qlab.data.schema import DELISTINGS_SCHEMA, PRICES_SCHEMA
from qlab.data.synthetic import generate_market
from qlab.engine.vector import simulate as vector_simulate
from qlab.features.basic import tr_index
from qlab.research5.costs import period_cost
from qlab.research5.signals import atr_fraction, ibs, rsi, sma_gap, z_reversal
from qlab.research5.sim import Candidates, SimSpec, simulate
from qlab.research5.strategy import (Config, Context, candidates, grid_configs, neighbors,
                                     vix_gate)
from qlab.validation.leakage import assert_point_in_time, check_point_in_time

ZERO = lambda t, a: 0.0  # noqa: E731


@pytest.fixture(scope="module")
def market():
    return generate_market(n_assets=30, seed=5, start="2010-01-01", end="2012-06-30")


@pytest.fixture(scope="module")
def bars(market):
    return normalize_prices(market.prices, market.delistings, market.calendar)


@pytest.fixture(scope="module")
def panel(market, bars):
    return panel_from_bars(bars, market.calendar)


def hilo(bars: pl.DataFrame, panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    t = np.searchsorted(panel.dates, bars["date"].to_numpy().astype("datetime64[D]"))
    n = np.searchsorted(panel.assets, bars["permaticker"].to_numpy())
    hi, lo = np.full(panel.shape, np.nan), np.full(panel.shape, np.nan)
    hi[t, n], lo[t, n] = bars["high_u"].to_numpy(), bars["low_u"].to_numpy()
    return hi, lo


def forced(entries: list[tuple[int, int]], **kw) -> Candidates:
    t = np.array([e[0] for e in entries])
    return Candidates(t, np.array([e[1] for e in entries]), np.arange(len(t), dtype=float), **kw)


# --- §12.1 look-ahead ---------------------------------------------------------------------------

def test_signals_are_point_in_time(panel, bars):
    hi, lo = hilo(bars, panel)
    cut = lambda m, p: m[: p.shape[0]]  # noqa: E731
    fns = {
        "z3": lambda p: z_reversal(p, 3),
        "sma200": lambda p: sma_gap(p, 200),
        "rsi2": lambda p: rsi(p, 2),
        "ibs": lambda p: ibs(p, cut(hi, p), cut(lo, p)),
        "atr14": lambda p: atr_fraction(p, cut(hi, p), cut(lo, p)),
    }
    for name, fn in fns.items():
        assert_point_in_time(fn, panel, name)


def _pipeline_nav(p: Panel, start: int = 60) -> np.ndarray:
    """Whole STR-TF pipeline on a panel: universe, signal, candidates, simulation -> NAV."""
    z, trend, ex = z_reversal(p, 3), sma_gap(p, 50) > 0, sma_gap(p, 5)
    days = np.cumsum(p.listed, axis=0)
    with np.errstate(invalid="ignore"):
        m = (np.nan_to_num(p.close_u) >= 5) & (days >= 60) & trend & (z <= -1.0)
    m[:start] = False
    ti, ai = np.nonzero(m)
    order = np.lexsort((ai, z[ti, ai], ti))
    c = Candidates(ti[order], ai[order], z[ti, ai][order])
    cost = lambda t, a: 0.001  # noqa: E731
    out = simulate(p, c, SimSpec(5, 5), np.full(p.shape, np.inf), start, p.shape[0], cost, cost,
                   ex)
    nav = np.full((p.shape[0], 1), np.nan)
    nav[start:, 0] = out.nav
    return nav


def test_whole_pipeline_is_point_in_time(panel):
    """Perturbing data after D must not change signals, positions or equity up to D."""
    report = check_point_in_time(_pipeline_nav, panel, "str_tf", n_cuts=8, min_day=61)
    assert report.ok, report
    assert np.nanstd(_pipeline_nav(panel)) > 0  # the pipeline actually trades


# --- §12.2 survivorship (real data) -------------------------------------------------------------

PANEL_R5 = Path("data/derived/sharadar_2026-09-25/panel_r5")


@pytest.mark.skipif(not PANEL_R5.exists(), reason="research 5 panel not built")
def test_universe_2000_contains_later_delisted_names():
    p, extra = load_panel(PANEL_R5)
    t = int(np.searchsorted(p.dates, np.datetime64("2000-01-03")))
    uni = np.asarray(extra["base_ok"][t]) & (np.asarray(extra["adv20"][t]) >= 20e6)
    later_delisted = np.asarray(p.delisting[t:]).any(axis=0)
    n, n_del = int(uni.sum()), int((uni & later_delisted).sum())
    print(f"universe 2000-01-03 (ADV20 >= 20M): {n} names, later delisted: {n_del}")
    assert n > 300 and n_del > 0.3 * n


# --- §12.3 split, §12.4 delisting ---------------------------------------------------------------

def _prices(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=PRICES_SCHEMA, orient="row")


def _calendar(n: int) -> list[date]:
    d = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-03-31"))
    return [x.astype(date) for x in d[np.is_busday(d)][:n]]


def test_split_in_held_position_does_not_jump_equity():
    cal = _calendar(30)
    # 2:1 split effective on day 15: unadjusted price halves, adjusted series is continuous.
    rows = [(1, d, 50.0, 50.0, 50.0, 50.0, 1e6, 50.0, 100.0 if i < 15 else 50.0)
            for i, d in enumerate(cal)]
    p = panel_from_bars(normalize_prices(_prices(rows), pl.DataFrame(schema=DELISTINGS_SCHEMA), cal),
                        cal)
    assert p.close_u[14, 0] == 100.0 and p.close_u[15, 0] == 50.0
    out = simulate(p, forced([(10, 0)]), SimSpec(1, 20), np.full(p.shape, np.inf), 0, 30,
                   ZERO, ZERO)
    assert out.n_positions[15] == 1
    np.testing.assert_allclose(out.nav, 1.0, atol=1e-12)


@pytest.mark.parametrize("kind,consideration,expected", [
    (BANKRUPTCY, None, -1.0), (PERFORMANCE, None, -0.30), (ACQUISITION, 12.0, 0.20)])
def test_delisting_in_held_position(kind, consideration, expected):
    cal = _calendar(30)
    rows = [(1, d, 10.0, 10.0, 10.0, 10.0, 1e6, 10.0, 10.0) for d in cal[:20]]
    dl = pl.DataFrame([(1, cal[19], kind, consideration)], schema=DELISTINGS_SCHEMA, orient="row")
    p = panel_from_bars(normalize_prices(_prices(rows), dl, cal), cal)
    out = simulate(p, forced([(5, 0)]), SimSpec(1, 50), np.full(p.shape, np.inf), 0, 30, ZERO, ZERO)
    assert out.trades["reason"].to_list() == ["delisted"]
    assert out.trades["gross_ret"][0] == pytest.approx(expected)
    assert out.nav[-1] == pytest.approx(1.0 + expected)


# --- §12.5 null strategy, §12.6 reproducibility -------------------------------------------------

def _random_candidates(panel: Panel, seed: int, per_day: int = 3) -> Candidates:
    rng = np.random.default_rng(seed)
    t, a = [], []
    for d in range(1, panel.shape[0] - 1):
        live = np.flatnonzero(panel.listed[d] & ~panel.delisting[d])
        pick = rng.choice(live, size=min(per_day, len(live)), replace=False)
        t += [d] * len(pick)
        a += pick.tolist()
    return forced(list(zip(t, a)))


def test_null_strategy_earns_the_universe_return(panel):
    """Random entries, zero costs: every trade earns exactly the asset's open-to-open total
    return, and on average what the universe earns over the same horizon."""
    c = _random_candidates(panel, seed=1)
    out = simulate(panel, c, SimSpec(10_000, 5), np.full(panel.shape, np.inf), 0, panel.shape[0],
                   ZERO, ZERO)
    idx = tr_index(panel)
    idx_open = np.vstack([np.ones((1, panel.shape[1])), idx[:-1]]) * (1.0 + panel.ret_co)
    tr = out.trades.filter(pl.col("reason") != "end")
    a, e, x = (tr[c].to_numpy() for c in ("asset", "entry_day", "exit_day"))
    np.testing.assert_allclose(tr["gross_ret"].to_numpy(), idx_open[x, a] / idx_open[e, a] - 1,
                               rtol=1e-10)
    # Universe: every listed asset held for the same 5 closes (open t+1 -> open t+6).
    h = 5
    fwd = idx_open[h:] / idx_open[:-h] - 1.0
    live = panel.listed[:-h] & panel.listed[h:] & panel.tradable[:-h]
    uni = fwd[live]
    uni = uni[np.isfinite(uni)]
    sample = tr.filter(pl.col("reason") == "time_stop")["gross_ret"].to_numpy()
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    assert abs(sample.mean() - uni.mean()) < 4 * se


def test_reproducible(panel):
    runs = [simulate(panel, _random_candidates(panel, seed=7), SimSpec(5, 5),
                     np.full(panel.shape, np.inf), 0, panel.shape[0], ZERO, ZERO) for _ in range(2)]
    np.testing.assert_array_equal(runs[0].nav, runs[1].nav)
    assert runs[0].trades.equals(runs[1].trades)


# --- simulator unit tests -----------------------------------------------------------------------

def test_single_trade_matches_vector_engine(panel):
    a = int(np.flatnonzero(panel.listed[0] & ~panel.delisting.any(axis=0))[0])
    s, k, c = 40, 7, 0.002
    out = simulate(panel, forced([(s, a)]), SimSpec(1, k), np.full(panel.shape, np.inf), 0,
                   panel.shape[0], lambda t, x: c, lambda t, x: c)
    targets = np.full(panel.shape, np.nan)
    targets[s] = 0.0
    targets[s, a] = 1.0
    targets[s + k] = 0.0
    ref = vector_simulate(panel, targets, c)
    assert out.trades["exit_day"][0] == s + k + 1
    np.testing.assert_allclose(out.nav, ref.nav, rtol=1e-12)


def test_capacity_cap_and_slots(panel):
    live = np.flatnonzero(panel.listed[20] & panel.tradable[21])[:4]
    adv = np.full(panel.shape, np.inf)
    adv[20, live[0]] = 1e6  # 1 % of 1M ADV = 10k USD = 0.01 of 1M capital
    out = simulate(panel, forced([(20, a) for a in live]), SimSpec(3, 5, capital=1e6), adv, 0,
                   40, ZERO, ZERO)
    tr = out.trades.sort("score")
    assert tr.height == 3  # only three slots
    assert tr["entry_value"][0] == pytest.approx(0.01)
    assert tr["entry_value"][1] == pytest.approx(1 / 3)


def test_moc_and_delayed_entry_timing(panel):
    a = int(np.flatnonzero(panel.listed[0] & ~panel.delisting.any(axis=0))[0])
    for delay, entry in ((0, 30), (1, 31), (2, 32)):
        out = simulate(panel, forced([(30, a)]), SimSpec(1, 3, entry_delay=delay),
                       np.full(panel.shape, np.inf), 0, 60, ZERO, ZERO)
        assert out.trades["entry_day"][0] == entry
    idx = tr_index(panel)
    moc = simulate(panel, forced([(30, a)]), SimSpec(1, 3, entry_delay=0),
                   np.full(panel.shape, np.inf), 0, 60, ZERO, ZERO)
    assert moc.trades["exit_day"][0] == 34  # three closes held after the close of 30
    exit_open = idx[33, a] * (1.0 + panel.ret_co[34, a])
    assert moc.trades["gross_ret"][0] == pytest.approx(exit_open / idx[30, a] - 1.0)


def test_period_cost():
    d = np.array(["2000-06-01", "2005-06-01", "2010-06-01"], dtype="datetime64[D]")
    np.testing.assert_allclose(period_cost(d) * 1e4, [25 * 1.5 + 5, 10 * 1.5 + 1, 5 * 1.5 + 0.5])
    np.testing.assert_allclose(period_cost(d, at_open=False, multiplier=2) * 1e4, [60, 22, 11])


def test_grid_and_neighbors():
    grid = grid_configs()
    assert len(grid) == 972 and len(set(grid)) == 972  # 3*3*2*3*3*2*3
    nb = neighbors(Config())
    assert len(nb) == 12  # middle values have two neighbours, trend_sma=200 and max_positions=10 one
    assert all(sum(getattr(n, k) != getattr(Config(), k) for k in
                   ("entry_z", "lookback_ret", "trend_sma", "exit_sma", "max_hold",
                    "max_positions", "min_adv")) == 1 for n in nb)


def test_candidates_ranked_filtered_and_earnings_excluded(panel, bars, tmp_path):
    hi, lo = hilo(bars, panel)
    days = np.cumsum(panel.listed, axis=0)
    earn = np.zeros(panel.shape, bool)
    extra = {"base_ok": (np.nan_to_num(panel.close_u) >= 5) & (days >= 60),
             "adv20": np.full(panel.shape, 1e9), "high_u": hi, "low_u": lo, "earn8k": earn}
    ctx = Context(panel, extra, 250, tmp_path, 0)
    cfg = Config(trend_sma=100, entry_z=1.0)
    c = candidates(ctx, cfg)
    assert len(c.t) > 20 and (c.t >= 250).all()
    z = z_reversal(panel, 3)
    assert (z[c.t, c.a] <= -1.0).all() and (sma_gap(panel, 100)[c.t, c.a] > 0).all()
    same_day = c.t[1:] == c.t[:-1]
    assert (c.score[1:][same_day] >= c.score[:-1][same_day]).all()
    earn[c.t[0] - 2, c.a[0]] = True  # filed two days before the first signal
    ctx2 = Context(panel, extra, 250, tmp_path, 0)
    c2 = candidates(ctx2, replace_cfg(cfg, exclude_earnings=True))
    assert len(c2.t) == len(c.t) - 1


def replace_cfg(cfg, **kw):
    from dataclasses import replace
    return replace(cfg, **kw)


def test_vix_gate_is_point_in_time():
    rng = np.random.default_rng(0)
    vix = 15 + 10 * np.abs(rng.standard_normal(600))
    vix[100] = np.nan
    for rule in ("abs25", "rel80"):
        full = vix_gate(vix, rule)
        assert not full[100]
        for t in (260, 400, 598):
            np.testing.assert_array_equal(vix_gate(vix[: t + 1], rule), full[: t + 1])
            noisy = vix.copy()
            noisy[t + 1:] = rng.uniform(0, 80, len(vix) - t - 1)
            np.testing.assert_array_equal(vix_gate(noisy, rule)[: t + 1], full[: t + 1])
    np.testing.assert_array_equal(vix_gate(vix, "abs25"), np.nan_to_num(vix) > 25)
