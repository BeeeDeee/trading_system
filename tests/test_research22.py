import numpy as np
import pytest
from scipy.stats import rankdata

from qlab.data.panel import Panel
from qlab.engine.costs import CostModel
from qlab.engine.vector import Decisions, simulate
from qlab.research22.compact import (LEAF_NAMES, STATE_NAMES, Compact, build_compact, daily_leaves,
                                     decision_schedule, market_states)
from qlab.research22.fitness import evaluate_tree, jitter_tree
from qlab.research22.grammar import Evaluator, GrammarError, node_type, tree_depth, tree_size, validate
from qlab.research22.kernels import portfolio, rank_rows
from qlab.research22.nulls import null_dataset, permute_forward, run_null_ga
from qlab.research22.search import Searcher, objectives, run_ga, run_random
from qlab.research22.trees import crossover, kind_of, mutate, normal_form, random_tree, simplify
from qlab.research22 import selection
from qlab.research22.dna import COMPONENTS, components_of, design, enrichment, frequency_table, replicated
from qlab.research22.selection import (break_even_cost, ga_candidates, gate_g1, gate_g5, ledger_candidates,
                                       representatives, select_run, streams)
from qlab.research22.stats22 import (MultiTest, by_fdr, by_qvalues, cluster_returns, min_track_record_length,
                                     multiple_testing, unique_columns)
from sklearn.metrics import adjusted_rand_score
import importlib.util
import json
from datetime import date
from pathlib import Path
from qlab.research22 import setup as r22_setup
from qlab.validation.vault import Vault, VaultError
from qlab.research22.store import load_ga_result, pairs_to_rows, save_result
from qlab.research22.late import (CANONICAL, ensemble_series, frozen_from_json, frozen_to_json, g6_decision,
                                  hash_files, late_report, matched_random, perf, prepare_frozen, rel_drawdown,
                                  worst_fold_rel_dd)
from qlab.schedule import period_starts
from qlab.validation.leakage import check_point_in_time, perturb_after

ZERO_COST = CostModel(floor_bps=0.0, tiers=((10 ** 9, 0.0),), default_bps=0.0, pre_decimal_multiplier=1.0)
TIER_COST = CostModel(floor_bps=5.0, tiers=((8, 3.0), (16, 6.0), (24, 12.0)), default_bps=25.0)
TOPK = 5


def make_market(n_days=520, n_assets=24, seed=0, zero_oc=False, holes=False):
    """Synthetic panel with two delistings (-100 % and -30 %), a churning top-18 universe and liquidity ranks."""
    rng = np.random.default_rng(seed)
    days = np.arange(np.datetime64("2019-01-07"), np.datetime64("2019-01-07") + np.timedelta64(int(n_days * 1.5), "D"))
    dates = days[((days.astype(np.int64) + 3) % 7) < 5][:n_days]
    shape = (n_days, n_assets)
    ret_co = rng.normal(0.0002, 0.008, shape)
    ret_oc = np.zeros(shape) if zero_oc else rng.normal(0.0002, 0.010, shape)
    tradable = np.ones(shape, bool)
    if holes:                                  # days without a usable open: positions can get stuck
        tradable &= rng.random(shape) > 0.04
    listed = np.ones(shape, bool)
    delisting = np.zeros(shape, bool)
    score = rng.normal(size=n_assets) + 0.5 * rng.normal(size=shape)
    order = np.argsort(-score, axis=1)
    liq_rank = np.empty(shape, np.float32)
    np.put_along_axis(liq_rank, order, np.arange(1, n_assets + 1, dtype=np.float32)[None, :].repeat(n_days, 0), axis=1)
    in_liq = liq_rank <= 18
    for asset, row, terminal in ((3, 300, -1.0), (7, 420, -0.3)):
        ret_co[row, asset] = terminal
        ret_co[row + 1:, asset] = 0.0
        ret_oc[row:, asset] = 0.0
        delisting[row, asset] = True
        listed[row + 1:, asset] = False
        tradable[row:, asset] = False
        in_liq[row - 8:row, asset] = True      # held / eligible in the weeks before the delisting
        in_liq[row:, asset] = False
    panel = Panel(dates, np.arange(n_assets), ret_co, ret_oc, tradable, listed, delisting,
                  np.full(shape, 10.0), rng.lognormal(15, 0.5, shape))
    return panel, {"in_liq1000": in_liq, "liq_rank": liq_rank}


@pytest.fixture(scope="module")
def market():
    return make_market()


@pytest.fixture(scope="module")
def compact(market):
    panel, extra = market
    return build_compact(panel, extra, 0, panel.shape[0] - 1, 0, TIER_COST, block=10)


# ------------------------------------------------------------------ kernels
def test_rank_rows_average_ties_and_nan():
    rng = np.random.default_rng(1)
    x = rng.integers(-3, 4, (50, 40)).astype(np.float32)
    x[:, ::5] = 0.0
    x[:, 2::9] = -0.0
    x[rng.random(x.shape) < 0.1] = np.nan
    out = rank_rows(x)
    for i in range(x.shape[0]):
        ok = np.isfinite(x[i])
        expect = (rankdata(x[i][ok], method="average") - 1) / ok.sum()
        np.testing.assert_allclose(out[i][ok], expect, atol=1e-6)
        assert np.isnan(out[i][~ok]).all()


def test_topk_tie_goes_to_lowest_column():
    score = np.array([[5, 5, 5, 5, 1, 1]], dtype=np.float32)
    wk = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    idx = np.arange(6, dtype=np.int64)[None, :]
    net, gross, turn, npos, nav = portfolio(score, wk, np.zeros((1, 6), bool), idx, np.zeros((1, 6)),
                                            np.ones((1, 6), bool), 1.0, 3)
    assert gross[0] == pytest.approx(1.0)       # columns 0, 1, 2
    assert npos[0] == 3 and turn[0] == pytest.approx(1.0) and nav[0] == pytest.approx(1.0)


def test_portfolio_rebalances_from_drifted_weights_and_scales_buys_to_cash():
    # week 1 holds columns 0 and 1 at no cost; the weights drift to 0.75 / 0.25 and week 2 rebalances them at 1 %.
    score = np.array([[2, 1, 0], [2, 1, 0]], dtype=np.float32)
    wk = np.array([[0.5, -0.5, 0.0], [0.0, 0.0, 0.0]])
    idx = np.tile(np.arange(3, dtype=np.int64), (2, 1))
    rate = np.array([[0.0] * 3, [0.01] * 3])
    net, gross, turn, npos, nav = portfolio(score, wk, np.zeros((2, 3), bool), idx, rate, np.ones((2, 3), bool), 1.0, 2)
    assert gross[0] == pytest.approx(0.0) and net[0] == pytest.approx(0.0) and turn[0] == pytest.approx(1.0)
    # sells 0.25 (cost 0.0025) -> cash 0.2475; the buy of 0.25 needs 0.2525, so it is scaled by 0.2475 / 0.2525
    assert turn[1] == pytest.approx(0.25 + 0.25 * 0.2475 / 0.2525) and nav[1] == pytest.approx(1 - 0.0025 - 0.0024504950495)
    assert npos.tolist() == [2, 2]


def test_untradable_position_is_held_blocks_the_buy_and_earns_its_return():
    # week 1 holds columns 1 and 2 (0.5 each, no costs); week 2 targets columns 0 and 2, so column 1 should be sold
    score = np.array([[1, 3, 2], [3, 1, 2]], dtype=np.float32)
    wk = np.array([[0.0, 0.0, 0.0], [0.2, 1.0, 0.0]])
    idx = np.tile(np.arange(3, dtype=np.int64), (2, 1))
    args = (score, wk, np.zeros((2, 3), bool), idx, np.zeros((2, 3)))
    free = portfolio(*args, np.ones((2, 3), bool), 1.0, 2)
    stuck = portfolio(*args, np.array([[True] * 3, [True, False, True]]), 1.0, 2)
    # tradable: column 1 is sold, column 0 bought: 0.5 * 1.2 + 0.5 * 1.0 - 1
    assert free[0][1] == pytest.approx(0.1) and free[3].tolist() == [2, 2]
    # stuck: no cash to buy column 0, column 1 keeps its +100 %: 0.5 * 2 + 0.5 * 1 - 1
    assert stuck[0][1] == pytest.approx(0.5) and stuck[2][1] == pytest.approx(0.0) and stuck[3].tolist() == [2, 2]


# ------------------------------------------------------------------ point in time
def test_leaves_are_point_in_time(market):
    panel, _ = market
    small = Panel(panel.dates[:330], panel.assets[:12], *(getattr(panel, f)[:330, :12] for f in (
        "ret_co", "ret_oc", "tradable", "listed", "delisting", "close_u", "dollar_volume")))

    def fn(p: Panel) -> np.ndarray:
        leaves = daily_leaves(p.ret_co, p.ret_oc, p.dollar_volume)
        return np.concatenate([leaves[name] for name in LEAF_NAMES], axis=1)

    report = check_point_in_time(fn, small, "research22 leaves", n_cuts=8, min_day=1, tol=1e-6)
    assert report.ok, report


def test_market_states_are_point_in_time():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0003, 0.01, 700)
    trend, high = market_states(r)
    for t in (150, 199, 230, 268, 269, 300, 450, 698):
        tr, hi = market_states(r[: t + 1])
        assert tr[t] == trend[t] and hi[t] == high[t]
    assert not high[:269].any() and not trend[:199].any()


def test_whole_compact_is_point_in_time(market):
    panel, extra = market
    last = 400
    base = build_compact(panel, extra, 0, last, 0, TIER_COST, block=10)
    rng = np.random.default_rng(5)
    noisy = perturb_after(panel, last, rng)
    extra2 = {k: v.copy() for k, v in extra.items()}
    extra2["in_liq1000"][last + 1:] = rng.random(extra2["in_liq1000"][last + 1:].shape) < 0.5
    extra2["liq_rank"][last + 1:] = rng.random(extra2["liq_rank"][last + 1:].shape) * 30
    other = build_compact(noisy, extra2, 0, last, 0, TIER_COST, block=10)
    for name in ("dec_rows", "idx", "wk", "dead", "rate"):
        np.testing.assert_array_equal(getattr(base, name), getattr(other, name), err_msg=name)
    for name in LEAF_NAMES:
        np.testing.assert_array_equal(base.leaves[name], other.leaves[name], err_msg=name)
    for name in STATE_NAMES:
        np.testing.assert_array_equal(base.states[name], other.states[name], err_msg=name)
    assert base.dec_rows[-1] + 1 <= last


def test_schedule_first_trading_day_of_week_and_embargo(market):
    panel, _ = market
    dec, nxt = decision_schedule(panel.dates, 0, panel.shape[0] - 1, 0)
    assert period_starts(panel.dates, "W")[dec].all() and (nxt[:-1] == dec[1:]).all()
    assert nxt[-1] + 1 <= panel.shape[0] - 1
    dec2, _ = decision_schedule(panel.dates, 300, panel.shape[0] - 1, 21)
    assert dec2.min() >= 321


# ------------------------------------------------------------------ data semantics
def test_holding_return_is_exact_product(market, compact):
    panel, _ = market
    c = compact
    for i in (0, 7, 20):
        f1, g = c.dec_rows[i] + 1, c.dec_rows[i + 1]
        for col in range(c.shape[1]):
            if c.idx[i, col] < 0:
                continue
            a = c.idx[i, col]
            p = 1.0 + panel.ret_oc[f1, a]
            for d in range(f1 + 1, g + 1):
                p *= (1.0 + panel.ret_co[d, a]) * (1.0 + panel.ret_oc[d, a])
            p *= 1.0 + panel.ret_co[g + 1, a]
            assert c.wk[i, col] == pytest.approx(p - 1.0, abs=1e-12)


def test_eligibility_needs_universe_and_tradable_open(market, compact):
    panel, extra = market
    c = compact
    for i in range(c.shape[0]):
        ids = c.idx[i][c.valid[i]]
        assert (np.diff(ids) > 0).all()                       # ascending asset order (tie rule)
        want = np.flatnonzero(extra["in_liq1000"][c.dec_rows[i]] & panel.tradable[c.dec_rows[i] + 1])
        np.testing.assert_array_equal(ids, want)


def test_delisted_names_are_flagged_dead(market, compact):
    c = compact
    hit = [(i, j) for i in range(c.shape[0]) for j in range(c.shape[1]) if c.valid[i, j] and c.dead[i, j]]
    assert hit, "the synthetic market has delistings inside the sample"
    for i, j in hit:
        a = c.idx[i, j]
        assert a in (3, 7)


# ------------------------------------------------------------------ parity with engine.vector
def _select(c: Compact, score: np.ndarray, topk: int) -> np.ndarray:
    """Independent top-k (ties: lowest column) -> (F, N_assets) equal weights."""
    n_assets = c.n_assets
    w = np.zeros((c.shape[0], n_assets))
    for i in range(c.shape[0]):
        s = np.where(c.valid[i] & np.isfinite(score[i]), score[i], -np.inf)
        k = min(topk, int(np.isfinite(s).sum()))
        sel = np.argsort(-s, kind="stable")[:k]
        w[i, c.idx[i, sel]] = 1.0 / k
    return w


@pytest.mark.parametrize("cost_model", [ZERO_COST, TIER_COST])
@pytest.mark.parametrize("holes", [False, True])
def test_weekly_portfolio_matches_vector_engine(cost_model, holes):
    # Intraday returns are zero so that the engine NAV at a close equals the NAV after the open trade.
    panel, extra = make_market(zero_oc=True, holes=holes)
    c = build_compact(panel, extra, 0, panel.shape[0] - 1, 0, cost_model, block=10)
    ev = Evaluator(c)
    score = ev.score(("leaf", "ret_20"))
    weekly = c.weekly(score, TOPK, 1.0)

    targets = Decisions(c.dec_rows, _select(c, score, TOPK))
    cost_rate = np.zeros(panel.shape)
    cost_rate[c.dec_rows + 1] = c.rate
    res = simulate(panel, targets, cost_rate)
    np.testing.assert_allclose(weekly.nav_trade, res.nav[c.dec_rows + 1], rtol=1e-10, atol=0)
    np.testing.assert_allclose(weekly.turnover, res.turnover[c.dec_rows + 1], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(weekly.n_positions, res.n_positions[c.dec_rows + 1])
    if holes:
        assert (weekly.n_positions > TOPK).any()                  # the test really exercises stuck positions
    # the weekly net return chains the NAVs: nav_open(i + 1) = nav_open(i) * (1 + net(i))
    nav_open = np.concatenate(([1.0], np.cumprod(1.0 + weekly.net)))
    cost_frac = 1.0 - weekly.nav_trade / nav_open[:-1]
    np.testing.assert_allclose(weekly.nav_trade, nav_open[:-1] * (1.0 - cost_frac), rtol=1e-12)


def test_benchmark_is_equal_weight_of_eligible(market):
    panel, extra = market
    c = build_compact(panel, extra, 0, panel.shape[0] - 1, 0, ZERO_COST, block=10)
    bench = c.benchmark(1.0)
    n_valid = c.valid.sum(axis=1)
    mean_wk = np.where(c.valid, np.nan_to_num(c.wk), 0.0).sum(axis=1) / n_valid
    assert (bench.n_positions == n_valid).all()
    np.testing.assert_allclose(bench.gross, mean_wk, atol=1e-12)


# ------------------------------------------------------------------ grammar and evaluator
def test_grammar_types_and_limits():
    ok = ("where", ("gt", ("leaf", "ret_20"), ("const", 0.5)), ("leaf", "vol_20"), ("neg", ("leaf", "ret_5")))
    assert node_type(ok) == "float" and tree_size(ok) == 7 and tree_depth(ok) == 3
    validate(ok, 5, 15)
    with pytest.raises(GrammarError):
        node_type(("add", ("leaf", "ret_20"), ("state", "mkt_trend")))      # bool where float expected
    with pytest.raises(GrammarError):
        node_type(("leaf", "no_such_terminal"))
    with pytest.raises(GrammarError):
        node_type(("const", 3.0))
    with pytest.raises(GrammarError):
        validate(("gt", ("leaf", "ret_20"), ("leaf", "ret_5")), 5, 15)       # a score must be float
    deep = ("leaf", "ret_5")
    for _ in range(6):
        deep = ("neg", deep)
    with pytest.raises(GrammarError):
        validate(deep, 5, 15)


def test_evaluator_cache_does_not_change_results(compact):
    trees = [("rank", ("sub", ("leaf", "ret_60"), ("leaf", "ret_5"))),
             ("add", ("rank", ("sub", ("leaf", "ret_60"), ("leaf", "ret_5"))), ("rank", ("leaf", "vol_20"))),
             ("where", ("state", "mkt_trend"), ("rank", ("leaf", "ret_20")), ("neg", ("rank", ("leaf", "ret_5")))),
             ("div", ("leaf", "ret_20"), ("sub", ("leaf", "vol_20"), ("leaf", "vol_20"))),
             ("rank", ("const", 1.0))]
    cached, fresh = Evaluator(compact), Evaluator(compact, cache_bytes=0)
    for t in trees * 2:
        np.testing.assert_array_equal(cached.score(t), fresh.score(t))
    assert cached.hits > 0
    div0 = cached.score(trees[3])
    assert np.all(div0[compact.valid] == 0.0)                  # protected division
    assert np.isnan(div0[~compact.valid]).all()                # padding is never selectable


# ------------------------------------------------------------------ fitness
def test_jitter_is_deterministic_and_stays_in_the_grammar():
    tree = ("add", ("leaf", "ret_20"), ("mul", ("leaf", "z20_vol_20"), ("const", 1.0)))
    a = jitter_tree(tree, np.random.default_rng([7, 0]))
    b = jitter_tree(tree, np.random.default_rng([7, 0]))
    assert a == b
    names = {n[1] for n in (a[1], a[2][1]) if n[0] == "leaf"}
    assert names <= set(LEAF_NAMES)


def test_fitness_is_finite_deterministic_and_flags_degenerate_scores(compact):
    ev = Evaluator(compact)
    tree = ("sub", ("leaf", "ret_60"), ("leaf", "ret_5"))
    f1 = evaluate_tree(ev, tree, seed=1, topk=TOPK)
    f2 = evaluate_tree(Evaluator(compact), tree, seed=1, topk=TOPK)
    assert f1 == f2 and f1.feasible and np.isfinite(f1.q25) and len(f1.fold_irs) == 8
    assert f1.turnover_year > 0
    flat = evaluate_tree(ev, ("rank", ("const", 1.0)), seed=1, topk=TOPK)
    assert not flat.feasible and flat.dispersion == 0.0


# ------------------------------------------------------------------ trees
def test_random_trees_respect_types_and_limits():
    rng = np.random.default_rng(0)
    ops = set()
    for _ in range(600):
        t = random_tree(rng)
        validate(t, 5, 15)
        ops.add(t[0])
        assert kind_of(random_tree(rng, "bool", 3, 7)) == "bool"
    assert {"where", "rank", "add"} <= ops


def test_variation_keeps_trees_valid():
    rng = np.random.default_rng(1)
    changed = 0
    for _ in range(400):
        a, b = random_tree(rng), random_tree(rng)
        for child in (crossover(rng, a, b), crossover(rng, b, a), mutate(rng, a), mutate(rng, b)):
            validate(child, 5, 15)
        changed += crossover(rng, a, b) != a
    assert changed > 100


def test_normal_form_rules():
    x, y = ("leaf", "ret_5"), ("leaf", "vol_20")
    one = ("const", 1.0)
    assert simplify(("neg", ("neg", x))) == x
    assert simplify(("rank", ("rank", x))) == ("rank", x)
    assert simplify(("where", ("state", "mkt_trend"), x, x)) == x
    assert simplify(("mul", x, one)) == x and simplify(("div", x, one)) == x
    assert normal_form(("add", y, x)) == normal_form(("add", x, y))
    assert normal_form(("mul", ("add", y, x), one)) == normal_form(("add", x, y))
    assert normal_form(("sub", y, x)) != normal_form(("sub", x, y))          # not commutative
    t = ("where", ("gt", ("neg", ("neg", x)), y), ("mul", one, y), ("rank", ("rank", x)))
    assert normal_form(normal_form(t)) == normal_form(t)


def test_normal_form_preserves_the_score(compact):
    rng = np.random.default_rng(2)
    ev = Evaluator(compact)
    for _ in range(60):
        t = random_tree(rng)
        np.testing.assert_allclose(ev.score(normal_form(t)), ev.score(t), atol=1e-6, equal_nan=True)


# ------------------------------------------------------------------ search
def _run(compact, seed=3, pop=20, gens=4):
    s = Searcher(Evaluator(compact), topk=TOPK)
    return run_ga(s, seed=seed, pop_size=pop, n_gen=gens), s


def test_ga_is_deterministic_valid_and_elitist(compact):
    r1, s1 = _run(compact)
    r2, _ = _run(compact)
    assert r1.population == r2.population and r1.ledger.rows == r2.ledger.rows
    assert len(r1.population) == 20 and len(r1.history) == 5
    for tree, fit in r1.population:
        validate(tree, 5, 15)
        assert tree == normal_form(tree) and fit == s1.ledger.rows[tree]
    best = [h["best_q25"] for h in r1.history if h["best_q25"] is not None]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(best, best[1:]))           # NSGA-II keeps the best of f1
    assert r1.n_backtests == 4 * len(r1.ledger)
    assert r1.front and all(f.feasible for _, f in r1.front)
    for _, f in r1.front:                                                     # first front is non-dominated
        assert not any(o[1].q25 > f.q25 and o[1].turnover_year < f.turnover_year and o[1].nodes < f.nodes
                       for o in r1.front)


def test_random_search_counts_distinct_normal_forms(compact):
    s = Searcher(Evaluator(compact), topk=TOPK)
    ledger = run_random(s, seed=1, n=40)
    assert len(ledger) == 40 and all(t == normal_form(t) for t in ledger.rows)
    s2 = Searcher(Evaluator(compact), topk=TOPK)
    assert run_random(s2, seed=1, n=40).rows == ledger.rows
    assert objectives(next(iter(ledger.rows.values()))) is not None


# ------------------------------------------------------------------ nulls
def test_permutation_null_keeps_structure_and_destroys_skill(compact):
    c = compact
    nul = permute_forward(c, np.random.default_rng(0))
    assert nul.leaves is c.leaves and nul.idx is c.idx and nul.rate is c.rate
    for i in range(c.shape[0]):
        n = int(c.valid[i].sum())
        np.testing.assert_array_equal(np.sort(nul.wk[i, :n]), np.sort(c.wk[i, :n]))
        assert nul.dead[i].sum() >= c.dead[i].sum()
    assert not np.array_equal(nul.wk, c.wk, equal_nan=True)
    assert nul.dead[c.dead].all()                                              # a delisted name is always paid out
    assert nul.dead[nul.wk < -0.25].all()                                      # terminal returns carry their payout
    assert null_dataset(c, 3).wk.tobytes() == null_dataset(c, 3).wk.tobytes()
    assert null_dataset(c, 3).wk.tobytes() != null_dataset(c, 4).wk.tobytes()
    oracle = np.where(c.valid, np.nan_to_num(c.wk), np.nan).astype(np.float32)   # perfect foresight on the real data
    real = c.weekly(oracle, TOPK, 1.0).gross.mean()
    fake = nul.weekly(oracle, TOPK, 1.0).gross.mean()
    assert real > fake + 0.005
    # the equal-weight benchmark only depends on the multiset of returns when there are no costs
    panel, extra = make_market()
    c0 = build_compact(panel, extra, 0, panel.shape[0] - 1, 0, ZERO_COST, block=10)
    nul0 = permute_forward(c0, np.random.default_rng(0))
    np.testing.assert_allclose(nul0.benchmark(1.0).gross, c0.benchmark(1.0).gross, atol=1e-12)
    assert (nul0.benchmark(1.0).n_positions == c0.valid.sum(axis=1)).all()      # no ghost positions of delisted names


def test_null_ga_runs_on_shared_evaluator(compact):
    ev = Evaluator(compact)
    r = run_null_ga(ev, compact, index=2, pop_size=12, n_gen=2, topk=TOPK)
    assert len(r.population) == 12 and np.isfinite(r.best_q25())
    assert ev.misses > 0


# ------------------------------------------------------------------ statistics helpers
def test_by_fdr_and_qvalues_known_case():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rej = by_fdr(p, 0.10)
    assert rej.tolist() == [True] + [False] * 9                      # thresholds i * 0.1 / (10 * H10)
    q = by_qvalues(p)
    assert q[0] == pytest.approx(0.001 * 10 * (1 / np.arange(1, 11)).sum())
    assert (np.diff(q) >= -1e-12).all() and (q >= p).all() and q.max() <= 1.0
    assert np.isnan(by_qvalues(np.array([0.01, np.nan]))[1])


def test_multiple_testing_finds_planted_effects_and_controls_errors():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.02, (250, 30))
    x[:, :3] += 0.01                                                  # t of about 8 on three columns
    mt = multiple_testing(x, alpha=0.10, n_boot=400, mean_block=4.0, seed=1)
    assert mt.stepm[:3].all() and mt.by[:3].all()
    assert mt.stepm[3:].sum() <= 1 and mt.by[3:].sum() <= 1
    again = multiple_testing(x, alpha=0.10, n_boot=400, mean_block=4.0, seed=1)
    np.testing.assert_array_equal(mt.p, again.p)
    assert multiple_testing(np.zeros((10, 0))).passed.size == 0


def test_cluster_returns_recovers_factor_groups():
    rng = np.random.default_rng(1)
    factors = rng.normal(size=(300, 3))
    truth = np.repeat([0, 1, 2], 15)
    x = factors[:, truth] + 0.4 * rng.normal(size=(300, 45))
    labels = cluster_returns(x)
    assert adjusted_rand_score(truth, labels) > 0.95
    flat = np.column_stack([x[:, :5], np.zeros(300), np.zeros(300)])
    assert len(cluster_returns(flat)) == 7


def test_unique_columns_and_min_track_record():
    a = np.random.default_rng(2).normal(size=(50, 3))
    x = np.column_stack([a[:, 0], a[:, 1], a[:, 0], a[:, 2], a[:, 1]])
    assert unique_columns(x).tolist() == [0, 1, 3]
    rng = np.random.default_rng(3)
    weak, strong = rng.normal(0.001, 0.01, 500), rng.normal(0.004, 0.01, 500)
    assert min_track_record_length(strong) < min_track_record_length(weak) < np.inf
    assert min_track_record_length(-np.abs(weak)) == np.inf


def test_break_even_cost_on_oracle_and_anti_oracle(compact):
    good = np.where(compact.valid, np.nan_to_num(compact.wk), np.nan).astype(np.float32)
    m = break_even_cost(compact, good, TOPK)
    assert 0.0 < m < 50.0
    mean_at = lambda k: float((compact.weekly(good, TOPK, k).net - compact.benchmark(k).net).mean())   # noqa: E731
    assert abs(mean_at(m)) < 1e-9 and mean_at(0.5 * m) > 0 > mean_at(2 * m)
    assert break_even_cost(compact, -good, TOPK) == 0.0


# ------------------------------------------------------------------ DNA
def test_components_design_and_frequency():
    t = ("where", ("state", "mkt_trend"), ("rank", ("leaf", "ret_20")), ("mul", ("const", 2.0), ("leaf", "vol_20")))
    assert components_of(t) == {"where", "mkt_trend", "rank", "ret_20", "mul", "const", "vol_20"}
    x = design([t, ("leaf", "ret_5")])
    assert x.shape == (2, len(COMPONENTS)) and x[0, COMPONENTS.index("rank")] and not x[1, COMPONENTS.index("rank")]
    freq = frequency_table({"a": [t, ("leaf", "ret_5")], "b": []})
    assert freq["a"][COMPONENTS.index("ret_5")] == 0.5 and np.isnan(freq["b"]).all()


def test_enrichment_detects_planted_component_and_reports_attainable_q():
    rng = np.random.default_rng(0)
    n = 160
    x = rng.random((n, 5)) < 0.4
    controls = rng.normal(size=(n, 3))
    strata = np.repeat([1, 2, 3, 4], n // 4)
    logit = 1.8 * x[:, 1] + 0.5 * controls[:, 0] - 0.9
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    res = enrichment(x, y, controls, strata, n_perm=300, seed=0)
    assert res["q"][1] < 0.10 and res["odds_ratio"][1] > 1.5
    assert (res["q"][[0, 2, 3, 4]] > 0.10).all()
    assert res["min_possible_q"] == pytest.approx(5 * (1 / np.arange(1, 6)).sum() / 301)
    # 35 components and 500 permutations cannot reach q < 0.10 (arithmetic); a small case shows the report says so
    assert 35 * (1 / np.arange(1, 36)).sum() / 501 > 0.10
    small = enrichment(rng.random((40, 12)) < 0.5, y[:40], controls[:40], strata[:40], n_perm=50)
    assert small["min_possible_q"] > 0.10 and (small["q"][np.isfinite(small["q"])] > 0.10).all()
    flat = enrichment(x, np.zeros(n, int), controls, strata, n_perm=10)
    assert np.isnan(flat["p"]).all()


def test_replicated_needs_both_halves_and_the_same_direction():
    def half(q, orr):
        return {"q": np.array(q + [np.nan] * (len(COMPONENTS) - len(q))),
                "odds_ratio": np.array(orr + [np.nan] * (len(COMPONENTS) - len(orr)))}
    a = half([0.01, 0.01, 0.2, 0.01], [3.0, 3.0, 3.0, 0.2])
    b = half([0.02, 0.5, 0.01, 0.01], [2.0, 4.0, 3.0, 0.3])
    assert replicated(a, b) == [COMPONENTS[0]]


# ------------------------------------------------------------------ selection pipeline
@pytest.fixture(scope="module")
def stages():
    panel, extra = make_market(n_days=1100, seed=11)
    dev = build_compact(panel, extra, 0, 700, 0, TIER_COST, block=10)
    val = build_compact(panel, extra, 701, 1099, 21, TIER_COST, block=10)
    ev_dev, ev_val = Evaluator(dev), Evaluator(val)
    ga = []
    for seed in (1, 6):
        res = run_ga(Searcher(ev_dev, topk=TOPK), seed=seed, pop_size=24, n_gen=3)
        ga.append(res)
    rnd = Searcher(ev_dev, topk=TOPK)
    run_random(rnd, seed=1, n=80)
    null_best = [run_null_ga(ev_dev, dev, i, pop_size=12, n_gen=2, topk=TOPK).best_q25() for i in range(3)]
    return dict(dev=dev, val=val, ev_dev=ev_dev, ev_val=ev_val, ga=ga, rnd=rnd.ledger, null_best=null_best)


def _run_select(st, **kw):
    return select_run(st["ga"], st["rnd"], st["null_best"], st["ev_dev"], st["dev"], st["ev_val"], st["val"],
                      n_boot=200, pbo_sample=150, pbo_blocks=8, topk=TOPK, **kw)


def test_validation_stage_starts_after_embargo(stages):
    assert stages["val"].dec_rows.min() >= 701 + 21 and stages["dev"].dec_rows.max() + 1 <= 700


def test_gate_g1_logic():
    g = gate_g1([1.0, 2.0, 3.0], [0.1, 0.2, 0.3, 0.4, 0.5, 5.0])
    assert g["median_real_best_f1"] == 2.0 and not g["passed"]          # p95 of the null is above 2
    assert gate_g1([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])["passed"]


def test_representatives_are_one_per_cluster_best_f1_first(stages):
    cands = ga_candidates(stages["ga"])
    assert cands and len({c.tree for c in cands}) == len(cands)
    net, bench = streams(stages["ev_dev"], stages["dev"], [c.tree for c in cands], TOPK)
    keep, labels, reps = representatives(net - bench[:, None], cands)
    assert len(reps) == len(np.unique(labels)) and len(set(reps)) == len(reps)
    f1 = [cands[j].fit.q25 for j in reps]
    assert f1 == sorted(f1, reverse=True)
    for j in reps:                                                     # no cluster-mate has a better dev f1
        mates = keep[labels == labels[list(keep).index(j)]]
        assert all(cands[m].fit.q25 <= cands[j].fit.q25 for m in mates)


def test_select_run_structure_and_determinism(stages):
    import json
    out = _run_select(stages, seed=3)
    for gate in ("G1", "G2", "G3", "G4"):
        assert gate in out
    assert 0.0 <= out["G4"]["pbo"] <= 1.0 and out["G3"]["n_reps"] >= 1
    assert len(out["ensemble_trees"]) <= 20
    ga_trees = {c.tree for c in ga_candidates(stages["ga"])}
    assert all(t in ga_trees for t in out["ensemble_trees"])
    assert json.dumps(out, sort_keys=True, default=str) == json.dumps(_run_select(stages, seed=3), sort_keys=True, default=str)


def test_select_run_ensemble_branch_and_diagnostics(stages, monkeypatch):
    def everything_passes(excess, *a, **k):
        k_cols = excess.shape[1]
        ones = np.ones(k_cols, bool)
        return MultiTest(np.full(k_cols, 5.0), np.full(k_cols, 0.001), ones, ones)

    monkeypatch.setattr(selection, "multiple_testing", everything_passes)
    out = _run_select(stages, seed=3, max_ensemble=4)
    assert len(out["ensemble_trees"]) == min(4, out["G3"]["n_reps"])
    diag = out["diagnostics"]
    assert set(diag["dsr"]) == {"0.3", "1.0", "3.0"} and all(0.0 <= v <= 1.0 for v in diag["dsr"].values())
    assert diag["min_trl_weeks"] > 0 and "champion_break_even_cost_multiplier" in diag


def test_gate_g5_runs_and_flags_unreachable_q(stages):
    g5 = gate_g5(stages["ga"], stages["rnd"], stages["ev_dev"], stages["dev"], stages["ev_val"], stages["val"],
                 n_perm=20, seed=0, topk=TOPK)
    assert set(g5) >= {"passed", "replicated", "enrichment", "frequency", "min_possible_q"}
    assert all(q > 0.10 for q in g5["min_possible_q"] if np.isfinite(q))     # 20 permutations cannot pass
    assert not g5["passed"]
    assert set(next(iter(g5["frequency"].values()))) == {"prior", "survivors", "robust"}


def test_search_result_round_trips_through_the_store(compact, tmp_path):
    res, _ = _run(compact)
    summary = {"seed": res.seed, "history": res.history, "population": pairs_to_rows(res.population),
               "front": pairs_to_rows(res.front), "n_backtests": res.n_backtests}
    save_result(tmp_path / "ga_3", summary, res.ledger)
    back = load_ga_result(tmp_path / "ga_3.json")
    assert back.seed == res.seed and back.population == res.population and back.front == res.front
    assert back.ledger.rows == res.ledger.rows and back.ledger.first_gen == res.ledger.first_gen
    assert back.n_backtests == res.n_backtests and back.history == res.history


# ------------------------------------------------------------------ late stage (G6)
def test_rel_drawdown_and_perf_known_values():
    net = np.array([0.10, -0.10, -0.10, 0.20])
    assert rel_drawdown(net, np.zeros(4)) == pytest.approx(1 - 0.891 / 1.1)
    assert rel_drawdown(net, net) == 0.0                                  # tracking the benchmark never draws down
    assert rel_drawdown(np.array([0.01, 0.01]), np.zeros(2)) == 0.0
    p = perf(np.array([0.10, -0.10, -0.10, 0.20]))
    assert p["max_dd"] == pytest.approx(1 - 0.891 / 1.1) and p["cagr"] == pytest.approx(1.0692 ** 13 - 1)


def test_g6_decision_requires_every_condition():
    assert g6_decision(0.5, 0.1, 0.2, 0.10, 0.08)
    assert not g6_decision(-0.1, 0.1, -1.0, 0.0, 0.1)                     # negative IR at 1x
    assert not g6_decision(0.5, -0.1, 0.2, 0.0, 0.1)                      # negative IR at 2x
    assert not g6_decision(0.5, 0.1, 0.6, 0.0, 0.1)                       # not above the random ensemble
    assert not g6_decision(0.5, 0.1, 0.2, 0.13, 0.08)                     # drawdown above 1.5 x worst dev fold
    assert g6_decision(0.5, 0.0, 0.2, 0.12, 0.08)                         # 2x IR of exactly 0 and dd at the bound pass


def test_matched_random_picks_nearest_in_turnover_and_size(stages):
    ledger = stages["rnd"]
    rows = [(t, f) for t, f in ledger.rows.items() if f.feasible]
    t0, f0 = rows[0]
    near = matched_random(ledger, f0.turnover_year, f0.nodes, n=5)
    assert len(near) == 5 and t0 in near and all(ledger.rows[t].feasible for t in near)
    far = matched_random(ledger, f0.turnover_year, f0.nodes, n=len(rows))
    dist = [abs(ledger.rows[t].turnover_year - f0.turnover_year) / f0.turnover_year
            + abs(ledger.rows[t].nodes - f0.nodes) / f0.nodes for t in far]
    assert dist == sorted(dist)


def test_hash_files_and_frozen_round_trip(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("one"), b.write_text("two")
    h = hash_files([str(a), str(b)])
    assert h == hash_files([str(a), str(b)]) and h != hash_files([str(b), str(a)])
    b.write_text("three")
    assert h != hash_files([str(a), str(b)])
    frozen = {"ensemble_trees": [("leaf", "ret_5")], "random_ensemble_trees": [], "matched_random_trees": [("neg", ("leaf", "vol_20"))],
              "champion_tree": ("leaf", "ret_20"), "canonical": dict(CANONICAL), "D": None}
    back = frozen_from_json(__import__("json").loads(__import__("json").dumps(frozen_to_json(frozen))))
    assert back == frozen


def test_late_report_on_frozen_selection(stages):
    st = stages
    sel = _run_select(st, seed=3)
    # make sure the late stage has trees to evaluate even if the noise market produced no passer
    members = sel["ensemble_trees"] or [c.tree for c in ga_candidates(st["ga"])[:3]]
    sel = {**sel, "ensemble_trees": members,
           "random_ensemble_trees": sel["random_ensemble_trees"] or members[:1],
           "champion_tree": sel["champion_tree"] or members[0],
           "ensemble_profile": sel["ensemble_profile"] or {"turnover_year": 40.0, "nodes": 4.0}}
    frozen = prepare_frozen(sel, st["rnd"], st["ev_dev"], st["dev"], "A", d_tree=("neg", ("leaf", "ret_5")),
                            n_matched=30, topk=TOPK)
    assert frozen["worst_dev_fold_rel_dd"] is not None and len(frozen["matched_random_trees"]) == 30
    rep = late_report(frozen, st["ev_val"], st["val"], topk=TOPK, n_boot=100, seed=1)
    for key in ("ir_1x", "ir_2x", "ir_3x", "ir_1x_ci90", "rel_drawdown", "ir_random_ensemble_1x", "ir_champion_1x",
                "ir_canonical_1x", "matched_random", "ir_D_1x", "ir_D_2x", "passed", "ensemble_perf_1x"):
        assert key in rep, key
    assert set(rep["ir_canonical_1x"]) == set(CANONICAL)
    lo, hi = rep["ir_1x_ci90"]
    assert lo <= hi and np.isfinite([lo, hi]).all()
    assert rep["passed"] == g6_decision(rep["ir_1x"], rep["ir_2x"], rep["ir_random_ensemble_1x"], rep["rel_drawdown"],
                                        frozen["worst_dev_fold_rel_dd"])
    again = late_report(frozen, st["ev_val"], st["val"], topk=TOPK, n_boot=100, seed=1)
    assert again == rep
    assert late_report({**frozen, "ensemble_trees": []}, st["ev_val"], st["val"])["passed"] is False


def test_ensemble_of_one_equals_the_single_stream(stages):
    tree = ("sub", ("leaf", "ret_60"), ("leaf", "ret_5"))
    net, bench = ensemble_series(stages["ev_val"], stages["val"], [tree], TOPK)
    n1, b1 = streams(stages["ev_val"], stages["val"], [tree], TOPK)
    np.testing.assert_array_equal(net, n1[:, 0])
    np.testing.assert_array_equal(bench, b1)
    assert worst_fold_rel_dd(stages["ev_dev"], stages["dev"], [tree], TOPK) >= 0.0


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_final_script_freeze_run_flow(stages, tmp_path, monkeypatch):
    fin = _load_script("r22_final")
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(r22_setup, "RESULTS", results)
    monkeypatch.setattr(r22_setup, "REGISTRY", tmp_path / "registry.sqlite")
    monkeypatch.setattr(r22_setup, "vault", lambda run: Vault(date(2019, 12, 31), tmp_path / f"v{run}" / "frozen.lock",
                                                              tmp_path / f"v{run}" / "vault.log"))
    monkeypatch.setattr(r22_setup, "load_compact", lambda run, stage, *a, **k: stages["val"])
    monkeypatch.setattr(fin, "ensure_clean", lambda: None)

    members = [c.tree for c in ga_candidates(stages["ga"])[:3]]
    sel = {"G1": None, "G2": None, "G3": {"passed": True}, "G4": None, "ensemble_trees": members,
           "random_ensemble_trees": members[:1], "random_ensemble_fallback": True, "champion_tree": members[0],
           "ensemble_profile": {"turnover_year": 40.0, "nodes": 4.0}}
    frozen = prepare_frozen(sel, stages["rnd"], stages["ev_dev"], stages["dev"], "A", n_matched=20, topk=TOPK)
    for run in ("A", "B"):
        (results / f"frozen_{run}.json").write_text(json.dumps(frozen_to_json({**frozen, "run": run}), default=float))

    fin.main(["freeze", "--run", "A"])
    monkeypatch.setenv(r22_setup.FINAL_ENV, fin.methodology_hash("A"))
    with pytest.raises(SystemExit):                              # B is not frozen yet: neither late stage may open
        fin.main(["run", "--run", "A"])
    fin.main(["freeze", "--run", "B"])
    with pytest.raises(VaultError):                              # a frozen methodology cannot be frozen again
        fin.main(["freeze", "--run", "A"])
    monkeypatch.setenv(r22_setup.FINAL_ENV, "not-the-hash")
    with pytest.raises(SystemExit):
        fin.main(["run", "--run", "A"])
    monkeypatch.setenv(r22_setup.FINAL_ENV, fin.methodology_hash("A"))
    fin.main(["run", "--run", "A", "--n-boot", "50"])
    out = json.loads((results / "final_A.json").read_text())
    assert out["run"] == "A" and out["n_members"] == 3 and out["methodology_hash"] == fin.methodology_hash("A")
    with pytest.raises(VaultError):                              # the late stage opens once
        fin.main(["run", "--run", "A", "--n-boot", "50"])
    assert fin.methodology_hash("A") != fin.methodology_hash("B")
    before = fin.methodology_hash("B")
    (results / "frozen_B.json").write_text("{}")                 # any change to a frozen file breaks the hash
    assert fin.methodology_hash("B") != before
    monkeypatch.setenv(r22_setup.FINAL_ENV, before)
    with pytest.raises(SystemExit):
        fin.main(["run", "--run", "B"])
    assert (tmp_path / "registry.sqlite").exists()
