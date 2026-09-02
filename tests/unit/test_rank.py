"""Opportunity construction and ranking: threshold, sort key, alphabetical ties."""

from __future__ import annotations

import random
from dataclasses import fields
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from scout.config.schema import ScoringConfig
from scout.domain.costs import CostEstimate
from scout.domain.edge import MIN_BIN_SAMPLES, BinKey, BinStats
from scout.domain.enums import Direction, MarketRegime, Regime, VolBucket
from scout.domain.features import FeatureRow
from scout.domain.market import Asset
from scout.domain.opportunity import Opportunity
from scout.domain.setup import Setup
from scout.domain.universe import UniverseEntry
from scout.scoring.rank import build_opportunity, rank

TS = datetime(2015, 6, 15, 20, 0, tzinfo=UTC)
REL = 1e-12

# Hand-computed: ev_net_r = 0.20 - 0.023 = 0.177; / 21.0 bars.
EV_R_LCB = 0.20
COST_R = 0.023
MEAN_BARS = 21.0
EV_NET_R = EV_R_LCB - COST_R
EV_PER_BAR_R = EV_NET_R / MEAN_BARS


def _setup(**overrides: Any) -> Setup:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "ts": TS,
        "strategy_id": "donchian_breakout_v1",
        "direction": Direction.LONG,
        "regime": Regime.TREND_UP,
        "reference_price": 100.0,
        "stop_price": 94.0,
        "target_price": 112.0,
        "max_hold_bars": 21,
        "trigger_note": "close>dc_high_55",
    }
    kwargs.update(overrides)
    return Setup(**kwargs)


def _feature_row(**overrides: Any) -> FeatureRow:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "ts": TS,
        "close": 100.0,
        "atr_14": 2.0,
        "atr_pct": 0.02,
        "vol_20": 0.2,
        "vol_60": 0.25,
        "ema_fast": 101.0,
        "ema_slow": 99.0,
        "ema_spread_atr": 1.0,
        "slope_atr_20": 0.5,
        "efficiency_ratio_20": 0.8,
        "efficiency_ratio_60": 0.7,
        "atr_percentile_1y": 0.4,
        "regime": Regime.TREND_UP,
        "vol_bucket": VolBucket.MID,
        "donchian_high_20": 102.0,
        "donchian_low_20": 90.0,
        "donchian_high_55": 105.0,
        "donchian_low_55": 85.0,
        "keltner_upper": 105.0,
        "keltner_lower": 93.0,
        "dist_to_high_atr": 2.5,
        "dist_to_low_atr": 7.5,
        "mom_252_skip21": 0.15,
        "mom_126_skip21": 0.08,
        "mom_21": 0.02,
        "mom_252_xs_pct": 0.92,
        "vol_xs_pct": 0.4,
        "xs_population": 400,
        "gap_atr": 0.1,
        "gap_abs_mean_20": 0.2,
        "overnight_var_share_60": 0.3,
        "market_regime": MarketRegime.RISK_ON,
        "spy_dd_252": 0.05,
        "spy_above_ma": True,
        "vix_close": 18.0,
        "beta_bench_90": 1.1,
        "corr_bench_90": 0.6,
        "bars_available": 400,
        "bars_since_gap": 50,
        "is_warm": True,
    }
    kwargs.update(overrides)
    return FeatureRow(**kwargs)


def _bin_stats(**overrides: Any) -> BinStats:
    kwargs: dict[str, Any] = {
        "key": BinKey("donchian_breakout_v1", Direction.LONG, VolBucket.MID),
        "as_of": TS,
        "n": MIN_BIN_SAMPLES,
        "mean_r": 0.30,
        "std_r": 1.0,
        "ev_r_lcb": EV_R_LCB,
        "mean_bars_held": MEAN_BARS,
        "win_rate": 0.4,
        "avg_win_r": 1.5,
        "avg_loss_r": -1.0,
        "target_rate": 0.3,
        "stop_rate": 0.5,
        "time_rate": 0.2,
        "median_r": 0.1,
        "p05_r": -1.2,
        "p95_r": 2.0,
    }
    kwargs.update(overrides)
    return BinStats(**kwargs)


def _cost(**overrides: Any) -> CostEstimate:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "ts": TS,
        "notional_usd": 7400.0,
        "expected_bars_held": MEAN_BARS,
        "entry_fee_bps": 0.47,
        "exit_fee_bps": 0.47,
        "spread_bps": 2.0,
        "slippage_bps": 9.2,
        "impact_bps": 0.4,
        "funding_bps": 0.0,
        "total_bps": 12.54,
        "cost_usd": 9.28,
        "cost_r": COST_R,
    }
    kwargs.update(overrides)
    return CostEstimate(**kwargs)


def _asset(**overrides: Any) -> Asset:
    kwargs: dict[str, Any] = {
        "asset_id": "AAPL",
        "symbol": "AAPL",
        "exchange": "NASDAQ",
        "quote_currency": "USD",
        "is_etf": False,
        "cluster": "INFO_TECH",
        "tick_size": Decimal("0.01"),
        "step_size": Decimal("1"),
        "min_notional_usd": Decimal("0"),
        "listed_at": None,
        "delisted_at": None,
        "delist_reason": None,
    }
    kwargs.update(overrides)
    return Asset(**kwargs)


def _entry(**overrides: Any) -> UniverseEntry:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "eligible": True,
        "reason": None,
        "adv_usd_30": 1_500_000_000.0,
        "spread_bps_est": 1.0,
        "bars_available": 400,
        "listed_days": 1000.0,
    }
    kwargs.update(overrides)
    return UniverseEntry(**kwargs)


def _opportunity(**overrides: Any) -> Opportunity:
    setup = overrides.pop("setup", None)
    if setup is None:
        setup = _setup(
            symbol=str(overrides.get("symbol", "AAPL")),
            strategy_id=str(overrides.get("strategy_id", "donchian_breakout_v1")),
        )
    kwargs: dict[str, Any] = {
        "symbol": setup.symbol,
        "ts": TS,
        "strategy_id": setup.strategy_id,
        "direction": Direction.LONG,
        "setup": setup,
        "ev_net_r": EV_NET_R,
        "ev_per_bar_r": EV_PER_BAR_R,
        "ev_r_lcb": EV_R_LCB,
        "ev_r_point": 0.30,
        "cost_r": COST_R,
        "expected_bars_held": MEAN_BARS,
        "bin_key": BinKey(setup.strategy_id, Direction.LONG, VolBucket.MID),
        "bin_n": MIN_BIN_SAMPLES,
        "bin_std_r": 1.0,
        "bin_win_rate": 0.4,
        "regime": Regime.TREND_UP,
        "vol_bucket": VolBucket.MID,
        "adv_usd_30": 1e8,
        "spread_bps_est": 1.0,
        "beta_bench_90": 1.1,
        "cluster": "INFO_TECH",
        "atr_pct": 0.02,
    }
    kwargs.update(overrides)
    return Opportunity(**kwargs)


def test_build_opportunity_populates_every_field() -> None:
    setup = _setup()
    row = _feature_row()
    stats = _bin_stats()
    cost = _cost()
    asset = _asset()
    entry = _entry()
    opp = build_opportunity(setup, row, stats, cost, asset, entry)

    expected: dict[str, object] = {
        "symbol": "AAPL",
        "ts": TS,
        "strategy_id": "donchian_breakout_v1",
        "direction": Direction.LONG,
        "setup": setup,
        "ev_net_r": EV_NET_R,
        "ev_per_bar_r": EV_PER_BAR_R,
        "ev_r_lcb": EV_R_LCB,
        "ev_r_point": 0.30,
        "cost_r": COST_R,
        "expected_bars_held": MEAN_BARS,
        "bin_key": stats.key,
        "bin_n": MIN_BIN_SAMPLES,
        "bin_std_r": 1.0,
        "bin_win_rate": 0.4,
        "regime": Regime.TREND_UP,
        "vol_bucket": VolBucket.MID,
        "adv_usd_30": 1_500_000_000.0,
        "spread_bps_est": 1.0,
        "beta_bench_90": 1.1,
        "cluster": "INFO_TECH",
        "atr_pct": 0.02,
    }
    names = {f.name for f in fields(Opportunity)}
    assert set(expected) == names
    for name in names:
        got = getattr(opp, name)
        want = expected[name]
        if isinstance(want, float):
            assert got == pytest.approx(want, rel=REL), name
        else:
            assert got == want, name


def test_expected_bars_held_floors_at_one() -> None:
    stats = _bin_stats(mean_bars_held=0.25)
    opp = build_opportunity(_setup(), _feature_row(), stats, _cost(), _asset(), _entry())
    assert opp.expected_bars_held == pytest.approx(1.0, rel=REL)
    assert opp.ev_per_bar_r == pytest.approx(EV_NET_R / 1.0, rel=REL)


def test_rank_filters_by_min_ev_net_r() -> None:
    keep = _opportunity(symbol="MSFT", ev_net_r=0.05, ev_per_bar_r=0.01)
    drop = _opportunity(symbol="AAPL", ev_net_r=0.049, ev_per_bar_r=0.02)
    on_threshold = _opportunity(symbol="IBM", ev_net_r=0.05, ev_per_bar_r=0.005)
    ranked = rank((keep, drop, on_threshold), ScoringConfig(min_ev_net_r=0.05))
    assert tuple(c.symbol for c in ranked) == ("MSFT", "IBM")


def test_rank_sorts_by_ev_per_bar_r_descending() -> None:
    slow = _opportunity(symbol="AAA", ev_net_r=0.20, ev_per_bar_r=0.005)
    fast = _opportunity(symbol="ZZZ", ev_net_r=0.12, ev_per_bar_r=0.008)
    mid = _opportunity(symbol="MMM", ev_net_r=0.10, ev_per_bar_r=0.006)
    ranked = rank((slow, fast, mid), ScoringConfig(min_ev_net_r=0.05))
    assert tuple(c.symbol for c in ranked) == ("ZZZ", "MMM", "AAA")


def test_rank_breaks_ties_alphabetically() -> None:
    # Same ev_per_bar_r: symbol then strategy_id, never input order.
    b_donchian = _opportunity(
        symbol="BBB",
        strategy_id="donchian_breakout_v1",
        ev_net_r=0.10,
        ev_per_bar_r=0.01,
    )
    a_xsec = _opportunity(
        symbol="AAA",
        strategy_id="xsec_momentum_v1",
        ev_net_r=0.10,
        ev_per_bar_r=0.01,
    )
    a_donchian = _opportunity(
        symbol="AAA",
        strategy_id="donchian_breakout_v1",
        ev_net_r=0.10,
        ev_per_bar_r=0.01,
    )
    ranked = rank(
        (b_donchian, a_xsec, a_donchian),
        ScoringConfig(min_ev_net_r=0.05),
    )
    assert [(c.symbol, c.strategy_id) for c in ranked] == [
        ("AAA", "donchian_breakout_v1"),
        ("AAA", "xsec_momentum_v1"),
        ("BBB", "donchian_breakout_v1"),
    ]


def test_rank_deterministic_under_shuffled_input() -> None:
    candidates = (
        _opportunity(symbol="MSFT", strategy_id="xsec_momentum_v1", ev_per_bar_r=0.009),
        _opportunity(symbol="AAPL", strategy_id="donchian_breakout_v1", ev_per_bar_r=0.009),
        _opportunity(symbol="IBM", ev_net_r=0.01, ev_per_bar_r=0.001),
        _opportunity(symbol="XOM", ev_net_r=0.20, ev_per_bar_r=0.012),
        _opportunity(symbol="T", ev_net_r=0.02, ev_per_bar_r=0.02),
        _opportunity(symbol="AAPL", strategy_id="xsec_momentum_v1", ev_per_bar_r=0.009),
    )
    cfg = ScoringConfig(min_ev_net_r=0.05)
    expected = rank(candidates, cfg)
    assert expected  # the 0.02 candidate is dropped
    rng = random.Random(20260827)
    for _ in range(20):
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        assert rank(tuple(shuffled), cfg) == expected


def test_rank_empty_is_empty() -> None:
    assert rank((), ScoringConfig()) == ()
