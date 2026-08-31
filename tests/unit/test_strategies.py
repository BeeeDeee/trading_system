"""Setup detection: geometric rules only; no costs, size, or other symbols."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from scout.config.schema import StrategyConfig
from scout.domain.enums import Direction, MarketRegime, Regime, VolBucket
from scout.domain.features import FeatureRow
from scout.domain.ports import Strategy
from scout.strategies.donchian_breakout import DonchianBreakout
from scout.strategies.registry import STRATEGY_FACTORIES, build_strategies
from scout.strategies.xsec_momentum import XSecMomentum
from scout.utils.errors import ScoutConfigError

TS = datetime(2015, 6, 15, 20, 0, tzinfo=UTC)


def _row(**overrides: Any) -> FeatureRow:
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
        "mom_252_xs_pct": 0.50,
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


def test_xsec_momentum_satisfies_protocol() -> None:
    strategy = XSecMomentum()
    assert isinstance(strategy, Strategy)
    assert Strategy not in XSecMomentum.__mro__


def test_donchian_breakout_satisfies_protocol() -> None:
    strategy = DonchianBreakout()
    assert isinstance(strategy, Strategy)
    assert Strategy not in DonchianBreakout.__mro__


def test_xsec_momentum_long_fires() -> None:
    # close=100, stop_atr=5, atr_14=2 → stop = 100 - 10 = 90; no target
    row = _row(mom_252_xs_pct=0.90)
    setup = XSecMomentum().detect(row)
    assert setup is not None
    assert setup.direction is Direction.LONG
    assert setup.target_price is None
    assert setup.stop_price == pytest.approx(90.0, rel=1e-9)
    assert setup.reference_price == pytest.approx(100.0, rel=1e-9)
    assert setup.max_hold_bars == 21
    assert setup.strategy_id == "xsec_momentum_v1"
    assert setup.trigger_note == "mom_252_xs_pct>=thr"
    assert setup.symbol == "AAPL"
    assert setup.ts == TS


def test_xsec_momentum_short_fires() -> None:
    # Bottom-decile rank. 0.10 is not used: 1.0 - 0.90 is 0.0999… in float,
    # so a print of exactly 0.10 would fail the `<= 1 - thr` check.
    # stop = 100 + 5 * 2 = 110
    row = _row(mom_252_xs_pct=0.05)
    setup = XSecMomentum().detect(row)
    assert setup is not None
    assert setup.direction is Direction.SHORT
    assert setup.target_price is None
    assert setup.stop_price == pytest.approx(110.0, rel=1e-9)
    assert setup.trigger_note == "mom_252_xs_pct<=1-thr"


def test_xsec_momentum_mid_rank_does_not_fire() -> None:
    assert XSecMomentum().detect(_row(mom_252_xs_pct=0.50)) is None


def test_xsec_momentum_not_warm_does_not_fire() -> None:
    assert XSecMomentum().detect(_row(mom_252_xs_pct=0.95, is_warm=False)) is None


def test_xsec_momentum_detect_is_pure() -> None:
    row = _row(mom_252_xs_pct=0.95)
    strategy = XSecMomentum()
    assert strategy.detect(row) == strategy.detect(row)


def test_xsec_momentum_does_not_gate_on_per_symbol_regime() -> None:
    row = _row(mom_252_xs_pct=0.95, regime=Regime.CHOP)
    setup = XSecMomentum().detect(row)
    assert setup is not None
    assert setup.regime is Regime.CHOP


def test_xsec_momentum_does_not_gate_on_market_regime() -> None:
    # Engine owns MARKET_REGIME_BLOCKED; detect is geometric only.
    row = _row(mom_252_xs_pct=0.95, market_regime=MarketRegime.RISK_OFF)
    assert XSecMomentum().detect(row) is not None


def test_xsec_momentum_nan_rank_does_not_fire() -> None:
    assert XSecMomentum().detect(_row(mom_252_xs_pct=float("nan"))) is None


def test_donchian_long_fires() -> None:
    # breakout_level = 105 + 0.10 * 2 = 105.2; close=106 > 105.2
    # stop = 106 - 3*2 = 100; target = 106 + 2*3*2 = 118
    row = _row(
        close=106.0,
        donchian_high_55=105.0,
        ema_spread_atr=0.20,
        regime=Regime.TREND_UP,
    )
    setup = DonchianBreakout().detect(row)
    assert setup is not None
    assert setup.direction is Direction.LONG
    assert setup.stop_price == pytest.approx(100.0, rel=1e-9)
    assert setup.target_price == pytest.approx(118.0, rel=1e-9)
    assert setup.max_hold_bars == 40
    assert setup.trigger_note == "close>dc_high_55+buf"
    assert setup.strategy_id == "donchian_breakout_v1"


def test_donchian_short_fires() -> None:
    # breakdown_level = 95 - 0.10 * 2 = 94.8; close=94 < 94.8
    # stop = 94 + 6 = 100; target = 94 - 12 = 82
    row = _row(
        close=94.0,
        donchian_low_55=95.0,
        ema_spread_atr=-0.20,
        regime=Regime.TREND_DOWN,
    )
    setup = DonchianBreakout().detect(row)
    assert setup is not None
    assert setup.direction is Direction.SHORT
    assert setup.stop_price == pytest.approx(100.0, rel=1e-9)
    assert setup.target_price == pytest.approx(82.0, rel=1e-9)
    assert setup.trigger_note == "close<dc_low_55-buf"


def test_donchian_close_on_channel_does_not_fire() -> None:
    # close == breakout_level = 105.2; must be strictly through the buffer
    row = _row(
        close=105.2,
        donchian_high_55=105.0,
        ema_spread_atr=1.0,
        regime=Regime.TREND_UP,
    )
    assert DonchianBreakout().detect(row) is None


def test_donchian_ema_spread_too_small_does_not_fire() -> None:
    row = _row(
        close=106.0,
        donchian_high_55=105.0,
        ema_spread_atr=0.19,
        regime=Regime.TREND_UP,
    )
    assert DonchianBreakout().detect(row) is None


def test_donchian_chop_does_not_fire() -> None:
    row = _row(close=106.0, donchian_high_55=105.0, regime=Regime.CHOP)
    assert DonchianBreakout().detect(row) is None


def test_donchian_not_warm_does_not_fire() -> None:
    row = _row(
        close=106.0,
        donchian_high_55=105.0,
        ema_spread_atr=1.0,
        regime=Regime.TREND_UP,
        is_warm=False,
    )
    assert DonchianBreakout().detect(row) is None


def test_from_params_rejects_unknown_key() -> None:
    with pytest.raises(ScoutConfigError, match="lookback"):
        XSecMomentum.from_params({"xs_threshold": 0.9, "lookback": 252})
    with pytest.raises(ScoutConfigError, match="donchian_n"):
        DonchianBreakout.from_params({"donchian_n": 20})


def test_from_params_binds_overrides() -> None:
    mom = XSecMomentum.from_params({"xs_threshold": 0.85, "max_hold_bars": 15})
    assert mom.xs_threshold == pytest.approx(0.85, rel=1e-9)
    assert mom.max_hold_bars == 15
    assert mom.stop_atr == pytest.approx(5.0, rel=1e-9)


def test_build_strategies_skips_disabled() -> None:
    built = build_strategies(
        [
            StrategyConfig(strategy_id="xsec_momentum_v1", enabled=False, params={}),
            StrategyConfig(strategy_id="donchian_breakout_v1", enabled=True, params={}),
        ]
    )
    assert len(built) == 1
    assert built[0].strategy_id == "donchian_breakout_v1"


def test_build_strategies_unknown_id_raises() -> None:
    with pytest.raises(ScoutConfigError, match="not_a_real_strategy_v1"):
        build_strategies(
            [StrategyConfig(strategy_id="not_a_real_strategy_v1", enabled=True, params={})]
        )


def test_no_range_fade_in_v1() -> None:
    assert set(STRATEGY_FACTORIES) == {"xsec_momentum_v1", "donchian_breakout_v1"}
    assert "range_fade_v1" not in STRATEGY_FACTORIES
