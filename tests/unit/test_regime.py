"""Per-symbol and market regime classifiers. Golden cases plus purity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from scout.config.schema import MarketRegimeConfig, RegimeConfig
from scout.domain.enums import MarketRegime, Regime
from scout.features.market import (
    DD_LOOKBACK,
    classify_market_regime,
    compute_market_columns,
)
from scout.features.regime import classify_regime, classify_regime_vectorized

REL = 1e-9

_CFG = RegimeConfig()
_MKT = MarketRegimeConfig()


def test_nan_gives_unknown() -> None:
    nan = float("nan")
    base = (0.5, 0.4, 1.0, 0.5)
    for i in range(4):
        args = list(base)
        args[i] = nan
        assert classify_regime(*args, _CFG) is Regime.UNKNOWN
    assert classify_regime(nan, nan, nan, nan, _CFG) is Regime.UNKNOWN


def test_strong_uptrend() -> None:
    assert (
        classify_regime(er_20=0.50, er_60=0.40, slope_atr=0.50, vol_pct=0.50, cfg=_CFG)
        is Regime.TREND_UP
    )


def test_strong_downtrend() -> None:
    assert (
        classify_regime(er_20=0.50, er_60=0.40, slope_atr=-0.50, vol_pct=0.50, cfg=_CFG)
        is Regime.TREND_DOWN
    )


def test_low_er_low_vol_is_range() -> None:
    assert (
        classify_regime(er_20=0.10, er_60=0.10, slope_atr=0.0, vol_pct=0.50, cfg=_CFG)
        is Regime.RANGE
    )


def test_low_er_high_vol_is_chop() -> None:
    assert (
        classify_regime(er_20=0.10, er_60=0.10, slope_atr=0.0, vol_pct=0.80, cfg=_CFG)
        is Regime.CHOP
    )


def test_high_er_flat_slope_is_chop() -> None:
    assert (
        classify_regime(er_20=0.50, er_60=0.40, slope_atr=0.0, vol_pct=0.50, cfg=_CFG)
        is Regime.CHOP
    )


def test_classifier_is_pure() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    er_20 = rng.uniform(0.0, 1.0, n)
    er_60 = rng.uniform(0.0, 1.0, n)
    slope = rng.uniform(-2.0, 2.0, n)
    vol = rng.uniform(0.0, 1.0, n)
    nan_idx = rng.integers(0, n, size=80)
    for arr in (er_20, er_60, slope, vol):
        arr[nan_idx[:20]] = np.nan
        nan_idx = nan_idx[20:]

    first = [
        classify_regime(float(a), float(b), float(c), float(d), _CFG)
        for a, b, c, d in zip(er_20, er_60, slope, vol, strict=True)
    ]
    second = [
        classify_regime(float(a), float(b), float(c), float(d), _CFG)
        for a, b, c, d in zip(er_20, er_60, slope, vol, strict=True)
    ]
    assert first == second
    assert all(isinstance(label, Regime) for label in first)

    vec = classify_regime_vectorized(
        pd.Series(er_20),
        pd.Series(er_60),
        pd.Series(slope),
        pd.Series(vol),
        _CFG,
    )
    assert list(vec) == first


@given(
    er_20=st.floats(allow_nan=True, allow_infinity=True),
    er_60=st.floats(allow_nan=True, allow_infinity=True),
    slope_atr=st.floats(allow_nan=True, allow_infinity=True),
    vol_pct=st.floats(allow_nan=True, allow_infinity=True),
)
def test_classifier_is_total(
    er_20: float, er_60: float, slope_atr: float, vol_pct: float
) -> None:
    a = classify_regime(er_20, er_60, slope_atr, vol_pct, _CFG)
    b = classify_regime(er_20, er_60, slope_atr, vol_pct, _CFG)
    assert a is b
    assert a in Regime


def test_market_nan_gives_unknown() -> None:
    nan = float("nan")
    assert classify_market_regime(True, nan, 0.5, _MKT) is MarketRegime.UNKNOWN
    assert classify_market_regime(True, 0.05, nan, _MKT) is MarketRegime.UNKNOWN
    assert classify_market_regime(False, nan, nan, _MKT) is MarketRegime.UNKNOWN


def test_spy_below_ma_with_20pct_drawdown_is_risk_off() -> None:
    assert classify_market_regime(False, 0.20, 0.50, _MKT) is MarketRegime.RISK_OFF

    n = 400
    close = pd.Series(np.full(n, 100.0))
    close.iloc[-1] = 80.0
    cols = compute_market_columns(close, _MKT)
    last = cols.iloc[-1]
    assert not bool(last["spy_above_ma"])
    assert last["spy_dd_252"] == pytest.approx(0.20, rel=REL)
    assert last["market_regime"] == MarketRegime.RISK_OFF


def test_market_above_ma_small_dd_is_risk_on() -> None:
    assert classify_market_regime(True, 0.05, 0.50, _MKT) is MarketRegime.RISK_ON


def test_market_default_is_neutral() -> None:
    # Above MA but stressed by drawdown: not RISK_ON, not RISK_OFF.
    assert classify_market_regime(True, 0.20, 0.50, _MKT) is MarketRegime.NEUTRAL
    # Below MA but not stressed.
    assert classify_market_regime(False, 0.05, 0.50, _MKT) is MarketRegime.NEUTRAL


def test_market_warmup_is_unknown() -> None:
    close = pd.Series(np.linspace(100.0, 120.0, DD_LOOKBACK - 1))
    cols = compute_market_columns(close, _MKT)
    assert (cols["market_regime"] == MarketRegime.UNKNOWN).all()
    assert cols["spy_dd_252"].isna().all()
