"""Candidate config -> decisions -> simulated returns, with shared per-signal caches."""

from dataclasses import dataclass, field

import numpy as np

from qlab.data.panel import Panel
from qlab.engine.vector import Decisions, SimResult, simulate
from qlab.features.basic import trailing_volatility
from qlab.schedule import period_starts
from qlab.strategies.grid import StrategyConfig
from qlab.strategies.portfolio import PortfolioSpec, build_decisions
from qlab.strategies.signals import Signal, build_signal, market_trend, market_vol_scale

TREND_FILTER_SMA = 200
VOL_TARGET = 0.15
VOL_LOOKBACK = 63


@dataclass
class Context:
    panel: Panel
    universe: np.ndarray        # (T, N) bool, point in time
    spy: int                    # column of SPY in the panel
    first_decision: int = 0     # no decisions before this day
    cost_rate: np.ndarray | float = 0.0
    cash_ret: np.ndarray | None = None
    _signal_key: str | None = field(default=None, repr=False)
    _signal: Signal | None = field(default=None, repr=False)
    _vol: np.ndarray | None = field(default=None, repr=False)
    _overlays: dict = field(default_factory=dict, repr=False)

    def signal(self, cfg: StrategyConfig) -> Signal:
        if self._signal_key != cfg.signal_key:  # one signal in memory at a time
            self._signal = None
            self._signal = build_signal(self.panel, cfg.family, cfg.param_dict)
            self._signal_key = cfg.signal_key
        return self._signal

    def volatility(self) -> np.ndarray:
        if self._vol is None:
            self._vol = trailing_volatility(self.panel, VOL_LOOKBACK)
        return self._vol

    def overlay(self, name: str) -> np.ndarray | None:
        if name == "none":
            return None
        if name not in self._overlays:
            if name == "trend_filter":
                self._overlays[name] = market_trend(self.panel, self.spy, TREND_FILTER_SMA)
            elif name == "vol_target":
                self._overlays[name] = market_vol_scale(self.panel, self.spy, VOL_TARGET,
                                                        VOL_LOOKBACK)
            else:
                raise ValueError(f"unknown overlay {name!r}")
        return self._overlays[name]

    def decision_days(self, rebalance: str) -> np.ndarray:
        days = period_starts(self.panel.dates, {"weekly": "W", "monthly": "M"}[rebalance])
        days &= np.arange(len(days)) >= self.first_decision
        days[self.first_decision] = True  # start invested on the first allowed day
        return days


def decisions_for(cfg: StrategyConfig, ctx: Context) -> Decisions:
    days = ctx.decision_days(cfg.rebalance)
    if cfg.family == "regime_market":
        on = market_trend(ctx.panel, ctx.spy, cfg.param_dict["sma"])
        idx = np.flatnonzero(days)
        weights = np.zeros((len(idx), ctx.panel.shape[1]))
        weights[:, ctx.spy] = on[idx]
        return Decisions(idx, weights)
    spec = PortfolioSpec(cfg.top_n, cfg.weighting, cfg.hysteresis, cfg.max_weight)
    vol = ctx.volatility() if cfg.weighting == "inverse_vol" else None
    return build_decisions(ctx.signal(cfg), ctx.universe, days, spec, vol, ctx.overlay(cfg.overlay))


def run_candidate(cfg: StrategyConfig, ctx: Context) -> SimResult:
    return simulate(ctx.panel, decisions_for(cfg, ctx), ctx.cost_rate, ctx.cash_ret)


def dense_targets(decisions: Decisions, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, np.nan)
    out[decisions.days] = decisions.weights
    return out
