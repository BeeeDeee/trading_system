"""Explicit strategy factory lookup. No entry points, no module scanning."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from scout.config.schema import StrategyConfig
from scout.domain.ports import Strategy
from scout.strategies.donchian_breakout import DonchianBreakout
from scout.strategies.xsec_momentum import XSecMomentum
from scout.utils.errors import ScoutConfigError

STRATEGY_FACTORIES: dict[str, Callable[[Mapping[str, Any]], Strategy]] = {
    "xsec_momentum_v1": XSecMomentum.from_params,
    "donchian_breakout_v1": DonchianBreakout.from_params,
}


def build_strategies(cfg: Sequence[StrategyConfig]) -> tuple[Strategy, ...]:
    """Explicit dict lookup. An unknown id raises at construction time."""
    out: list[Strategy] = []
    for sc in cfg:
        if not sc.enabled:
            continue
        factory = STRATEGY_FACTORIES.get(sc.strategy_id)
        if factory is None:
            raise ScoutConfigError(
                f"unknown strategy_id {sc.strategy_id!r}; "
                f"known: {sorted(STRATEGY_FACTORIES)}"
            )
        out.append(factory(sc.params))
    return tuple(out)
