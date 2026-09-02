"""Penalty-only size multiplier. Never a bonus, never creates a trade."""

from __future__ import annotations

from scout.config.schema import SentimentConfig
from scout.domain.enums import Direction
from scout.domain.sentiment import SentimentView


def sentiment_multiplier(
    view: SentimentView | None,
    direction: Direction,
    cfg: SentimentConfig,
) -> float:
    """Size penalty in [0.0, 1.0].

    1.0        = no effect (neutral, disabled, missing, or aligned)
    (0.0, 1.0) = size penalty
    0.0        = veto (caller rejects with SENTIMENT_VETO)
    """
    if not cfg.enabled or view is None or view.is_neutral:
        return 1.0

    aligned = view.score * direction.sign
    if aligned >= 0.0:
        return 1.0

    severity = abs(aligned) * view.confidence
    if severity >= cfg.veto_threshold:
        return 0.0
    return max(cfg.min_multiplier, 1.0 - cfg.penalty_slope * severity)
