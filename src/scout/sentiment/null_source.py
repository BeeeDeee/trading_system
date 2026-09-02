"""Always-empty sentiment source. Neutral by construction."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from scout.domain.sentiment import SentimentObservation


class NullSentimentSource:
    """Returns no observations. Views built from it are neutral; multiplier is 1.0."""

    source_id: str = "null"

    def observations(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> tuple[SentimentObservation, ...]:
        del symbols, start, end
        return ()
