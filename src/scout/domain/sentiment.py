from dataclasses import dataclass
from datetime import datetime

from scout.domain._checks import require_aware


@dataclass(frozen=True, slots=True)
class SentimentObservation:
    """One raw, timestamped observation. Point-in-time discipline lives in the
    two timestamp fields and nowhere else.
    """

    symbol: str
    source: str  # e.g. "vix_term", "short_interest"
    event_ts: datetime  # when the event/publication happened
    available_ts: datetime  # when WE could first have known it. MUST be >= event_ts.
    raw_score: float  # source-native
    score: float  # normalised to [-1, 1]
    source_weight: float  # (0, 1], static per source, from config
    sample_size: int  # articles/posts behind this observation
    payload_hash: str  # dedupe key

    def __post_init__(self) -> None:
        require_aware(self.event_ts, "event_ts")
        require_aware(self.available_ts, "available_ts")
        if self.available_ts < self.event_ts:
            raise ValueError("available_ts must be >= event_ts")


@dataclass(frozen=True, slots=True)
class SentimentView:
    """Aggregated sentiment for one symbol as of one decision timestamp.
    Built only from observations with available_ts <= ts.
    """

    symbol: str
    ts: datetime
    score: float  # [-1, 1], decay-weighted
    confidence: float  # [0, 1], from sample size, freshness, agreement
    sample_size: int
    freshness_bars: float  # bars since the newest contributing observation
    source_count: int
    disagreement: float  # [0, 1]: weighted stdev across sources
    is_stale: bool  # freshness_bars > config max; then treat neutral

    def __post_init__(self) -> None:
        require_aware(self.ts, "ts")

    @property
    def is_neutral(self) -> bool:
        """True when stale, empty, or confidence below the config floor. A
        neutral view produces a multiplier of exactly 1.0.

        The config floor is applied by the aggregator (M3.3) when it sets
        `is_stale`. This property has no config access, so it treats stale or
        empty (sample_size == 0 or source_count == 0) as neutral.
        """
        return self.is_stale or self.sample_size <= 0 or self.source_count == 0
