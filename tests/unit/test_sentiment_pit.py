"""Point-in-time discipline for sentiment. Filter on available_ts, never event_ts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scout.config.schema import SentimentConfig
from scout.domain.sentiment import SentimentObservation
from scout.sentiment.aggregate import build_view

T = datetime(2015, 6, 15, 14, 0, tzinfo=UTC)


def test_filters_on_available_ts_not_event_ts() -> None:
    """Observation published at T, first knowable at T+6h. A decision at T+1h sees nothing."""
    obs = SentimentObservation(
        symbol="AAPL",
        source="vix_term",
        event_ts=T,
        available_ts=T + timedelta(hours=6),
        raw_score=-1.0,
        score=-1.0,
        source_weight=1.0,
        sample_size=10,
        payload_hash="lagged",
    )
    cfg = SentimentConfig()
    too_early = build_view((obs,), "AAPL", T + timedelta(hours=1), cfg)
    assert too_early.source_count == 0
    assert too_early.sample_size == 0
    assert too_early.is_neutral is True

    # Same observation is visible once available_ts is reached (n_eff floors may
    # still mark the view stale; membership is what this test locks).
    known = build_view((obs,), "AAPL", T + timedelta(hours=6), cfg)
    assert known.source_count == 1
    assert known.sample_size == 10
    assert known.score == pytest.approx(-1.0)


def test_available_ts_equal_to_decision_ts_is_visible() -> None:
    obs = SentimentObservation(
        symbol="AAPL",
        source="vix_term",
        event_ts=T,
        available_ts=T,
        raw_score=0.5,
        score=0.5,
        source_weight=1.0,
        sample_size=10,
        payload_hash="on-bar",
    )
    view = build_view((obs,), "AAPL", T, SentimentConfig())
    assert view.source_count == 1
    assert view.score == pytest.approx(0.5)
