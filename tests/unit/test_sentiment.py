"""Sentiment plumbing: null source, aggregation, penalty-only multiplier."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scout.config.schema import SentimentConfig
from scout.domain.enums import Direction
from scout.domain.ports import SentimentSource
from scout.domain.sentiment import SentimentObservation, SentimentView
from scout.sentiment.aggregate import _observation_weight, build_view
from scout.sentiment.multiplier import sentiment_multiplier
from scout.sentiment.null_source import NullSentimentSource

TS = datetime(2015, 6, 15, 20, 0, tzinfo=UTC)


def _obs(**overrides: Any) -> SentimentObservation:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "source": "vix_term",
        "event_ts": TS - timedelta(hours=1),
        "available_ts": TS,
        "raw_score": 0.0,
        "score": 0.0,
        "source_weight": 1.0,
        "sample_size": 10,
        "payload_hash": "h0",
    }
    kwargs.update(overrides)
    return SentimentObservation(**kwargs)


def _view(**overrides: Any) -> SentimentView:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "ts": TS,
        "score": -0.30,
        "confidence": 0.50,
        "sample_size": 10,
        "freshness_bars": 0.0,
        "source_count": 1,
        "disagreement": 0.0,
        "is_stale": False,
    }
    kwargs.update(overrides)
    return SentimentView(**kwargs)


def _enabled(**overrides: Any) -> SentimentConfig:
    return SentimentConfig(enabled=True, **overrides)


def test_null_source_returns_empty() -> None:
    src = NullSentimentSource()
    assert src.source_id == "null"
    assert src.observations(("AAPL", "MSFT"), TS, TS) == ()
    assert isinstance(src, SentimentSource)
    assert SentimentSource not in NullSentimentSource.__mro__


def test_no_observations_is_neutral() -> None:
    view = build_view((), "AAPL", TS, _enabled())
    assert view.is_neutral is True
    assert view.score == 0.0
    assert view.source_count == 0
    assert sentiment_multiplier(view, Direction.LONG, _enabled()) == 1.0


def test_disabled_is_neutral() -> None:
    view = _view(score=-0.80, confidence=0.90)
    assert sentiment_multiplier(view, Direction.LONG, SentimentConfig(enabled=False)) == 1.0


def test_stale_view_is_neutral() -> None:
    view = _view(score=-0.80, confidence=0.90, is_stale=True)
    assert view.is_neutral is True
    assert sentiment_multiplier(view, Direction.LONG, _enabled()) == 1.0


def test_aligned_sentiment_gives_no_bonus() -> None:
    cfg = _enabled()
    long_view = _view(score=0.60, confidence=0.80)
    assert sentiment_multiplier(long_view, Direction.LONG, cfg) == 1.0
    short_view = _view(score=-0.80, confidence=0.90)
    assert sentiment_multiplier(short_view, Direction.SHORT, cfg) == 1.0


def test_opposed_sentiment_penalises() -> None:
    # 10-SENTIMENT.md §4: LONG, score=-0.30, confidence=0.50
    # severity = 0.30 * 0.50 = 0.15; 1 - 0.85 * 0.15 = 0.8725
    view = _view(score=-0.30, confidence=0.50)
    assert sentiment_multiplier(view, Direction.LONG, _enabled()) == pytest.approx(0.8725)


def test_severe_opposition_vetoes() -> None:
    # 10-SENTIMENT.md §4: LONG, score=-0.80, confidence=0.90
    # severity = 0.72 >= veto_threshold 0.70
    view = _view(score=-0.80, confidence=0.90)
    assert sentiment_multiplier(view, Direction.LONG, _enabled()) == 0.0


def test_decay_halves_at_half_life() -> None:
    cfg = SentimentConfig()
    now = _obs(payload_hash="now")
    aged = _obs(
        event_ts=TS - timedelta(hours=cfg.half_life_hours + 1),
        available_ts=TS - timedelta(hours=cfg.half_life_hours),
        payload_hash="aged",
    )
    w_now = _observation_weight(now, TS, cfg)
    w_aged = _observation_weight(aged, TS, cfg)
    assert w_aged == pytest.approx(0.5 * w_now)

    # Equal weights 1 : 0.5 on scores 1 and 0 ⇒ score = 1 / 1.5
    mixed = build_view(
        (
            _obs(score=1.0, payload_hash="now"),
            _obs(
                event_ts=TS - timedelta(hours=cfg.half_life_hours + 1),
                available_ts=TS - timedelta(hours=cfg.half_life_hours),
                score=0.0,
                payload_hash="aged",
            ),
        ),
        "AAPL",
        TS,
        cfg,
    )
    assert mixed.score == pytest.approx(2.0 / 3.0)


def test_min_multiplier_floor() -> None:
    # Defaults never bind min_multiplier (veto fires first). Force the floor.
    cfg = _enabled(veto_threshold=1.0, penalty_slope=2.0, min_multiplier=0.40)
    view = _view(score=-1.0, confidence=0.50)  # severity 0.50; 1 - 2*0.50 = 0
    assert sentiment_multiplier(view, Direction.LONG, cfg) == pytest.approx(0.40)


@given(
    score=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
    confidence=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    sample_size=st.integers(0, 500),
    source_count=st.integers(0, 8),
    is_stale=st.booleans(),
    enabled=st.booleans(),
    direction=st.sampled_from([Direction.LONG, Direction.SHORT]),
    veto=st.floats(0.05, 1.0, allow_nan=False, allow_infinity=False),
    slope=st.floats(0.0, 3.0, allow_nan=False, allow_infinity=False),
    min_mult=st.floats(0.01, 0.99, allow_nan=False, allow_infinity=False),
)
def test_multiplier_range(
    score: float,
    confidence: float,
    sample_size: int,
    source_count: int,
    is_stale: bool,
    enabled: bool,
    direction: Direction,
    veto: float,
    slope: float,
    min_mult: float,
) -> None:
    cfg = SentimentConfig(
        enabled=enabled,
        veto_threshold=veto,
        penalty_slope=slope,
        min_multiplier=min_mult,
    )
    view = _view(
        score=score,
        confidence=confidence,
        sample_size=sample_size,
        source_count=source_count,
        is_stale=is_stale,
    )
    mult = sentiment_multiplier(view, direction, cfg)
    assert 0.0 <= mult <= 1.0
    assert sentiment_multiplier(None, direction, cfg) == 1.0
