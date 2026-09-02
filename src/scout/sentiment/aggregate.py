"""Aggregate point-in-time observations into a SentimentView."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta

from scout.config.schema import SentimentConfig
from scout.domain.sentiment import SentimentObservation, SentimentView
from scout.utils.errors import ScoutLookaheadError

# v1 is daily. M7's 4h clock must use a 4-hour duration when that task lands.
_BAR_DURATION = timedelta(days=1)


def build_view(
    observations: Sequence[SentimentObservation],
    symbol: str,
    ts: datetime,
    cfg: SentimentConfig,
) -> SentimentView:
    """Decay-weighted view of observations knowable at `ts` for `symbol`.

    Filters on `available_ts`, never `event_ts`. Empty input is a neutral view.
    """
    contributing: list[tuple[SentimentObservation, float]] = []
    for obs in observations:
        if obs.symbol != symbol:
            continue
        if obs.available_ts > ts:
            continue
        age_hours = (ts - obs.available_ts).total_seconds() / 3600.0
        if age_hours > cfg.max_age_hours:
            continue
        weight = _observation_weight(obs, ts, cfg)
        if weight <= 0.0:
            continue
        contributing.append((obs, weight))

    for obs, _weight in contributing:
        if obs.available_ts > ts:
            raise ScoutLookaheadError(
                f"contributing observation {obs.payload_hash!r} has "
                f"available_ts {obs.available_ts.isoformat()} > ts {ts.isoformat()}"
            )

    if not contributing:
        return _empty_view(symbol, ts)

    weights = [w for _obs, w in contributing]
    scores = [obs.score for obs, _w in contributing]
    sum_w = sum(weights)
    score = sum(s * w for s, w in zip(scores, weights, strict=True)) / sum_w

    sum_w2 = sum(w * w for w in weights)
    n_eff = (sum_w * sum_w) / sum_w2
    sample_confidence = min(1.0, n_eff / cfg.n_eff_full)

    per_source_scores, per_source_weights = _per_source_means(contributing)
    disagreement = min(1.0, max(0.0, _weighted_stdev(per_source_scores, per_source_weights)))
    agreement_confidence = 1.0 - disagreement

    newest = max(obs.available_ts for obs, _w in contributing)
    freshness_bars = (ts - newest) / _BAR_DURATION
    freshness_confidence = 0.5 ** (freshness_bars / cfg.freshness_half_life_bars)

    confidence = sample_confidence * agreement_confidence * freshness_confidence
    is_stale = (
        freshness_bars > cfg.max_freshness_bars
        or n_eff < cfg.min_n_eff
        or confidence < cfg.min_confidence
    )
    return SentimentView(
        symbol=symbol,
        ts=ts,
        score=score,
        confidence=confidence,
        sample_size=sum(obs.sample_size for obs, _w in contributing),
        freshness_bars=freshness_bars,
        source_count=len({obs.source for obs, _w in contributing}),
        disagreement=disagreement,
        is_stale=is_stale,
    )


def _observation_weight(
    obs: SentimentObservation,
    ts: datetime,
    cfg: SentimentConfig,
) -> float:
    age_hours = (ts - obs.available_ts).total_seconds() / 3600.0
    if age_hours < 0.0 or age_hours > cfg.max_age_hours:
        return 0.0
    decay = 0.5 ** (age_hours / cfg.half_life_hours)
    return float(decay * obs.source_weight * math.log1p(obs.sample_size))


def _empty_view(symbol: str, ts: datetime) -> SentimentView:
    return SentimentView(
        symbol=symbol,
        ts=ts,
        score=0.0,
        confidence=0.0,
        sample_size=0,
        freshness_bars=0.0,
        source_count=0,
        disagreement=0.0,
        is_stale=False,
    )


def _per_source_means(
    contributing: Sequence[tuple[SentimentObservation, float]],
) -> tuple[list[float], list[float]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for obs, weight in contributing:
        grouped[obs.source].append((obs.score, weight))
    means: list[float] = []
    totals: list[float] = []
    for pairs in grouped.values():
        src_scores = [score for score, _w in pairs]
        src_weights = [weight for _score, weight in pairs]
        means.append(_weighted_mean(src_scores, src_weights))
        totals.append(sum(src_weights))
    return means, totals


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights, strict=True)) / total


def _weighted_stdev(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _weighted_mean(values, weights)
    total = sum(weights)
    var = sum(w * (v - mu) ** 2 for v, w in zip(values, weights, strict=True)) / total
    return math.sqrt(max(var, 0.0))
