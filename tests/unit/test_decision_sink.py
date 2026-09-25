"""ParquetDecisionSink must keep a stable schema across flushes."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from scout.domain.audit import DecisionRecord
from scout.domain.enums import RejectionReason
from scout.storage.decision_sink import ParquetDecisionSink

TS = datetime(2015, 1, 5, 21, 0, tzinfo=UTC)


def _reject() -> DecisionRecord:
    return DecisionRecord(
        run_id="test",
        decision_ts=TS,
        symbol="AAA",
        strategy_id="xsec_momentum_v1",
        stage="GATE",
        accepted=False,
        rejection_reason=RejectionReason.NO_SETUP,
        direction=None,
        regime=None,
        vol_bucket=None,
        reference_price=None,
        stop_price=None,
        target_price=None,
        reward_risk_ratio=None,
        bin_key=None,
        bin_n=None,
        ev_r_point=None,
        ev_r_lcb=None,
        cost_r=None,
        ev_net_r=None,
        ev_per_bar_r=None,
        rank=None,
        sentiment_score=None,
        sentiment_confidence=None,
        sentiment_multiplier=None,
        portfolio_heat_pct=None,
        cluster=None,
        size_multiplier=None,
        qty=None,
        notional_usd=None,
        adv_usd_30=None,
        spread_bps_est=None,
        features_json=None,
    )


def test_parquet_sink_flush_all_null_then_qty(tmp_path: Path) -> None:
    path = tmp_path / "decisions.parquet"
    sink = ParquetDecisionSink(path, flush_every=1)
    sink.write([_reject()])
    sink.flush()
    sized = replace(_reject(), accepted=True, rejection_reason=None, qty=10.0, notional_usd=1000.0)
    sink.write([sized])
    sink.close()
    table = pq.read_table(path)
    assert table.num_rows == 2
    qty = table.column("qty").to_pylist()
    assert qty[0] is None
    assert qty[1] == 10.0
