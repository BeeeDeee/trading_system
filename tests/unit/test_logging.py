from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from io import StringIO

from scout.utils.logging import JsonFormatter, RunIdFilter, configure_logging


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="scout.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_run_id() -> None:
    payload = json.loads(JsonFormatter().format(_record(run_id="abc123")))
    assert payload["run_id"] == "abc123"
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "scout.test"
    assert "ts" in payload


def test_json_formatter_omits_absent_optional_fields() -> None:
    payload = json.loads(JsonFormatter().format(_record(run_id="r1")))
    assert "symbol" not in payload
    assert "decision_ts" not in payload
    assert "strategy_id" not in payload
    assert "correlation_id" not in payload


def test_json_formatter_includes_optional_fields() -> None:
    ts = datetime(2021, 3, 4, 20, 0, tzinfo=UTC)
    payload = json.loads(
        JsonFormatter().format(
            _record(
                run_id="r1",
                decision_ts=ts,
                symbol="AAPL",
                strategy_id="xsec_momentum_v1",
                correlation_id="c-9",
            )
        )
    )
    assert payload["decision_ts"] == ts.isoformat()
    assert payload["symbol"] == "AAPL"
    assert payload["strategy_id"] == "xsec_momentum_v1"
    assert payload["correlation_id"] == "c-9"


def test_configure_logging_stamps_run_id() -> None:
    stream = StringIO()
    logger = configure_logging("run-42", stream=stream)
    logger.info("accepted")
    line = stream.getvalue().strip()
    payload = json.loads(line)
    assert payload["run_id"] == "run-42"
    assert payload["msg"] == "accepted"


def test_run_id_filter_overwrites_record() -> None:
    record = _record(run_id="old")
    assert RunIdFilter("new").filter(record) is True
    assert record.run_id == "new"  # type: ignore[attr-defined]
