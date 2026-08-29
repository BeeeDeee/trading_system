from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, TextIO

_OPTIONAL_FIELDS: tuple[str, ...] = (
    "decision_ts",
    "symbol",
    "strategy_id",
    "correlation_id",
)


class RunIdFilter(logging.Filter):
    """Attach `run_id` to every record so the JSON formatter can emit it."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


class JsonFormatter(logging.Formatter):
    """JSON-lines formatter. Every record carries `run_id`."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": getattr(record, "run_id", None),
        }
        for name in _OPTIONAL_FIELDS:
            value = getattr(record, name, None)
            if value is None:
                continue
            if isinstance(value, datetime):
                payload[name] = value.isoformat()
            else:
                payload[name] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(
    run_id: str,
    *,
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure the `scout` logger with JSON lines and a fixed `run_id`."""
    logger = logging.getLogger("scout")
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RunIdFilter(run_id))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
