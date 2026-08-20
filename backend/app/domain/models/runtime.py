"""Runtime command and pipeline result models."""

from dataclasses import dataclass, field
from datetime import datetime

from app.domain.models.enums import CommandStatus, RuntimeCommandType


@dataclass(frozen=True, slots=True)
class RuntimeCommand:
    command_id: str
    type: RuntimeCommandType
    timestamp: datetime
    requested_by: str
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    processed_at: datetime
    generated_order_intent_ids: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeCommandResult:
    command_id: str
    status: CommandStatus
    message: str | None = None
