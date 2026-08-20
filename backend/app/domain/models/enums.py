"""Domain enumerations."""

from enum import Enum


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class RiskDecisionStatus(str, Enum):
    APPROVE = "approve"
    MODIFY = "modify"
    REJECT = "reject"
    HALT = "halt"


class ExecutionStatus(str, Enum):
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class RuntimeCommandType(str, Enum):
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    HALT = "halt"
    RECONCILE = "reconcile"


class CommandStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"


class RuntimeState(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    HALTED = "halted"
