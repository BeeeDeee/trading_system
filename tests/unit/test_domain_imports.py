"""Import-level checks for the minimal backend skeleton."""

from app.domain.models.enums import RiskDecisionStatus
from app.domain.ports.trading import ExecutionPort, OrderPlanner, RiskManager


def test_core_contracts_are_importable() -> None:
    assert RiskDecisionStatus.APPROVE == "approve"
    assert ExecutionPort is not None
    assert OrderPlanner is not None
    assert RiskManager is not None
