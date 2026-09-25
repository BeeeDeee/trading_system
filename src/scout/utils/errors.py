class ScoutError(Exception):
    """Base error for the scout package."""


class ScoutConfigError(ScoutError):
    """Invalid configuration; fail at startup."""


class ScoutDataError(ScoutError):
    """Missing, corrupt, or stale data."""


class ScoutExecutionError(ScoutError):
    """Broker rejected the order or was unreachable."""


class ScoutLookaheadError(ScoutError):
    """A causality assertion tripped. Must never be caught."""
