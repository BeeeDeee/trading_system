from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_UP, Decimal

from scout.utils.errors import ScoutError

CENTS = Decimal("0.01")


def to_decimal(value: Decimal | float | int | str) -> Decimal:
    """Convert to Decimal. Floats go through str; `Decimal(0.1)` is not 0.1."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    return Decimal(str(value))


def round_qty(qty: Decimal, step_size: Decimal) -> Decimal:
    """Round quantity DOWN to a multiple of `step_size`. Never up into extra risk."""
    if qty < 0:
        raise ScoutError(f"qty must be non-negative, got {qty}")
    return _to_increment(qty, step_size, ROUND_DOWN)


def round_stop(price: Decimal, tick_size: Decimal, *, is_long: bool) -> Decimal:
    """Round a stop AWAY from entry: down for a long, up for a short."""
    rounding = ROUND_DOWN if is_long else ROUND_UP
    return _to_increment(price, tick_size, rounding)


def round_entry(price: Decimal, tick_size: Decimal, *, is_long: bool) -> Decimal:
    """Round an entry to the conservative side: up for a long, down for a short."""
    rounding = ROUND_UP if is_long else ROUND_DOWN
    return _to_increment(price, tick_size, rounding)


def round_target(price: Decimal, tick_size: Decimal, *, is_long: bool) -> Decimal:
    """Round a target to the conservative side: down for a long, up for a short."""
    rounding = ROUND_DOWN if is_long else ROUND_UP
    return _to_increment(price, tick_size, rounding)


def round_usd(amount: Decimal) -> Decimal:
    """Round a non-negative USD amount down to cents."""
    if amount < 0:
        raise ScoutError(f"round_usd requires a non-negative amount, got {amount}")
    return amount.quantize(CENTS, rounding=ROUND_DOWN)


def _to_increment(value: Decimal, increment: Decimal, rounding: str) -> Decimal:
    if increment <= 0:
        raise ScoutError(f"increment must be positive, got {increment}")
    if value < 0:
        raise ScoutError(f"price must be non-negative, got {value}")
    return (value / increment).to_integral_value(rounding) * increment
