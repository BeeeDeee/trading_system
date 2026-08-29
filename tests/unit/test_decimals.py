from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scout.utils.decimals import (
    CENTS,
    round_entry,
    round_qty,
    round_stop,
    round_target,
    round_usd,
    to_decimal,
)
from scout.utils.errors import ScoutError

STEPS = (
    Decimal("1"),
    Decimal("0.1"),
    Decimal("0.01"),
    Decimal("0.001"),
    Decimal("0.0001"),
)


def test_to_decimal_via_str_not_binary_float() -> None:
    assert to_decimal(0.1) == Decimal("0.1")
    assert to_decimal(Decimal("1.25")) == Decimal("1.25")
    assert to_decimal(3) == Decimal("3")
    assert to_decimal("0.01") == Decimal("0.01")


def test_round_qty_never_rounds_up() -> None:
    assert round_qty(Decimal("1.9"), Decimal("1")) == Decimal("1")
    assert round_qty(Decimal("1.0"), Decimal("1")) == Decimal("1")
    assert round_qty(Decimal("0.3"), Decimal("1")) == Decimal("0")
    assert round_qty(Decimal("1.234"), Decimal("0.01")) == Decimal("1.23")


@given(
    qty=st.decimals(
        min_value=Decimal("0"),
        max_value=Decimal("1000000"),
        allow_nan=False,
        allow_infinity=False,
        places=8,
    ),
    step=st.sampled_from(STEPS),
)
def test_step_rounding_always_down(qty: Decimal, step: Decimal) -> None:
    out = round_qty(qty, step)
    assert out <= qty
    assert (out / step) == (out / step).to_integral_value()
    assert out + step > qty


def test_round_stop_away_from_entry() -> None:
    tick = Decimal("0.01")
    assert round_stop(Decimal("10.004"), tick, is_long=True) == Decimal("10.00")
    assert round_stop(Decimal("10.001"), tick, is_long=False) == Decimal("10.01")
    assert round_stop(Decimal("10.00"), tick, is_long=True) == Decimal("10.00")


def test_round_entry_conservative() -> None:
    tick = Decimal("0.01")
    assert round_entry(Decimal("10.001"), tick, is_long=True) == Decimal("10.01")
    assert round_entry(Decimal("10.009"), tick, is_long=False) == Decimal("10.00")


def test_round_target_conservative() -> None:
    tick = Decimal("0.01")
    assert round_target(Decimal("12.009"), tick, is_long=True) == Decimal("12.00")
    assert round_target(Decimal("8.001"), tick, is_long=False) == Decimal("8.01")


def test_round_usd_down_to_cents() -> None:
    assert Decimal("0.01") == CENTS
    assert round_usd(Decimal("12.349")) == Decimal("12.34")
    assert round_usd(Decimal("12.34")) == Decimal("12.34")


def test_round_qty_rejects_negative() -> None:
    with pytest.raises(ScoutError, match="non-negative"):
        round_qty(Decimal("-1"), Decimal("1"))


def test_round_qty_rejects_non_positive_step() -> None:
    with pytest.raises(ScoutError, match="positive"):
        round_qty(Decimal("1"), Decimal("0"))
