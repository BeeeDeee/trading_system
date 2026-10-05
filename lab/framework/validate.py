"""JSON Schema validation of cards, message payloads and outboxes. Errors are returned, not raised,
so they can be sent back to the agent as feedback."""

import json
from functools import cache
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMAS = Path(__file__).parent / "schemas"
SCHEMA_VERSION = 1


@cache
def _schema(name: str) -> dict:
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text())


def _errors(schema: dict, obj) -> list[str]:
    v = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in sorted(v.iter_errors(obj), key=lambda e: list(map(str, e.absolute_path)))]


def card_errors(card: dict) -> list[str]:
    return _errors(_schema("hypothesis"), card)


def specified_missing(card: dict) -> list[str]:
    """Fields required for SPECIFIED that are missing or empty."""
    required = _schema("hypothesis")["x-specified-required"]
    return [f for f in required if not card.get(f)]


def grid_problems(card: dict, both_sides: bool = False) -> list[str]:
    """G2 tests one grid step each way around the primary value, so the grid must allow that: the value is
    in the grid, a numeric grid is strictly increasing, and there is at least one neighbor. `both_sides`
    also asks for a neighbor on each side of a numeric value (advice: a value at the edge of its grid, such as
    a weight of 1.0 under gross <= 1, is tested on one side only)."""
    out = []
    for name, p in (card.get("signal", {}).get("params") or {}).items():
        grid, value = p.get("grid", []), p.get("value")
        if value not in grid:
            out.append(f"param {name}: value {value!r} is not in its grid {grid}")
            continue
        numeric = all(isinstance(g, (int, float)) and not isinstance(g, bool) for g in grid)
        if numeric and any(b <= a for a, b in zip(grid, grid[1:])):
            out.append(f"param {name}: numeric grid must be strictly increasing, got {grid}")
        elif both_sides and numeric and grid.index(value) in (0, len(grid) - 1):
            out.append(f"param {name}: value {value} needs a grid neighbor on each side, got {grid}")
        elif len(grid) < 2:
            out.append(f"param {name}: grid needs at least one alternative to {value!r}")
    return out


def payload_errors(msg_type: str, payload: dict) -> list[str]:
    defs = _schema("messages")["$defs"]
    if msg_type not in defs:
        return [f"unknown message type {msg_type!r}"]
    return _errors(defs[msg_type], payload)


def outbox_errors(outbox: dict) -> list[str]:
    return _errors(_schema("outbox"), outbox)
