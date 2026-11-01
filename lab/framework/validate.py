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


def ml_problems(card: dict) -> list[str]:
    """Checks of the `ml` block that need code, not schema: the model can run here, the refit schedule fits the
    compute budget, daily horizon only, and the hyperparameters in signal.params exist for the model."""
    ml = card.get("ml")
    if not ml:
        return []
    from lab.framework import ml as mlmod
    out = []
    model = ml["model"]
    if not mlmod.available().get(model, False):
        out.append(f"ml.model {model}: torch is not installed here (extra `deep`); choose ridge, logit, rf, gbm or mlp")
    if {"5m", "15m"} & set((card.get("signal") or {}).get("timeframes") or []):
        out.append("ml cards are daily: the intraday branch has no ml contract")
    every = int(ml["retrain_every_days"])
    if model in mlmod.DEEP and every < 21:
        out.append("ml.retrain_every_days >= 21 for lstm / tcn")
    n = len(mlmod.fit_schedule(6500, every, 504, int(ml["horizon_days"])))
    limit = mlmod.MAX_FITS[mlmod.kind(model)]
    if n > limit:
        out.append(f"ml: refit every {every} rows means about {n} fits over 25 years; the limit for {model} is {limit} "
                   f"(raise retrain_every_days to at least {int(6500 / limit) + 1})")
    if int(ml["horizon_days"]) > 60:
        out.append("ml.horizon_days <= 60")
    hh = (card.get("holding_period") or {}).get("typical_days")
    if hh and ml["horizon_days"] > 3 * float(hh) + 5:
        out.append(f"ml.horizon_days {ml['horizon_days']} is far longer than holding_period.typical_days {hh}: the label "
                   "should match what the position earns")
    return out
