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


def payload_errors(msg_type: str, payload: dict) -> list[str]:
    defs = _schema("messages")["$defs"]
    if msg_type not in defs:
        return [f"unknown message type {msg_type!r}"]
    return _errors(defs[msg_type], payload)


def outbox_errors(outbox: dict) -> list[str]:
    return _errors(_schema("outbox"), outbox)
