"""The blackboard: one SQLite file (`lab.db`) shared by all agents through the framework.

Only framework code (running as the labcore user) writes it. Agents never open it: they get a rendered
workspace and leave an outbox that the framework applies (see `invocations.py`).

Append-only tables (`transitions`, `gate_results`, `trials`) are protected by triggers, the same way as
`qlab.validation.registry`. In `messages` only the delivery columns (`handled_*`, `reacted_at`) may change,
and each only once.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    family TEXT NOT NULL,
    title TEXT NOT NULL,
    author_agent TEXT NOT NULL,
    status TEXT NOT NULL,
    card_json TEXT NOT NULL,          -- card content of the current version (no status/history)
    card_sha256 TEXT NOT NULL,
    reject_stage TEXT,                -- set when status = REJECTED / PARKED / RETIRED
    reject_code TEXT,
    objection_rounds INTEGER NOT NULL DEFAULT 0,
    priority REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT NOT NULL REFERENCES hypotheses(id),
    version INTEGER NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    actor TEXT NOT NULL,
    invocation_id TEXT,
    message_id INTEGER,
    reason TEXT NOT NULL,
    ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    from_agent TEXT NOT NULL,
    to_agent TEXT NOT NULL,
    hypothesis_id TEXT,
    payload_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    invocation_id TEXT,               -- the agent run that sent it (NULL for framework messages)
    created_at TEXT NOT NULL,
    handled_at TEXT,                  -- consumed by the recipient (its inbox)
    handled_by TEXT,                  -- invocation id or framework component that consumed it
    reacted_at TEXT                   -- the framework applied the message's state effects (tick.py)
);
CREATE INDEX IF NOT EXISTS messages_inbox ON messages(to_agent, handled_at);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    gate TEXT NOT NULL,
    evaluator TEXT NOT NULL,          -- 'stub' only in the walking skeleton and tests
    code_sha256 TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,             -- running | done | error
    error TEXT
);

CREATE TABLE IF NOT EXISTS gate_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    hypothesis_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    gate TEXT NOT NULL,
    passed INTEGER NOT NULL,
    reason_code TEXT,
    metrics_json TEXT NOT NULL,
    thresholds_sha256 TEXT NOT NULL,
    evaluator TEXT NOT NULL,
    ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    config_sha256 TEXT NOT NULL,
    sharpe REAL,
    n_obs INTEGER,
    ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_invocations (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    model TEXT NOT NULL,
    hypothesis_id TEXT,
    workspace TEXT NOT NULL,
    input_message_ids TEXT NOT NULL,  -- JSON list rendered into the inbox
    input_sha256 TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    exit_code INTEGER,
    outcome TEXT,                     -- applied | outbox_rejected | failed | timeout
    error TEXT,
    n_turns INTEGER,
    tokens_in INTEGER,
    tokens_out INTEGER,
    transcript_path TEXT,
    output_json TEXT                  -- the outbox as applied (or as rejected)
);

CREATE TABLE IF NOT EXISTS budget (
    day TEXT NOT NULL,
    agent TEXT NOT NULL,
    planned INTEGER NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, agent)
);

CREATE TABLE IF NOT EXISTS locks (
    hypothesis_id TEXT PRIMARY KEY,
    invocation_id TEXT NOT NULL,
    lease_until TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS canary_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    framework_sha256 TEXT NOT NULL,   -- lab/framework + lab/canaries + gates.yaml
    ok INTEGER NOT NULL,
    detail_json TEXT NOT NULL,
    ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS counters (
    name TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
"""

APPEND_ONLY = ("transitions", "gate_results", "trials", "canary_runs")

_TRIGGERS = "".join(f"""
CREATE TRIGGER IF NOT EXISTS {t}_no_update BEFORE UPDATE ON {t}
BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;
CREATE TRIGGER IF NOT EXISTS {t}_no_delete BEFORE DELETE ON {t}
BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;
""" for t in APPEND_ONLY) + """
CREATE TRIGGER IF NOT EXISTS messages_no_delete BEFORE DELETE ON messages
BEGIN SELECT RAISE(ABORT, 'messages are append-only'); END;
CREATE TRIGGER IF NOT EXISTS messages_delivery_only BEFORE UPDATE ON messages
WHEN (OLD.handled_at IS NOT NULL AND (NEW.handled_at IS NOT OLD.handled_at OR NEW.handled_by IS NOT OLD.handled_by))
  OR (OLD.reacted_at IS NOT NULL AND NEW.reacted_at IS NOT OLD.reacted_at)
  OR NEW.type IS NOT OLD.type OR NEW.from_agent IS NOT OLD.from_agent OR NEW.to_agent IS NOT OLD.to_agent
  OR NEW.hypothesis_id IS NOT OLD.hypothesis_id OR NEW.payload_json IS NOT OLD.payload_json
  OR NEW.idempotency_key IS NOT OLD.idempotency_key OR NEW.created_at IS NOT OLD.created_at
BEGIN SELECT RAISE(ABORT, 'a message can only be marked handled/reacted, once each'); END;
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def dumps(obj) -> str:
    """Canonical JSON (sorted keys, no whitespace) used for hashes and stored payloads."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30, isolation_level=None)  # explicit transactions only
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")
    return con


def init(path: Path) -> sqlite3.Connection:
    con = connect(path)
    con.executescript(SCHEMA + _TRIGGERS)
    return con


class Tx:
    """`with Tx(con):` = BEGIN IMMEDIATE ... COMMIT, rollback on any exception. Nested use is a no-op."""

    def __init__(self, con: sqlite3.Connection):
        self.con, self.outer = con, False

    def __enter__(self):
        if not self.con.in_transaction:
            self.con.execute("BEGIN IMMEDIATE")
            self.outer = True
        return self.con

    def __exit__(self, exc_type, *_):
        if self.outer:
            self.con.execute("ROLLBACK" if exc_type else "COMMIT")
        return False


def next_id(con: sqlite3.Connection, name: str) -> int:
    with Tx(con):
        con.execute("INSERT INTO counters (name, value) VALUES (?, 1) "
                    "ON CONFLICT(name) DO UPDATE SET value = value + 1", (name,))
        return int(con.execute("SELECT value FROM counters WHERE name = ?", (name,)).fetchone()[0])
