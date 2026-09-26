"""Append-only registry of trials (spec §9.8).

Every run of candidates and every change of anything that affects selection is a trial. Rows cannot
be updated or deleted (enforced by SQLite triggers), so the counts used for the Deflated Sharpe
cannot shrink after the fact.
"""

import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

KINDS = ("candidates", "grid_change", "filter_change", "methodology_eval", "other")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    kind TEXT NOT NULL,
    n_configs INTEGER NOT NULL,
    note TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS trials_no_update BEFORE UPDATE ON trials
BEGIN SELECT RAISE(ABORT, 'trial registry is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trials_no_delete BEFORE DELETE ON trials
BEGIN SELECT RAISE(ABORT, 'trial registry is append-only'); END;
"""


def config_hash(config) -> str:
    """Stable hash of a JSON-serializable configuration (key order does not matter)."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def current_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class TrialRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def record(self, kind: str, config, n_configs: int = 1, note: str = "") -> int:
        if kind not in KINDS:
            raise ValueError(f"unknown trial kind {kind!r}")
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO trials (ts, git_commit, config_hash, kind, n_configs, note) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), current_commit(), config_hash(config),
                 kind, n_configs, note))
            return int(cur.lastrowid)

    def total_configs(self, kind: str | None = None) -> int:
        """Number of configurations tried (sum of n_configs), optionally for one kind."""
        query = "SELECT COALESCE(SUM(n_configs), 0) FROM trials"
        args: tuple = ()
        if kind:
            query, args = query + " WHERE kind = ?", (kind,)
        with self._connect() as con:
            return int(con.execute(query, args).fetchone()[0])

    @property
    def n_meth(self) -> int:
        """Methodology variants evaluated on the stitched WFO out-of-sample (DSR trials)."""
        return self.total_configs("methodology_eval")
