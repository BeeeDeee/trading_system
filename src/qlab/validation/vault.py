"""OOS vault protecting the final holdout (spec §9.7).

Research code asks the vault for the last day it may see. Before the holdout can be opened the
methodology must be frozen (its hash written to a lock file); opening is logged and allowed once
per data snapshot. It is local code and can be bypassed deliberately, but not by accident and not
without a trace.
"""

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np


class VaultError(RuntimeError):
    pass


@dataclass(frozen=True)
class Vault:
    boundary: date            # last day of the development period
    lock_path: Path
    log_path: Path

    def last_visible_index(self, dates: np.ndarray) -> int:
        """Number of leading `dates` inside the development period."""
        return int(np.searchsorted(np.asarray(dates, dtype="datetime64[D]"),
                                   np.datetime64(self.boundary, "D"), side="right"))

    def check_dev_only(self, dates: np.ndarray) -> None:
        if len(dates) and np.asarray(dates, dtype="datetime64[D]").max() > np.datetime64(self.boundary):
            raise VaultError(f"data after the holdout boundary {self.boundary} requested outside "
                             "a final evaluation")

    def freeze(self, methodology_hash: str) -> None:
        if self.lock_path.exists():
            raise VaultError(f"methodology already frozen in {self.lock_path}")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text(json.dumps(
            {"methodology_hash": methodology_hash, "ts": datetime.now(timezone.utc).isoformat()}))

    def open_final(self, methodology_hash: str, snapshot_id: str, commit: str) -> None:
        """Authorize the one final evaluation of the frozen methodology on this snapshot."""
        if not self.lock_path.exists():
            raise VaultError("methodology is not frozen; call freeze() first")
        frozen = json.loads(self.lock_path.read_text())["methodology_hash"]
        if frozen != methodology_hash:
            raise VaultError(f"methodology hash {methodology_hash} differs from frozen {frozen}")
        entries = self.log()
        if any(e["snapshot_id"] == snapshot_id for e in entries):
            raise VaultError(f"holdout of snapshot {snapshot_id} was already opened")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                                "snapshot_id": snapshot_id, "commit": commit,
                                "methodology_hash": methodology_hash}) + "\n")

    def log(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text().splitlines() if line]
