from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scout.config.schema import ScoutConfig


def read_snapshot_id(path: str | Path) -> str:
    snapshot_path = Path(path)
    if not snapshot_path.is_file():
        return ""
    raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    value = raw.get("data_snapshot_id")
    if value is None:
        return ""
    return str(value)


def config_hash(cfg: ScoutConfig) -> str:
    payload = cfg.model_dump(mode="json", exclude={"run": {"mode"}})
    payload["data_snapshot_id"] = read_snapshot_id(cfg.data.snapshot_id_path)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
