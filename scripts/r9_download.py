"""Research 9: daily klines of every USDT spot pair ever listed on Binance (prereg §3).

    uv run python scripts/r9_download.py binance_2026-10-03 [--last-month 2026-08]

Idempotent (files from research 8 are reused). Writes spot_manifest_r9.json next to the raw files.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from qlab.research8 import download as dl
from qlab.research9.data import tradable_spot

ap = argparse.ArgumentParser()
ap.add_argument("snapshot")
ap.add_argument("--last-month", default="2026-08")
ap.add_argument("--workers", type=int, default=24)
a = ap.parse_args()
raw = Path("data/raw") / a.snapshot
all_spot = set(dl.list_symbols("spot_1d"))
syms = tradable_spot(all_spot)
print(f"spot symbols: {len(all_spot)}, USDT pairs kept: {len(syms)}")
res = dl.fetch_many("spot_1d", syms, raw, a.last_month, a.workers)
errors = [r for r in res if "error" in r]
(raw / "spot_manifest_r9.json").write_text(json.dumps({
    "snapshot": a.snapshot, "last_month": a.last_month, "created_at": datetime.now(timezone.utc).isoformat(),
    "symbols": syms, "files": sum(r.get("files", 0) for r in res), "errors": errors}, indent=1))
print(f"files: {sum(r.get('files', 0) for r in res)}, errors: {len(errors)}")
sys.exit(1 if errors else 0)
