"""Research 15: 15-minute klines from data.binance.vision (prereg §3).

    uv run python scripts/r15_download.py binance_2026-10-03 [--last-month 2026-09]

Primary: spot and perp 15m of BTCUSDT and ETHUSDT. Secondary: perp 15m of every symbol that was ever in the
monthly point-in-time top 10 (qlab.research15.universe) from 2020-03 to the last month. Membership is computed
from the daily perp candles already in data/raw/<snap>/perp_1d. Idempotent; every zip is checked against its
CHECKSUM. Writes r15_manifest.json (members per month, file counts, errors) next to the raw files.
"""

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from qlab.research8 import download as dl
from qlab.research15 import universe as uni

PRIMARY = ["BTCUSDT", "ETHUSDT"]

ap = argparse.ArgumentParser()
ap.add_argument("snapshot")
ap.add_argument("--last-month", default="2026-09")
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
raw = Path("data/raw") / a.snapshot

spot_symbols = {p.name for p in (raw / "spot_1d").iterdir()} | set(dl.list_symbols("spot_1d"))
perps = uni.eligible(sorted(p.name for p in (raw / "perp_1d").iterdir()), spot_symbols)
print(f"eligible crypto perps: {len(perps)}")
volumes = {s: uni.daily_quote_volume(raw / "perp_1d" / s) for s in perps}
y, m = map(int, a.last_month.split("-"))
months = [date(yy, mm, 1) for yy in range(2020, y + 1) for mm in range(1, 13)
          if (yy, mm) >= (2020, 3) and (yy, mm) <= (y, m)]
members = uni.monthly_members(volumes, months)
secondary = sorted({s for v in members.values() for s in v})
print(f"secondary basket: {len(secondary)} symbols over {len(months)} months")

res = dl.fetch_many("spot_15m", PRIMARY, raw, a.last_month, a.workers)
res += dl.fetch_many("perp_15m", PRIMARY + secondary, raw, a.last_month, a.workers)
errors = [r for r in res if "error" in r]
(raw / "r15_manifest.json").write_text(json.dumps({
    "snapshot": a.snapshot, "last_month": a.last_month, "created_at": datetime.now(timezone.utc).isoformat(),
    "rule": {"top_n": uni.TOP_N, "window_days": uni.WINDOW_DAYS, "min_history_days": uni.MIN_HISTORY_DAYS,
             "excluded": sorted(uni.EXCLUDED)},
    "primary": PRIMARY, "secondary": secondary, "members": members,
    "files": {f"{r['kind']}/{r['symbol']}": r.get("files", 0) for r in res}, "errors": errors}, indent=1))
print(f"files: {sum(r.get('files', 0) for r in res)}, new: {sum(r.get('new', 0) for r in res)}, errors: {len(errors)}")
sys.exit(1 if errors else 0)
