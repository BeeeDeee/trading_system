"""Normalize a Sharadar Parquet snapshot into bars (spec §4.4).

Usage: python scripts/build_bars.py data/parquet/sharadar_YYYY-MM-DD
Output: data/derived/<snapshot>/bars/part-*.parquet, delistings.parquet, sic_history.parquet
"""

import json
import sys
import time
from pathlib import Path

from qlab.data.sharadar import Snapshot, build_bars

src = Path(sys.argv[1])
out = Path("data/derived") / src.name / "bars"
t0 = time.time()
summary = build_bars(Snapshot(src), out)
summary["seconds"] = round(time.time() - t0, 1)
print(json.dumps(summary, indent=2))
