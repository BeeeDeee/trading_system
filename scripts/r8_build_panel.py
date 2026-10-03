"""Research 8: FRED T-bill + raw Binance files -> data/derived/<snap>/r8_panel.npz.

    uv run python scripts/r8_build_panel.py binance_2026-10-03
"""

import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

import polars as pl

from qlab.research8 import panel as P
from qlab.research8.setup import HOLDOUT, WARMUP_START, panel_path

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"

snapshot = sys.argv[1]
raw = Path("data/raw") / snapshot
manifest = json.loads((raw / "manifest.json").read_text())
fred_path = raw / "fred_DTB3.csv"
if not fred_path.exists():
    blob = urllib.request.urlopen(FRED, timeout=60).read()
    fred_path.write_bytes(blob)
    (raw / "fred_DTB3.sha256").write_text(hashlib.sha256(blob).hexdigest() + "\n")
fred = pl.read_csv(io.BytesIO(fred_path.read_bytes()), infer_schema_length=0)
pairs = {p: tuple(v) for p, v in manifest["pairs"].items() if v}
p = P.build(raw, pairs, fred, WARMUP_START, HOLDOUT[1])
P.save(p, panel_path(snapshot))
print(f"panel {len(p.dates)} days x {len(p.symbols)} symbols -> {panel_path(snapshot)}")
