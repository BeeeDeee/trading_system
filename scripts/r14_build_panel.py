"""Research 14: hourly panel of BTC/ETH (+ secondary basket) from Binance 1h zips (prereg §3).

    uv run python scripts/r14_build_panel.py binance_2026-10-03

Reads the raw zips of market_relations_2026_oct read-only, writes data/derived/<snap>/r14_1h/<SYM>.npz and
prints an audit (gaps, duplicates, OHLC consistency, SHA256 of the zip set) as JSON.
"""

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from qlab.research14.setup import PRIMARY, RAW_1H, SECONDARY, hour_grid, panel_dir

snapshot = sys.argv[1]
FUNDING = Path("data/raw") / snapshot / "funding"
grid = hour_grid()
out_dir = panel_dir(snapshot)
out_dir.mkdir(parents=True, exist_ok=True)


def read_csvs(files: list[Path]) -> np.ndarray:
    rows = []
    for f in files:
        with zipfile.ZipFile(f) as z:
            txt = z.read(z.namelist()[0]).decode()
        lines = [ln for ln in txt.splitlines() if ln and ln[0].isdigit()]
        if lines:
            rows.append(np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",", ndmin=2, usecols=range(6)))
    return np.vstack(rows)


audit = {"snapshot": snapshot, "symbols": {}}
for sym in PRIMARY + SECONDARY:
    files = sorted((RAW_1H / sym).glob(f"{sym}-1h-*.zip"))
    sha = hashlib.sha256()
    for f in files:
        sha.update(f.name.encode() + f.read_bytes())
    k = read_csvs(files)
    ts = k[:, 0].astype(np.int64)
    ts = np.where(ts > 10**14, ts // 1000, ts)                   # microseconds since 2025
    h = ts // 3_600_000
    off_hour = int((ts % 3_600_000 != 0).sum())
    h, idx = np.unique(h, return_index=True)
    n_dup = len(ts) - len(h)
    o, hi, lo, c = (k[idx, j] for j in (1, 2, 3, 4))
    bad_ohlc = int(((hi < np.maximum(o, c) - 1e-12) | (lo > np.minimum(o, c) + 1e-12)).sum())
    inside = (h >= grid[0]) & (h <= grid[-1])
    pos = h[inside] - grid[0]
    close = np.full(len(grid), np.nan)
    real = np.zeros(len(grid), bool)
    close[pos], real[pos] = c[inside], True
    first = int(np.argmax(real))
    filled = close.copy()
    for t in range(first + 1, len(grid)):                       # forward fill after the first candle
        if not real[t]:
            filled[t] = filled[t - 1]
    funding = np.full(len(grid), np.nan)
    ffiles = sorted((FUNDING / sym).glob(f"{sym}-fundingRate-*.zip"))
    if ffiles:
        rows = []
        for f in ffiles:
            with zipfile.ZipFile(f) as z:
                lines = z.read(z.namelist()[0]).decode().splitlines()[1:]
            rows += [ln.split(",") for ln in lines if ln]
        fh = np.array([int(r[0]) for r in rows]) // 3_600_000
        fr = np.array([float(r[2]) for r in rows])
        ok = (fh >= grid[0]) & (fh <= grid[-1])
        funding[fh[ok] - grid[0]] = fr[ok]
    np.savez(out_dir / f"{sym}.npz", hours=grid, close=filled, real=real, funding=funding)
    missing = int((~real[first:]).sum())
    runs, cur = [], 0
    for r in real[first:]:
        cur = 0 if r else cur + 1
        if cur == 1:
            runs.append(1)
        elif cur > 1:
            runs[-1] += 1
    audit["symbols"][sym] = {
        "n_zips": len(files), "sha256_zips": sha.hexdigest(), "n_candles": int(len(ts)),
        "first_hour_utc": str((grid[first] * 3600).astype("datetime64[s]")),
        "duplicates": n_dup, "off_hour_timestamps": off_hour, "bad_ohlc": bad_ohlc,
        "missing_hours": missing, "n_gaps": len(runs), "longest_gap_h": max(runs, default=0),
        "funding_first_utc": (str((grid[int(np.argmax(~np.isnan(funding)))] * 3600).astype("datetime64[s]"))
                              if np.isfinite(funding).any() else None),
        "n_funding": int(np.isfinite(funding).sum()),
    }
print(json.dumps(audit, indent=1))
