"""Research 15: 15m panel from the Binance zips (prereg §3).

    uv run python scripts/r15_build_panel.py binance_2026-10-03 [--secondary]

Writes data/derived/<snap>/r15_15m/{spot,perp}/<SYM>.npz and prints an audit as JSON (gaps, duplicates,
off-grid timestamps, OHLC consistency, SHA256 of the zip set). Spot: BTC, ETH. Perp: BTC, ETH and, with
--secondary, every member of the point-in-time basket (r15_manifest.json). Raw files are read-only.
"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from qlab.research15.setup import PER_DAY, PRIMARY, grid_range, panel_dir, raw_dir

ap = argparse.ArgumentParser()
ap.add_argument("snapshot")
ap.add_argument("--secondary", action="store_true")
a = ap.parse_args()
first_g, end_g = grid_range()
FUNDING = Path("data/raw") / a.snapshot / "funding"


def read_csvs(files: list[Path]) -> np.ndarray:
    rows = []
    for f in files:
        with zipfile.ZipFile(f) as z:
            txt = z.read(z.namelist()[0]).decode()
        lines = [ln for ln in txt.splitlines() if ln and ln[0].isdigit()]
        if lines:
            rows.append(np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",", ndmin=2, usecols=range(6)))
    return np.vstack(rows)


def funding_15m(sym: str, t0: int, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    files = sorted((FUNDING / sym).glob(f"{sym}-fundingRate-*.zip"))
    for f in files:
        with zipfile.ZipFile(f) as z:
            lines = z.read(z.namelist()[0]).decode().splitlines()[1:]
        for ln in lines:
            if not ln:
                continue
            p = ln.split(",")
            i = int(p[0]) // 900_000 - t0                        # calc_time floored to its 15m candle
            if 0 <= i < n:
                out[i] = float(p[2])
    return out


def build(market: str, sym: str) -> dict:
    files = sorted(raw_dir(a.snapshot, market, sym).glob(f"{sym}-15m-*.zip"))
    sha = hashlib.sha256()
    for f in files:
        sha.update(f.name.encode() + f.read_bytes())
    k = read_csvs(files)
    ts = k[:, 0].astype(np.int64)
    ts = np.where(ts > 10**14, ts // 1000, ts)                   # microseconds since 2025
    off_grid = int((ts % 900_000 != 0).sum())
    q = ts // 900_000
    q, idx = np.unique(q, return_index=True)                     # first record of a shifted-time duplicate wins
    n_dup = len(ts) - len(q)
    o, h, lo, c = (k[idx, j] for j in (1, 2, 3, 4))
    bad = int(((h < np.maximum(o, c) - 1e-12) | (lo > np.minimum(o, c) + 1e-12)).sum())
    t0 = max(first_g, int(q[0]) // 16 * 16)
    n = end_g - t0
    pos = q - t0
    ok = (pos >= 0) & (pos < n)
    real = np.zeros(n, bool)
    real[pos[ok]] = True
    arr = {key: np.full(n, np.nan) for key in "ohlc"}
    for key, v in zip("ohlc", (o, h, lo, c)):
        arr[key][pos[ok]] = v[ok]
    first = int(np.argmax(real))
    close = arr["c"]
    for t in range(first + 1, n):                                # carry the close over missing candles
        if not real[t]:
            close[t] = close[t - 1]
            arr["o"][t] = arr["h"][t] = arr["l"][t] = close[t]
    fund = funding_15m(sym, t0, n) if market == "perp" else np.full(n, np.nan)
    out = panel_dir(a.snapshot, market)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{sym}.npz", t0=t0, o=arr["o"], h=arr["h"], l=arr["l"], c=close, real=real,
                        funding=fund)
    runs, cur = [], 0
    for r in real[first:]:
        cur = 0 if r else cur + 1
        if cur == 1:
            runs.append(1)
        elif cur > 1:
            runs[-1] += 1
    return {"n_zips": len(files), "sha256_zips": sha.hexdigest(), "n_candles": int(len(ts)),
            "first_utc": str(np.datetime64((t0 + first) * 900, "s")),
            "duplicates": n_dup, "off_grid_timestamps": off_grid, "bad_ohlc": bad,
            "missing": int((~real[first:]).sum()), "n_gaps": len(runs), "longest_gap_candles": max(runs, default=0),
            "n_funding": int(np.isfinite(fund).sum())}


audit = {"snapshot": a.snapshot, "symbols": {}}
for sym in PRIMARY:
    audit["symbols"][f"spot/{sym}"] = build("spot", sym)
    audit["symbols"][f"perp/{sym}"] = build("perp", sym)
if a.secondary:
    manifest = json.loads((Path("data/raw") / a.snapshot / "r15_manifest.json").read_text())
    for sym in manifest["secondary"]:
        audit["symbols"][f"perp/{sym}"] = build("perp", sym)
print(json.dumps(audit, indent=1))
