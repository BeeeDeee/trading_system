"""Research 8: download the Binance snapshot (perp + spot daily klines, funding), verify checksums.

    uv run python scripts/r8_download.py binance_2026-10-03 [--last-month 2026-09]

Stages: 1) daily klines of every USDT-M perp ever listed; 2) candidate set = perps with a Binance spot
pair that were ever in the top 40 by 30-day quote volume on a Monday (superset of the top-20
universe and of the top-40 robustness check; volumes only, no returns); 3) funding + spot klines of
the candidates; 4) mark-price klines of the candidates (prereg v1.1). Idempotent: a rerun downloads only what is missing. Writes manifest.json.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from qlab.research8 import data as D, download as dl

TOP_CANDIDATES = 40


def candidates(raw: Path, perps: list[str], spot_symbols: set[str]) -> list[str]:
    vols = []
    for p in perps:
        if D.spot_pair(p, spot_symbols) is None:
            continue
        k = D.read_klines(raw / "perp_1d" / p)
        if k.height:
            vols.append(k.select("date", "quote_volume").with_columns(pl.lit(p).alias("sym")))
    v = (pl.concat(vols).sort("sym", "date")
           .with_columns(pl.col("quote_volume").rolling_mean(30, min_samples=1).over("sym").alias("qv30"))
           .filter(pl.col("date").dt.weekday() == 1)
           .with_columns(pl.col("qv30").rank("ordinal", descending=True).over("date").alias("rk")))
    ever = set(v.filter(pl.col("rk") <= TOP_CANDIDATES)["sym"].unique().to_list())
    return sorted(ever | {"BTCUSDT", "ETHUSDT"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--last-month", default="2026-09")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    raw = Path("data/raw") / a.snapshot
    raw.mkdir(parents=True, exist_ok=True)

    perps = [s for s in dl.list_symbols("perp_1d") if D.PERP_RE.match(s)]
    spot_symbols = set(dl.list_symbols("spot_1d"))
    print(f"USDT-M perps ever listed: {len(perps)}, spot symbols: {len(spot_symbols)}")
    r1 = dl.fetch_many("perp_1d", perps, raw, a.last_month, a.workers)
    cand = candidates(raw, perps, spot_symbols)
    print(f"candidates (ever top {TOP_CANDIDATES} by 30d volume, with spot pair): {len(cand)}")
    spots = sorted({D.spot_pair(p, spot_symbols)[0] for p in cand})
    r2 = dl.fetch_many("funding", cand, raw, a.last_month, a.workers)
    r3 = dl.fetch_many("spot_1d", spots, raw, a.last_month, a.workers)
    r4 = dl.fetch_many("mark_1d", cand, raw, a.last_month, a.workers)     # prereg v1.1: liquidation by mark price
    errors = [r for r in r1 + r2 + r3 + r4 if "error" in r]
    manifest = {"snapshot": a.snapshot, "last_month": a.last_month,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "n_perps": len(perps), "n_spot_symbols": len(spot_symbols), "candidates": cand,
                "pairs": {p: D.spot_pair(p, spot_symbols) for p in cand},
                "files": {k: sum(r.get("files", 0) for r in rs) for k, rs in
                          (("perp_1d", r1), ("funding", r2), ("spot_1d", r3), ("mark_1d", r4))},
                "errors": errors}
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("files",)}), f"errors: {len(errors)}")
    for e in errors[:20]:
        print("  ", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
