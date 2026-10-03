"""Research 8 data audit (prereg §3): coverage, gaps, invalid prices, funding schedule, pair mapping.

    uv run python scripts/r8_audit.py binance_2026-10-03   -> docs/research8/DATA_AUDIT.md

No return statistics. Over the holdout only data-integrity counts are reported.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from qlab.research8 import data as D
from qlab.research8.setup import DEV

snapshot = sys.argv[1]
raw = Path("data/raw") / snapshot
man = json.loads((raw / "manifest.json").read_text())
pairs = {p: v for p, v in man["pairs"].items() if v}
rows, problems = [], []


def gaps(dates: pl.Series) -> int:
    if dates.len() < 2:
        return 0
    d = dates.diff().dt.total_days().drop_nulls()
    return int((d - 1).clip(lower_bound=0).sum())


for p, (spot, mult) in sorted(pairs.items()):
    k = D.read_klines(raw / "perp_1d" / p)
    s = D.read_klines(raw / "spot_1d" / spot)
    f = D.read_funding(raw / "funding" / p)
    bad_k = int(((k["close"] <= 0) | (k["low"] <= 0) | (k["high"] < k["low"])).sum()) if k.height else 0
    bad_s = int(((s["close"] <= 0) | (s["high"] < s["low"])).sum()) if s.height else 0
    off_hour = int((f["ts"].dt.minute() != 0).sum()) if f.height else 0
    j = k.select("date", pl.col("close").alias("pc")).join(
        s.select("date", (pl.col("close") * mult).alias("sc")), on="date")
    ratio = (j["pc"] / j["sc"]) if j.height else pl.Series([], dtype=pl.Float64)
    med = float(ratio.median()) if j.height else float("nan")
    if j.height and abs(med - 1) > 0.10:
        problems.append(f"{p}: medián perp/spot {med:.3f} (špatné párování nebo multiplikátor)")
    dev = j.filter(pl.col("date") <= DEV[1])
    big_basis_dev = int(((dev["pc"] / dev["sc"] - 1).abs() > 0.05).sum()) if dev.height else 0
    rows.append({"perp": p, "spot": spot, "mult": mult,
                 "perp_from": str(k["date"].min()) if k.height else "–", "perp_to": str(k["date"].max()) if k.height else "–",
                 "spot_from": str(s["date"].min()) if s.height else "–", "spot_to": str(s["date"].max()) if s.height else "–",
                 "perp_gaps": gaps(k["date"]) if k.height else 0, "spot_gaps": gaps(s["date"]) if s.height else 0,
                 "bad_rows": bad_k + bad_s, "funding_events": f.height, "funding_off_hour": off_hour,
                 "ratio_median": round(med, 4), "dev_days_basis_gt5pct": big_basis_dev})
    if bad_k + bad_s:
        problems.append(f"{p}: {bad_k + bad_s} řádků s nekladnou cenou nebo high < low")
    if k.height and not f.height:
        problems.append(f"{p}: svíčky bez fundingu")

df = pl.DataFrame(rows)
out = Path("docs/research8/DATA_AUDIT.md")
lines = [f"# Výzkum 8 – audit dat (`{snapshot}`)", "",
         f"Perpů USDT-M v historii: {man['n_perps']}, kandidátů (kdy top 40 podle objemu, se spotem): "
         f"{len(man['candidates'])}, s platným párem: {len(pairs)}. Soubory: {man['files']}. "
         f"Chyby stahování: {len(man['errors'])}. Všechny soubory ověřené proti `.CHECKSUM` (sha256).", "",
         "Bez výnosových statistik. Sloupec `dev_days_basis_gt5pct` jen pro vývojové období.", "",
         "## Nalezené problémy", ""] + ([f"- {x}" for x in problems] or ["- žádné"]) + [
         "", "## Souhrn", "",
         f"- mezery v denních svíčkách perpů celkem: {int(df['perp_gaps'].sum())}, spotu: {int(df['spot_gaps'].sum())}",
         f"- funding události mimo celou hodinu: {int(df['funding_off_hour'].sum())}",
         f"- delistované (perp končí před 2026-09-30): {int((df['perp_to'] < '2026-09-30').sum())}",
         "", "## Po symbolech", "",
         "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
lines += ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows]
out.write_text("\n".join(lines) + "\n")
print(f"{len(rows)} symbols, {len(problems)} problems -> {out}")
for x in problems[:30]:
    print("  ", x)
