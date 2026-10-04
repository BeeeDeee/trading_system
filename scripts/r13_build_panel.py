"""Research 13: panel_r9 + daily high/low from the same raw klines -> data/derived/<snap>/panel_r13 (prereg §3).

    uv run python scripts/r13_build_panel.py binance_2026-10-03     -> docs/research13/DATA_AUDIT.md

Rows are filtered exactly as in qlab.research9.panel.build. Checks: the rebuilt close equals panel_r9
bitwise, high/low exist exactly on the panel_r9 candle rows, and every matrix of panel_r13 equals panel_r9.
"""

import sys
from pathlib import Path

import numpy as np

from qlab.data.panel import load_panel, save_panel
from qlab.research8 import data as D8
from qlab.research9.setup import panel_dir as r9_dir
from qlab.research13.setup import panel_dir

snapshot = sys.argv[1]
raw = Path("data/raw") / snapshot
p, extra = load_panel(r9_dir(snapshot))
syms = (r9_dir(snapshot) / "symbols.txt").read_text().split()
T, N = p.shape
high, low, close = (np.full((T, N), np.nan) for _ in range(3))
i0 = p.dates[0]
for j, s in enumerate(syms):
    k = D8.read_klines(raw / "spot_1d" / s)
    d = (k["date"].to_numpy().astype("datetime64[D]") - i0).astype(int)
    keep = (d >= 0) & (d < T) & (k["open"].to_numpy() > 0) & (k["close"].to_numpy() > 0)
    d = d[keep]
    high[d, j], low[d, j], close[d, j] = (k[c].to_numpy()[keep] for c in ("high", "low", "close"))

assert np.array_equal(close, np.asarray(p.close_u), equal_nan=True), "close differs from panel_r9"
candle = np.isfinite(np.asarray(extra["open"]))
assert np.array_equal(np.isfinite(high), candle) and np.array_equal(np.isfinite(low), candle), "row mask differs"
opn = np.asarray(extra["open"])
with np.errstate(invalid="ignore"):
    bad_high = int((high < np.fmax(opn, close)).sum())
    bad_low = int((low > np.fmin(opn, close)).sum())
    bad_hl = int((high < low).sum())

save_panel(p, panel_dir(snapshot), {**{k: np.asarray(v) for k, v in extra.items()}, "high": high, "low": low})
(panel_dir(snapshot) / "symbols.txt").write_text("\n".join(syms) + "\n")

p13, x13 = load_panel(panel_dir(snapshot))
fields = ("dates", "assets", "ret_co", "ret_oc", "tradable", "listed", "delisting", "close_u", "dollar_volume")
same = all(np.array_equal(getattr(p13, f), getattr(p, f), equal_nan=True) for f in fields)
same &= all(np.array_equal(x13[k], extra[k], equal_nan=True) for k in extra)
assert same, "panel_r13 differs from panel_r9"

lines = [f"# Výzkum 13 – audit dat (`{snapshot}`)", "",
         f"`panel_r13` = `panel_r9` + `extra_high`, `extra_low` ze stejných raw svíček a se stejnými filtry řádků.", "",
         f"- panel {p.dates[0]} → {p.dates[-1]}, {T} dní, {N} párů",
         f"- všechny matice `panel_r9` (vč. `open`, `qv`) bitově shodné: **{'ano' if same else 'NE'}**",
         "- close přepočtený z raw svíček bitově shodný s `panel_r9`: **ano**; high/low existují přesně na řádcích svíček",
         f"- svíček: {int(candle.sum())}; high < max(open, close): {bad_high}; low > min(open, close): {bad_low}; "
         f"high < low: {bad_hl}", ""]
Path("docs/research13").mkdir(parents=True, exist_ok=True)
Path("docs/research13/DATA_AUDIT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
