"""Research 9: raw spot klines -> data/derived/<snap>/panel_r9 (+ audit, no return statistics).

    uv run python scripts/r9_build_panel.py binance_2026-10-03     -> docs/research9/DATA_AUDIT.md
"""

import json
import sys
from pathlib import Path

import numpy as np

from qlab.research9 import panel as P
from qlab.research9.setup import HOLDOUT, PANEL_START, save

snapshot = sys.argv[1]
raw = Path("data/raw") / snapshot
man = json.loads((raw / "spot_manifest_r9.json").read_text())
syms = man["symbols"]
p, extra, syms = P.build(raw, syms, PANEL_START, HOLDOUT[1])
keep = p.listed.any(axis=0)
idx = np.flatnonzero(keep)
from qlab.data.panel import Panel  # noqa: E402
p = Panel(p.dates, np.arange(len(idx), dtype=np.int64), *(getattr(p, f)[:, idx] for f in
          ("ret_co", "ret_oc", "tradable", "listed", "delisting", "close_u", "dollar_volume")))
extra = {k: v[:, idx] for k, v in extra.items()}
syms = [syms[i] for i in idx]
save(snapshot, p, extra, syms)
last = [str(p.dates[np.flatnonzero(p.tradable[:, j])[-1]]) for j in range(len(syms))]
first = [str(p.dates[np.flatnonzero(p.tradable[:, j])[0]]) for j in range(len(syms))]
n_del = int(p.delisting.sum())
traded_per_day = p.tradable.sum(axis=1)
lines = [f"# Výzkum 9 – audit dat (`{snapshot}`)", "",
         f"Spotových párů USDT (bez stablecoinů, wrapped, fiat a pákových tokenů): {len(man['symbols'])}, "
         f"s daty v panelu: {len(syms)}. Soubory: {man['files']}, chyby stahování: {len(man['errors'])}, "
         "vše ověřeno proti `.CHECKSUM`.", "",
         f"- panel {p.dates[0]} → {p.dates[-1]}, {len(p.dates)} dní",
         f"- **poslední den dat: {max(last)}** (konec holdoutu {HOLDOUT[1]}); BTC {last[syms.index('BTCUSDT')]}, "
         f"ETH {last[syms.index('ETHUSDT')]}",
         f"- první den BTC {first[syms.index('BTCUSDT')]}, ETH {first[syms.index('ETHUSDT')]}",
         f"- delistingových řádků (konec dat nebo mezera > 7 dní): {n_del}",
         f"- obchodovatelných párů za den: min {int(traded_per_day.min())}, 2018-04-01 "
         f"{int(traded_per_day[np.searchsorted(p.dates, np.datetime64('2018-04-01'))])}, "
         f"2022-01-01 {int(traded_per_day[np.searchsorted(p.dates, np.datetime64('2022-01-01'))])}, "
         f"konec {int(traded_per_day[-1])}",
         f"- řádky s |ret_oc| > 90 % nebo |ret_co| > 50 % (k ověření, nemění se): "
         f"{int((np.abs(p.ret_oc) > 0.9).sum())} / {int((np.abs(p.ret_co) > 0.5).sum())}", ""]
Path("docs/research9").mkdir(parents=True, exist_ok=True)
Path("docs/research9/DATA_AUDIT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
