"""Report backtestu → JSON pro záložku Backtest (stejné statistiky jako živá aréna: arena.stats).

Holm a DSR jsou tady jen v rámci tohoto backtestu (30 variant). Do rozhodování na dni 45/90 nevstupují (SPEC §14.2)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from arena import stats
from arena.config import Config
from arena.snapshot import _dump
from arena.util import wall_ms

DAY = 86_400_000
LIMITS = [
    "Fill bez knihy: první obchod po latenci + extra slippage, bez dopadu na cenu. Náklady jsou proto spíš optimistické; "
    "sloupce net při 1,5× a 2× nákladech ukazují citlivost.",
    "obi: bookDepth (30s snímky z data.binance.vision), živě lokální kniha. Filtry (tick, step, minNotional) jsou dnešní.",
    "Explorativní: stejné zmrazené parametry jako živě, žádné ladění. Holm a DSR jsou jen v rámci backtestu a do "
    "rozhodnutí na dni 45/90 nevstupují.",
    "Období před 2026-07-22 je mimo vzorek priorů (research/obi_prior); 2026-07-22 až 10-19 je in-sample.",
]


def _month(ms: int) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m")


def _curve(tr: Sequence[stats.T], day0: int, ndays: int) -> dict[str, list[Any]]:
    net = np.zeros(ndays)
    for t in tr:
        i = t.exit_time // DAY - day0
        if 0 <= i < ndays:
            net[i] += t.net_bps
    return {"t": [(day0 + i) * DAY for i in range(ndays)], "net": [round(x, 3) for x in np.cumsum(net)]}


def _variant(vid: str, tr: Sequence[stats.T], ctl: Sequence[stats.T], day0: int, ndays: int) -> dict[str, Any]:
    g = stats.group_stats(vid, tr)
    months: dict[str, list[float]] = {}
    for t in tr:
        months.setdefault(_month(t.exit_time), []).append(t.net_bps)
    sides = {s: [t.net_bps for t in tr if t.side == s] for s in ("LONG", "SHORT")}
    hist = np.histogram([t.net_bps for t in tr], bins=np.arange(-300, 325, 25))[0] if tr else []
    dp = stats.daily_pnl(tr) if tr else np.zeros(0)
    return {
        "variant": vid, "curve": _curve(tr, day0, ndays), "control_curve": _curve(ctl, day0, ndays),
        "by_symbol": g.by_symbol, "exit_mix": g.exit_mix, "cost_bps": g.cost_bps,
        "monthly": [{"m": m, "n": len(x), "net_mean": float(np.mean(x))} for m, x in sorted(months.items())],
        "sides": {s: {"n": len(x), "net_mean": float(np.mean(x)) if x else None} for s, x in sides.items()},
        "hist": {"edges": list(range(-300, 325, 25)), "counts": [int(x) for x in hist]},
        "pnl_usdt": float(dp.sum()) if len(dp) else 0.0,
    }


def build_report(cfg: Config, rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], site_data: Path,
                 run_dir: Path | None = None) -> dict[str, Any]:
    run_id = str(meta["id"])
    all_t = stats.from_rows(rows)
    clean = [t for t in all_t if not t.tainted]
    variants = {vid: list(v.symbols) for vid, v in sorted(cfg.variants.items()) if v.family != "xp"}
    period = (int(meta["end_ms"]) - int(meta["start_ms"])) + 1
    lb = stats.leaderboard(clean, variants, (), False, period)
    day0, ndays = int(meta["start_ms"]) // DAY, period // DAY
    out = site_data / "backtest" / run_id
    by_var: dict[str, list[stats.T]] = {}
    for t in clean:
        by_var.setdefault(t.variant, []).append(t)
    for vid in variants:
        ctl = by_var.get(f"ctl.{vid.split('.')[0]}", []) if not vid.startswith("ctl.") else []
        _dump(out / "variants" / f"{vid}.json", _variant(vid, by_var.get(vid, []), ctl, day0, ndays))
    summary = {
        **meta, "generated_at": wall_ms(), "n_trades": len(clean), "n_tainted_excluded": len(all_t) - len(clean),
        "leaderboard": {k: g.to_json() for k, g in lb.items()}, "heatmap": stats.heatmap(clean), "limits": LIMITS,
    }
    _dump(out / "summary.json", summary)
    idx_path = site_data / "backtest" / "index.json"
    try:
        idx = json.loads(idx_path.read_text())
    except (OSError, ValueError):
        idx = {"runs": []}
    idx["runs"] = [r for r in idx["runs"] if r["id"] != run_id] + [{
        "id": run_id, "label": meta.get("label", run_id), "from": meta["from"], "to": meta["to"],
        "n_trades": len(clean), "generated_at": summary["generated_at"], "sample": meta.get("sample", "")}]
    idx["runs"].sort(key=lambda r: r["from"], reverse=True)
    _dump(idx_path, idx)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(run_dir / "trades.jsonl.gz", "wt") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        (run_dir / "meta.json").write_text(json.dumps(dict(meta), indent=1))
    return summary
