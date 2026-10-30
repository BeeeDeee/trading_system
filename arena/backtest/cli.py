"""`arena backtest`: stáhne historii, spustí všechny sloty po symbolech a zapíše report do webu."""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from pathlib import Path

from arena.backtest.data import Store, _day_ms, dates, fetch_all
from arena.backtest.engine import DAY_MS, _iso, history_start, run_backtest
from arena.backtest.report import build_report
from arena.config import Config
from arena.slots import CTL_LOOKBACK_BARS

log = logging.getLogger(__name__)

PRIOR_FROM, PRIOR_TO = "2026-07-22", "2026-10-19"   # období priorů obi (SPEC §5.5), in-sample


def sample_kind(start: str, end: str) -> str:
    if end < PRIOR_FROM:
        return "mimo vzorek priorů (out-of-sample)"
    if start >= PRIOR_FROM and end <= PRIOR_TO:
        return "in-sample (období priorů)"
    return "překrývá období priorů (částečně in-sample)"


def _exchange_info(db_path: str) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        r = con.execute("SELECT value FROM meta WHERE key='exchange_info'").fetchone()
    finally:
        con.close()
    if not r:
        raise SystemExit("chybí exchange_info v meta DB (uloží ho `arena run` při startu)")
    return str(r[0])


def backtest(cfg: Config, args: argparse.Namespace) -> int:
    start, end = args.date_from, args.date_to
    root = Path(cfg.paths.db).parent / "bt"
    store = Store(root / "data")
    symbols = [s.upper() for s in args.symbols.split(",")] if args.symbols else list(cfg.universe)
    hist = history_start(cfg, start)
    loop_day = _iso(_day_ms(start) - CTL_LOOKBACK_BARS * 300_000)
    if not args.no_fetch:
        t0 = time.monotonic()
        need = sorted(set(symbols) | {"BTCUSDT"})
        miss = fetch_all(store, need, hist, end, ticks_from=loop_day, workers=args.workers,
                        tick_symbols=symbols)
        bad = {k: v for k, v in miss.items() if v}
        log.info("stažení hotovo za %.0f s", time.monotonic() - t0)
        if bad:
            print("CHYBĚJÍCÍ DATA:", {k: v[:5] + (["…"] if len(v) > 5 else []) for k, v in bad.items()})
            if not args.force:
                print("Přerušeno (--force přeskočí, chybějící dny pak nemají obchody).")
                return 3
    if args.fetch_only:
        return 0
    ei = _exchange_info(cfg.paths.db)
    t0 = time.monotonic()
    shards = run_backtest(cfg, root / "data", start, end, ei, symbols, args.workers)
    rows = [r for s in shards for r in s["rows"]]
    run_id = args.label or f"{start}_{end}"
    meta = {"id": run_id, "label": args.label or f"{start} až {end}", "from": start, "to": end,
            "start_ms": _day_ms(start), "end_ms": _day_ms(end) + DAY_MS - 1, "days": len(dates(start, end)),
            "sample": sample_kind(start, end), "symbols": symbols, "config_hash": cfg.config_hash,
            "cost_model_version": cfg.paper.cost_model_version, "runtime_s": round(time.monotonic() - t0),
            "n_signals": sum(s["n_signals"] for s in shards), "n_ticks": sum(s["n_ticks"] for s in shards),
            "open_at_end": sum(s["open_at_end"] for s in shards)}
    site_data = Path(cfg.paths.site) / "data"
    rep = build_report(cfg, rows, meta, site_data, root / "runs" / run_id)
    print(f"backtest {run_id}: {rep['n_trades']} obchodů (tainted vynecháno {rep['n_tainted_excluded']}), "
          f"{meta['n_signals']} signálů, {meta['runtime_s']} s → {site_data / 'backtest' / run_id}")
    return 0
