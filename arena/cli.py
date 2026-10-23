"""Příkazová řádka `arena` (SPEC §16)."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import logging
import logging.handlers
import signal
import sys
from pathlib import Path

from arena.config import Config, ConfigError, load_config, load_signer, write_lock

# Příkazy z SPEC §16, které zatím nejsou implementované (fáze, ve které vzniknou).
_PENDING = {
    "replay": "P3", "stats": "P4", "calibrate": "P7", "reconcile": "P5",
    "kill": "P5", "unkill": "P5", "flatten": "P5",
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arena", description="arena5m: 5m strategie na Binance USDⓈ-M")
    p.add_argument("--config", help="adresář konfigurace (default ./config nebo $ARENA_CONFIG_DIR)")
    p.add_argument("--data-root", help="přesměruje paths.* a recorder.raw_dir pod tento adresář (vývoj, testy)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--log-file", help="log do souboru s rotací 50 MB × 3 (jinak stderr/journald)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="P0: konektivita, čas, klíč, futures, one-way, margin/páka, poplatky, filtry")
    c.add_argument("--env", choices=["testnet", "mainnet"], help="přepíše env z konfigurace (jen read-only)")

    sub.add_parser("lock-variants", help="zaregistruje nové varianty do config/variants.lock.json")
    sub.add_parser("config-show", help="vypíše env, config_hash a seznam variant")
    sub.add_parser("universe-check", help="30denní obrat a aktuální spread universa (jen výpis, D5)")

    r = sub.add_parser("run", help="hlavní proces (P1: jen datová vrstva)")
    r.add_argument("--live", action="store_true", help="povolí live executor (navíc nutné live.enabled: true)")

    bb = sub.add_parser("backfill-book", help="předplní bar_book z bookDepth (source=bookDepth)")
    bb.add_argument("--days", type=int, required=True)

    bv = sub.add_parser("book-validate", help="lokální kniha vs bookDepth za den D (§4.3)")
    bv.add_argument("--date", help="YYYY-MM-DD, default včerejšek UTC")

    for name, phase in _PENDING.items():
        sub.add_parser(name, help=f"(fáze {phase}, zatím neimplementováno)")
    return p


def _setup_logging(verbose: bool, log_file: str | None, default_level: int) -> None:
    level = logging.DEBUG if verbose else default_level
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler: logging.Handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=50 * 2**20, backupCount=3)
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def apply_data_root(cfg: Config, root: str | None) -> Config:
    if not root:
        return cfg
    r = Path(root)
    return dataclasses.replace(
        cfg,
        paths=dataclasses.replace(cfg.paths, db=str(r / "arena.db"), site=str(r / "site"), kill_file=str(r / "KILL")),
        recorder=dataclasses.replace(cfg.recorder, raw_dir=str(r / "raw")),
    )


async def _run(cfg: Config, live: bool) -> int:
    from arena.binance.rest import RestClient
    from arena.db import Database
    from arena.engine import Engine

    if live:
        print("live executor přijde ve fázi P5; běží jen datová vrstva", file=sys.stderr)
    db = Database(cfg.paths.db)
    async with RestClient(cfg.data_endpoints.rest) as rest:   # P1: jen veřejná tržní data
        engine = Engine(cfg, db, rest)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, engine.stop.set)
        await engine.run()
    db.close()
    return 0


async def _universe_check(cfg: Config) -> int:
    from arena.binance.rest import RestClient

    async with RestClient(cfg.endpoints_by_env["mainnet"].rest) as rest:
        print(f"{'symbol':9} {'obrat 30d/den [M USDT]':>24} {'5m obrat [M]':>13} {'spread [bps]':>13}")
        for sym in cfg.universe:
            kl = await rest.klines(sym, "1d", limit=31)
            kl = kl[:-1][-30:]   # bez rozpracovaného dne
            qv = sum(float(k[7]) for k in kl) / max(len(kl), 1)
            d = await rest.depth(sym, 5)
            bid, ask = float(d["bids"][0][0]), float(d["asks"][0][0])
            spread = (ask - bid) / ((ask + bid) / 2) * 1e4
            print(f"{sym:9} {qv / 1e6:>24.1f} {qv / 288 / 1e6:>13.2f} {spread:>13.2f}")
    print("(jen výpis; universe je zmrazené, D5)")
    return 0


async def _book_validate(cfg: Config, date: str | None) -> int:
    from arena.data.book_validate import render, validate_day
    from arena.db import Database

    d = date or (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).isoformat()
    db = Database(cfg.paths.db)
    summary = await validate_day(db, cfg.universe, d, cfg.book.bands_pct)
    db.close()
    print(render(summary))
    return 0 if summary["passed"] else 1


async def _backfill_book(cfg: Config, days: int) -> int:
    from arena.data.book_validate import backfill_book
    from arena.db import Database

    db = Database(cfg.paths.db)
    out = await backfill_book(db, cfg.universe, days, cfg.book.bands_pct)
    db.close()
    for sym, n in out.items():
        print(f"{sym}: {n} barů z bookDepth")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _setup_logging(args.verbose, args.log_file, logging.INFO if args.cmd == "run" else logging.WARNING)
    try:
        if args.cmd == "lock-variants":
            cfg = load_config(args.config, check_lock=False)
            added = write_lock(cfg)
            print("přidáno: " + (", ".join(added) if added else "nic"))
            return 0
        if args.cmd == "config-show":
            cfg = load_config(args.config)
            print(f"env={cfg.env} config_hash={cfg.config_hash}")
            for vid, v in sorted(cfg.variants.items()):
                print(f"  {vid:10} {v.params_hash} {','.join(v.symbols)}")
            return 0
        if args.cmd == "check":
            from arena.check import run_check

            cfg = load_config(args.config, env_override=args.env)
            rep = asyncio.run(run_check(cfg, load_signer(cfg.env)))
            print(rep.render())
            if rep.failed:
                print("VÝSLEDEK: SELHALO")
                return 1
            if rep.incomplete:
                print("VÝSLEDEK: NEÚPLNÉ (část kontrol přeskočena)")
                return 3
            print("VÝSLEDEK: OK")
            return 0
        if args.cmd == "universe-check":
            return asyncio.run(_universe_check(load_config(args.config)))
        if args.cmd == "run":
            return asyncio.run(_run(apply_data_root(load_config(args.config), args.data_root), args.live))
        if args.cmd == "backfill-book":
            return asyncio.run(_backfill_book(apply_data_root(load_config(args.config), args.data_root), args.days))
        if args.cmd == "book-validate":
            return asyncio.run(_book_validate(apply_data_root(load_config(args.config), args.data_root), args.date))
    except ConfigError as e:
        print(f"chyba konfigurace: {e}", file=sys.stderr)
        return 2
    print(f"`arena {args.cmd}` přijde ve fázi {_PENDING[args.cmd]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
