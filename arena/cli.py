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
_PENDING: dict[str, str] = {}


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

    st = sub.add_parser("stats", help="statistiky variant (SPEC §13)")
    st.add_argument("--variant")
    st.add_argument("--mode", choices=["paper", "live"], default="paper")
    st.add_argument("--clean", action="store_true", help="bez tainted obchodů")

    sub.add_parser("calibrate", help="návrh parametrů nákladového modelu z live/paper dvojčat (nic nemění)")
    rc = sub.add_parser("reconcile", help="jednorázová rekonciliace: výpis rozdílů (--fix provede akce §9)")
    rc.add_argument("--fix", action="store_true")
    sub.add_parser("kill", help="kill switch: live se zavře a nic nového se neotevře (paper běží dál)")
    sub.add_parser("unkill", help="zruší kill switch")
    fl = sub.add_parser("flatten", help="zruší všechny a5- příkazy a zavře pozice z DB (s potvrzením)")
    fl.add_argument("--yes", action="store_true")
    rr = sub.add_parser("risk-reset", help="ruční reset: live_halt, zmrazené symboly, vypnuté sloty")
    rr.add_argument("--what", choices=["all", "halt", "frozen", "slots"], default="all")

    rp = sub.add_parser("replay", help="offline běh slotů nad nahranými daty (determinismus)")
    rp.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    rp.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD (včetně)")
    rp.add_argument("--ticks", help="adresář s aggTrades (recorder nebo data.binance.vision zipy)")
    rp.add_argument("--book", help="adresář s bookDepth soubory (zip nebo csv.gz)")
    rp.add_argument("--src-db", help="zdrojová DB s bary a bar_book (default paths.db)")

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

    db = Database(cfg.paths.db)
    async with RestClient(cfg.data_endpoints.rest) as rest:   # P1: jen veřejná tržní data
        engine = Engine(cfg, db, rest, live_flag=live)
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


def _admin(cfg: Config, args: argparse.Namespace) -> int:
    from arena.db import Database

    if args.cmd == "kill":
        Path(cfg.paths.kill_file).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.paths.kill_file).write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n")
        print(f"kill switch aktivní ({cfg.paths.kill_file}): live se zavře, nové vstupy nebudou")
        return 0
    if args.cmd == "unkill":
        Path(cfg.paths.kill_file).unlink(missing_ok=True)
        print("kill switch zrušen")
        return 0
    db = Database(cfg.paths.db)
    try:
        if args.cmd == "risk-reset":
            from arena.risk import RiskGuard

            before = RiskGuard(cfg, db, live_flag=False).reset(args.what)
            print(f"reset {args.what}; předtím: {before}")
            return 0
        if args.cmd == "calibrate":
            from arena.calibrate import calibrate, render

            print(render(calibrate(db)))
            return 0
        from arena import stats

        q = "SELECT * FROM trades WHERE mode=?" + (" AND tainted=0" if args.clean else "")
        trades = stats.from_rows(db.execute(q, (args.mode,)).fetchall())
        variants = {vid: list(v.symbols) for vid, v in sorted(cfg.variants.items())}
        lb = stats.leaderboard(trades, variants)
        keys = [args.variant] if args.variant else sorted(lb)
        print(f"{'varianta':10} {'n':>5} {'/den':>5} {'net':>7} {'95% CI':>17} {'gross':>7} {'nákl.':>6} "
              f"{'Sharpe':>6} {'Δctl':>6} {'p_holm':>6} {'DSR':>5}  stav")
        for k in keys:
            g = lb[k]
            f = lambda x, d=1: "—" if x is None or x != x else f"{x:.{d}f}"  # noqa: E731
            print(f"{k:10} {g.n:5} {f(g.trades_per_day):>5} {f(g.net_mean):>7} "
                  f"{'[' + f(g.net_ci[0]) + ', ' + f(g.net_ci[1]) + ']':>17} {f(g.gross_mean):>7} "
                  f"{f(g.cost_bps.get('total')):>6} {f(g.sharpe, 2):>6} {f(g.vs_control.get('delta')):>6} "
                  f"{f(g.p_holm, 3):>6} {f(g.dsr, 2):>5}  {g.badge}")
        return 0
    finally:
        db.close()


async def _exchange_admin(cfg: Config, args: argparse.Namespace) -> int:
    """reconcile / flatten proti burze cfg.env (podepsané)."""
    from arena.binance.filters import parse_exchange_info
    from arena.binance.rest import RestClient
    from arena.db import Database
    from arena.execution.live import LiveExecutor
    from arena.reconcile import Reconciler
    from arena.risk import RiskGuard

    signer = load_signer(cfg.env)
    if signer is None:
        print(f"chybí klíč pro {cfg.env}", file=sys.stderr)
        return 2
    db = Database(cfg.paths.db)
    async with RestClient(cfg.endpoints.rest, signer=signer) as rest:
        await rest.sync_time()
        syms = tuple(cfg.live.slots) or cfg.universe
        filters = parse_exchange_info(await rest.exchange_info(), syms)
        ex = LiveExecutor(rest, filters, db, rest.clock.now_ms, lambda s: None)
        ex.restore()
        if args.cmd == "reconcile":
            diffs = await Reconciler(rest, ex, RiskGuard(cfg, db, False), syms).run(fix=args.fix)
            print("\n".join(str(d) for d in diffs) or "bez rozdílů")
            return 0
        # flatten: a5- příkazy zrušit, pozice známé z DB zavřít; cizí pozice a příkazy se nechají
        from arena.execution.ids import is_ours

        algos = [a for a in await rest.open_algo_orders() if is_ours(str(a.get("clientAlgoId")))]
        orders = [o for o in await rest.open_orders() if is_ours(str(o.get("clientOrderId")))]
        print(f"a5- algo příkazů {len(algos)}, a5- příkazů {len(orders)}, pozic z DB {len(ex.positions)}:")
        for p in ex.positions.values():
            print(f"  {p.slot_id} {p.side} qty {p.qty} stav {p.state}")
        if not args.yes and input("Provést? napiš ANO: ").strip() != "ANO":
            print("nic se nezměnilo")
            return 1
        for a in algos:
            await rest.cancel_algo_order(str(a["clientAlgoId"]))
        for o in orders:
            await rest.cancel_order(str(o["symbol"]), str(o["clientOrderId"]))
        for p in list(ex.positions.values()):
            await ex.market_exit(p, "KILL")
        print("hotovo; filly zapíše běžící proces nebo příští rekonciliace")
    db.close()
    return 0


def _replay(cfg: Config, args: argparse.Namespace) -> int:
    from arena.db import Database
    from arena.replay import run_replay, summary

    src = Database(args.src_db or cfg.paths.db)
    res = run_replay(cfg, src, args.date_from, args.date_to, Path(args.ticks) if args.ticks else None,
                     Path(args.book) if args.book else None)
    src.close()
    print(summary(res))
    return 0


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
        if args.cmd in ("stats", "calibrate", "kill", "unkill", "risk-reset"):
            return _admin(apply_data_root(load_config(args.config), args.data_root), args)
        if args.cmd in ("reconcile", "flatten"):
            return asyncio.run(_exchange_admin(apply_data_root(load_config(args.config), args.data_root), args))
        if args.cmd == "replay":
            return _replay(apply_data_root(load_config(args.config), args.data_root), args)
        if args.cmd == "backfill-book":
            return asyncio.run(_backfill_book(apply_data_root(load_config(args.config), args.data_root), args.days))
        if args.cmd == "book-validate":
            return asyncio.run(_book_validate(apply_data_root(load_config(args.config), args.data_root), args.date))
    except ConfigError as e:
        print(f"chyba konfigurace: {e}", file=sys.stderr)
        return 2
    print(f"neznámý příkaz {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
