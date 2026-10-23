"""Příkazová řádka `arena` (SPEC §16)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from arena.config import ConfigError, load_config, load_signer, write_lock

# Příkazy z SPEC §16, které zatím nejsou implementované (fáze, ve které vzniknou).
_PENDING = {
    "universe-check": "P1", "run": "P3", "replay": "P3", "backfill-book": "P2", "book-validate": "P1",
    "stats": "P4", "calibrate": "P7", "reconcile": "P5", "kill": "P5", "unkill": "P5", "flatten": "P5",
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arena", description="arena5m: 5m strategie na Binance USDⓈ-M")
    p.add_argument("--config", help="adresář konfigurace (default ./config nebo $ARENA_CONFIG_DIR)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="P0: konektivita, čas, klíč, futures, one-way, margin/páka, poplatky, filtry")
    c.add_argument("--env", choices=["testnet", "mainnet"], help="přepíše env z konfigurace (jen read-only)")

    sub.add_parser("lock-variants", help="zaregistruje nové varianty do config/variants.lock.json")
    sub.add_parser("config-show", help="vypíše env, config_hash a seznam variant")

    for name, phase in _PENDING.items():
        sub.add_parser(name, help=f"(fáze {phase}, zatím neimplementováno)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
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
    except ConfigError as e:
        print(f"chyba konfigurace: {e}", file=sys.stderr)
        return 2
    print(f"`arena {args.cmd}` přijde ve fázi {_PENDING[args.cmd]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
