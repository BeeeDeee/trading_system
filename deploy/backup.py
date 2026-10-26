#!/usr/bin/env python3
"""Denní záloha arena5m: konzistentní snímek arena.db (sqlite backup API), gzip, rotace,
a inkrementální kopie surových aggTrades (soubory jsou po zápisu neměnné).
Volitelně rsync mimo server: ARENA_BACKUP_REMOTE=user@host:/cesta/ (klíč v ~arena/.ssh)."""
import gzip
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("ARENA_DATA_ROOT", "/srv/arena"))
DEST = Path(os.environ.get("ARENA_BACKUP_DIR", str(ROOT / "backup")))
KEEP = int(os.environ.get("ARENA_BACKUP_KEEP", "14"))


def backup_db() -> Path:
    DEST.mkdir(parents=True, exist_ok=True)
    tmp = DEST / "arena.snapshot.tmp"
    out = DEST / f"arena-{time.strftime('%Y%m%d-%H%M', time.gmtime())}.db.gz"
    src = sqlite3.connect(f"file:{ROOT / 'arena.db'}?mode=ro", uri=True)
    dst = sqlite3.connect(tmp)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    chk = sqlite3.connect(tmp)
    ok = chk.execute("pragma integrity_check").fetchone()[0]
    chk.close()
    if ok != "ok":
        tmp.unlink()
        sys.exit(f"integrity_check selhal: {ok}")
    with open(tmp, "rb") as f, gzip.open(out, "wb", compresslevel=6) as g:
        shutil.copyfileobj(f, g)
    tmp.unlink()
    for old in sorted(DEST.glob("arena-*.db.gz"))[:-KEEP]:
        old.unlink()
    return out


def sync_raw() -> int:
    n = 0
    base = ROOT / "raw"
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        q = DEST / "raw" / p.relative_to(base)
        if q.exists() and q.stat().st_size == p.stat().st_size:
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        n += 1
    return n


def main() -> None:
    out = backup_db()
    n = sync_raw()
    print(f"db: {out} ({out.stat().st_size // 1024} kB), raw: +{n} souborů")
    remote = os.environ.get("ARENA_BACKUP_REMOTE")
    if remote:
        subprocess.run(["rsync", "-a", "--delete", f"{DEST}/", remote], check=True)
        print(f"rsync -> {remote}")


if __name__ == "__main__":
    main()
