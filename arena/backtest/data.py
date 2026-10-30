"""Historická data pro backtest: 5m klines, bookDepth, fundingRate (zip) a aggTrades (převedené na npz).

Vše z https://data.binance.vision (USDⓈ-M, futures/um). Stahování je idempotentní: co už je na disku, se nestahuje.
aggTrades se po převodu na `{SYM}-{date}.npz` (T, a, p) maže, protože zip je zhruba 3× větší než npz."""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import httpx
import numpy as np

from arena.data.bars import BAR_MS
from arena.db import Bar

log = logging.getLogger(__name__)

BASE = "https://data.binance.vision/data/futures/um"


def dates(start: str, end: str) -> list[str]:
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [(a + dt.timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def months(start: str, end: str) -> list[str]:
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    out, y, m = [], a.year, a.month
    while (y, m) <= (b.year, b.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _day_ms(d: str) -> int:
    return int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


class Store:
    """Adresář s daty backtestu: klines/, bookDepth/, fundingRate/, ticks/."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def kline(self, sym: str, d: str) -> Path:
        return self.root / "klines" / sym / f"{sym}-5m-{d}.zip"

    def book(self, sym: str, d: str) -> Path:
        return self.root / "bookDepth" / sym / f"{sym}-bookDepth-{d}.zip"

    def funding(self, sym: str, ym: str) -> Path:
        return self.root / "fundingRate" / sym / f"{sym}-fundingRate-{ym}.zip"

    def ticks(self, sym: str, d: str) -> Path:
        return self.root / "ticks" / sym / f"{sym}-{d}.npz"


def _get(client: httpx.Client, url: str, dest: Path, retries: int = 4) -> str:
    """'ok' / 'have' / 'missing' (404). Zápis přes .part, aby půlka souboru nikdy neplatila."""
    if dest.exists():
        return "have"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for i in range(retries):
        try:
            with client.stream("GET", url) as r:
                if r.status_code == 404:
                    return "missing"
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            part.replace(dest)
            return "ok"
        except (httpx.HTTPError, OSError):
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
    return "missing"


def agg_zip_to_npz(zip_path: Path, out: Path) -> int:
    """aggTrades zip → npz se sloupci T (ms), a (agg id), p (cena), seřazeno podle (T, a), bez duplicit."""
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(z.namelist()[0])
    b = raw.replace(b"true", b"1").replace(b"false", b"0")
    if b[:1].isalpha():
        b = b[b.index(b"\n") + 1:] if b"\n" in b else b""
    del raw
    arr = np.fromstring(b.strip().replace(b"\n", b","), sep=",").reshape(-1, 7) if b.strip() else np.empty((0, 7))
    a = arr[:, 0].astype(np.int64)
    t = arr[:, 5].astype(np.int64)
    p = arr[:, 1].copy()
    order = np.lexsort((a, t))
    a, t, p = a[order], t[order], p[order]
    keep = np.ones(len(a), dtype=bool)
    keep[1:] = a[1:] != a[:-1]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, T=t[keep], a=a[keep], p=p[keep])
    tmp.replace(out)
    return int(keep.sum())


def _fetch_ticks(args: tuple[str, str, str, str]) -> tuple[str, str, str]:
    root, sym, d, _ = args
    st = Store(Path(root))
    out = st.ticks(sym, d)
    if out.exists():
        return sym, d, "have"
    zp = st.root / "ticks" / sym / f"{sym}-aggTrades-{d}.zip"
    with httpx.Client(timeout=120, follow_redirects=True) as c:
        res = _get(c, f"{BASE}/daily/aggTrades/{sym}/{sym}-aggTrades-{d}.zip", zp)
    if res == "missing":
        return sym, d, "missing"
    agg_zip_to_npz(zp, out)
    zp.unlink()
    return sym, d, "ok"


def fetch_all(store: Store, symbols: list[str], start: str, end: str, ticks_from: str | None = None,
              workers: int = 3, tick_symbols: list[str] | None = None) -> dict[str, list[str]]:
    """Stáhne klines + bookDepth (od `start`), fundingRate (měsíce) a aggTrades (od `ticks_from`, default start).
    Vrací {druh: [chybějící soubory]}."""
    missing: dict[str, list[str]] = {"klines": [], "bookDepth": [], "fundingRate": [], "ticks": []}
    jobs: list[tuple[str, str, Path, str]] = []
    for s in symbols:
        for d in dates(start, end):
            jobs.append(("klines", f"{BASE}/daily/klines/{s}/5m/{s}-5m-{d}.zip", store.kline(s, d), f"{s} {d}"))
            jobs.append(("bookDepth", f"{BASE}/daily/bookDepth/{s}/{s}-bookDepth-{d}.zip", store.book(s, d), f"{s} {d}"))
        for ym in months(start, end):
            jobs.append(("fundingRate", f"{BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{ym}.zip",
                         store.funding(s, ym), f"{s} {ym}"))

    def one(j: tuple[str, str, Path, str]) -> tuple[str, str, str]:
        with httpx.Client(timeout=120, follow_redirects=True) as c:
            return j[0], j[3], _get(c, j[1], j[2])

    with ThreadPoolExecutor(6) as ex:
        for kind, label, res in ex.map(one, jobs):
            if res == "missing":
                missing[kind].append(label)
    tick_jobs = [(str(store.root), s, d, "") for s in (tick_symbols or symbols) for d in dates(ticks_from or start, end)]
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        for sym, d, res in ex.map(_fetch_ticks, tick_jobs, chunksize=1):
            done += 1
            if res == "missing":
                missing["ticks"].append(f"{sym} {d}")
            if done % 50 == 0:
                log.info("ticky: %d/%d", done, len(tick_jobs))
    return missing


# --- čtení ---------------------------------------------------------------------------------------

def read_bars(store: Store, sym: str, start: str, end: str) -> list[Bar]:
    out: list[Bar] = []
    for d in dates(start, end):
        p = store.kline(sym, d)
        if not p.exists():
            continue
        with zipfile.ZipFile(p) as z:
            rd = csv.reader(io.StringIO(z.read(z.namelist()[0]).decode()))
            for r in rd:
                if r and r[0].isdigit():
                    out.append(Bar.from_rest(sym, r))
    return out


def read_funding(store: Store, sym: str, start: str, end: str) -> list[tuple[int, float]]:
    lo, hi = _day_ms(start), _day_ms(end) + 86_400_000
    out: list[tuple[int, float]] = []
    for ym in months(start, end):
        p = store.funding(sym, ym)
        if not p.exists():
            continue
        with zipfile.ZipFile(p) as z:
            for r in csv.reader(io.StringIO(z.read(z.namelist()[0]).decode())):
                if r and r[0].isdigit() and lo <= int(r[0]) < hi:
                    out.append((int(r[0]), float(r[2])))
    return sorted(out)


class TickStore:
    """aggTrades jednoho symbolu po dnech (npz), v paměti nejvýš dva dny."""

    def __init__(self, store: Store, sym: str) -> None:
        self.store, self.sym = store, sym
        self._days: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}

    def _day(self, d: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if d not in self._days:
            if len(self._days) >= 2:
                self._days.pop(next(iter(self._days)))
            p = self.store.ticks(self.sym, d)
            if p.exists():
                z = np.load(p)
                self._days[d] = (z["T"], z["a"], z["p"])
            else:
                self._days[d] = None
        return self._days[d]

    def window(self, lo: int, hi: int) -> Iterator[tuple[int, int, float]]:
        """Obchody s T v [lo, hi) vzestupně (T, a, p). Chybějící den = žádné obchody."""
        d0 = dt.datetime.fromtimestamp(lo / 1000, dt.timezone.utc).date()
        d1 = dt.datetime.fromtimestamp((hi - 1) / 1000, dt.timezone.utc).date()
        d = d0
        while d <= d1:
            arrs = self._day(d.isoformat())
            if arrs is not None:
                T, a, p = arrs
                i, j = int(np.searchsorted(T, lo, "left")), int(np.searchsorted(T, hi, "left"))
                if j > i:
                    yield from zip(T[i:j].tolist(), a[i:j].tolist(), p[i:j].tolist())
            d += dt.timedelta(days=1)


def has_day(store: Store, sym: str, d: str) -> bool:
    return store.ticks(sym, d).exists()


__all__ = ["BAR_MS", "Store", "TickStore", "dates", "months", "fetch_all", "read_bars", "read_funding",
           "agg_zip_to_npz", "has_day"]
