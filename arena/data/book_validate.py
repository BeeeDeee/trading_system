"""Validace lokální knihy proti `bookDepth` z data.binance.vision (SPEC §4.3, akceptace P1).

`bookDepth` CSV: `timestamp, percentage, depth, notional`; kumulativní notional pro ±0,2/1/2/3/4/5 %,
záporné `percentage` = bid. Snapshot zhruba každých 30 s, timestamp = reálný čas (nástraha 16)."""

from __future__ import annotations

import bisect
import datetime as dt
import io
import statistics
import zipfile
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from arena.db import Database

BOOKDEPTH_URL = "https://data.binance.vision/data/futures/um/daily/bookDepth/{sym}/{sym}-bookDepth-{date}.zip"
PAIR_TOLERANCE_MS = 5_000


def _ts(s: str) -> int:
    return int(dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def parse_bookdepth_csv(text: str) -> dict[int, dict[float, tuple[float, float]]]:
    """→ {ts: {pct: (bid_notional, ask_notional)}}; jen časy, kde je bid i ask daného pásma."""
    raw: dict[int, dict[float, list[float | None]]] = {}
    for line in text.splitlines():
        if not line or line[0].isalpha():
            continue   # hlavička
        p = line.split(",")
        ts, pct, notional = _ts(p[0]), float(p[1]), float(p[3])
        slot = raw.setdefault(ts, {}).setdefault(abs(pct), [None, None])
        slot[0 if pct < 0 else 1] = notional
    out: dict[int, dict[float, tuple[float, float]]] = {}
    for ts, bands in raw.items():
        good = {pct: (b, a) for pct, (b, a) in bands.items() if b is not None and a is not None}
        if good:
            out[ts] = good  # type: ignore[assignment]
    return out


async def fetch_bookdepth(symbol: str, date: str, client: httpx.AsyncClient | None = None
                          ) -> dict[int, dict[float, tuple[float, float]]] | None:
    """Stáhne a rozparsuje denní `bookDepth`. None, když soubor (ještě) neexistuje."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=60, follow_redirects=True)
    try:
        r = await client.get(BOOKDEPTH_URL.format(sym=symbol, date=date))
        if r.status_code == 404:
            return None
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            text = z.read(z.namelist()[0]).decode()
        return parse_bookdepth_csv(text)
    finally:
        if own:
            await client.aclose()


@dataclass(frozen=True)
class BandResult:
    symbol: str
    pct: float
    n_pairs: int
    obi_corr: float | None
    median_abs_dnotional_pct: float | None     # medián |live − bookDepth| / bookDepth × 100, bid i ask
    median_dnotional_pct: float | None         # se znaménkem (live níž = záporné → chybějící vzdálené úrovně)

    @property
    def passed(self) -> bool:
        return (self.obi_corr is not None and self.obi_corr >= 0.9 and self.median_abs_dnotional_pct is not None
                and self.median_abs_dnotional_pct <= 10.0)


def compare(symbol: str, live: dict[int, dict[float, tuple[float, float]]],
            ref: dict[int, dict[float, tuple[float, float]]], pcts: tuple[float, ...]) -> list[BandResult]:
    """Páruje každý `bookDepth` snapshot s nejbližším živým vzorkem (± 5 s)."""
    live_ts = sorted(live)
    out = []
    for pct in pcts:
        lo, ro, dn_abs, dn = [], [], [], []
        for ts, bands in ref.items():
            if pct not in bands or not live_ts:
                continue
            i = bisect.bisect_left(live_ts, ts)
            cands = [live_ts[j] for j in (i - 1, i) if 0 <= j < len(live_ts)]
            best = min(cands, key=lambda t: abs(t - ts))
            if abs(best - ts) > PAIR_TOLERANCE_MS or pct not in live[best]:
                continue
            rb, ra = bands[pct]
            lb, la = live[best][pct]
            if rb + ra <= 0 or lb + la <= 0:
                continue
            ro.append((rb - ra) / (rb + ra))
            lo.append((lb - la) / (lb + la))
            for lv, rv in ((lb, rb), (la, ra)):
                if rv > 0:
                    dn.append((lv - rv) / rv * 100)
                    dn_abs.append(abs(lv - rv) / rv * 100)
        corr = float(np.corrcoef(lo, ro)[0, 1]) if len(lo) >= 10 else None
        out.append(BandResult(symbol, pct, len(lo), corr,
                              statistics.median(dn_abs) if dn_abs else None,
                              statistics.median(dn) if dn else None))
    return out


def live_bands_for_day(db: Database, symbol: str, date: str) -> dict[int, dict[float, tuple[float, float]]]:
    start = _ts(f"{date} 00:00:00")
    rows = db.book_bands(symbol, start - PAIR_TOLERANCE_MS, start + 86_400_000 + PAIR_TOLERANCE_MS)
    out: dict[int, dict[float, tuple[float, float]]] = {}
    for r in rows:
        out.setdefault(int(r["ts"]), {})[float(r["pct"])] = (float(r["bid_notional"]), float(r["ask_notional"]))
    return out


async def validate_day(db: Database, symbols: tuple[str, ...], date: str,
                       pcts: tuple[float, ...] = (0.2, 1.0, 2.0, 5.0)) -> dict[str, Any]:
    results: list[BandResult] = []
    missing: list[str] = []
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for sym in symbols:
            ref = await fetch_bookdepth(sym, date, client)
            if ref is None:
                missing.append(sym)
                continue
            results.extend(compare(sym, live_bands_for_day(db, sym, date), ref, pcts))
    # akceptace P1: obi1 a obi2 na každém symbolu
    gate = [r for r in results if r.pct in (1.0, 2.0)]
    summary = {
        "date": date,
        "missing_bookdepth": missing,
        "passed": bool(gate) and all(r.passed for r in gate) and not missing,
        "results": [r.__dict__ | {"passed": r.passed} for r in results],
    }
    db.event("INFO" if summary["passed"] else "WARN", "book-validate", f"book-validate {date}: "
             f"{'OK' if summary['passed'] else 'NEPROŠLO'}", summary)
    db.meta_set("book_validate_last", date)
    return summary


def render(summary: dict[str, Any]) -> str:
    lines = [f"book-validate {summary['date']}: {'OK' if summary['passed'] else 'NEPROŠLO'}"]
    if summary["missing_bookdepth"]:
        lines.append(f"  bookDepth chybí: {', '.join(summary['missing_bookdepth'])}")
    lines.append(f"  {'symbol':9} {'pct':>4} {'párů':>5} {'corr obi':>9} {'med|Δ|%':>8} {'medΔ%':>7}  ")
    for r in summary["results"]:
        corr = f"{r['obi_corr']:.3f}" if r["obi_corr"] is not None else "—"
        ma = f"{r['median_abs_dnotional_pct']:.1f}" if r["median_abs_dnotional_pct"] is not None else "—"
        md = f"{r['median_dnotional_pct']:+.1f}" if r["median_dnotional_pct"] is not None else "—"
        flag = "ok" if r["passed"] else ("--" if r["pct"] not in (1.0, 2.0) else "FAIL")
        lines.append(f"  {r['symbol']:9} {r['pct']:>4g} {r['n_pairs']:>5} {corr:>9} {ma:>8} {md:>7}  {flag}")
    return "\n".join(lines)


async def backfill_book(db: Database, symbols: tuple[str, ...], days: int, pcts: tuple[float, ...],
                        today: dt.date | None = None) -> dict[str, int]:
    """`arena backfill-book --days N` (SPEC §5.5): předplní `book_bands` a `bar_book` ze `bookDepth`
    (source=bookDepth), aby `obi` nemusela čekat 288 barů. Živá data mají přednost (INSERT OR IGNORE)."""
    from arena.data.recorder import aggregate_bar_book

    today = today or dt.datetime.now(dt.timezone.utc).date()
    dates = [(today - dt.timedelta(days=d)).isoformat() for d in range(days, 0, -1)]
    out: dict[str, int] = {}
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for sym in symbols:
            n_bars = 0
            for date in dates:
                ref = await fetch_bookdepth(sym, date, client)
                if ref is None:
                    continue
                rows = [(sym, ts, p, b, a, None, "bookDepth") for ts, bands in sorted(ref.items())
                        for p, (b, a) in bands.items() if p in pcts]
                db.insert_book_bands(rows)   # type: ignore[arg-type]
                day0 = _ts(f"{date} 00:00:00")
                flat = [{"ts": r[1], "pct": r[2], "bid_notional": r[3], "ask_notional": r[4]} for r in rows]
                agg = []
                j = 0
                for ot in range(day0, day0 + 86_400_000, 300_000):
                    while j < len(flat) and flat[j]["ts"] < ot:
                        j += 1
                    k = j
                    while k < len(flat) and flat[k]["ts"] < ot + 300_000:
                        k += 1
                    agg.extend(aggregate_bar_book(sym, ot, flat[j:k], pcts, source="bookDepth"))
                db.upsert_bar_book(agg, replace=False)
                n_bars += len(agg) // max(len(pcts), 1)
            out[sym] = n_bars
    db.event("INFO", "backfill-book", f"bookDepth {dates[0]}..{dates[-1]}", out)
    return out
