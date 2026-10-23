"""Recorder (SPEC §4.4): surové aggTrades do denních gzip CSV, 30s vzorky pásem knihy, agregace na bar."""

from __future__ import annotations

import asyncio
import datetime as dt
import gzip
import logging
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Mapping

from arena.data.bars import BAR_MS
from arena.data.book import LocalBook
from arena.db import Database

log = logging.getLogger(__name__)

AGG_HEADER = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"


def utc_date(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d")


def agg_row(e: Mapping[str, Any]) -> str:
    """aggTrade event → řádek ve formátu data.binance.vision (ceny jako původní řetězce)."""
    return f"{e['a']},{e['p']},{e['q']},{e['f']},{e['l']},{e['T']},{'true' if e['m'] else 'false'}\n"


@dataclass
class _Segment:
    date: str
    path: Path
    fh: IO[str]


@dataclass
class AggTradeRecorder:
    """`{raw_dir}/aggTrades/{SYM}/{YYYY-MM-DD}.csv.gz`, po restartu nový segment `{date}.partN.csv.gz`.

    Zápis do bufferu, flush (gzip Z_SYNC_FLUSH) nejpozději po `flush_every_s`. Datum podle `T` (čas obchodu)."""

    raw_dir: Path
    flush_every_s: float = 5.0
    on_event: Callable[[str, str, dict[str, Any]], None] | None = None

    _seg: dict[str, _Segment] = field(default_factory=dict, init=False)
    _buf: dict[str, list[str]] = field(default_factory=dict, init=False)
    last_id: dict[str, int] = field(default_factory=dict, init=False)
    rows_written: int = field(default=0, init=False)

    def on_agg_trade(self, e: Mapping[str, Any]) -> None:
        sym = e["s"]
        aid = int(e["a"])
        prev = self.last_id.get(sym)
        if prev is not None and aid != prev + 1 and aid > prev:
            if self.on_event is not None:
                self.on_event("WARN", "mezera v aggTrades", {"symbol": sym, "prev": prev, "next": aid,
                                                            "missing": aid - prev - 1})
        if prev is not None and aid <= prev:
            return   # duplicita po reconnectu
        self.last_id[sym] = aid
        date = utc_date(int(e["T"]))
        seg = self._seg.get(sym)
        if seg is None or seg.date != date:
            self._flush_symbol(sym)
            if seg is not None:
                seg.fh.close()
            self._seg[sym] = self._open(sym, date)
        self._buf.setdefault(sym, []).append(agg_row(e))

    def _open(self, sym: str, date: str) -> _Segment:
        d = self.raw_dir / "aggTrades" / sym
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{date}.csv.gz"
        n = 1
        while path.exists():
            n += 1
            path = d / f"{date}.part{n}.csv.gz"
        fh = gzip.open(path, "wt", encoding="ascii", newline="")
        fh.write(AGG_HEADER)
        log.info("recorder: nový segment %s", path)
        return _Segment(date, path, fh)

    def _flush_symbol(self, sym: str) -> None:
        buf = self._buf.get(sym)
        seg = self._seg.get(sym)
        if not buf or seg is None:
            return
        seg.fh.write("".join(buf))
        self.rows_written += len(buf)
        buf.clear()
        seg.fh.flush()
        seg.fh.buffer.flush(zlib.Z_SYNC_FLUSH)   # type: ignore[attr-defined]  # data čitelná i po pádu

    def flush(self) -> None:
        for sym in list(self._buf):
            self._flush_symbol(sym)

    def close(self) -> None:
        self.flush()
        for seg in self._seg.values():
            seg.fh.close()
        self._seg.clear()

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.flush_every_s)
            except asyncio.TimeoutError:
                pass
            self.flush()


def aggregate_bar_book(symbol: str, open_time: int, rows: list[Any], pcts: tuple[float, ...],
                       min_ok: int = 8, source: str = "live") -> list[tuple[str, int, float, float | None, int, int, str]]:
    """`rows` = řádky `book_bands` (ts, pct, bid_notional, ask_notional) pro symbol a bar.

    obi = průměr (bid − ask)/(bid + ask) přes vzorky v [open_time, open_time + 300 s) – stejně jako
    `research/obi_prior/analyze.py`. `book_ok` = aspoň `min_ok` vzorků (z 10)."""
    out = []
    for pct in pcts:
        vals = []
        for r in rows:
            if abs(float(r["pct"]) - pct) > 1e-9 or not (open_time <= int(r["ts"]) < open_time + BAR_MS):
                continue
            b, a = float(r["bid_notional"]), float(r["ask_notional"])
            if b + a > 0:
                vals.append((b - a) / (b + a))
        n = len(vals)
        obi = sum(vals) / n if n else None
        out.append((symbol, open_time, pct, obi, n, 1 if n >= min_ok else 0, source))
    return out


@dataclass
class BookSampler:
    """Každých 30 s na :00 a :30 (čas burzy) spočítá pásma knihy pro všechny symboly (SPEC §4.3).

    Vzorek se nezapíše, když kniha není synchronizovaná nebo je diff starší než `max_diff_age_s`."""

    db: Database
    books: Mapping[str, LocalBook]
    pcts: tuple[float, ...]
    server_now_ms: Callable[[], int]
    local_now_ms: Callable[[], int]
    every_s: int = 30
    max_diff_age_s: float = 2.0
    on_tick: Callable[[int], None] | None = None   # např. preventivní resnapshot

    samples_ok: int = field(default=0, init=False)
    samples_bad: int = field(default=0, init=False)
    last_sample: dict[str, int] = field(default_factory=dict, init=False)

    def sample(self, ts: int) -> list[tuple[str, int, float, float, float, float, str]]:
        rows = []
        local = self.local_now_ms()
        for sym, book in self.books.items():
            if not book.is_fresh(local, self.max_diff_age_s * 1000):
                self.samples_bad += 1
                continue
            bands = book.bands(self.pcts)
            if bands is None:
                self.samples_bad += 1
                continue
            self.samples_ok += 1
            self.last_sample[sym] = ts
            rows.extend((sym, ts, b.pct, b.bid_notional, b.ask_notional, b.mid, "live") for b in bands)
        if rows:
            self.db.insert_book_bands(rows)
        return rows

    async def run(self, stop: asyncio.Event) -> None:
        step = self.every_s * 1000
        while not stop.is_set():
            now = self.server_now_ms()
            nxt = (now // step + 1) * step
            try:
                await asyncio.wait_for(stop.wait(), timeout=(nxt - now) / 1000)
                return
            except asyncio.TimeoutError:
                pass
            self.sample(nxt)
            if self.on_tick is not None:
                self.on_tick(nxt)
