"""`arena replay`: běh slotů nad nahranými daty (SPEC §16, P3). Deterministický: stejné vstupy → stejné obchody.

Zdroje: bary a bar_book ze zdrojové DB (nebo REST klines), ticky z recorderu (`{dir}/aggTrades/{SYM}/{date}*.csv.gz`)
nebo z data.binance.vision (`{dir}/{SYM}-aggTrades-{date}.zip`), volitelně bookDepth (`--book DIR`).
Bez knihy je fill = cena prvního obchodu po latenci (TradeFillModel)."""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import heapq
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from arena.binance.filters import parse_exchange_info
from arena.config import Config
from arena.data.bars import BAR_MS
from arena.data.book_validate import parse_bookdepth_csv
from arena.data.recorder import aggregate_bar_book
from arena.db import Database
from arena.execution.base import TradeRecord
from arena.execution.paper import PaperExecutor, TradeFillModel
from arena.slots import FeatureSource, SlotRunner
from arena.strategies.registry import build_slots, build_strategies, global_warmup
from arena.util import canonical_json, sha256_hex

REPLAY_DECISION_LAG_MS = 500   # bar close → rozhodnutí; živě změřeno typicky 0,2–0,5 s (CHANGELOG P3)


def _day_ms(d: str) -> int:
    return int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def _dates(start: str, end: str) -> list[str]:
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [(a + dt.timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def iter_agg_trades(ticks_dir: Path, symbol: str, dates: list[str]) -> Iterator[dict[str, Any]]:
    """aggTrades symbolu za dny vzestupně podle (T, a), bez duplicit (segmenty po restartu se překrývají)."""
    for d in dates:
        rows: list[dict[str, Any]] = []
        for p in sorted((ticks_dir / "aggTrades" / symbol).glob(f"{d}*.csv.gz")):
            with gzip.open(p, "rt") as f:
                try:
                    rows.extend(csv.DictReader(f))
                except EOFError:
                    pass   # segment useknutý pádem: co se přečetlo, platí
        z = ticks_dir / f"{symbol}-aggTrades-{d}.zip"
        if z.exists():
            with zipfile.ZipFile(z) as zf:
                text = zf.read(zf.namelist()[0]).decode()
            rd = csv.reader(io.StringIO(text))
            for r in rd:
                if r and r[0].isdigit():
                    rows.append({"agg_trade_id": r[0], "price": r[1], "quantity": r[2], "transact_time": r[5],
                                 "is_buyer_maker": r[6]})
        seen: set[int] = set()
        out = []
        for r in rows:
            a = int(r["agg_trade_id"])
            if a in seen:
                continue
            seen.add(a)
            out.append({"s": symbol, "a": a, "p": r["price"], "q": r["quantity"], "T": int(r["transact_time"]),
                        "m": str(r["is_buyer_maker"]).lower() == "true"})
        out.sort(key=lambda t: (t["T"], t["a"]))
        yield from out


def _tick_events(ticks_dir: Path, symbol: str, dates: list[str]) -> Iterator[tuple[int, int, int, str, Any]]:
    for t in iter_agg_trades(ticks_dir, symbol, dates):
        yield t["T"], 1, t["a"], symbol, t


@dataclass
class ReplayResult:
    trades: list[TradeRecord]
    n_signals: int
    fingerprint: str


def _prepare_db(cfg: Config, src: Database, start_ms: int, end_ms: int, warmup: int,
                book_dir: Path | None) -> Database:
    db = Database(":memory:")
    lo = start_ms - (warmup + 1) * BAR_MS
    for sym in cfg.universe:
        db.upsert_bars(src.bars(sym, lo, end_ms))
        rows = src.execute("SELECT * FROM bar_book WHERE symbol=? AND open_time>=? AND open_time<=?",
                           (sym, lo, end_ms)).fetchall()
        db.upsert_bar_book([tuple(r) for r in rows])
        if book_dir is not None:
            for p in sorted(book_dir.glob(f"{sym}-bookDepth-*")):
                text = (zipfile.ZipFile(p).read(zipfile.ZipFile(p).namelist()[0]).decode() if p.suffix == ".zip"
                        else gzip.open(p, "rt").read())
                parsed = parse_bookdepth_csv(text)
                flat = sorted(({"ts": ts, "pct": pct, "bid_notional": b, "ask_notional": a}
                               for ts, bands in parsed.items() for pct, (b, a) in bands.items()),
                              key=lambda r: r["ts"])
                if not flat:
                    continue
                first = flat[0]["ts"] - flat[0]["ts"] % BAR_MS
                agg, j = [], 0
                for ot in range(first, flat[-1]["ts"] + 1, BAR_MS):
                    k = j
                    while k < len(flat) and flat[k]["ts"] < ot + BAR_MS:
                        k += 1
                    agg.extend(aggregate_bar_book(sym, ot, flat[j:k], cfg.book.bands_pct, source="bookDepth"))
                    j = k
                db.upsert_bar_book(agg, replace=False)
    ei = src.meta_get("exchange_info")
    if ei:
        db.meta_set("exchange_info", ei)
    return db


def run_replay(cfg: Config, src: Database, start: str, end: str, ticks_dir: Path | None,
               book_dir: Path | None = None) -> ReplayResult:
    strategies = build_strategies(cfg)
    slots = build_slots(cfg)
    warmup = global_warmup(strategies)
    start_ms, end_ms = _day_ms(start), _day_ms(end) + 86_400_000 - BAR_MS
    db = _prepare_db(cfg, src, start_ms, end_ms, warmup, book_dir)
    ei = db.meta_get("exchange_info")
    if not ei:
        raise RuntimeError("chybí exchange_info v meta zdrojové DB (uloží ho `arena run` při startu)")
    filters = parse_exchange_info(json.loads(ei), cfg.universe)
    # replay má vlastní in-memory DB: nezávisí na cooldownech ani pozicích ze zdrojové DB
    fill = TradeFillModel(dict(cfg.paper.extra_slippage_bps))
    ex = PaperExecutor(filters, fill, cfg.paper.notional_usdt, cfg.paper.entry_latency_ms,
                       cfg.paper.trigger_latency_ms, cfg.paper.outage_penalty_bps,
                       cost_model_version=cfg.paper.cost_model_version)
    trades: list[TradeRecord] = []
    runner = SlotRunner(cfg, db, slots, ex, FeatureSource(db, warmup))
    runner.db.meta_set("first_bar_time", str(start_ms))
    orig_closed = ex.on_closed

    def closed(rec: TradeRecord) -> None:
        trades.append(rec)
        if orig_closed is not None:
            orig_closed(rec)

    ex.on_closed = closed

    dates = _dates(start, end)
    streams: list[Iterator[tuple[int, int, int, str, Any]]] = []
    if ticks_dir is not None:
        for sym in cfg.universe:
            streams.append(_tick_events(ticks_dir, sym, dates))
    bar_events = ((ot + BAR_MS + REPLAY_DECISION_LAG_MS, 0, 0, "", ot) for ot in range(start_ms, end_ms + 1, BAR_MS))
    streams.append(bar_events)
    n_signals = 0
    for ts, kind, _, sym, payload in heapq.merge(*streams, key=lambda e: (e[0], e[1], e[2], e[3])):
        if kind == 1:
            ex.on_trade(sym, payload)
        else:
            ot = int(payload)
            have = frozenset(s for s in cfg.universe if db.execute(
                "SELECT 1 FROM bars WHERE symbol=? AND open_time=?", (s, ot)).fetchone())
            if have:
                n_signals += len(runner.on_bar_close(ot, have, ts))
    fp = sha256_hex(canonical_json([t.row() for t in trades]))
    db.close()
    return ReplayResult(trades, n_signals, fp)


def summary(res: ReplayResult) -> str:
    lines = [f"signálů {res.n_signals}, obchodů {len(res.trades)}, otisk {res.fingerprint[:16]}"]
    by: dict[str, list[float]] = {}
    for t in res.trades:
        by.setdefault(t.slot_id.rsplit(".", 1)[0], []).append(t.net_bps)
    for vid, xs in sorted(by.items()):
        lines.append(f"  {vid:10} n={len(xs):4} net {sum(xs) / len(xs):+7.1f} bps")
    return "\n".join(lines)
