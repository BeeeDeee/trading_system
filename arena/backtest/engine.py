"""Backtest: stejné strategie, SlotRunner a paper executor (TrailingSim nad aggTrades) jako živý paper, nad historií.

Rozdíly proti živému paperu (jdou do reportu jako omezení):
- fill bez knihy: první obchod po latenci + `extra_slippage_bps` (jako `arena replay`), tedy bez dopadu na cenu;
- ticky se podávají jen symbolům s otevřenou nebo čekající pozicí (jinak by 1 mld. obchodů trvalo hodiny);
- `obi` počítá z `bookDepth` (30s snímky z data.binance.vision), živě z lokální knihy (book-validate: korelace ≥ 0,9);
- filtry (tick, step, minNotional) jsou dnešní, ne historické.
Prvních 7 dní před `start` je rozběh (ctl potřebuje týden signálů cílové strategie); jejich obchody se zahodí."""

from __future__ import annotations

import json
import logging
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from arena.backtest.data import Store, TickStore, _day_ms, read_bars, read_funding
from arena.binance.filters import parse_exchange_info
from arena.config import Config
from arena.data.bars import BAR_MS
from arena.db import Database
from arena.execution.base import TradeRecord
from arena.replay import REPLAY_DECISION_LAG_MS, book_bar_rows, build_executor
from arena.features import BarArrays, BookArrays, book_arrays, compute
from arena.slots import CTL_LOOKBACK_BARS, FeatureSource, SlotRunner, contiguous_tail
from arena.strategies.registry import build_slots, build_strategies, global_warmup

log = logging.getLogger(__name__)

DAY_MS = 86_400_000


class ArrayFeatureSource(FeatureSource):
    """Featury z celých sloupců načtených jednou do paměti (řez místo SQL na každý bar): stejné výsledky jako
    `FeatureSource`, jen ~10× rychlejší. Bar a kniha se berou z DB při `load`, pozdější zápisy se nevidí."""

    def load(self, symbols: list[str]) -> ArrayFeatureSource:
        self._bars: dict[str, BarArrays] = {}
        self._book: dict[str, BookArrays] = {}
        self._idx: dict[str, dict[int, int]] = {}
        for sym in symbols:
            rows = self.db.bars(sym)
            arr = BarArrays.from_rows(rows)
            self._bars[sym] = arr
            self._idx[sym] = {int(t): i for i, t in enumerate(arr.open_time)}
            bb = self.db.execute("SELECT open_time, pct, obi, book_ok FROM bar_book WHERE symbol=?", (sym,)).fetchall()
            self._book[sym] = book_arrays(arr.open_time, bb)
        return self

    def _slice(self, symbol: str, t: int) -> tuple[BarArrays, BookArrays, int] | None:
        i = self._idx[symbol].get(t)
        if i is None:
            return None
        lo = max(0, i - self.warmup + 1)
        a = self._bars[symbol]
        bars = BarArrays(*(x[lo:i + 1] for x in (a.open_time, a.o, a.h, a.l, a.c, a.qv, a.tbqv)))
        b = self._book[symbol]
        book = BookArrays({p: x[lo:i + 1] for p, x in b.obi.items()}, b.ok[lo:i + 1])
        return bars, book, lo

    def arrays(self, symbol: str, t: int) -> BarArrays | None:
        sl = self._slice(symbol, t)
        return None if sl is None else contiguous_tail(sl[0])

    def features(self, symbol: str, t: int, btc: BarArrays | None) -> dict[str, float] | None:
        sl = self._slice(symbol, t)
        if sl is None:
            return None
        bars = contiguous_tail(sl[0])
        n = len(bars)
        book = BookArrays({p: x[len(x) - n:] for p, x in sl[1].obi.items()}, sl[1].ok[len(sl[1].ok) - n:])
        use_btc = btc if (btc is not None and symbol != "BTCUSDT") else None
        return compute(bars, book, use_btc, self.donchian_ns, self.ema_ns)


def warmup_days(cfg: Config) -> int:
    strategies = {k: v for k, v in build_strategies(cfg).items() if v.family != "xp"}
    return global_warmup(strategies) // 288 + 1


def _iso(ms: int) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date().isoformat()


def history_start(cfg: Config, start: str) -> str:
    """První den klines/bookDepth: rozběh ctl (7 dní) + warmup featur."""
    return _iso(_day_ms(start) - (CTL_LOOKBACK_BARS // 288 + warmup_days(cfg) + 1) * DAY_MS)


def run_shard(cfg: Config, store_root: str, symbols: list[str], start: str, end: str,
              exchange_info: str) -> dict[str, Any]:
    t0 = time.monotonic()
    store = Store(Path(store_root))
    strategies = {k: v for k, v in build_strategies(cfg).items() if v.family != "xp"}
    slots = [s for s in build_slots(cfg) if s.strategy.family != "xp" and s.symbol in symbols]
    warmup = global_warmup(strategies)
    start_ms, end_ms = _day_ms(start), _day_ms(end) + DAY_MS - BAR_MS
    loop_ms = start_ms - CTL_LOOKBACK_BARS * BAR_MS
    hist = history_start(cfg, start)

    db = Database(":memory:")
    closes: dict[str, dict[int, float]] = {}
    for sym in sorted(set(symbols) | {"BTCUSDT"}):
        bars = read_bars(store, sym, hist, end)
        db.upsert_bars(bars)
        closes[sym] = {b.open_time: b.c for b in bars}
        if sym in symbols:
            for p in sorted((store.root / "bookDepth" / sym).glob(f"{sym}-bookDepth-*.zip")):
                with zipfile.ZipFile(p) as z:
                    db.upsert_bar_book(book_bar_rows(cfg, sym, z.read(z.namelist()[0]).decode()), replace=False)
    db.meta_set("exchange_info", exchange_info)
    filters = parse_exchange_info(json.loads(exchange_info), cfg.universe)
    ex = build_executor(cfg, filters)
    runner = SlotRunner(cfg, db, slots, ex, ArrayFeatureSource(db, warmup).load(sorted(set(symbols) | {"BTCUSDT"})))
    db.meta_set("first_bar_time", str(loop_ms))
    trades: list[TradeRecord] = []
    orig_closed = ex.on_closed

    def closed(rec: TradeRecord) -> None:
        trades.append(rec)
        if orig_closed is not None:
            orig_closed(rec)

    ex.on_closed = closed
    tick = {s: TickStore(store, s) for s in symbols}
    fund = {s: [f for f in read_funding(store, s, _iso(loop_ms), end) if f[0] >= loop_ms] for s in symbols}
    fpos = dict.fromkeys(symbols, 0)
    fed_prev: set[str] = set()
    prev_ts = loop_ms
    n_signals = n_ticks = 0
    sym_set = frozenset(symbols)
    for ot in range(loop_ms, end_ms + 1, BAR_MS):
        ts = ot + BAR_MS + REPLAY_DECISION_LAG_MS
        active = sorted({p.symbol for p in ex.positions.values()})
        for sym in active:
            if sym not in fed_prev:
                ex._last_agg.pop(sym, None)   # ticky se přeskakovaly: bez toho by mezera ID označila pozici jako tainted
            for T, a, p in tick[sym].window(prev_ts, ts):
                ex.on_trade(sym, {"p": p, "T": T, "a": a})
                n_ticks += 1
        fed_prev = set(active)
        for sym in symbols:
            while fpos[sym] < len(fund[sym]) and fund[sym][fpos[sym]][0] < ts:
                t_f, rate = fund[sym][fpos[sym]]
                mark = closes[sym].get(t_f - t_f % BAR_MS) or ex.fill.last.get(sym)
                if mark:
                    ex.on_funding(sym, t_f, rate, mark)
                fpos[sym] += 1
        have = frozenset(s for s in sym_set if ot in closes[s])
        for s in have:
            ex.fill.last[s] = closes[s][ot]
        if have:
            n_signals += len(runner.on_bar_close(ot, have, ts))
        prev_ts = ts
    rows = [t.row() for t in trades if t.entry_time >= start_ms]
    db.close()
    log.info("backtest %s: %d obchodů, %d signálů, %d ticků, %.0f s", ",".join(symbols), len(rows), n_signals,
             n_ticks, time.monotonic() - t0)
    return {"symbols": symbols, "rows": rows, "n_signals": n_signals, "n_ticks": n_ticks,
            "seconds": time.monotonic() - t0, "open_at_end": len(ex.positions)}


def run_backtest(cfg: Config, store_root: Path, start: str, end: str, exchange_info: str,
                 symbols: list[str] | None = None, workers: int = 3) -> list[dict[str, Any]]:
    """Jeden shard na symbol (symboly jsou nezávislé; BTC bary dostane každý kvůli `ll`)."""
    symbols = symbols or list(cfg.universe)
    # nejtěžší (BTC) první, ať se na konci nečeká na jediný proces
    order = sorted(symbols, key=lambda s: s != "BTCUSDT")
    with ProcessPoolExecutor(workers) as pool:
        futs = [pool.submit(run_shard, cfg, str(store_root), [s], start, end, exchange_info) for s in order]
        return [f.result() for f in futs]
