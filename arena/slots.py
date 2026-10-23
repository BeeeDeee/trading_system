"""SlotRunner: na uzavřeném baru spočítá featury, zeptá se strategií a pošle intenty executoru.

Sdílí ho `engine.py` (live paper) i `replay.py`, takže replay testuje přesně stejnou cestu."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from arena.config import Config
from arena.data.bars import BAR_MS
from arena.db import Database
from arena.execution.base import TradeRecord
from arena.execution.paper import PaperExecutor
from arena.features import BarArrays, book_arrays, compute
from arena.strategies.base import BarContext, Intent
from arena.strategies.registry import Slot
from arena.util import wall_ms

log = logging.getLogger(__name__)

CTL_LOOKBACK_BARS = 7 * 288


def contiguous_tail(bars: BarArrays) -> BarArrays:
    """Nejdelší souvislý konec řady (bez děr), featury se nesmí počítat přes mezeru."""
    t = bars.open_time
    if len(t) < 2:
        return bars
    gaps = np.nonzero(np.diff(t) != BAR_MS)[0]
    if len(gaps) == 0:
        return bars
    start = int(gaps[-1]) + 1
    return BarArrays(*(a[start:] for a in (bars.open_time, bars.o, bars.h, bars.l, bars.c, bars.qv, bars.tbqv)))


@dataclass
class FeatureSource:
    """Featury ze `bars` a `bar_book` v DB. Okno = globální warmup."""

    db: Database
    warmup: int
    donchian_ns: tuple[int, ...] = (12, 24, 48)
    ema_ns: tuple[int, ...] = (576,)

    def arrays(self, symbol: str, t: int) -> BarArrays | None:
        rows = self.db.bars(symbol, end=t, limit=self.warmup)
        if not rows or rows[-1].open_time != t:
            return None
        return contiguous_tail(BarArrays.from_rows(rows))

    def features(self, symbol: str, t: int, btc: BarArrays | None) -> dict[str, float] | None:
        bars = self.arrays(symbol, t)
        if bars is None:
            return None
        rows = self.db.execute("SELECT open_time, pct, obi, book_ok FROM bar_book WHERE symbol=? AND open_time>=? "
                               "AND open_time<=?", (symbol, int(bars.open_time[0]), t)).fetchall()
        book = book_arrays(bars.open_time, rows)
        use_btc = btc if (btc is not None and symbol != "BTCUSDT") else None
        return compute(bars, book, use_btc, self.donchian_ns, self.ema_ns)


@dataclass
class SlotRunner:
    cfg: Config
    db: Database
    slots: list[Slot]
    executor: PaperExecutor
    features: FeatureSource
    # live (P5): (slot, intent, signal_id, action, paper_trade_id, bar_time, decided_at, decision_px)
    on_signal: Callable[[Slot, Intent, int, str, str | None, int, int, float], None] | None = None

    cooldown_until: dict[str, int] = field(default_factory=dict)   # slot_id → první bar, kdy smí vstoupit
    seq: dict[str, int] = field(default_factory=dict)
    last_features: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.executor.on_closed = self.on_trade_closed
        self.executor.on_state = self.save_state
        cd = self.cfg.engine.cooldown_bars * BAR_MS
        for r in self.db.execute("SELECT slot_id, MAX(exit_time) AS t FROM trades WHERE mode='paper' "
                                 "GROUP BY slot_id").fetchall():
            self.cooldown_until[r["slot_id"]] = (int(r["t"]) // BAR_MS) * BAR_MS + cd
        for r in self.db.execute("SELECT slot_id, trade_id FROM trades WHERE mode='paper'").fetchall():
            n = int(str(r["trade_id"]).rsplit("-", 1)[-1])
            self.seq[r["slot_id"]] = max(self.seq.get(r["slot_id"], 0), n)
        states = {r["slot_id"]: json.loads(r["state_json"]) for r in self.db.execute(
            "SELECT slot_id, state_json FROM slot_state WHERE mode='paper' AND status='in_position'").fetchall()}
        self.executor.restore(states)
        for sid, st in states.items():
            self.seq[sid] = max(self.seq.get(sid, 0), int(st["seq"]))
        for s in self.slots:
            self.db.execute("INSERT OR IGNORE INTO slots VALUES (?,?,?,?,?,?,?,?)",
                            (s.slot_id, s.strategy.family, s.strategy.variant_id, s.symbol,
                             json.dumps(dict(s.strategy.params), sort_keys=True),
                             self.cfg.variants[s.strategy.variant_id].params_hash, int(s.is_control),
                             wall_ms()))

    # --- persistence --------------------------------------------------------

    def save_state(self, slot_id: str, state: dict[str, Any] | None) -> None:
        self.db.execute("INSERT OR REPLACE INTO slot_state VALUES (?,?,?,?,?)",
                        (slot_id, "paper", "in_position" if state else "flat",
                         json.dumps(state) if state else None, wall_ms()))

    def on_trade_closed(self, rec: TradeRecord) -> None:
        row = rec.row()
        cols = ",".join(row)
        self.db.execute(f"INSERT OR REPLACE INTO trades({cols}) VALUES ({','.join('?' * len(row))})",
                        tuple(row.values()))
        self.cooldown_until[rec.slot_id] = (rec.exit_time // BAR_MS) * BAR_MS + self.cfg.engine.cooldown_bars * BAR_MS
        self.db.event("INFO", "paper", f"{rec.slot_id} {rec.exit_reason} {rec.net_bps:+.1f} bps",
                      {"trade_id": rec.trade_id, **rec.extra})

    # --- bar ----------------------------------------------------------------

    def ctl_extra(self, slot: Slot, t: int) -> dict[str, float]:
        first = self.db.meta_get("first_bar_time")
        if first is None or t - int(first) < CTL_LOOKBACK_BARS * BAR_MS:
            return {}
        target = f"{slot.strategy.params['target']}.v00.{slot.symbol}"
        r = self.db.execute("SELECT COUNT(*) n, SUM(side='LONG') l FROM signals WHERE slot_id=? AND bar_time>?",
                            (target, t - CTL_LOOKBACK_BARS * BAR_MS)).fetchone()
        n = int(r["n"] or 0)
        return {"ctl_entry_prob": n / CTL_LOOKBACK_BARS, "ctl_long_share": (int(r["l"] or 0) / n) if n else 0.5}

    def on_bar_close(self, t: int, symbols: frozenset[str], decided_at: int,
                     stale_symbols: frozenset[str] = frozenset()) -> list[int]:
        """Vrací ID zapsaných signálů."""
        if self.db.meta_get("first_bar_time") is None:
            self.db.meta_set("first_bar_time", str(t))
        closes: dict[str, float] = {}
        feats: dict[str, dict[str, float] | None] = {}
        btc = self.features.arrays("BTCUSDT", t) if "BTCUSDT" in symbols else None
        for sym in sorted(symbols):
            if sym in stale_symbols:
                feats[sym] = None
                continue
            f = self.features.features(sym, t, btc)
            feats[sym] = f
            if f is not None:
                closes[sym] = f["c"]
                self.last_features[sym] = f
        self.executor.on_bar_close(t, decided_at, closes)

        ids: list[int] = []
        stale_book: set[str] = set()
        for slot in self.slots:
            f = feats.get(slot.symbol)
            if f is None:
                continue
            if slot.strategy.family == "ll" or (slot.is_control and slot.strategy.params.get("target") == "ll"):
                if "ret1_btc" not in f:
                    continue   # BTC pro tento bar chybí
            extra = self.ctl_extra(slot, t) if slot.is_control else {}
            in_pos = slot.slot_id in self.executor.positions
            cooling = t < self.cooldown_until.get(slot.slot_id, 0)
            ctx = BarContext(slot.symbol, t, f, slot.slot_id,
                             "in_position" if in_pos else ("cooldown" if cooling else "flat"), extra=extra)
            if slot.strategy.is_stale(ctx):
                stale_book.add(slot.symbol)
                continue
            intent = slot.strategy.on_bar(ctx)
            if intent is None:
                continue
            action = "suppressed_position" if in_pos else ("suppressed_cooldown" if cooling else "entered")
            sid = self._log_signal(slot, t, intent, action, decided_at)
            ids.append(sid)
            paper_tid = None
            if action == "entered":
                self.seq[slot.slot_id] = self.seq.get(slot.slot_id, 0) + 1
                paper_tid = self.executor.open(slot.slot_id, slot.symbol, self.seq[slot.slot_id], intent, sid, t,
                                               decided_at, f["c"], self.cfg.config_hash).trade_id
            if self.on_signal is not None and not slot.is_control:
                self.on_signal(slot, intent, sid, action, paper_tid, t, decided_at, f["c"])
        for sym in sorted(stale_book):
            self.db.event("WARN", "obi", "bar bez platné knihy (book_ok=0), obi mlčí", {"symbol": sym, "bar": t})
        return ids

    def _log_signal(self, slot: Slot, t: int, intent: Intent, action: str, decided_at: int) -> int:
        cur = self.db.execute("INSERT INTO signals(slot_id, bar_time, side, intent_json, action, decided_at, "
                              "config_hash) VALUES (?,?,?,?,?,?,?)",
                              (slot.slot_id, t, intent.side, json.dumps(asdict(intent)), action, decided_at,
                               self.cfg.config_hash))
        return int(cur.lastrowid)


@dataclass
class FundingTracker:
    """Funding z markPrice streamu: když přijde event s E ≥ předchozí `T` (next funding time), funding proběhl
    v čase T se sazbou `r` z posledního eventu před ním (SPEC §7)."""

    on_funding: Callable[[str, int, float, float], None]
    last: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    def on_mark(self, e: Mapping[str, Any]) -> None:
        sym = e["s"]
        prev = self.last.get(sym)
        if prev is not None:
            t_f = int(prev["T"])
            if int(e["E"]) >= t_f > int(prev["E"]):
                rate = float(prev["r"])
                if math.isfinite(rate):
                    self.on_funding(sym, t_f, rate, float(prev["p"]))
        self.last[sym] = e
