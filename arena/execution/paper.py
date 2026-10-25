"""Paper executor (SPEC §7): realistické fill, TrailingSim, funding, výpadky. Fixní notional, bez skládání.

Čas je vždy explicitní (ms UTC, čas burzy): live engine volá `advance(now)` z časovače, replay ho volá
s časy z nahraných dat. Díky tomu je replay deterministický."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Mapping

from arena.binance.filters import SymbolFilters
from arena.data.bars import BAR_MS
from arena.data.book import LocalBook
from arena.data.ticks import TickDispatcher
from arena.execution.base import FillModel, FillQuote, OrderSide, TradeRecord, bps, side_sign
from arena.execution.ids import paper_trade_id
from arena.execution.sizing import entry_qty, plan_exits
from arena.execution.trailing_sim import ExitTrigger, TrailingSim
from arena.strategies.base import Intent

log = logging.getLogger(__name__)

EXIT_REASONS = {"TS": "trailing", "HS": "hard_stop", "TP": "take_profit", "TX": "time", "RC": "reconciled"}


# --- fill modely ---------------------------------------------------------------------------------

@dataclass
class BookFillModel:
    """Live paper: market fill projde lokální knihu. Když kniha není synchronizovaná nebo je starší než 1 s,
    použije bookTicker (nebo poslední obchod) + `stale_book_penalty_bps` (SPEC §4.3)."""

    books: Mapping[str, LocalBook]
    ticks: TickDispatcher
    local_now_ms: Callable[[], int]
    stale_book_penalty_bps: float = 1.0
    extra_slippage_bps: Mapping[str, float] = field(default_factory=dict)
    max_book_age_ms: float = 1000.0

    def _adj(self, symbol: str, side: OrderSide, px: float, extra_bps: float = 0.0) -> float:
        k = (self.extra_slippage_bps.get(symbol, 0.0) + extra_bps) / 1e4
        return px * (1 + k) if side == "BUY" else px * (1 - k)

    def market(self, symbol: str, side: OrderSide, qty: float, t_ms: int,
               trade: dict[str, Any] | None = None) -> FillQuote | None:
        book = self.books.get(symbol)
        if book is not None and book.is_fresh(self.local_now_ms(), self.max_book_age_ms):
            mf = book.simulate_market(side, qty)
            if mf is not None and mf.complete:
                return FillQuote(self._adj(symbol, side, mf.vwap), "book")
        bt = self.ticks.book_ticker.get(symbol)
        if bt is not None:
            px = float(bt["a"]) if side == "BUY" else float(bt["b"])
            return FillQuote(self._adj(symbol, side, px, self.stale_book_penalty_bps), "ticker")
        last = self.last_price(symbol)
        if last is not None:
            return FillQuote(self._adj(symbol, side, last, self.stale_book_penalty_bps), "trade")
        return None

    def mid(self, symbol: str) -> float | None:
        book = self.books.get(symbol)
        if book is not None and book.synced:
            b = book.best()
            if b is not None:
                return (b[0] + b[1]) / 2
        bt = self.ticks.book_ticker.get(symbol)
        return (float(bt["a"]) + float(bt["b"])) / 2 if bt else None

    def last_price(self, symbol: str) -> float | None:
        t = self.ticks.last_trade.get(symbol)
        return float(t["p"]) if t else None


@dataclass
class TradeFillModel:
    """Replay bez knihy: fill = cena prvního obchodu v čase ≥ t (předá ho executor) + extra slippage."""

    extra_slippage_bps: Mapping[str, float] = field(default_factory=dict)
    last: dict[str, float] = field(default_factory=dict)

    def market(self, symbol: str, side: OrderSide, qty: float, t_ms: int,
               trade: dict[str, Any] | None = None) -> FillQuote | None:
        px = float(trade["p"]) if trade is not None else self.last.get(symbol)
        if px is None:
            return None
        k = self.extra_slippage_bps.get(symbol, 0.0) / 1e4
        return FillQuote(px * (1 + k) if side == "BUY" else px * (1 - k), "trade")

    def mid(self, symbol: str) -> float | None:
        return self.last.get(symbol)

    def last_price(self, symbol: str) -> float | None:
        return self.last.get(symbol)


# --- pozice --------------------------------------------------------------------------------------

@dataclass
class PaperPosition:
    slot_id: str
    symbol: str
    seq: int
    trade_id: str
    side: Literal["LONG", "SHORT"]
    signal_id: int | None
    entry_bar: int                    # open_time signálního baru
    decision_ts: int
    decision_px: float
    arrival_mid: float | None
    intent: dict[str, Any]
    config_hash: str
    status: Literal["entering", "open", "exiting"] = "entering"
    due_ts: int = 0                   # čas plánovaného fillu (vstup nebo market výstup)
    qty: float = 0.0
    entry_px: float = 0.0
    entry_time: int = 0
    entry_fee: float = 0.0
    entry_source: str = ""
    plan: dict[str, Any] = field(default_factory=dict)
    sim: TrailingSim | None = None
    funding: float = 0.0
    tainted: int = 0
    exit_kind: str | None = None
    exit_trigger_px: float | None = None
    exit_trigger_ts: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["sim"] = self.sim.to_json() if self.sim else None
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> PaperPosition:
        d = dict(d)
        d["sim"] = TrailingSim.from_json(d["sim"]) if d.get("sim") else None
        return cls(**d)


@dataclass
class PaperExecutor:
    filters: Mapping[str, SymbolFilters]
    fill: FillModel
    notional_usdt: float = 1000.0
    entry_latency_ms: int = 250
    trigger_latency_ms: int = 100
    outage_penalty_bps: float = 2.0
    taker_fee: Mapping[str, float] = field(default_factory=dict)    # skutečné sazby účtu, default VIP0
    maker_fee: Mapping[str, float] = field(default_factory=dict)
    cost_model_version: int = 1
    on_closed: Callable[[TradeRecord], None] | None = None
    on_state: Callable[[str, dict[str, Any] | None], None] | None = None     # (slot_id, state|None)
    on_event: Callable[[str, str, dict[str, Any]], None] | None = None
    # mezera v aggTrade ID (reconnect WS, restart): (symbol, první chybějící ID, poslední chybějící ID).
    # Volající stáhne chybějící obchody a zavolá `fill_gap`; mezitím se nové obchody symbolu drží v bufferu.
    # Bez callbacku (replay) mezera jen označí otevřené pozice symbolu jako tainted.
    on_gap: Callable[[str, int, int], None] | None = None

    positions: dict[str, PaperPosition] = field(default_factory=dict)
    _last_agg: dict[str, int] = field(default_factory=dict)
    _hold: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def taker(self, sym: str) -> float:
        return self.taker_fee.get(sym, 0.0005)

    def maker(self, sym: str) -> float:
        return self.maker_fee.get(sym, 0.0002)

    # --- vstup --------------------------------------------------------------

    def open(self, slot_id: str, symbol: str, seq: int, intent: Intent, signal_id: int | None, bar_time: int,
             decision_ts: int, decision_px: float, config_hash: str) -> PaperPosition:
        if slot_id in self.positions:
            raise RuntimeError(f"{slot_id}: pozice už existuje")
        pos = PaperPosition(slot_id, symbol, seq, paper_trade_id(slot_id, seq), intent.side, signal_id, bar_time,
                            decision_ts, decision_px, self.fill.mid(symbol), asdict(intent), config_hash,
                            due_ts=decision_ts + self.entry_latency_ms)
        self.positions[slot_id] = pos
        self._save(pos)
        return pos

    def _fill_entry(self, pos: PaperPosition, now: int, trade: dict[str, Any] | None) -> None:
        f = self.filters[pos.symbol]
        intent = Intent(**pos.intent)
        try:
            qty = entry_qty(f, self.notional_usdt, pos.decision_px)
        except ValueError as e:
            self._emit("ERROR", "paper vstup pod filtry", {"slot": pos.slot_id, "err": str(e)})
            self._drop(pos)
            return
        side: OrderSide = "BUY" if pos.side == "LONG" else "SELL"
        q = self.fill.market(pos.symbol, side, float(qty), now, trade)
        if q is None:
            self._emit("ERROR", "paper vstup bez ceny", {"slot": pos.slot_id})
            self._drop(pos)
            return
        plan = plan_exits(intent, f, q.px, qty)
        pos.qty, pos.entry_px, pos.entry_time, pos.entry_source = float(qty), q.px, now, q.source
        pos.entry_fee = pos.qty * q.px * self.taker(pos.symbol)
        tp = float(plan.tp_px) if plan.tp_px is not None else None
        last = self.fill.last_price(pos.symbol) or q.px
        if tp is not None and ((pos.side == "LONG" and tp <= last) or (pos.side == "SHORT" and tp >= last)):
            pos.notes.append("tp_skipped_marketable")
            self._emit("INFO", "tp_skipped_marketable", {"slot": pos.slot_id, "tp": tp, "last": last})
            tp = None
        pos.notes.extend(plan.notes)
        pos.plan = {"callback_rate": float(plan.callback_rate) if plan.callback_rate is not None else None,
                    "activate_px": float(plan.activate_px) if plan.activate_px is not None else None,
                    "hard_stop_px": float(plan.hard_stop_px), "tp_px": tp, "max_hold_bars": plan.max_hold_bars}
        pos.sim = TrailingSim(pos.side, q.px, float(plan.hard_stop_px), pos.plan["callback_rate"],
                              pos.plan["activate_px"], tp, start_ts=now)
        pos.status = "open"
        self._save(pos)

    # --- události trhu ------------------------------------------------------

    def _open_with_sim(self, symbol: str) -> list[PaperPosition]:
        return [p for p in self.positions.values() if p.symbol == symbol and p.status == "open" and p.sim is not None]

    def on_trade(self, symbol: str, trade: dict[str, Any]) -> None:
        """aggTrade (dict s p, T, a). Pořadí obchodů zajišťuje TickDispatcher."""
        ts, px, aid = int(trade["T"]), float(trade["p"]), int(trade["a"])
        if symbol in self._hold:
            self._hold[symbol].append(trade)
            return
        prev = self._last_agg.get(symbol)
        if prev is None:   # po startu: navázat na poslední obchod, který viděly simy (po přehrání výpadku)
            prev = max((p.sim.last_agg for p in self._open_with_sim(symbol) if p.sim and p.sim.last_agg >= 0),
                       default=None)
        if prev is not None and aid <= prev:
            return
        if prev is not None and aid > prev + 1 and self._open_with_sim(symbol):
            if self.on_gap is not None:
                self._hold[symbol] = [trade]
                self._emit("WARN", "mezera v aggTrades u otevřených pozic, doplňuje se",
                           {"symbol": symbol, "from_id": prev + 1, "to_id": aid - 1})
                self.on_gap(symbol, prev + 1, aid - 1)
                return
            self._taint(symbol, "mezera v aggTrades", {"from_id": prev + 1, "to_id": aid - 1})
        self._last_agg[symbol] = aid
        if isinstance(self.fill, TradeFillModel):
            self.fill.last[symbol] = px
        for pos in [p for p in self.positions.values() if p.symbol == symbol]:
            if pos.status in ("entering", "exiting") and pos.due_ts <= ts:
                self._resolve(pos, ts, trade)
            if pos.status == "open" and pos.sim is not None:
                trig = pos.sim.on_trade(px, ts, aid)
                if trig is not None:
                    self._on_trigger(pos, trig)
                    if pos.status == "exiting" and pos.due_ts <= ts:
                        self._resolve(pos, ts, trade)

    def advance(self, now: int) -> None:
        """Live: dořeš fill, jejichž čas nastal (časovač engine)."""
        for pos in list(self.positions.values()):
            if pos.status == "open" and pos.sim is not None and pos.sim.pending is not None \
                    and now >= pos.sim.pending.ts + self.trigger_latency_ms:
                trig = pos.sim.flush()
                if trig is not None:
                    self._on_trigger(pos, trig)
            if pos.status in ("entering", "exiting") and pos.due_ts <= now:
                self._resolve(pos, now, None)

    def _on_trigger(self, pos: PaperPosition, trig: ExitTrigger) -> None:
        pos.exit_kind, pos.exit_trigger_px, pos.exit_trigger_ts = trig.kind, trig.trigger_px, trig.ts
        if trig.is_market:
            pos.status, pos.due_ts = "exiting", trig.ts + self.trigger_latency_ms
            self._save(pos)
        else:   # TP LIMIT: maker fill přesně za limit v čase obchodu
            self._close(pos, trig.trigger_px, trig.ts, self.maker(pos.symbol), "book")

    def _resolve(self, pos: PaperPosition, now: int, trade: dict[str, Any] | None) -> None:
        if pos.status == "entering":
            self._fill_entry(pos, now, trade)
            return
        side: OrderSide = "SELL" if pos.side == "LONG" else "BUY"
        q = self.fill.market(pos.symbol, side, pos.qty, now, trade)
        if q is None:
            self._emit("ERROR", "paper výstup bez ceny, zkusí se znovu", {"slot": pos.slot_id})
            pos.due_ts = now + 1000
            return
        self._close(pos, q.px, now, self.taker(pos.symbol), q.source)

    def on_bar_close(self, open_time: int, decided_at: int, closes: Mapping[str, float]) -> None:
        """Časový výstup: na uzavření baru entry_bar + max_hold_bars (market přes knihu)."""
        for pos in list(self.positions.values()):
            if pos.status != "open" or pos.symbol not in closes:
                continue
            if open_time >= pos.entry_bar + int(pos.plan["max_hold_bars"]) * BAR_MS:
                if pos.sim is not None and pos.sim.pending is not None:
                    continue   # výstup z burzy už běží
                pos.exit_kind, pos.exit_trigger_px, pos.exit_trigger_ts = "TX", closes[pos.symbol], decided_at
                pos.status, pos.due_ts = "exiting", decided_at + self.entry_latency_ms
                self._save(pos)
            elif pos.sim is not None:
                self._save(pos)   # průběžně ulož HWM a poslední zpracovaný obchod

    def on_funding(self, symbol: str, funding_time: int, rate: float, mark_px: float) -> None:
        """`pnl -= sign × notional × fundingRate` pro pozice otevřené v čase fundingu (SPEC §7)."""
        for pos in self.positions.values():
            if pos.symbol == symbol and pos.status in ("open", "exiting") and pos.entry_time < funding_time:
                pos.funding -= side_sign(pos.side) * pos.qty * mark_px * rate
                self._save(pos)

    # --- výpadek ------------------------------------------------------------

    def _taint(self, symbol: str, why: str, data: dict[str, Any]) -> None:
        for pos in self._open_with_sim(symbol):
            if not pos.tainted:
                pos.tainted = 1
                self._save(pos)
        self._emit("WARN", f"tainted: {why}", {"symbol": symbol, **data})

    def fill_gap(self, symbol: str, trades: list[dict[str, Any]], complete: bool, now: int) -> None:
        """Doplnění mezery z `on_gap`: obchody z REST přehrát (výstup v mezeře = fill bez knihy → tainted),
        neúplná data → tainted, pak zpracovat obchody držené v bufferu."""
        held = self._hold.pop(symbol, [])
        if complete:
            self.replay_outage(symbol, trades, now, None, restart=False)
        else:
            self._taint(symbol, "mezeru v aggTrades se nepodařilo doplnit", {"n": len(trades)})
        if held:
            self._last_agg[symbol] = int(held[0]["a"]) - 1
        for tr in held:
            self.on_trade(symbol, tr)

    def replay_outage(self, symbol: str, trades: list[dict[str, Any]], now: int,
                      bar_closes: Mapping[int, float] | None = None, restart: bool = True,
                      complete: bool = True) -> None:
        """Přehraj chybějící aggTrades do TrailingSim (po restartu nebo z `fill_gap`). Kniha za výpadek není:
        výstup, který nastal během výpadku, má fill = cena obchodu ± `outage_penalty_bps` a `tainted = 1` (SPEC §7).
        Pozice, která výpadek přečkala bez výstupu, tainted není, pokud jsou přehraná data úplná (`complete`)."""
        k = self.outage_penalty_bps / 1e4
        for pos in [p for p in self.positions.values() if p.symbol == symbol]:
            if restart and pos.status == "entering":
                self._drop(pos)   # vstup za výpadku se neprovedl
                continue
            if pos.status != "open" or pos.sim is None:
                continue
            if not complete:
                pos.tainted = 1
            deadline = pos.entry_bar + (int(pos.plan["max_hold_bars"]) + 1) * BAR_MS
            for tr in trades:
                ts = int(tr["T"])
                if ts >= deadline and bar_closes:
                    # časový výstup nastal během výpadku: close baru ± penalizace
                    px = bar_closes.get(deadline - BAR_MS)
                    if px is not None:
                        pos.exit_kind, pos.exit_trigger_px, pos.exit_trigger_ts = "TX", px, deadline
                        pos.tainted = 1
                        self._close(pos, px * (1 - side_sign(pos.side) * k), deadline, self.taker(symbol), "outage")
                        break
                trig = pos.sim.on_trade(float(tr["p"]), ts, int(tr["a"]))
                if trig is None and tr is trades[-1]:
                    trig = pos.sim.flush()
                if trig is not None:
                    pos.exit_kind, pos.exit_trigger_px, pos.exit_trigger_ts = trig.kind, trig.trigger_px, trig.ts
                    pos.tainted = 1
                    if trig.is_market:
                        px = float(tr["p"]) * (1 - side_sign(pos.side) * k)
                        self._close(pos, px, ts, self.taker(symbol), "outage")
                    else:
                        self._close(pos, trig.trigger_px, trig.ts, self.maker(symbol), "outage")
                    break
            else:
                self._save(pos)

    # --- uzavření -----------------------------------------------------------

    def _close(self, pos: PaperPosition, exit_px: float, exit_time: int, fee_rate: float, source: str) -> None:
        sg = side_sign(pos.side)
        notional = pos.qty * pos.entry_px
        gross = sg * (exit_px - pos.entry_px) * pos.qty
        exit_fee = pos.qty * exit_px * fee_rate
        net = gross - pos.entry_fee - exit_fee + pos.funding
        sim = pos.sim
        trig_px = pos.exit_trigger_px if pos.exit_trigger_px is not None else exit_px
        rec = TradeRecord(
            trade_id=pos.trade_id, slot_id=pos.slot_id, mode="paper", signal_id=pos.signal_id, side=pos.side,
            qty=pos.qty, notional=notional, decision_px=pos.decision_px, arrival_mid=pos.arrival_mid,
            entry_px=pos.entry_px, entry_time=pos.entry_time, entry_fee=pos.entry_fee,
            exit_trigger_px=trig_px, exit_px=exit_px, exit_time=exit_time, exit_fee=exit_fee,
            exit_reason=EXIT_REASONS.get(pos.exit_kind or "", pos.exit_kind or "?"), funding=pos.funding,
            gross_pnl=gross, net_pnl=net, gross_bps=bps(gross, notional), net_bps=bps(net, notional),
            entry_slip_bps=sg * (pos.entry_px - pos.decision_px) / pos.decision_px * 1e4,
            exit_slip_bps=sg * (trig_px - exit_px) / trig_px * 1e4 if trig_px else 0.0,
            mfe_bps=(sg * (sim.mfe_px - pos.entry_px) / pos.entry_px * 1e4) if sim and sim.mfe_px else None,
            mae_bps=(sg * (sim.mae_px - pos.entry_px) / pos.entry_px * 1e4) if sim and sim.mae_px else None,
            bars_held=max(0, (exit_time - (pos.entry_bar + BAR_MS)) // BAR_MS),
            tainted=pos.tainted, cost_model_version=self.cost_model_version, config_hash=pos.config_hash,
            extra={"entry_source": pos.entry_source, "exit_source": source, "notes": pos.notes, "plan": pos.plan},
        )
        del self.positions[pos.slot_id]
        if self.on_state is not None:
            self.on_state(pos.slot_id, None)
        if self.on_closed is not None:
            self.on_closed(rec)

    def _drop(self, pos: PaperPosition) -> None:
        self.positions.pop(pos.slot_id, None)
        if self.on_state is not None:
            self.on_state(pos.slot_id, None)

    def _save(self, pos: PaperPosition) -> None:
        if self.on_state is not None:
            self.on_state(pos.slot_id, pos.to_json())

    def _emit(self, level: str, msg: str, data: dict[str, Any]) -> None:
        log.log(logging.WARNING if level in ("WARN", "ERROR") else logging.INFO, "%s %s", msg, data)
        if self.on_event is not None:
            self.on_event(level, msg, data)

    def restore(self, states: Mapping[str, dict[str, Any]]) -> None:
        for slot_id, st in states.items():
            self.positions[slot_id] = PaperPosition.from_json(st)
