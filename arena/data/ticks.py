"""Dispatcher ticků (SPEC §4.3): aggTrade → odběratelé symbolu (TrailingSim otevřených paper pozic),
poslední bookTicker a markPrice na symbol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

TradeHandler = Callable[[Mapping[str, Any]], None]


@dataclass
class TickDispatcher:
    book_ticker: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    mark: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    last_trade: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    _subs: dict[str, list[TradeHandler]] = field(default_factory=dict)

    def subscribe(self, symbol: str, handler: TradeHandler) -> None:
        self._subs.setdefault(symbol, []).append(handler)

    def unsubscribe(self, symbol: str, handler: TradeHandler) -> None:
        subs = self._subs.get(symbol, [])
        if handler in subs:
            subs.remove(handler)

    def on_agg_trade(self, e: Mapping[str, Any]) -> None:
        sym = e["s"]
        prev = self.last_trade.get(sym)
        # pořadí podle T / a; starší nebo duplicitní trade (reconnect) se zahodí
        if prev is not None and int(e["a"]) <= int(prev["a"]):
            return
        self.last_trade[sym] = e
        for h in list(self._subs.get(sym, ())):
            h(e)

    def on_book_ticker(self, e: Mapping[str, Any]) -> None:
        self.book_ticker[e["s"]] = e

    def on_mark_price(self, e: Mapping[str, Any]) -> None:
        self.mark[e["s"]] = e
