"""TrailingSim: simulace výstupů na burze podle last price, CONTRACT_PRICE (SPEC §7).

- trailing LONG (SELL): s activatePrice se aktivuje prvním obchodem ≥ activatePrice, jinak hned při vstupu
  s HWM = vstupní cena; po aktivaci HWM = max(HWM, last), spustí se, když last ≤ HWM × (1 − cb/100).
  SHORT zrcadlově (LWM, last ≥ LWM × (1 + cb/100)).
- tvrdý stop: LONG last ≤ stop, SHORT last ≥ stop.
- TP LIMIT: fill jen když proběhne obchod STRIKTNĚ za limitní cenou (dotek nestačí), cena = limit.
- dva výstupy se stejným časem obchodu → vyhrává horší výsledek (konzervativně). Výstup se proto potvrdí až
  obchodem s vyšším časem nebo voláním `flush()`.

Čistá logika bez I/O a bez času zdi; ceny jako float (paper a statistika)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

ExitKind = Literal["TS", "HS", "TP"]


@dataclass(frozen=True)
class ExitTrigger:
    kind: ExitKind
    trigger_px: float      # TS: teoretická úroveň HWM×(1−cb); HS: stop; TP: limit
    ts: int                # čas obchodu, který výstup spustil
    agg_id: int
    is_market: bool        # TS/HS → market fill přes knihu; TP → maker fill za limit


@dataclass
class TrailingSim:
    side: Literal["LONG", "SHORT"]
    entry_px: float
    hard_stop_px: float
    callback_pct: float | None = None
    activate_px: float | None = None
    tp_px: float | None = None
    start_ts: int = 0            # obchody před tímto časem (před fillem vstupu) se ignorují

    activated: bool = False
    extreme: float | None = None   # HWM (LONG) / LWM (SHORT) od aktivace
    last_ts: int = 0
    last_agg: int = -1
    mfe_px: float | None = None
    mae_px: float | None = None
    pending: ExitTrigger | None = None
    done: ExitTrigger | None = None

    def __post_init__(self) -> None:
        if self.callback_pct is not None and self.activate_px is None and self.extreme is None:
            self.activated = True
            self.extreme = self.entry_px

    @property
    def trail_level(self) -> float | None:
        if not self.activated or self.extreme is None or self.callback_pct is None:
            return None
        k = self.callback_pct / 100
        return self.extreme * (1 - k) if self.side == "LONG" else self.extreme * (1 + k)

    def _worse(self, a: ExitTrigger, b: ExitTrigger) -> ExitTrigger:
        """Horší výsledek pro pozici: LONG nižší výstupní cena, SHORT vyšší."""
        if self.side == "LONG":
            return a if a.trigger_px <= b.trigger_px else b
        return a if a.trigger_px >= b.trigger_px else b

    def on_trade(self, px: float, ts: int, agg_id: int) -> ExitTrigger | None:
        """Zpracuj jeden aggTrade. Vrací potvrzený výstup (když obchod s vyšším časem uzavře dávku)."""
        if self.done is not None or ts < self.start_ts or agg_id <= self.last_agg:
            return None
        confirmed = None
        if self.pending is not None and ts > self.pending.ts:
            confirmed = self.done = self.pending
            self.pending = None
            return confirmed
        self.last_ts, self.last_agg = ts, agg_id
        if self.mfe_px is None:
            self.mfe_px = self.mae_px = px
        if self.side == "LONG":
            self.mfe_px, self.mae_px = max(self.mfe_px, px), min(self.mae_px, px)  # type: ignore[arg-type]
        else:
            self.mfe_px, self.mae_px = min(self.mfe_px, px), max(self.mae_px, px)  # type: ignore[arg-type]

        cand: list[ExitTrigger] = []
        long = self.side == "LONG"
        # TP: striktně za limitem
        if self.tp_px is not None and ((long and px > self.tp_px) or (not long and px < self.tp_px)):
            cand.append(ExitTrigger("TP", self.tp_px, ts, agg_id, False))
        # tvrdý stop
        if (long and px <= self.hard_stop_px) or (not long and px >= self.hard_stop_px):
            cand.append(ExitTrigger("HS", self.hard_stop_px, ts, agg_id, True))
        # trailing
        if self.callback_pct is not None:
            if not self.activated and self.activate_px is not None:
                if (long and px >= self.activate_px) or (not long and px <= self.activate_px):
                    self.activated = True
                    self.extreme = px
            if self.activated:
                assert self.extreme is not None
                self.extreme = max(self.extreme, px) if long else min(self.extreme, px)
                lvl = self.trail_level
                assert lvl is not None
                if (long and px <= lvl) or (not long and px >= lvl):
                    cand.append(ExitTrigger("TS", lvl, ts, agg_id, True))
        for c in cand:
            self.pending = c if self.pending is None else self._worse(self.pending, c)
        return None

    def flush(self) -> ExitTrigger | None:
        """Potvrď čekající výstup (konec dávky, čas fillu, časový výstup)."""
        if self.pending is not None and self.done is None:
            self.done, self.pending = self.pending, None
            return self.done
        return None

    # --- persistence (slot_state.state_json) --------------------------------

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> TrailingSim:
        d = dict(d)
        for k in ("pending", "done"):
            if d.get(k) is not None:
                d[k] = ExitTrigger(**d[k])
        return cls(**d)
