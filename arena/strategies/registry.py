"""Sestavení strategií a slotů z konfigurace."""

from __future__ import annotations

from dataclasses import dataclass

from arena.config import Config
from arena.strategies.base import Strategy
from arena.strategies.ctl import Control
from arena.strategies.families import BookImbalance, Breakout, LeadLag
from arena.strategies.xp import ExecProbe

FAMILY_CLASSES = {"brk": Breakout, "obi": BookImbalance, "ll": LeadLag, "xp": ExecProbe}


@dataclass(frozen=True)
class Slot:
    slot_id: str           # "<variant_id>.<symbol>", u kontrol "ctl.<family>.<symbol>"
    strategy: Strategy
    symbol: str

    @property
    def is_control(self) -> bool:
        return self.strategy.family == "ctl"


def build_strategies(cfg: Config) -> dict[str, Strategy]:
    out: dict[str, Strategy] = {}
    for vid, v in sorted(cfg.variants.items()):
        if v.family in FAMILY_CLASSES:
            out[vid] = FAMILY_CLASSES[v.family](v.family, vid, dict(v.params), v.symbols)
    for vid, v in sorted(cfg.variants.items()):
        if v.family == "ctl":
            target = out[f"{v.params['target']}.v00"]
            out[vid] = Control("ctl", vid, dict(v.params), v.symbols, target)
    return out


def build_slots(cfg: Config) -> list[Slot]:
    return [Slot(f"{vid}.{sym}", s, sym) for vid, s in build_strategies(cfg).items() for sym in s.symbols]


def global_warmup(strategies: dict[str, Strategy]) -> int:
    """max warmup_bars všech strategií + 288, min. 864 (SPEC §4.2)."""
    return max(864, max(s.warmup_bars for s in strategies.values()) + 288)
