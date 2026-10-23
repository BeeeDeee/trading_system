"""Kontrolní rodina ctl (SPEC §5.7): náhodný vstup s frekvencí a směrem jako V00, výstupy identické s V00."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Mapping

from arena.strategies.base import BarContext, Intent, Side, Strategy


def slot_rng(slot_id: str, bar_time: int) -> random.Random:
    """RNG seedovaný hashem (slot_id, bar_time): deterministický v live i replay."""
    seed = int.from_bytes(hashlib.sha256(f"{slot_id}|{bar_time}".encode()).digest()[:8], "big")
    return random.Random(seed)


@dataclass(frozen=True)
class Control:
    family: str
    variant_id: str
    params: Mapping[str, Any]
    symbols: tuple[str, ...]
    target: Strategy          # V00 cílové rodiny

    @property
    def warmup_bars(self) -> int:
        return self.target.warmup_bars

    def is_stale(self, ctx: BarContext) -> bool:
        return self.target.is_stale(ctx)

    def exits(self, ctx: BarContext, side: Side, reason: str) -> Intent | None:
        return self.target.exits(ctx, side, reason)

    def on_bar(self, ctx: BarContext) -> Intent | None:
        if self.is_stale(ctx):
            return None
        # empirická frekvence a podíl LONG u V00 za posledních 7 dní dodá engine; jinak prior z konfigurace
        p = float(ctx.extra.get("ctl_entry_prob", self.params["prior_entry_prob"]))
        long_share = float(ctx.extra.get("ctl_long_share", self.params["prior_long_share"]))
        rng = slot_rng(ctx.slot_id, ctx.bar_time)
        if rng.random() >= p:
            return None
        side: Side = "LONG" if rng.random() < long_share else "SHORT"
        return self.exits(ctx, side, f"ctl náhodný vstup (p={p:.4f})")
