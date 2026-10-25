"""Rodina xp, exec probe: testovací provoz live executoru na testnetu (ne součást experimentu, nepočítá se do N).

Náhodné vstupy (RNG seedovaný hashem slotu a baru, stejně jako ctl) s rotací výstupních mechanismů, aby každý
symbol procvičil všechny cesty executoru: trailing od vstupu, trailing s aktivací, jen tvrdý stop a časový výstup,
TP LIMIT a úzký trailing. Smysl: za dny nasbírat stovky round tripů a ověřit ochranu pozic, `ALGO_UPDATE`,
rekonciliaci a risk guard, na což by vzácné signály strategií potřebovaly týdny."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from arena.strategies.base import BarContext, Intent, Side
from arena.strategies.ctl import slot_rng

# název → (callback %, aktivace %, tvrdý stop %, TP vzdálenost %, držení v barech); None = nepoužít
MECHANISMS: dict[str, tuple[float | None, float | None, float, float | None, int]] = {
    "trail": (0.2, None, 1.0, None, 3),
    "trail_act": (0.2, 0.05, 1.0, None, 4),
    "stop_time": (None, None, 0.5, None, 2),
    "tp": (0.3, None, 1.0, 0.15, 4),
    "tight": (0.1, None, 0.15, None, 3),
}


@dataclass(frozen=True)
class ExecProbe:
    family: str
    variant_id: str
    params: Mapping[str, Any]
    symbols: tuple[str, ...]

    @property
    def warmup_bars(self) -> int:
        return 2

    def is_stale(self, ctx: BarContext) -> bool:
        return False

    def exits(self, ctx: BarContext, side: Side, reason: str) -> Intent | None:
        mech = str(reason.split(":", 1)[0])
        cb, act, hs, tp_pct, hold = MECHANISMS[mech]
        tp = None
        if tp_pct is not None:
            c = ctx.features.get("c")
            if not c:
                return None
            tp = c * (1 + (1 if side == "LONG" else -1) * tp_pct / 100)
        return Intent(side, cb, act, hs, tp, hold, reason)

    def on_bar(self, ctx: BarContext) -> Intent | None:
        if ctx.slot_status != "flat":     # bez šumu v tabulce signálů, probe se nepotlačuje, jen čeká
            return None
        rng = slot_rng(ctx.slot_id, ctx.bar_time)
        p = float(self.params["entry_prob"])
        if rng.random() >= p:
            return None
        side: Side = "LONG" if rng.random() < float(self.params["long_share"]) else "SHORT"
        names = list(self.params["mechanisms"])
        mech = names[int(rng.random() * len(names)) % len(names)]
        return self.exits(ctx, side, f"{mech}: exec probe (p={p:.3f})")
