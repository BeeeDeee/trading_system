"""Rodiny brk, obi, ll (SPEC §5.4–5.6). Parametry přicházejí z config/strategies/*.yaml."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from arena.features import W
from arena.strategies.base import (BarContext, Intent, Side, common_callback, common_hard_stop, finite)


@dataclass(frozen=True)
class _Base:
    family: str
    variant_id: str
    params: Mapping[str, Any]
    symbols: tuple[str, ...]

    @property
    def warmup_bars(self) -> int:
        return W + 1

    def is_stale(self, ctx: BarContext) -> bool:
        return False

    def _allowed(self, side: Side) -> bool:
        return self.params["direction"] == "both" or side == "LONG"

    def _trailing_exits(self, ctx: BarContext, side: Side, reason: str) -> Intent | None:
        p = self.params
        cb = common_callback(ctx.features.get("atr14_pct", math.nan), p["k_cb"], p["cb_min"], p["cb_max"])
        if cb is None:
            return None
        return Intent(side=side, callback_pct=cb, activate_offset_pct=None,
                      hard_stop_pct=common_hard_stop(cb, p["hs_mult"], p["hs_min_pct"]),
                      take_profit_px=None, max_hold_bars=int(p["max_hold_bars"]), reason=reason)


@dataclass(frozen=True)
class Breakout(_Base):
    """brk: LONG `c > don_hi_N` a `lvz288 ≥ vz`, SHORT `c < don_lo_N` a `lvz288 ≥ vz`."""

    @property
    def warmup_bars(self) -> int:
        return max(W + 1, int(self.params["N"]) + 1)

    def exits(self, ctx: BarContext, side: Side, reason: str) -> Intent | None:
        it = self._trailing_exits(ctx, side, reason)
        am = self.params["activate_mult"]
        if it is None or am is None:
            return it
        # v09: aktivace trailingu až při zisku activate_mult × callback; do té doby jen tvrdý stop
        return Intent(it.side, it.callback_pct, round(am * it.callback_pct, 6), it.hard_stop_pct,
                      it.take_profit_px, it.max_hold_bars, it.reason)

    def on_bar(self, ctx: BarContext) -> Intent | None:
        f, p = ctx.features, self.params
        n = int(p["N"])
        c, hi, lo = f.get("c", math.nan), f.get(f"don_hi_{n}", math.nan), f.get(f"don_lo_{n}", math.nan)
        if not finite(c, hi, lo):
            return None
        if p["vz"] is not None:
            lvz = f.get("lvz288", math.nan)
            if not finite(lvz) or lvz < p["vz"]:
                return None
        if c > hi and self._allowed("LONG"):
            return self.exits(ctx, "LONG", f"c {c:g} > don_hi_{n} {hi:g}")
        if c < lo and self._allowed("SHORT"):
            return self.exits(ctx, "SHORT", f"c {c:g} < don_lo_{n} {lo:g}")
        return None


@dataclass(frozen=True)
class BookImbalance(_Base):
    """obi: LONG `obi{p}_z ≥ zt`, SHORT `≤ −zt`, jen při `book_ok`. Bar bez knihy je `stale`."""

    def is_stale(self, ctx: BarContext) -> bool:
        return ctx.features.get("book_ok", 0.0) != 1.0

    def exits(self, ctx: BarContext, side: Side, reason: str) -> Intent | None:
        it = self._trailing_exits(ctx, side, reason)
        tm = self.params["tp_mult"]
        if it is None or tm is None:
            return it
        c = ctx.features["c"]
        sign = 1 if side == "LONG" else -1
        tp = c * (1 + sign * tm * it.callback_pct / 100)
        return Intent(it.side, it.callback_pct, it.activate_offset_pct, it.hard_stop_pct, tp,
                      it.max_hold_bars, it.reason)

    def on_bar(self, ctx: BarContext) -> Intent | None:
        if self.is_stale(ctx):
            return None
        f, p = ctx.features, self.params
        z = f.get(f"obi{float(p['p']):g}_z", math.nan)
        if not finite(z):
            return None
        zt = float(p["zt"])
        side: Side | None = "LONG" if z >= zt else ("SHORT" if z <= -zt else None)
        if side is None or not self._allowed(side):
            return None
        if p["cvd_filter"]:
            cz = f.get("cvd12_z", math.nan)
            if not finite(cz) or (side == "LONG" and cz < 0) or (side == "SHORT" and cz > 0):
                return None
        return self.exits(ctx, side, f"obi{float(p['p']):g}_z {z:+.2f}")


@dataclass(frozen=True)
class LeadLag(_Base):
    """ll: `rb = ret1_btc / sigma288_btc`; LONG alt když `rb ≥ k` a `ret1_alt ≤ λ × beta288 × ret1_btc`."""

    @property
    def warmup_bars(self) -> int:
        ema = self.params["regime_ema"]
        return max(W + 2, 2 * int(ema) if ema else 0)

    def exits(self, ctx: BarContext, side: Side, reason: str) -> Intent | None:
        p = self.params
        if p["k_cb"] is None:
            # v09: bez trailingu, jen tvrdý stop + časový výstup
            return Intent(side, None, None, float(p["fixed_hard_stop_pct"]), None, int(p["max_hold_bars"]), reason)
        return self._trailing_exits(ctx, side, reason)

    def on_bar(self, ctx: BarContext) -> Intent | None:
        f, p = ctx.features, self.params
        rb1, ra1 = f.get("ret1_btc", math.nan), f.get("ret1", math.nan)
        sig, beta = f.get("sigma288_btc", math.nan), f.get("beta288", math.nan)
        if int(p["trigger_bars"]) == 2:
            rb_ret = rb1 + f.get("ret1_btc_prev", math.nan)
            ra_ret = ra1 + f.get("ret1_prev", math.nan)
            scale = sig * math.sqrt(2)
        else:
            rb_ret, ra_ret, scale = rb1, ra1, sig
        if not finite(rb_ret, ra_ret, scale, beta) or scale <= 0:
            return None
        rb = rb_ret / scale
        k, lam = float(p["k"]), float(p["lam"])
        side: Side | None = None
        if rb >= k and ra_ret <= lam * beta * rb_ret:
            side = "LONG"
        elif rb <= -k and ra_ret >= lam * beta * rb_ret:
            side = "SHORT"
        if side is None or not self._allowed(side):
            return None
        ema_n = p["regime_ema"]
        if ema_n:
            ema, cb = f.get(f"ema_{int(ema_n)}_btc", math.nan), f.get("c_btc", math.nan)
            if not finite(ema, cb) or (side == "LONG" and cb <= ema) or (side == "SHORT" and cb >= ema):
                return None
        return self.exits(ctx, side, f"rb {rb:+.2f}, alt {ra_ret * 1e4:+.1f} bps")
