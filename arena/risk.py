"""Risk guard (SPEC §10). Stav se drží v `meta` (JSON), aby přežil restart."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from arena.config import Config
from arena.db import Database

log = logging.getLogger(__name__)

DAY_MS = 86_400_000
PRIVATE_WS_MAX_DOWN_MS = 30_000


@dataclass
class RiskGuard:
    cfg: Config
    db: Database
    live_flag: bool                                   # `arena run --live`
    on_alert: Callable[[str, str], None] = lambda title, msg: None

    live_halt: bool = False                           # denní ztráta účtu → ruční reset
    slot_disabled_until: dict[str, int] = field(default_factory=dict)   # -1 = natrvalo (ruční reset)
    frozen_symbols: dict[str, str] = field(default_factory=dict)        # symbol → důvod (FOREIGN_POSITION…)
    day_start: dict[str, float] = field(default_factory=dict)           # {"day": ms, "equity": x}
    private_down_since: int | None = None
    stale_symbols: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        raw = self.db.meta_get("risk_state")
        if raw:
            st = json.loads(raw)
            self.live_halt = bool(st.get("live_halt"))
            self.slot_disabled_until = {k: int(v) for k, v in st.get("slot_disabled_until", {}).items()}
            self.frozen_symbols = dict(st.get("frozen_symbols", {}))
            self.day_start = dict(st.get("day_start", {}))

    @property
    def live_allowed(self) -> bool:
        """Live executor se vytvoří jen s `live.enabled: true` A flagem `--live` (SPEC §10)."""
        return bool(self.cfg.live.enabled and self.live_flag)

    def kill_active(self) -> bool:
        return Path(self.cfg.paths.kill_file).exists()

    def save(self) -> None:
        self.db.meta_set("risk_state", json.dumps({
            "live_halt": self.live_halt, "slot_disabled_until": self.slot_disabled_until,
            "frozen_symbols": self.frozen_symbols, "day_start": self.day_start}))

    # --- vstupní brána ------------------------------------------------------

    def can_enter(self, slot_id: str, symbol: str, now: int, open_positions: int) -> tuple[bool, str]:
        if not self.live_allowed:
            return False, "live vypnuto"
        if self.kill_active():
            return False, "kill switch"
        if self.live_halt:
            return False, "live_halt (denní ztráta účtu)"
        if symbol in self.frozen_symbols:
            return False, f"symbol zmrazen: {self.frozen_symbols[symbol]}"
        until = self.slot_disabled_until.get(slot_id)
        if until is not None and (until < 0 or now < until):
            return False, "slot disabled" + (" natrvalo" if until < 0 else "")
        if open_positions >= self.cfg.risk.max_live_positions:
            return False, "max. otevřených live pozic"
        if symbol in self.stale_symbols:
            return False, "stale data"
        if self.private_down_since is not None and now - self.private_down_since > PRIVATE_WS_MAX_DOWN_MS:
            return False, "private WS odpojený > 30 s"
        return True, ""

    # --- pravidla po obchodu a na equity ------------------------------------

    def on_live_trade_closed(self, slot_id: str, allocation_usdt: float, now: int) -> None:
        """Denní ztráta slotu ≥ 3 % alokace → disabled do UTC půlnoci; kumulativní DD ≥ 15 % → natrvalo."""
        day0 = now // DAY_MS * DAY_MS
        r = self.db.execute("SELECT COALESCE(SUM(net_pnl),0) FROM trades WHERE mode='live' AND slot_id=? "
                            "AND exit_time>=?", (slot_id, day0)).fetchone()
        if r[0] <= -self.cfg.risk.slot_daily_loss_pct / 100 * allocation_usdt:
            self.slot_disabled_until[slot_id] = day0 + DAY_MS
            self.on_alert("slot disabled", f"{slot_id}: denní ztráta {r[0]:.2f} USDT")
        pnl = [x[0] for x in self.db.execute("SELECT net_pnl FROM trades WHERE mode='live' AND slot_id=? "
                                             "ORDER BY exit_time", (slot_id,)).fetchall()]
        peak = cum = dd = 0.0
        for x in pnl:
            cum += x
            peak = max(peak, cum)
            dd = max(dd, peak - cum)
        if dd >= self.cfg.risk.slot_max_dd_pct / 100 * allocation_usdt:
            self.slot_disabled_until[slot_id] = -1
            self.on_alert("slot disabled natrvalo", f"{slot_id}: kumulativní DD {dd:.2f} USDT")
        self.save()

    def on_equity(self, equity: float, now: int) -> bool:
        """Equity účtu (wallet + unrealized). Vrací True, když se právě spustil live_halt (zavřít vše)."""
        day0 = now // DAY_MS * DAY_MS
        if self.day_start.get("day") != day0:
            self.day_start = {"day": day0, "equity": equity}
            # denní disable slotů končí o půlnoci
            self.slot_disabled_until = {k: v for k, v in self.slot_disabled_until.items() if v < 0 or v > now}
            self.save()
            return False
        start = float(self.day_start["equity"])
        if not self.live_halt and start > 0 and (equity - start) / start <= -self.cfg.risk.account_daily_loss_pct / 100:
            self.live_halt = True
            self.save()
            self.on_alert("LIVE HALT", f"denní ztráta účtu {(equity - start):.2f} USDT ({(equity / start - 1) * 100:.2f} %)")
            return True
        return False

    def freeze(self, symbol: str, reason: str) -> None:
        if self.frozen_symbols.get(symbol) != reason:
            self.frozen_symbols[symbol] = reason
            self.save()
            self.on_alert("symbol zmrazen", f"{symbol}: {reason}")

    def reset(self, what: str = "all") -> dict[str, Any]:
        """Ruční reset (CLI): live_halt, zmrazené symboly a natrvalo vypnuté sloty."""
        before = {"live_halt": self.live_halt, "frozen": dict(self.frozen_symbols),
                  "disabled": dict(self.slot_disabled_until)}
        if what in ("all", "halt"):
            self.live_halt = False
        if what in ("all", "frozen"):
            self.frozen_symbols.clear()
        if what in ("all", "slots"):
            self.slot_disabled_until.clear()
        self.save()
        return before
