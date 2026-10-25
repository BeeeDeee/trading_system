"""Snapshot pro dashboard (D7, SPEC §15): `snapshot.json` každých 15 s, `variants/<id>.json` a `costs.json`
každých 5 min. Zápis atomicky (tmp + os.replace). Dashboard nikdy nečte DB (nástraha 13), ve snapshotu nejsou
žádné klíče (nástraha 14)."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from arena import stats
from arena.config import Config
from arena.data.bars import BAR_MS
from arena.db import Database
from arena.util import atomic_write_text, wall_ms

log = logging.getLogger(__name__)

DAY = 86_400_000


def _clean(v: Any) -> Any:
    if isinstance(v, float):
        return None if not math.isfinite(v) else round(v, 6)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return _clean(v.item())
    return v


def _dump(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(_clean(obj), ensure_ascii=False, separators=(",", ":")))


@dataclass
class SnapshotWriter:
    cfg: Config
    db: Database
    site_dir: Path
    status: Callable[[], Mapping[str, Any]]                  # engine.status()
    positions: Callable[[], list[dict[str, Any]]]            # otevřené paper pozice pro přehled
    live_variants: Callable[[], list[str]] = lambda: []
    live_positions: Callable[[], list[dict[str, Any]]] = lambda: []
    calibrated: Callable[[], bool] = lambda: False

    leaderboard_cache: dict[str, Any] = field(default_factory=dict)
    last_heavy_ms: int = 0

    @property
    def data_dir(self) -> Path:
        return self.site_dir / "data"

    # --- 5 min: statistiky, varianty, náklady -------------------------------

    def _trades(self, mode: str = "paper", exclude_tainted: bool = False) -> list[stats.T]:
        q = "SELECT * FROM trades WHERE mode=? AND cost_model_version=? AND slot_id NOT LIKE 'xp.%'"   # exec probe mimo experiment
        rows = self.db.execute(q + (" AND tainted=0" if exclude_tainted else ""),
                               (mode, self.cfg.paper.cost_model_version)).fetchall()
        return stats.from_rows(rows)

    def write_heavy(self) -> None:
        t0 = time.monotonic()
        variants = {vid: list(v.symbols) for vid, v in sorted(self.cfg.variants.items()) if v.family != "xp"}
        first = self.db.meta_get("first_bar_time")
        period = (wall_ms() - int(first)) if first else None
        out: dict[str, Any] = {}
        all_trades: dict[bool, list[stats.T]] = {}
        for excl in (False, True):
            trades = self._trades(exclude_tainted=excl)
            all_trades[excl] = trades
            lb = stats.leaderboard(trades, variants, self.live_variants(), self.calibrated(), period)
            out["clean" if excl else "all"] = {k: g.to_json() for k, g in lb.items()}
        self.leaderboard_cache = {
            "computed_at": wall_ms(),
            "leaderboard": out,
            "heatmap": stats.heatmap(all_trades[False]),
            "n_trades": len(all_trades[False]),
        }
        trades = all_trades[False]
        live = self._trades("live")
        for vid in variants:
            self._write_variant(vid, [t for t in trades if t.variant == vid], [t for t in live if t.variant == vid],
                                [t for t in trades if t.variant == f"ctl.{vid.split('.')[0]}"])
        self._write_costs(trades, live)
        self.last_heavy_ms = wall_ms()
        log.info("snapshot: statistiky %.1f s", time.monotonic() - t0)

    def _write_variant(self, vid: str, paper: list[stats.T], live: list[stats.T], ctl: list[stats.T]) -> None:
        def curve(tr: list[stats.T]) -> dict[str, list[Any]]:
            tr = sorted(tr, key=lambda t: t.exit_time)
            return {"t": [t.exit_time for t in tr], "net": list(np.cumsum([t.net_bps for t in tr])),
                    "gross": list(np.cumsum([t.gross_bps for t in tr]))}

        def daily(tr: list[stats.T]) -> dict[str, list[Any]]:
            dp = stats.daily_pnl(tr)
            d0 = min((t.day for t in tr), default=0)
            return {"day": [(d0 + i) * DAY for i in range(len(dp))], "cum_pnl": list(np.cumsum(dp))}

        hist = np.histogram([t.net_bps for t in paper], bins=np.arange(-300, 325, 25))[0] if paper else []
        rows = self.db.execute(
            "SELECT trade_id, slot_id, mode, side, entry_time, exit_time, entry_px, exit_px, exit_reason, net_bps, "
            "gross_bps, entry_slip_bps, exit_slip_bps, bars_held, tainted FROM trades WHERE slot_id LIKE ? "
            "ORDER BY exit_time DESC LIMIT 200", (f"{vid}.%",)).fetchall()
        _dump(self.data_dir / "variants" / f"{vid}.json", {
            "variant": vid, "generated_at": wall_ms(),
            "paper_curve": curve(paper), "control_curve": curve(ctl), "live_curve": curve(live),
            "paper_daily": daily(paper), "live_daily": daily(live),
            "by_symbol": stats.group_stats(vid, paper).by_symbol if paper else {},
            "exit_mix": stats.group_stats(vid, paper).exit_mix if paper else {},
            "hist": {"edges": list(range(-300, 325, 25)), "counts": [int(x) for x in hist]},
            "trades": [dict(r) for r in rows],
        })

    def _write_costs(self, paper: list[stats.T], live: list[stats.T]) -> None:
        def per_symbol(tr: list[stats.T]) -> dict[str, Any]:
            out = {}
            for sym in sorted({t.symbol for t in tr}):
                xs = [t for t in tr if t.symbol == sym]
                es = np.array([t.entry_slip_bps for t in xs])
                xx = np.array([t.exit_slip_bps for t in xs])
                out[sym] = {"n": len(xs), "fee_bps": float(np.mean([t.fee_bps for t in xs])),
                            "entry_slip_med": float(np.median(es)), "entry_slip_p90": float(np.percentile(es, 90)),
                            "exit_slip_med": float(np.median(xx)), "exit_slip_p90": float(np.percentile(xx, 90)),
                            "funding_bps": float(np.mean([t.funding_bps for t in xs]))}
            return out

        lat = {r["kind"]: {"n": r["n"], "avg_ms": r["a"]} for r in self.db.execute(
            "SELECT kind, COUNT(*) n, AVG(ms) a FROM latencies GROUP BY kind").fetchall()}
        _dump(self.data_dir / "costs.json", {
            "generated_at": wall_ms(), "cost_model_version": self.cfg.paper.cost_model_version,
            "paper": per_symbol(paper), "live": per_symbol(live), "latencies": lat,
            "model_error": self._model_error(),
        })

    def _model_error(self) -> dict[str, Any]:
        """live − paper dvojče na páru obchodů (SPEC §13): medián a p90 rozdílu net_bps a složek."""
        rows = self.db.execute(
            "SELECT l.slot_id, l.net_bps - p.net_bps d, l.entry_slip_bps - p.entry_slip_bps de, "
            "l.exit_slip_bps - p.exit_slip_bps dx FROM trades l JOIN trades p ON p.trade_id = l.twin_trade_id "
            "WHERE l.mode='live' AND l.slot_id NOT LIKE 'xp.%'").fetchall()
        if not rows:
            return {"n": 0}
        d = np.array([r["d"] for r in rows])
        return {"n": len(rows), "net_med": float(np.median(d)), "net_p90": float(np.percentile(d, 90)),
                "entry_med": float(np.median([r["de"] for r in rows])),
                "exit_med": float(np.median([r["dx"] for r in rows]))}

    # --- 15 s: přehled a health ---------------------------------------------

    def health(self) -> dict[str, Any]:
        now = wall_ms()
        st = dict(self.status())
        bars_last = st.get("bars_last", {})
        # sekundy od uzavření posledního uzavřeného baru (normálně 0–300 s; > 320 s = další bar se zpozdil)
        st["bar_age_s"] = {s: (now - (int(t) + BAR_MS)) / 1000 for s, t in bars_last.items()}
        day_ago = (now - DAY) // BAR_MS * BAR_MS
        st["book_ok_24h"] = {r["symbol"]: r["ok"] for r in self.db.execute(
            "SELECT symbol, AVG(book_ok) ok FROM bar_book WHERE pct=1.0 AND source='live' AND open_time>=? "
            "GROUP BY symbol", (day_ago,)).fetchall()}
        st["book_validate"] = self.db.meta_get("book_validate_last")
        bv = self.db.execute("SELECT data_json FROM events WHERE component='book-validate' ORDER BY id DESC "
                             "LIMIT 1").fetchone()
        st["book_validate_result"] = json.loads(bv["data_json"]) if bv and bv["data_json"] else None
        st["kill_switch"] = Path(self.cfg.paths.kill_file).exists()
        st["errors_24h"] = self.db.execute("SELECT COUNT(*) FROM events WHERE level='ERROR' AND ts>=?",
                                           (now - DAY,)).fetchone()[0]
        st["last_reconcile"] = self.db.meta_get("last_reconcile")
        return st

    def _market(self) -> dict[str, Any]:
        """Posledních 288 uzavřených 5m barů na symbol (24 h) pro sparklines a 24h změnu."""
        out: dict[str, Any] = {}
        for sym in self.cfg.universe:
            rows = self.db.execute("SELECT open_time, c FROM bars WHERE symbol=? ORDER BY open_time DESC LIMIT 288",
                                   (sym,)).fetchall()[::-1]
            if rows:
                out[sym] = {"t0": rows[0][0], "step": BAR_MS, "close": [r[1] for r in rows]}
        return out

    def _equity_curve(self, now: int, max_points: int = 240) -> dict[str, list[Any]]:
        rows = self.db.execute("SELECT ts, wallet, unrealized FROM equity WHERE ts>=? ORDER BY ts",
                               (now - DAY,)).fetchall()
        if len(rows) > max_points:
            step = len(rows) / max_points
            rows = [rows[int(i * step)] for i in range(max_points)] + [rows[-1]]
        return {"t": [r[0] for r in rows], "equity": [r[1] + (r[2] or 0.0) for r in rows]}

    def _recent_trades(self, limit: int = 40) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT trade_id, slot_id, mode, side, entry_px, exit_px, exit_time, exit_reason, net_bps, net_pnl, tainted "
            "FROM trades ORDER BY exit_time DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def _activity(self, now: int) -> dict[str, Any]:
        """Signály za 24 h: počty podle akce a po hodinách (jen `entered`)."""
        since = now - DAY
        acts = self.db.execute("SELECT action, COUNT(*) FROM signals WHERE decided_at>=? AND slot_id NOT LIKE 'xp.%' GROUP BY action",
                               (since,)).fetchall()
        hours = [0] * 24
        for (t,) in self.db.execute("SELECT decided_at FROM signals WHERE decided_at>=? AND action='entered' AND slot_id NOT LIKE 'xp.%'",
                                    (since,)).fetchall():
            hours[min(23, int((t - since) // 3_600_000))] += 1
        return {"by_action": {a: n for a, n in acts}, "entered_by_hour": hours, "since": since}

    def write_snapshot(self) -> None:
        now = wall_ms()
        day0 = now // DAY * DAY
        today = self.db.execute(
            "SELECT mode, COUNT(*) n, SUM(net_pnl) pnl, SUM(entry_fee + exit_fee) fee, SUM(funding) funding "
            "FROM trades WHERE exit_time>=? AND tainted=0 AND slot_id NOT LIKE 'xp.%' GROUP BY mode", (day0,)).fetchall()
        eq = self.db.execute("SELECT * FROM equity ORDER BY ts DESC LIMIT 1").fetchone()
        events = self.db.execute("SELECT ts, level, component, msg FROM events ORDER BY id DESC LIMIT 200").fetchall()
        _dump(self.data_dir / "snapshot.json", {
            "generated_at": now, "env": self.cfg.env, "data_env": self.cfg.data_env,
            "live_enabled": self.cfg.live.enabled, "config_hash": self.cfg.config_hash,
            "cost_model_version": self.cfg.paper.cost_model_version,
            "overview": {
                "equity": dict(eq) if eq else None,
                "today": {r["mode"]: dict(r) for r in today},
                "paper_positions": self.positions(),
                "live_positions": self.live_positions(),
                "ledger_diffs": json.loads(self.db.meta_get("ledger_diffs") or "[]"),
            },
            "health": self.health(),
            "stats": self.leaderboard_cache,
            "events": [dict(r) for r in events],
            "market": self._market(),
            "equity_curve": self._equity_curve(now),
            "recent_trades": self._recent_trades(),
            "activity": self._activity(now),
            "universe": list(self.cfg.universe),
            "live_slots": {sym: {"variant": sl.variant, "notional": sl.notional_usdt}
                           for sym, sl in self.cfg.live.slots.items()},
        })
