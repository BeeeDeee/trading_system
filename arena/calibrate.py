"""`arena calibrate` (SPEC §7, P7): z párů live/paper dvojčat navrhne parametry nákladového modelu.

Nic nepřepisuje: vlastník návrh potvrdí a zvýší `paper.cost_model_version`."""

from __future__ import annotations

from typing import Any

import numpy as np

from arena.db import Database

MIN_PAIRS = 30


def calibrate(db: Database) -> dict[str, Any]:
    rows = db.execute(
        "SELECT l.slot_id, l.entry_px le, p.entry_px pe, l.exit_px lx, p.exit_px px, l.side, "
        "l.entry_slip_bps les, p.entry_slip_bps pes, l.exit_slip_bps lxs, p.exit_slip_bps pxs, "
        "l.net_bps ln, p.net_bps pn FROM trades l JOIN trades p ON p.trade_id = l.twin_trade_id "
        "WHERE l.mode='live' AND p.mode='paper' AND l.tainted=0 AND p.tainted=0").fetchall()
    by_sym: dict[str, list[Any]] = {}
    for r in rows:
        by_sym.setdefault(r["slot_id"].rsplit(".", 1)[1], []).append(r)
    out: dict[str, Any] = {"symbols": {}, "latency": {}}
    for sym, rs in sorted(by_sym.items()):
        sg = np.array([1 if r["side"] == "LONG" else -1 for r in rs])
        d_entry = sg * (np.array([r["le"] for r in rs]) - np.array([r["pe"] for r in rs])) / np.array(
            [r["pe"] for r in rs]) * 1e4
        d_exit = -sg * (np.array([r["lx"] for r in rs]) - np.array([r["px"] for r in rs])) / np.array(
            [r["px"] for r in rs]) * 1e4
        d_net = np.array([r["ln"] - r["pn"] for r in rs])
        out["symbols"][sym] = {
            "n_pairs": len(rs), "enough": len(rs) >= MIN_PAIRS,
            "entry_diff_bps_med": float(np.median(d_entry)), "entry_diff_bps_p90": float(np.percentile(d_entry, 90)),
            "exit_diff_bps_med": float(np.median(d_exit)), "exit_diff_bps_p90": float(np.percentile(d_exit, 90)),
            "net_diff_bps_med": float(np.median(d_net)),
            # kladný rozdíl = live horší než paper → přirážka, záporný se nepřebírá (konzervativně)
            "proposed_extra_slippage_bps": round(max(0.0, float(np.median((d_entry + d_exit) / 2))), 2),
        }
    for kind in ("decision_to_send", "send_to_ack", "ack_to_fill", "trigger_to_fill"):
        ms = [r[0] for r in db.execute("SELECT ms FROM latencies WHERE kind=?", (kind,)).fetchall()]
        if ms:
            out["latency"][kind] = {"n": len(ms), "med": float(np.median(ms)), "p90": float(np.percentile(ms, 90))}
    lat = out["latency"]
    if {"decision_to_send", "send_to_ack", "ack_to_fill"} <= set(lat):
        # paper: fill v t_d + entry_latency_ms → rozhodnutí → odeslání → ack → fill (součet mediánů)
        out["proposed_entry_latency_ms"] = int(sum(lat[k]["med"] for k in ("decision_to_send", "send_to_ack",
                                                                             "ack_to_fill")))
    if "trigger_to_fill" in lat:
        out["proposed_trigger_latency_ms"] = int(lat["trigger_to_fill"]["med"])
    return out


def render(c: dict[str, Any]) -> str:
    lines = ["Kalibrace nákladového modelu (návrh, konfigurace se nemění):"]
    for sym, s in c["symbols"].items():
        flag = "" if s["enough"] else f"  (málo párů < {MIN_PAIRS})"
        lines.append(f"  {sym:9} párů {s['n_pairs']:4}  vstup Δ med {s['entry_diff_bps_med']:+.2f} p90 "
                     f"{s['entry_diff_bps_p90']:+.2f}  výstup Δ med {s['exit_diff_bps_med']:+.2f} p90 "
                     f"{s['exit_diff_bps_p90']:+.2f}  → extra_slippage_bps {s['proposed_extra_slippage_bps']}{flag}")
    for k, v in c["latency"].items():
        lines.append(f"  latence {k:17} n {v['n']:4} med {v['med']:.0f} ms p90 {v['p90']:.0f} ms")
    if "proposed_entry_latency_ms" in c:
        lines.append(f"  → entry_latency_ms {c['proposed_entry_latency_ms']}")
    if "proposed_trigger_latency_ms" in c:
        lines.append(f"  → trigger_latency_ms {c['proposed_trigger_latency_ms']}")
    lines.append("Po potvrzení: uprav config/arena.yaml (paper.*) a zvyš paper.cost_model_version.")
    return "\n".join(lines)
