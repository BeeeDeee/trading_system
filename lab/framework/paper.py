"""Deterministic paper trading (G5): the admitted strategy run on forward data only. No LLM, no orders.

Every cycle, for each hypothesis in PAPER:
1. ingest phase (network, `update_forward`): fetch the closed days after the snapshot for its instruments and
   its benchmark (lab.framework.forward, append-only),
2. judge phase (sandbox, `run`): extend the historical view with those days, run the strategy over the whole
   history (warm-up) with the lab engine, and append the new forward days to `paper_days`. Stored days are
   never recomputed: a later data revision cannot rewrite the paper record.
3. Sentinel kill rule: paper drawdown > `kill_drawdown_vs_backtest` x the dev drawdown -> RETIRED.

Without real orders there is no fill tracking: paper = the model's own execution on data that did not exist
when the hypothesis was judged. That is what makes it the lab's only clean evidence (PLAN §6.1).
"""

import json
from dataclasses import replace
from datetime import date

import numpy as np

from lab.framework import data, forward, metrics
from lab.framework.blackboard import Lab
from lab.framework.db import Tx, now
from lab.framework.gates import thresholds
from lab.framework.states import S


def admitted_on(lab: Lab, hid: str) -> date:
    ts = lab.con.execute("SELECT ts FROM transitions WHERE hypothesis_id = ? AND to_status = 'PAPER' "
                         "ORDER BY id DESC LIMIT 1", (hid,)).fetchone()[0]
    return date.fromisoformat(ts[:10])


def _instruments(ctx) -> dict[str, list[str]]:
    """Forward-fetchable instruments per dataset: the strategy's and the benchmark's crypto columns. For a
    top-n universe only the pairs that were among the 3n most liquid at the end of the snapshot."""
    v = ctx.view
    out = {}
    for j, (name, cls) in enumerate(zip(v.instruments, v.asset_class)):
        if not cls.startswith("crypto"):
            continue
        ds = "binance_perp_1d" if cls == "crypto_perp" else "binance_spot_1d"
        out.setdefault(ds, []).append(name)
    if ctx.card["universe"]["kind"] == "crypto_top_n" and "binance_spot_1d" in out:
        n = 3 * int(ctx.card["universe"]["n"])
        dv = np.nan_to_num(v.dollar_volume[-30:]).mean(axis=0)
        keep = {v.instruments[j] for j in np.argsort(-dv)[:n]} | set(ctx.bench_weights)
        out["binance_spot_1d"] = [i for i in out["binance_spot_1d"] if i in keep]
    return out


def update_forward(lab: Lab, evaluator, today: date | None = None) -> list[str]:
    log = []
    for h in lab.hypotheses(S.PAPER):
        ctx = evaluator._context(lab, h["id"], lab.card(h["id"]))
        snapshot_end = ctx.view.dates[-1].astype(object)
        for ds, names in _instruments(ctx).items():
            new = sum(forward.update(lab.paths.home, ds, n, snapshot_end, today) for n in names)
            log.append(f"{h['id']} forward {ds}: {new} new instrument-days")
    return log


def extended_context(lab: Lab, evaluator, hid: str):
    ctx = evaluator._context(lab, hid, lab.card(hid))
    view = ctx.view
    for ds in forward.SUPPORTED:
        view = forward.extend(view, lab.paths.home, ds)
    if ctx.card["universe"]["kind"] == "crypto_top_n":
        crypto = [i for i, c in enumerate(view.asset_class) if c.startswith("crypto")]
        mask = np.zeros(view.shape, dtype=bool)
        mask[:, crypto] = data.crypto_top_n(replace(view.columns([view.instruments[i] for i in crypto]),
                                                    universe=None), int(ctx.card["universe"]["n"]))
        view = replace(view, universe=mask)
    return replace(ctx, view=view, data_end=len(view.dates), runs={}), len(ctx.view.dates)


def run(lab: Lab, evaluator, hid: str) -> int:
    """Append new forward days of `hid` to paper_days. Returns the number of days added."""
    ctx, snapshot_rows = extended_context(lab, evaluator, hid)
    start_day = np.datetime64(admitted_on(lab, hid))
    first = max(snapshot_rows, int(np.searchsorted(ctx.view.dates, start_day, side="right")))
    end = len(ctx.view.dates)
    if first >= end:
        return 0
    params = evaluator.primary(ctx.card)
    r = evaluator.run(ctx, params, end)
    bench = evaluator.benchmark(ctx, end)
    last = lab.con.execute("SELECT MAX(date) FROM paper_days WHERE hypothesis_id = ?", (hid,)).fetchone()[0]
    held = r.sim.held
    added = 0
    with Tx(lab.con):
        for t in range(first, end):
            d = str(ctx.view.dates[t])
            if last and d <= last:
                continue
            entries = int(((np.abs(held[t]) > 1e-6) & (np.abs(held[t - 1]) <= 1e-6)).sum())
            lab.con.execute("INSERT INTO paper_days (hypothesis_id, date, ret, bench_ret, cash_ret, gross, turnover,"
                            " entries, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (hid, d, float(r.returns[t]), float(bench[t]), float(ctx.view.cash_ret[t]),
                             float(r.sim.gross[t]), float(r.sim.turnover[t]), entries, now()))
            added += 1
    return added


def record(lab: Lab, hid: str) -> dict:
    rows = lab.con.execute("SELECT * FROM paper_days WHERE hypothesis_id = ? ORDER BY date", (hid,)).fetchall()
    ret = np.array([r["ret"] for r in rows])
    return {"days": len(rows), "first": rows[0]["date"] if rows else None, "last": rows[-1]["date"] if rows else None,
            "ret": ret, "bench": np.array([r["bench_ret"] for r in rows]),
            "cash": np.array([r["cash_ret"] for r in rows]), "entries": sum(r["entries"] for r in rows),
            "max_dd": metrics.max_dd(ret) if len(ret) else 0.0}


def dev_max_dd(lab: Lab, hid: str) -> float | None:
    row = lab.con.execute("SELECT metrics_json FROM gate_results WHERE hypothesis_id = ? AND gate = 'G1' AND passed = 1 "
                          "ORDER BY id DESC LIMIT 1", (hid,)).fetchone()
    return json.loads(row[0]).get("max_dd") if row else None


def sentinel_check(lab: Lab, hid: str) -> S:
    """Kill rule on the paper record. Returns the state after the check."""
    config, _ = thresholds(lab.paths.gates)
    rec, dd_dev = record(lab, hid), dev_max_dd(lab, hid)
    limit = config["sentinel"]["kill_drawdown_vs_backtest"]
    if dd_dev and rec["days"] and rec["max_dd"] > limit * dd_dev:
        reason = (f"paper drawdown {rec['max_dd']:.1%} above {limit} x the dev drawdown {dd_dev:.1%}")
        lab.send("ALERT", "sentinel", "human", hid, {"severity": "critical", "text": f"{hid} retired: {reason}"})
        lab.transition(hid, S.RETIRED, "sentinel", reason, reason_code="kill_drawdown")
        return S.RETIRED
    return S.PAPER


def ready(lab: Lab, hid: str) -> bool:
    config, _ = thresholds(lab.paths.gates)
    g5 = config["G5"]
    classes = set(lab.card(hid).get("asset_classes") or [])
    months = min([g5["min_months"].get(c, g5["min_months"]["default"]) for c in classes] or [g5["min_months"]["default"]])
    rec = record(lab, hid)
    if not rec["days"]:
        return False
    span = (date.fromisoformat(rec["last"]) - date.fromisoformat(rec["first"])).days
    return span >= months * 30.4 and rec["entries"] >= g5["min_position_entries"]
