#!/usr/bin/env python3
"""Replay the stored days through the pinned engine and compare with what the cloud runs stored.

  replay_check.py DATA_DIR

For every stored day D: settle(prev day, prices/D) -> plan(settled, scores/D, free decisions of D) must reproduce
days/D (everything but regime/notes/code). This proves that the seeded state, the engine and the config on this
server are the very ones that produced the cloud results. The LLM is not involved: scores and the "Claude volně"
decisions are the stored ones.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evening.store import Store, write_json  # noqa: E402
from evening import runner  # noqa: E402

IGNORE = {"regime", "notes", "code", "run_no"}


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if not path and k in IGNORE:
                continue
            if k not in a or k not in b:
                yield f"{path}/{k}: only in {'stored' if k in a else 'replay'}"
            else:
                yield from diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield f"{path}: len {len(a)} vs {len(b)}"
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from diff(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > 1e-6:
            yield f"{path}: {a} vs {b}"
    elif a != b:
        yield f"{path}: {a!r} vs {b!r}"


def rebuild_free(volny):
    """The stored decisions lack stop_price/new_stop (the engine moved them into orders/positions); put them back."""
    orders = {o.get("ticker"): o for o in volny.get("orders") or [] if o.get("side", "").lower() == "buy" or o.get("action") == "BUY"}
    stops = {p["ticker"]: p.get("stop") for p in volny.get("positions") or []}
    out = []
    for x in volny.get("decisions") or []:
        x = dict(x)
        if x.get("action") == "BUY" and orders.get(x["ticker"], {}).get("stop_price"):
            x["stop_price"] = orders[x["ticker"]]["stop_price"]
        if x.get("action") == "HOLD" and stops.get(x["ticker"]):
            x["new_stop"] = stops[x["ticker"]]
        out.append(x)
    return out


def main(root):
    st = Store(root)
    work = os.path.join(root, "work", "_replay")
    os.makedirs(work, exist_ok=True)
    prev, bad = None, 0
    for d in st.ids("days"):
        stored = st.get("days", d)
        run = st.get("runs", d) or {}
        if run.get("status") == "catchup":
            prev = stored                              # no scores on catch-up days; the stored doc is the baseline
            continue
        pp = "-"
        if prev:
            write_json(os.path.join(work, "prev.json"), prev)
            pp = os.path.join(work, "prev.json")
        write_json(os.path.join(work, "today.json"), st.get("prices", d))
        runner.eng("settle", runner.CONFIG, pp, os.path.join(work, "today.json"), os.path.join(work, "settled.json"))
        free = {"decisions": rebuild_free(stored["variants"].get("volny") or {})}
        write_json(os.path.join(work, "free.json"), free)
        write_json(os.path.join(work, "scores.json"), st.get("scores", d))
        runner.eng("plan", runner.CONFIG, os.path.join(work, "settled.json"), os.path.join(work, "scores.json"), os.path.join(work, "free.json"), os.path.join(work, "out.json"))
        out = json.load(open(os.path.join(work, "out.json")))
        ds = list(diff(stored, out["day"]))
        trades = {f"{t['variant']}-{t['ticker']}-{t['entry_date']}": t for t in out.get("trades") or []}
        for tid, t in trades.items():
            s = st.get("trades", tid)
            if s is None:
                ds.append(f"trade {tid}: missing in stored")
            else:
                ds += [f"trade {tid}{x}" for x in diff(s, t)]
        print(f"{d}: {'OK' if not ds else str(len(ds)) + ' differences'}")
        for x in ds[:8]:
            print("   ", x)
        bad += bool(ds)
        prev = stored
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1])
