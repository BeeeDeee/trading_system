"""Read-only views for people: funnel, rejection reasons, the per-hypothesis trace, invocation log.
The dashboard (step 5) renders the same data."""

import json
from collections import Counter

from lab.framework.blackboard import Lab
from lab.framework.states import FUNNEL


def funnel(lab: Lab) -> dict[str, int]:
    counts = Counter(h["status"] for h in lab.hypotheses())
    return {s.value: counts.get(s.value, 0) for s in FUNNEL}


def rejection_reasons(lab: Lab) -> list[dict]:
    rows = lab.con.execute("SELECT status, reject_stage, reject_code, COUNT(*) n FROM hypotheses "
                           "WHERE reject_stage IS NOT NULL GROUP BY 1, 2, 3 ORDER BY n DESC")
    return [dict(r) for r in rows]


def trace(lab: Lab, hid: str) -> list[dict]:
    """Everything that happened to one hypothesis, in time order: the 'conversation' of the agents."""
    ev = []
    for r in lab.con.execute("SELECT * FROM transitions WHERE hypothesis_id = ?", (hid,)):
        ev.append({"ts": r["ts"], "kind": "transition", "actor": r["actor"],
                   "text": f"{r['from_status'] or '∅'} -> {r['to_status']} (v{r['version']}): {r['reason']}"})
    # Messages about this hypothesis, plus hypothesis-less ones that caused its transitions (its creation,
    # an ingest that unblocked it), and every agent run that wrote any of them.
    msgs = lab.con.execute(
        "SELECT * FROM messages WHERE hypothesis_id = ? OR id IN "
        "(SELECT message_id FROM transitions WHERE hypothesis_id = ? AND message_id IS NOT NULL)",
        (hid, hid)).fetchall()
    for r in msgs:
        ev.append({"ts": r["created_at"], "kind": "message", "actor": r["from_agent"],
                   "text": f"{r['type']} -> {r['to_agent']}: {_summary(r['type'], json.loads(r['payload_json']))}"
                           + ("" if r["handled_at"] else "  [unread]")})
    inv_ids = {r["invocation_id"] for r in msgs if r["invocation_id"]}
    for r in lab.con.execute("SELECT * FROM agent_invocations WHERE hypothesis_id = ? OR id IN (%s)"
                             % ",".join("?" * len(inv_ids)), (hid, *inv_ids)):
        ev.append({"ts": r["started_at"], "kind": "invocation", "actor": r["agent"],
                   "text": f"run {r['id']} ({r['model']}) -> {r['outcome'] or 'running'}"
                           + (f": {r['error']}" if r["error"] else "")})
    for r in lab.con.execute("SELECT * FROM gate_results WHERE hypothesis_id = ?", (hid,)):
        ev.append({"ts": r["ts"], "kind": "gate", "actor": "gatekeeper",
                   "text": f"{r['gate']} v{r['version']} {'PASS' if r['passed'] else 'FAIL'}"
                           f"{' ' + r['reason_code'] if r['reason_code'] else ''} [{r['evaluator']}]"})
    return sorted(ev, key=lambda e: e["ts"])


def _summary(msg_type: str, p: dict) -> str:
    match msg_type:
        case "NEW_HYPOTHESIS" | "REVISION":
            return p["card"].get("title", "") + (f" ({p['reason']})" if p.get("reason") else "")
        case "DATA_REQUEST" | "DATA_READY":
            return p["dataset"]
        case "IMPL_DONE":
            return f"{len(p['files'])} files: {p['summary']}"
        case "GATE_RESULT":
            return f"{p['gate']} {'pass' if p['passed'] else 'fail'} {p.get('reason_code') or ''}".strip()
        case "OBJECTION":
            return f"to {p['return_to']}: " + "; ".join(f"{i['check']}: {i['finding']}" for i in p["items"])
        case "VERDICT":
            return f"{p['decision']}: {p['reason']}"
        case "QUESTION":
            return p["question"]
        case "ALERT":
            return f"[{p['severity']}] {p['text']}"
        case "LESSON":
            return f"{p['outcome']}: {p['lesson']}"
        case "KNOWLEDGE":
            return f"[{p['kind']}] {p['title']}: {p['statement']}"
        case "FAMILY_MERGE":
            return f"{p['from_family']} -> {p['into_family']}: {p['reason']}"
    return json.dumps(p)[:120]


def invocations(lab: Lab, limit: int = 50) -> list[dict]:
    rows = lab.con.execute("SELECT id, agent, model, hypothesis_id, started_at, ended_at, outcome, error "
                           "FROM agent_invocations ORDER BY started_at DESC LIMIT ?", (limit,))
    return [dict(r) for r in rows]


EXPORT_TABLES = {"hypotheses": "id", "transitions": "id", "messages": "id", "gate_results": "id", "trials": "id",
                 "agent_invocations": "started_at", "lessons": "message_id", "knowledge": "id",
                 "family_merges": "ts", "canary_runs": "id"}
DROP_COLUMNS = {"agent_invocations": {"workspace", "transcript_path", "output_json"}}


def export(lab: Lab, out_dir) -> str:
    """One JSON file per table (sorted, stable), so the nightly commit diff shows what changed."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for table, order in EXPORT_TABLES.items():
        rows = [{k: v for k, v in dict(r).items() if k not in DROP_COLUMNS.get(table, set())}
                for r in lab.con.execute(f"SELECT * FROM {table} ORDER BY {order}")]
        (out / f"{table}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str) + "\n")
        counts[table] = len(rows)
    return " ".join(f"{t}={n}" for t, n in counts.items())


def disk_usage(lab: Lab) -> list[tuple[str, int]]:
    """Sizes of LAB_HOME's top-level entries and the free space of its disk (bytes), largest first."""
    import shutil
    from pathlib import Path

    def size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    rows = [(f"LAB_HOME/{p.name}", size(p)) for p in sorted(lab.paths.home.iterdir())]
    rows.sort(key=lambda r: -r[1])
    free = shutil.disk_usage(lab.paths.home)
    return rows + [("-- free on this disk", free.free), ("-- disk total", free.total)]
