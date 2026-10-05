"""The deterministic phase of every orchestrator cycle.

1. React to new messages in id order (each message once, `reacted_at`): create and revise hypotheses,
   check specs and data, ingest, start G0, apply Skeptic objections and verdicts, park and reopen.
2. Drive state-based work: G1 -> G2 -> G3 -> SKEPTIC_REVIEW, Sentinel admission, G5.
3. Repeat until nothing moves.

Agents never call this. Everything here is code, so a hypothesis can pass a gate only through it.
"""

import json
from difflib import SequenceMatcher

from lab.framework import catalog, sentinel, validate
from lab.framework.blackboard import Lab, LabError
from lab.framework.db import Tx, now
from lab.framework.gates import Evaluator, run_gate, thresholds
from lab.framework.states import HUMAN, S

AUTO_GATES = {S.IMPLEMENTED: "G1", S.GATE_1: "G2", S.GATE_2: "G3"}
DUPLICATE_SIMILARITY = 0.90


def tick(lab: Lab, evaluator: Evaluator | None, max_rounds: int = 50) -> list[str]:
    """Run the deterministic phase. Without an evaluator, gates are not run (messages still are)."""
    log: list[str] = []
    for _ in range(max_rounds):
        moved = _react_all(lab, evaluator, log)
        moved |= _drive_states(lab, evaluator, log)
        if not moved:
            break
    return log


# ---------------------------------------------------------------------------- reactions to messages

def _react_all(lab: Lab, evaluator, log) -> bool:
    rows = lab.con.execute("SELECT * FROM messages WHERE reacted_at IS NULL ORDER BY id").fetchall()
    reacted = 0
    for m in rows:
        payload = json.loads(m["payload_json"])
        if evaluator is None and _needs_evaluator(m, payload):
            continue  # leave it for a tick that can run gates
        reacted += 1
        try:
            note = REACTIONS.get(m["type"], _no_reaction)(lab, m, payload, evaluator)
        except LabError as e:
            # A refused reaction (stale state, invalid entry) is answered, never retried forever.
            note = f"refused: {e}"
            if m["from_agent"] not in ("system", "gatekeeper", "sentinel"):
                lab.send("QUESTION", "system", m["from_agent"], m["hypothesis_id"],
                         {"question": f"Your {m['type']} (message {m['id']}) was refused: {e}",
                          "in_reply_to": m["id"]})
        with Tx(lab.con):
            lab.con.execute("UPDATE messages SET reacted_at = ? WHERE id = ?", (now(), m["id"]))
            if m["to_agent"] in ("system", "gatekeeper"):
                lab.con.execute("UPDATE messages SET handled_at = ?, handled_by = 'tick' "
                                "WHERE id = ? AND handled_at IS NULL", (now(), m["id"]))
        if note:
            log.append(f"msg {m['id']} {m['type']} {m['from_agent']}->{m['to_agent']}: {note}")
    return reacted > 0


def _needs_evaluator(m, payload) -> bool:
    return m["type"] in ("IMPL_DONE", "DATA_READY") or (m["type"] == "VERDICT" and m["from_agent"] == "skeptic"
                                         and payload["decision"] == "no_objection")


def _no_reaction(lab, m, payload, evaluator):
    return None


def _new_hypothesis(lab, m, payload, evaluator):
    hid = lab.create_hypothesis(payload["card"], m["from_agent"], invocation_id=m["invocation_id"],
                                message_id=m["id"])
    return f"created {hid}; " + _spec_and_data(lab, hid, m)


def _revision(lab, m, payload, evaluator):
    version = lab.revise(m["hypothesis_id"], payload["card"], m["from_agent"], payload["reason"])
    return f"{m['hypothesis_id']} v{version}; " + _spec_and_data(lab, m["hypothesis_id"], m)


def _spec_and_data(lab: Lab, hid: str, m) -> str:
    """IDEA -> SPECIFIED (spec complete, not a duplicate) -> DATA_READY or BLOCKED_DATA."""
    card, author = lab.card(hid), m["from_agent"]
    ask_to = HUMAN if author == HUMAN else "scout"
    missing = validate.specified_missing(card)
    if missing:
        lab.send("QUESTION", "system", ask_to, hid,
                 {"question": f"{hid} stays in IDEA: missing {', '.join(missing)}", "in_reply_to": m["id"]})
        return "stays IDEA (incomplete)"
    grid = validate.grid_problems(card)
    if grid:
        lab.send("QUESTION", "system", ask_to, hid,
                 {"question": f"{hid} stays in IDEA: " + "; ".join(grid), "in_reply_to": m["id"]})
        return "stays IDEA (parameter grid)"
    dup = _duplicate_of(lab, hid, card)
    if dup:
        lab.send("QUESTION", "system", ask_to, hid,
                 {"question": f"{hid} looks like a duplicate of {dup}; revise to make the difference explicit "
                              "or let the Chair park it", "in_reply_to": m["id"]})
        return f"stays IDEA (duplicate of {dup}?)"
    lab.transition(hid, S.SPECIFIED, author, "spec complete", invocation_id=m["invocation_id"], message_id=m["id"])
    return _check_data(lab, hid)


def _duplicate_of(lab: Lab, hid: str, card: dict) -> str | None:
    text = card["signal"]["description"].lower()
    for other in lab.hypotheses():
        if other["id"] == hid or lab.canonical_family(other["family"]) != lab.canonical_family(card["family"]):
            continue
        o = json.loads(other["card_json"])
        if o["title"].strip().lower() == card["title"].strip().lower() or SequenceMatcher(
                None, text, o.get("signal", {}).get("description", "").lower()).ratio() >= DUPLICATE_SIMILARITY:
            return other["id"]
    return None


def _check_data(lab: Lab, hid: str) -> str:
    res = catalog.resolve(lab.paths.catalog, lab.card(hid)["data_requirements"])
    if res.ok:
        lab.transition(hid, S.DATA_READY, "system", "all data requirements resolve in the catalog")
        return "DATA_READY"
    for req in res.missing:
        lab.send("DATA_REQUEST", "system", "archivist", hid,
                 {"dataset": req["dataset"], "description": req.get("description", ""),
                  "frequency": req["frequency"], "period": [str(p) for p in req.get("period", [])]})
    if res.problems:   # data exists but cannot be used as requested (range, no loader): the owner decides
        lab.send("QUESTION", "system", HUMAN, hid, {"question": f"{hid} blocked: " + "; ".join(res.problems)})
    reason = "; ".join([f"missing dataset {r['dataset']}" for r in res.missing] + res.problems)
    lab.transition(hid, S.BLOCKED_DATA, "system", reason)
    return f"BLOCKED_DATA ({reason})"


def _data_ready(lab, m, payload, evaluator):
    """Ingest (by the evaluator: fetch, validate, dev/holdout split, catalog entry), then unblock."""
    try:
        entry = evaluator.ingest(lab, payload)
    except ValueError as e:
        raise LabError(f"ingest of {payload['dataset']} failed: {e}") from None
    unblocked = []
    for h in lab.hypotheses(S.BLOCKED_DATA):
        if catalog.resolve(lab.paths.catalog, lab.card(h["id"])["data_requirements"]).ok:
            lab.transition(h["id"], S.DATA_READY, "system", f"dataset {payload['dataset']} ingested",
                           message_id=m["id"])
            unblocked.append(h["id"])
    return f"ingested {payload['dataset']} (holdout from {entry['holdout_from']}); unblocked {unblocked or 'none'}"


def _impl_done(lab, m, payload, evaluator):
    out = run_gate(lab, m["hypothesis_id"], "G0", evaluator)
    return f"G0 {'passed' if out.passed else 'failed: ' + str(out.reason_code)}"


def _objection(lab, m, payload, evaluator):
    hid = m["hypothesis_id"]
    h = lab.hypothesis(hid)
    if h["status"] != S.SKEPTIC_REVIEW:
        raise LabError(f"{hid} is {h['status']}, not in SKEPTIC_REVIEW")
    limit = thresholds(lab.paths.gates)[0]["skeptic"]["max_objection_rounds"]
    rounds = h["objection_rounds"] + 1
    with Tx(lab.con):
        lab.con.execute("UPDATE hypotheses SET objection_rounds = ? WHERE id = ?", (rounds, hid))
    if rounds > limit:
        lab.transition(hid, S.REJECTED, "system", f"objection round {rounds} > {limit}",
                       message_id=m["id"], reason_code="objection_limit",
                       stage="SKEPTIC_REVIEW")
        return "REJECTED (objection limit)"
    to = S.DATA_READY if payload["return_to"] == "builder" else S.IDEA
    checks = ", ".join(i["check"] for i in payload["items"])
    lab.transition(hid, to, "skeptic", f"objection round {rounds}: {checks}", message_id=m["id"],
                   invocation_id=m["invocation_id"])
    return f"back to {to} (round {rounds})"


def _verdict(lab, m, payload, evaluator):
    if m["to_agent"] != "system":
        return None
    hid, sender, decision = m["hypothesis_id"], m["from_agent"], payload["decision"]
    status = lab.hypothesis(hid)["status"]
    code = payload.get("reason_code") or decision
    if sender == "skeptic" and decision == "no_objection":
        if status != S.SKEPTIC_REVIEW:
            raise LabError(f"{hid} is {status}, not in SKEPTIC_REVIEW")
        out = run_gate(lab, hid, "G4", evaluator)
        return f"G4 {'passed' if out.passed else 'failed'}"
    if sender == "skeptic" and decision == "reject":
        lab.transition(hid, S.REJECTED, "skeptic", payload["reason"], message_id=m["id"], reason_code=code)
        return "REJECTED by skeptic"
    if sender == "archivist" and decision == "infeasible":
        lab.transition(hid, S.PARKED, "archivist", payload["reason"], message_id=m["id"], reason_code=code)
        return "PARKED (data infeasible)"
    if sender == "chair" and decision == "park":
        lab.transition(hid, S.PARKED, "chair", payload["reason"], message_id=m["id"], reason_code=code)
        return "PARKED by chair"
    if sender == "chair" and decision == "reopen":
        lab.transition(hid, S.IDEA, "chair", f"reopened: {payload['reason']}", message_id=m["id"])
        return "reopened to IDEA (needs a REVISION)"
    return None


def _lesson(lab, m, payload, evaluator):
    from lab.framework import lessons
    lessons.check(lab, m["hypothesis_id"], payload)
    with Tx(lab.con):
        lab.con.execute("INSERT OR IGNORE INTO lessons (message_id, hypothesis_id, ts) VALUES (?, ?, ?)",
                        (m["id"], m["hypothesis_id"], now()))
    lessons.render(lab)
    return f"lesson on {m['hypothesis_id']} recorded"


def _family_merge(lab, m, payload, evaluator):
    lab.merge_family(payload["from_family"], payload["into_family"], m["from_agent"], payload["reason"],
                     message_id=m["id"])
    return f"family {payload['from_family']} merged into {lab.canonical_family(payload['into_family'])}"


REACTIONS = {"FAMILY_MERGE": _family_merge, "LESSON": _lesson, "NEW_HYPOTHESIS": _new_hypothesis, "REVISION": _revision, "DATA_READY": _data_ready,
             "IMPL_DONE": _impl_done, "OBJECTION": _objection, "VERDICT": _verdict}


# ---------------------------------------------------------------------------- state-driven work

def _locked(lab: Lab, hid: str) -> bool:
    return lab.con.execute("SELECT 1 FROM locks WHERE hypothesis_id = ? AND lease_until > ?",
                           (hid, now())).fetchone() is not None


def _drive_states(lab: Lab, evaluator, log) -> bool:
    moved = False
    for h in lab.hypotheses():
        hid, status = h["id"], S(h["status"])
        if _locked(lab, hid):
            continue
        if status in AUTO_GATES and evaluator is not None:
            out = run_gate(lab, hid, AUTO_GATES[status], evaluator)
            log.append(f"{hid} {AUTO_GATES[status]} {'passed' if out.passed else 'failed: ' + str(out.reason_code)}")
            moved = True
        elif status == S.GATE_3:
            lab.transition(hid, S.SKEPTIC_REVIEW, "system", "G1-G3 passed, Skeptic review before the holdout")
            log.append(f"{hid} -> SKEPTIC_REVIEW")
            moved = True
        elif status == S.HOLDOUT:
            new = sentinel.admit_to_paper(lab, hid)
            if new != S.HOLDOUT:
                log.append(f"{hid} sentinel -> {new}")
                moved = True
        elif status == S.PAPER and evaluator is not None and getattr(evaluator, "paper_ready", None) \
                and evaluator.paper_ready(lab, hid):
            out = run_gate(lab, hid, "G5", evaluator)
            log.append(f"{hid} G5 {'passed' if out.passed else 'failed'}")
            moved = True
    return moved

