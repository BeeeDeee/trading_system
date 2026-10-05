"""One agent run = one invocation: render a workspace, let the agent work, apply its outbox.

    start()  framework: lease the hypothesis, write the workspace (inbox, card, catalog, registry, gate
             metrics, empty outbox), record the invocation.
    (agent)  reads the workspace, writes files under it, stages messages with `lab send` (-> outbox.json).
    finish() framework: validate the whole outbox against the role of the invocation (taken from lab.db,
             never from the workspace), copy accepted files into the repo, store the messages, mark the
             inbox handled, release the lease. All or nothing, and a second finish() is a no-op.

The agent never sees lab.db, real data or other workspaces; in production the workspace is the only
directory its OS user can write.
"""

import json
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from lab.framework import agents, catalog, states, validate
from lab.framework.blackboard import Lab, LabError, sha256
from lab.framework.db import Tx, dumps, now

LEASE = timedelta(hours=2)
OUTBOX = "outbox.json"


@dataclass(frozen=True)
class Invocation:
    id: str
    agent: str
    hypothesis_id: str | None
    workspace: Path


def start(lab: Lab, agent: str, hid: str | None = None, model: str = "stub", task: str | None = None) -> Invocation:
    if agent not in states.LLM_AGENTS:
        raise LabError(f"{agent} is not an agent")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    inv_id = f"{agent}-{stamp}-{secrets.token_hex(3)}"
    if hid:
        lab.hypothesis(hid)
        _lease(lab, hid, inv_id)
    ws = lab.paths.workspaces / inv_id
    try:
        inbox = [_message_dict(m) for m in lab.inbox(agent, hid)]
        _render(lab, ws, inv_id, agent, hid, inbox, task)
        with Tx(lab.con):
            lab.con.execute(
                "INSERT INTO agent_invocations (id, agent, model, hypothesis_id, workspace, input_message_ids,"
                " input_sha256, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (inv_id, agent, model, hid, str(ws), dumps([m["id"] for m in inbox]),
                 sha256({"inbox": inbox, "hid": hid, "card": lab.card(hid) if hid else None}), now()))
    except Exception:
        _release(lab, inv_id)
        raise
    return Invocation(inv_id, agent, hid, ws)


def _lease(lab: Lab, hid: str, inv_id: str) -> None:
    with Tx(lab.con):
        row = lab.con.execute("SELECT invocation_id, lease_until FROM locks WHERE hypothesis_id = ?",
                              (hid,)).fetchone()
        if row and row["lease_until"] > now():
            raise LabError(f"{hid} is leased by {row['invocation_id']} until {row['lease_until']}")
        until = (datetime.now(timezone.utc) + LEASE).isoformat(timespec="microseconds")
        lab.con.execute("INSERT INTO locks (hypothesis_id, invocation_id, lease_until) VALUES (?, ?, ?) "
                        "ON CONFLICT(hypothesis_id) DO UPDATE SET invocation_id = excluded.invocation_id,"
                        " lease_until = excluded.lease_until", (hid, inv_id, until))


def _release(lab: Lab, inv_id: str) -> None:
    with Tx(lab.con):
        lab.con.execute("DELETE FROM locks WHERE invocation_id = ?", (inv_id,))


def _message_dict(m) -> dict:
    return {"id": m["id"], "type": m["type"], "from": m["from_agent"], "to": m["to_agent"],
            "hypothesis_id": m["hypothesis_id"], "payload": json.loads(m["payload_json"]),
            "created_at": m["created_at"]}


def _render(lab: Lab, ws: Path, inv_id: str, agent: str, hid: str | None, inbox: list[dict],
            task: str | None) -> None:
    ws.mkdir(parents=True)
    write = lambda name, obj: (ws / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))  # noqa: E731
    write("context.json", {"invocation_id": inv_id, "agent": agent, "hypothesis_id": hid, "task": task})
    write("inbox.json", inbox)
    write(OUTBOX, {"invocation_id": inv_id, "messages": []})
    write("registry.json", [
        {"id": h["id"], "version": h["version"], "title": h["title"], "family": h["family"],
         "status": h["status"], "terminal": {"stage": h["reject_stage"], "reason_code": h["reject_code"]}
         if h["reject_stage"] else None} for h in lab.hypotheses()])
    shutil.copy(lab.paths.catalog, ws / "catalog.yaml")
    if hid:
        shutil.copy(lab.paths.card(hid), ws / "card.yaml")
        rows = lab.con.execute("SELECT gate, version, passed, reason_code, metrics_json, evaluator, ts "
                               "FROM gate_results WHERE hypothesis_id = ? ORDER BY id", (hid,))
        write("gate_results.json", [{**dict(r), "metrics": json.loads(r["metrics_json"])} for r in rows])
        existing = lab.paths.strategies / hid
        if existing.exists():
            shutil.copytree(existing, ws / "strategy", ignore=shutil.ignore_patterns("__pycache__"))
    (ws / "strategy").mkdir(exist_ok=True)
    agents.render_context(lab, ws, agents.context_items(agent), hid)


# ---------------------------------------------------------------------------- staged mode (agent side)

def stage(ws: Path, msg_type: str, to: str, hid: str | None, payload: dict) -> int:
    """Append one message to the workspace outbox after a local check (the real check is in finish())."""
    ctx = json.loads((ws / "context.json").read_text())
    errors = validate.payload_errors(msg_type, payload)
    if msg_type in ("NEW_HYPOTHESIS", "REVISION") and isinstance(payload.get("card"), dict):
        errors += [f"card: {e}" for e in validate.card_errors(payload["card"])]
    rule = states.MESSAGE_RULES.get(msg_type, {})
    if ctx["agent"] not in rule.get("from", ()):
        errors.append(f"{ctx['agent']} cannot send {msg_type}")
    if to not in rule.get("to", ()):
        errors.append(f"{msg_type} cannot be addressed to {to}")
    if errors:
        raise LabError("; ".join(errors))
    box = json.loads((ws / OUTBOX).read_text())
    box["messages"].append({"type": msg_type, "to": to, "hypothesis_id": hid, "payload": payload})
    (ws / OUTBOX).write_text(json.dumps(box, indent=2, ensure_ascii=False))
    return len(box["messages"])


def check_card(ws: Path, rel: str) -> tuple[list[str], list[str]]:
    """What the framework will say about a card, before it is sent: (errors, warnings). Errors keep the card
    in IDEA or refuse it; warnings are advice. Uses only workspace files (catalog, registry)."""
    card = read_card(ws, rel)
    if not isinstance(card, dict):
        return [f"{rel} is not a YAML mapping"], []
    card = {k: v for k, v in card.items() if k not in CARD_DROP}
    errors = validate.card_errors(card)
    if errors:
        return errors, []
    errors += [f"missing or empty for SPECIFIED: {f}" for f in validate.specified_missing(card)]
    errors += validate.grid_problems(card)
    warnings = [w for w in validate.grid_problems(card, both_sides=True) if w not in errors]
    if card.get("data_requirements"):
        res = catalog.resolve(ws / "catalog.yaml", card["data_requirements"])
        errors += [f"dataset {r['dataset']} is not in the catalog (the card would go to BLOCKED_DATA)"
                   for r in res.missing]
        errors += res.problems
    registry = json.loads((ws / "registry.json").read_text())
    title = card["title"].strip().lower()
    warnings += [f"same title as {h['id']} ({h['status']})" for h in registry
                 if h["title"].strip().lower() == title]
    same_family = [h["id"] for h in registry if h["family"] == card["family"]]
    if same_family:
        warnings.append(f"family {card['family']!r} already has {len(same_family)} hypotheses ({', '.join(same_family)}); "
                        "the duplicate check compares signal descriptions within the family")
    return errors, warnings


CARD_DROP = {"id", "version", "status", "history", "terminal", "author_agent"}


def read_card(ws: Path, rel: str) -> dict:
    path = _inside(ws, rel)
    return yaml.safe_load(path.read_text())


def _inside(ws: Path, rel: str) -> Path:
    path = (ws / rel).resolve()
    if not path.is_relative_to(ws.resolve()) or not path.is_file():
        raise LabError(f"{rel} is not a file inside the workspace")
    return path


# ---------------------------------------------------------------------------- finish (framework side)

def finish(lab: Lab, inv_id: str, exit_code: int = 0, *, transcript_path: str | None = None,
           usage: dict | None = None, violations: list[str] | None = None, timed_out: bool = False,
           note: str | None = None) -> str:
    """Apply the outbox, or nothing. `violations` (from the transcript audit) discard the whole run and raise
    an ALERT to the owner: an agent that tried to step outside its workspace does not get its messages in."""
    row = lab.con.execute("SELECT * FROM agent_invocations WHERE id = ?", (inv_id,)).fetchone()
    if row is None:
        raise LabError(f"unknown invocation {inv_id}")
    if row["outcome"]:
        return row["outcome"]  # idempotent
    agent, hid, ws = row["agent"], row["hypothesis_id"], Path(row["workspace"])
    usage, box, plan = usage or {}, None, None
    if violations:
        outcome, error = "policy_violation", "; ".join(violations)
    elif timed_out:
        outcome, error = "timeout", note or "wall-clock limit"
    elif exit_code != 0:
        outcome, error = "failed", note or f"exit code {exit_code}"   # a crashed run's outbox is not applied
    else:
        try:
            box = json.loads((ws / OUTBOX).read_text())
            plan = _check_outbox(lab, box, inv_id, agent, hid, ws)
            outcome, error = "applied", None
        except (LabError, ValueError, OSError) as e:
            outcome, error = "outbox_rejected", str(e)

    if outcome == "applied":
        for src, dst in plan["files"]:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        with Tx(lab.con):
            for k, (msg_type, to, mhid, payload) in enumerate(plan["messages"]):
                lab.send(msg_type, agent, to, mhid, payload, invocation_id=inv_id, key=f"{inv_id}:{k}")
            lab.mark_handled(json.loads(row["input_message_ids"]), inv_id)
            _finish_row(lab, inv_id, exit_code, outcome, error, box, transcript_path, usage)
    else:
        with Tx(lab.con):
            _finish_row(lab, inv_id, exit_code, outcome, error, box, transcript_path, usage)
        if outcome == "policy_violation":
            lab.send("ALERT", "system", "human", hid, {"severity": "critical", "text":
                     f"{agent} run {inv_id} discarded, policy violation: {error}"[:2000]})
    _release(lab, inv_id)
    return outcome


def _finish_row(lab, inv_id, exit_code, outcome, error, box, transcript_path, usage):
    lab.con.execute(
        "UPDATE agent_invocations SET ended_at = ?, exit_code = ?, outcome = ?, error = ?, output_json = ?,"
        " transcript_path = ?, n_turns = ?, tokens_in = ?, tokens_out = ? WHERE id = ?",
        (now(), exit_code, outcome, error, dumps(box) if box is not None else None, transcript_path,
         usage.get("n_turns"), usage.get("tokens_in"), usage.get("tokens_out"), inv_id))


def _check_outbox(lab: Lab, box: dict, inv_id: str, agent: str, hid: str | None, ws: Path) -> dict:
    """Validate everything before anything is applied. Returns the messages and file copies to do."""
    errors = validate.outbox_errors(box)
    if errors:
        raise LabError("outbox: " + "; ".join(errors))
    if box["invocation_id"] != inv_id:
        raise LabError("outbox belongs to another invocation")
    messages, files = [], []
    for i, m in enumerate(box["messages"]):
        mhid, payload = m.get("hypothesis_id"), m["payload"]
        if hid and mhid not in (hid, None):
            raise LabError(f"message {i}: this run may only write about {hid}, not {mhid}")
        try:
            lab.check_message(m["type"], agent, m["to"], mhid, payload)
        except LabError as e:
            raise LabError(f"message {i}: {e}") from None
        if m["type"] == "IMPL_DONE":
            if agent != "builder" or mhid != hid:
                raise LabError(f"message {i}: IMPL_DONE only from the builder for its leased hypothesis")
            root = (ws / "strategy").resolve()
            for rel in payload["files"]:
                src = _inside(ws, rel)
                if not src.is_relative_to(root):
                    raise LabError(f"message {i}: {rel} is not inside strategy/")
                files.append((src, lab.paths.strategies / hid / src.relative_to(root)))
        if m["type"] == "DATA_READY":
            src = _inside(ws, payload["source_file"])
            files.append((src, lab.paths.lab / "data" / "sources" / f"{payload['dataset']}{src.suffix}"))
        messages.append((m["type"], m["to"], mhid, payload))
    return {"messages": messages, "files": files}
