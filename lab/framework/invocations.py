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

from lab.framework import states, validate
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


def start(lab: Lab, agent: str, hid: str | None = None, model: str = "stub") -> Invocation:
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
        _render(lab, ws, inv_id, agent, hid, inbox)
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


def _render(lab: Lab, ws: Path, inv_id: str, agent: str, hid: str | None, inbox: list[dict]) -> None:
    ws.mkdir(parents=True)
    write = lambda name, obj: (ws / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))  # noqa: E731
    write("context.json", {"invocation_id": inv_id, "agent": agent, "hypothesis_id": hid})
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


# ---------------------------------------------------------------------------- staged mode (agent side)

def stage(ws: Path, msg_type: str, to: str, hid: str | None, payload: dict) -> int:
    """Append one message to the workspace outbox after a local check (the real check is in finish())."""
    ctx = json.loads((ws / "context.json").read_text())
    errors = validate.payload_errors(msg_type, payload)
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
           usage: dict | None = None) -> str:
    row = lab.con.execute("SELECT * FROM agent_invocations WHERE id = ?", (inv_id,)).fetchone()
    if row is None:
        raise LabError(f"unknown invocation {inv_id}")
    if row["outcome"]:
        return row["outcome"]  # idempotent
    agent, hid, ws = row["agent"], row["hypothesis_id"], Path(row["workspace"])
    usage, box, plan = usage or {}, None, None
    if exit_code != 0:
        outcome, error = "failed", f"exit code {exit_code}"   # a crashed run's partial outbox is not applied
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
