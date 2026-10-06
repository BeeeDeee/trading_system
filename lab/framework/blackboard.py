"""Blackboard API: hypotheses, transitions and typed messages over `lab.db`.

This module is the only code that writes those tables. Every write checks the permission tables in
`states.py`, so the rules hold no matter which caller (handler, CLI, orchestrator) asks.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass

import yaml

from lab.framework import db, states, validate
from lab.framework.db import Tx, dumps, now
from lab.framework.paths import LabPaths
from lab.framework.states import S

CARD_ORDER = ("title", "author_agent", "family", "asset_classes", "universe", "mechanism", "signal",
              "mechanism_test", "data_requirements", "holding_period", "market_exposure", "references", "falsification_criteria",
              "notes")


class LabError(RuntimeError):
    """A refused action (permission, schema, guard). The message is shown to the caller or agent."""


def sha256(obj) -> str:
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


@dataclass
class Lab:
    paths: LabPaths
    con: sqlite3.Connection

    @classmethod
    def open(cls, paths: LabPaths) -> "Lab":
        return cls(paths, db.init(paths.db))

    # ---------------------------------------------------------------- hypotheses

    def hypothesis(self, hid: str) -> sqlite3.Row:
        row = self.con.execute("SELECT * FROM hypotheses WHERE id = ?", (hid,)).fetchone()
        if row is None:
            raise LabError(f"unknown hypothesis {hid}")
        return row

    def card(self, hid: str) -> dict:
        return json.loads(self.hypothesis(hid)["card_json"])

    def hypotheses(self, status: str | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM hypotheses", ()
        if status:
            q, args = q + " WHERE status = ?", (status,)
        return self.con.execute(q + " ORDER BY id", args).fetchall()

    def create_hypothesis(self, card: dict, actor: str, *, invocation_id: str | None = None,
                          message_id: int | None = None) -> str:
        errors = validate.card_errors(card)
        if errors:
            raise LabError("invalid card: " + "; ".join(errors))
        if not states.allowed(None, S.IDEA, actor):
            raise LabError(f"{actor} cannot create hypotheses")
        card = {**card, "author_agent": actor}
        with Tx(self.con):
            hid = f"H-{db.next_id(self.con, 'hypothesis'):04d}"
            ts = now()
            self.con.execute(
                "INSERT INTO hypotheses (id, version, family, title, author_agent, status, card_json, card_sha256,"
                " created_at, updated_at) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)",
                (hid, card["family"], card["title"], actor, S.IDEA, dumps(card), sha256(card), ts, ts))
            self._log_transition(hid, 1, None, S.IDEA, actor, "created", invocation_id, message_id)
        self.sync_card(hid)
        return hid

    def revise(self, hid: str, card: dict, actor: str, reason: str) -> int:
        """New version of the card. Only in IDEA (spec fixes, Skeptic returns, reopened PARKED)."""
        h = self.hypothesis(hid)
        if h["status"] != S.IDEA:
            raise LabError(f"{hid} is {h['status']}; a card can only be revised in IDEA")
        errors = validate.card_errors(card)
        if errors:
            raise LabError("invalid card: " + "; ".join(errors))
        if card["family"] != h["family"]:
            raise LabError("the family of a hypothesis cannot change (it would reset the trial count)")
        card = {**card, "author_agent": h["author_agent"]}
        version = h["version"] + 1
        with Tx(self.con):
            self.con.execute("UPDATE hypotheses SET version = ?, title = ?, card_json = ?, card_sha256 = ?,"
                             " updated_at = ? WHERE id = ?",
                             (version, card["title"], dumps(card), sha256(card), now(), hid))
            self._log_transition(hid, version, S.IDEA, S.IDEA, actor, f"revision v{version}: {reason}")
        self.sync_card(hid)
        return version

    def transition(self, hid: str, to: S, actor: str, reason: str, *, invocation_id: str | None = None,
                   message_id: int | None = None, reason_code: str | None = None,
                   stage: str | None = None) -> None:
        """Move a hypothesis. For terminal states `stage` says where it died (default: the state it left)."""
        h = self.hypothesis(hid)
        frm = S(h["status"])
        if not states.allowed(frm, to, actor):
            raise LabError(f"{actor} cannot move {hid} from {frm} to {to}")
        gate = states.GATE_FOR.get(to)
        if gate and not self.gate_passed(hid, h["version"], gate):
            raise LabError(f"{hid} v{h['version']} has no passing {gate} result; cannot enter {to}")
        terminal = to in states.TERMINAL
        with Tx(self.con):
            self.con.execute(
                "UPDATE hypotheses SET status = ?, reject_stage = ?, reject_code = ?, updated_at = ? WHERE id = ?",
                (to, (stage or frm) if terminal else None, reason_code if terminal else None, now(), hid))
            self._log_transition(hid, h["version"], frm, to, actor, reason, invocation_id, message_id)
        self.sync_card(hid)

    def _log_transition(self, hid, version, frm, to, actor, reason, invocation_id=None, message_id=None):
        self.con.execute(
            "INSERT INTO transitions (hypothesis_id, version, from_status, to_status, actor, invocation_id,"
            " message_id, reason, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (hid, version, frm, to, actor, invocation_id, message_id, reason, now()))

    def history(self, hid: str) -> list[dict]:
        rows = self.con.execute("SELECT * FROM transitions WHERE hypothesis_id = ? ORDER BY id", (hid,))
        return [{"ts": r["ts"], "version": r["version"], "from": r["from_status"], "to": r["to_status"],
                 "agent": r["actor"], "invocation_id": r["invocation_id"], "reason": r["reason"]} for r in rows]

    def sync_card(self, hid: str) -> None:
        """Write the human-readable card (content + status + history) to lab/hypotheses/<id>.yaml."""
        h, card = self.hypothesis(hid), self.card(hid)
        out = {"id": hid, "version": h["version"]}
        out.update((k, card[k]) for k in CARD_ORDER if k in card)
        out["status"] = h["status"]
        if h["reject_stage"]:
            out["terminal"] = {"stage": h["reject_stage"], "reason_code": h["reject_code"]}
        out["history"] = self.history(hid)
        path = self.paths.card(hid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Generated by the lab framework from lab.db. Do not edit; send a REVISION.\n"
                        + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=110))

    # ---------------------------------------------------------------- messages

    def send(self, msg_type: str, sender: str, to: str, hid: str | None, payload: dict, *,
             invocation_id: str | None = None, key: str | None = None) -> int:
        """Store a typed message (idempotent on `key`). Returns its id."""
        self.check_message(msg_type, sender, to, hid, payload)
        key = key or sha256([sender, to, msg_type, hid, payload, invocation_id])
        with Tx(self.con):
            old = self.con.execute("SELECT id FROM messages WHERE idempotency_key = ?", (key,)).fetchone()
            if old:
                return int(old["id"])
            cur = self.con.execute(
                "INSERT INTO messages (type, from_agent, to_agent, hypothesis_id, payload_json, schema_version,"
                " idempotency_key, invocation_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (msg_type, sender, to, hid, dumps(payload), validate.SCHEMA_VERSION, key, invocation_id, now()))
            return int(cur.lastrowid)

    def check_message(self, msg_type: str, sender: str, to: str, hid: str | None, payload: dict) -> None:
        """Raise LabError unless `sender` may send this message. No side effects."""
        rule = states.MESSAGE_RULES.get(msg_type)
        if rule is None:
            raise LabError(f"unknown message type {msg_type}")
        if sender not in rule["from"]:
            raise LabError(f"{sender} cannot send {msg_type}")
        if to not in rule["to"]:
            raise LabError(f"{msg_type} cannot be addressed to {to}")
        errors = validate.payload_errors(msg_type, payload)
        if errors:
            raise LabError(f"invalid {msg_type} payload: " + "; ".join(errors))
        if msg_type == "VERDICT" and payload["decision"] not in states.VERDICT_DECISIONS.get(sender, ()):
            raise LabError(f"{sender} cannot take the decision {payload['decision']!r}")
        if msg_type == "VERDICT" and sender == "skeptic" and payload["decision"] == "no_objection":
            required = yaml.safe_load(self.paths.gates.read_text())["skeptic"]["checklist"]
            checklist = payload.get("checklist") or {}
            gaps = [c for c in required if checklist.get(c, {}).get("status") != "ok"]
            if gaps:
                raise LabError("no_objection needs every checklist item with status ok and evidence; "
                               f"missing or concern: {', '.join(gaps)} (send an OBJECTION instead)")
        if msg_type in ("OBJECTION", "VERDICT", "IMPL_DONE", "REVISION", "LESSON") and hid is None:
            raise LabError(f"{msg_type} must name a hypothesis")
        if hid is not None:
            self.hypothesis(hid)

    def inbox(self, agent: str, hid: str | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM messages WHERE to_agent = ? AND handled_at IS NULL", [agent]
        if hid:
            q += " AND (hypothesis_id = ? OR hypothesis_id IS NULL)"
            args.append(hid)
        return self.con.execute(q + " ORDER BY id", args).fetchall()

    def mark_handled(self, message_ids: list[int], by: str) -> None:
        with Tx(self.con):
            for mid in message_ids:
                self.con.execute("UPDATE messages SET handled_at = ?, handled_by = ? "
                                 "WHERE id = ? AND handled_at IS NULL", (now(), by, mid))

    # ---------------------------------------------------------------- gates and trials

    def gate_passed(self, hid: str, version: int, gate: str) -> bool:
        row = self.con.execute("SELECT passed FROM gate_results WHERE hypothesis_id = ? AND version = ? AND gate = ?"
                               " ORDER BY id DESC LIMIT 1", (hid, version, gate)).fetchone()
        return bool(row and row["passed"])

    # ---------------------------------------------------------------- families (multiple-testing groups)

    def canonical_family(self, family: str) -> str:
        seen = set()
        while family not in seen:
            seen.add(family)
            row = self.con.execute("SELECT into_family FROM family_merges WHERE from_family = ?", (family,)).fetchone()
            if row is None:
                return family
            family = row[0]
        return family

    def family_members(self, family: str) -> list[str]:
        """Every family name merged into the same canonical family (trials are counted over all of them)."""
        root = self.canonical_family(family)
        names = {r[0] for r in self.con.execute("SELECT family FROM hypotheses")} | {
            r[0] for r in self.con.execute("SELECT from_family FROM family_merges")} | {root}
        return sorted(f for f in names if self.canonical_family(f) == root)

    def merge_family(self, from_family: str, into_family: str, actor: str, reason: str,
                     message_id: int | None = None) -> None:
        if actor not in ("chair", "human"):
            raise LabError(f"{actor} cannot merge families")
        known = {r[0] for r in self.con.execute("SELECT family FROM hypotheses")}
        for f in (from_family, into_family):
            if f not in known:
                raise LabError(f"unknown family {f!r}")
        if self.canonical_family(from_family) != from_family:
            raise LabError(f"{from_family!r} is already merged into {self.canonical_family(from_family)!r}")
        into = self.canonical_family(into_family)
        if into == from_family:
            raise LabError("a family cannot be merged into itself")
        with Tx(self.con):
            self.con.execute("INSERT INTO family_merges (from_family, into_family, actor, reason, message_id, ts) "
                             "VALUES (?, ?, ?, ?, ?, ?)", (from_family, into, actor, reason, message_id, now()))

    def _in_family(self, family: str) -> tuple[str, list[str]]:
        members = self.family_members(family)
        return ",".join("?" * len(members)), members

    def family_trials(self, family: str) -> int:
        q, members = self._in_family(family)
        return int(self.con.execute(f"SELECT COUNT(*) FROM trials WHERE family IN ({q})", members).fetchone()[0])

    def family_trial_sharpes(self, family: str) -> list[float]:
        q, members = self._in_family(family)
        return [r[0] for r in self.con.execute(
            f"SELECT sharpe FROM trials WHERE family IN ({q}) AND sharpe IS NOT NULL", members)]

    def family_holdout_attempts(self, family: str) -> int:
        q, members = self._in_family(family)
        return int(self.con.execute(
            "SELECT COUNT(*) FROM gate_results g JOIN hypotheses h ON h.id = g.hypothesis_id "
            f"WHERE g.gate = 'G4' AND h.family IN ({q})", members).fetchone()[0])
