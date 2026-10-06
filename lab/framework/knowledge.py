"""The lab's knowledge base: durable findings that hold across hypotheses, rendered for every agent.

Per-hypothesis lessons (lessons.py) say what one card taught. Knowledge entries say what the lab now knows:
market regularities (`market`), data problems (`data`), methodological pitfalls (`method`), how the judge
behaves (`framework`) and how the lab should work (`process`). They come from the Librarian, the Chair and the
owner, are append-only, and a correction is a new entry that `supersedes` the old one.

`market` entries obey the lessons' rule: no metrics, because the Scout reads them and dev-period numbers fed
back into proposals are an untracked adaptive search. Other kinds may carry numbers (counts, dates, sizes).
"""

import json

from lab.framework import lessons
from lab.framework.blackboard import Lab, LabError
from lab.framework.db import Tx, dumps, now

KINDS = ("market", "data", "method", "framework", "process")
TITLES = {"market": "Markets and mechanisms", "data": "Data", "method": "Methodology",
          "framework": "The judge (framework behaviour)", "process": "How the lab works"}


def add(lab: Lab, payload: dict, actor: str, message_id: int | None = None) -> str:
    if payload["kind"] == "market":
        for f in ("title", "statement"):
            m = lessons.NUMBERS.search(payload[f])
            if m:
                raise LabError(f"{f}: {m.group(0)!r} looks like a metric; market knowledge is qualitative")
    sup = payload.get("supersedes")
    if sup and lab.con.execute("SELECT 1 FROM knowledge WHERE id = ?", (sup,)).fetchone() is None:
        raise LabError(f"unknown knowledge entry {sup}")
    with Tx(lab.con):
        n = lab.con.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0] + 1
        kid = f"K-{n:04d}"
        lab.con.execute("INSERT INTO knowledge (id, kind, title, statement, evidence, confidence, supersedes, actor,"
                        " message_id, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (kid, payload["kind"], payload["title"], payload["statement"], dumps(payload["evidence"]),
                         payload["confidence"], sup, actor, message_id, now()))
    render(lab)
    return kid


def entries(lab: Lab) -> list[dict]:
    rows = [dict(r) for r in lab.con.execute("SELECT * FROM knowledge ORDER BY id")]
    superseded = {r["supersedes"] for r in rows if r["supersedes"]}
    for r in rows:
        r["evidence"] = json.loads(r["evidence"])
        r["current"] = r["id"] not in superseded
    return rows


def path(lab: Lab):
    return lab.paths.lab / "knowledge" / "KNOWLEDGE.md"


def render(lab: Lab) -> None:
    rows = entries(lab)
    out = ["# Lab knowledge base", "",
           "Durable findings across hypotheses (per-hypothesis lessons are in lessons.md). Append-only: a correction",
           "is a new entry that supersedes the old one. Written by the Librarian, the Chair and the owner;",
           "rendered by the framework.", ""]
    for kind in KINDS:
        cur = [r for r in rows if r["kind"] == kind and r["current"]]
        if not cur:
            continue
        out += [f"## {TITLES[kind]}", ""]
        for r in cur:
            out += [f"### {r['id']} {r['title']}", "", r["statement"], "",
                    f"*confidence {r['confidence']} · {r['ts'][:10]} · {r['actor']}"
                    + (f" · supersedes {r['supersedes']}" if r["supersedes"] else "")
                    + " · evidence: " + "; ".join(r["evidence"]) + "*", ""]
    old = [r for r in rows if not r["current"]]
    if old:
        out += ["## Superseded", ""] + [f"- {r['id']} {r['title']}" for r in old] + [""]
    f = path(lab)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(out))
