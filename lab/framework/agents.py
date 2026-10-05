"""Agent definitions (`lab/agents/agents.yaml`) and the framework context copied into their workspaces.

The definition is framework data: model, prompt file, tools, allowed `lab` verbs, timeout, and the context
items an agent gets. Context items are produced here, never by an agent.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from lab.framework import factsheets
from lab.framework.paths import LAB_DIR, LabPaths

AGENTS_DIR = LAB_DIR / "agents"
EXAMPLE_CARD = LAB_DIR / "examples" / "etf_sector_momentum.yaml"


@dataclass(frozen=True)
class AgentSpec:
    name: str
    model: str
    prompt_file: Path
    tools: tuple[str, ...]
    lab_verbs: tuple[str, ...]
    timeout_s: float
    effort: str | None
    context: tuple[str, ...]

    @property
    def prompt(self) -> str:
        return self.prompt_file.read_text()


def spec(agent: str, config: Path = AGENTS_DIR / "agents.yaml") -> AgentSpec:
    cfg = yaml.safe_load(config.read_text())
    if agent not in cfg["agents"]:
        raise KeyError(f"no headless definition for agent {agent!r} in {config.name}")
    a = {**cfg["defaults"], **cfg["agents"][agent]}
    return AgentSpec(agent, a["model"], config.parent / a["prompt"], tuple(a["tools"]), tuple(a["lab_verbs"]),
                     float(a["timeout_s"]), a.get("effort"), tuple(a.get("context", ())))


def context_items(agent: str) -> tuple[str, ...]:
    try:
        return spec(agent).context
    except KeyError:
        return ()   # agents without a headless definition yet (stubs) get no extra context


def render_context(lab, ws: Path, items: tuple[str, ...], hid: str | None = None) -> None:
    paths: LabPaths = lab.paths
    for item in items:
        match item:
            case "factsheets":
                try:
                    shutil.copy(factsheets.get(paths.home, paths.catalog), ws / "factsheets.md")
                except (OSError, KeyError, ValueError) as e:   # no data on this machine (tests, CI)
                    (ws / "factsheets.md").write_text(f"# Fact sheets unavailable\n\n{type(e).__name__}: {e}\n")
            case "prior_studies":
                shutil.copy(AGENTS_DIR / "context" / "prior_studies.md", ws / "prior_studies.md")
            case "card_schema":
                shutil.copy(LAB_DIR / "framework" / "schemas" / "hypothesis.schema.json", ws / "hypothesis.schema.json")
            case "card_example":
                shutil.copy(EXAMPLE_CARD, ws / "card_example.yaml")
            case "gates":
                shutil.copy(paths.gates, ws / "gates.yaml")
            case "history":
                if hid:
                    (ws / "history.json").write_text(json.dumps(history(lab, hid), indent=2, ensure_ascii=False))
            case "blocked":
                (ws / "blocked.json").write_text(json.dumps(blocked(lab), indent=2, ensure_ascii=False))
            case "family":
                if hid:
                    (ws / "family.json").write_text(json.dumps(family(lab, hid), indent=2, ensure_ascii=False))
            case _:
                raise ValueError(f"unknown context item {item!r}")


def history(lab, hid: str) -> dict:
    """Everything said and done about one hypothesis, with full payloads (IMPL_DONE summaries, earlier
    objections and verdicts, gate results) and the transitions, in time order."""
    msgs = [{"id": r["id"], "ts": r["created_at"], "type": r["type"], "from": r["from_agent"], "to": r["to_agent"],
             "payload": json.loads(r["payload_json"])}
            for r in lab.con.execute("SELECT * FROM messages WHERE hypothesis_id = ? ORDER BY id", (hid,))]
    trans = [{"ts": r["ts"], "version": r["version"], "from": r["from_status"], "to": r["to_status"],
              "actor": r["actor"], "reason": r["reason"]}
             for r in lab.con.execute("SELECT * FROM transitions WHERE hypothesis_id = ? ORDER BY id", (hid,))]
    h = lab.hypothesis(hid)
    return {"hypothesis_id": hid, "version": h["version"], "status": h["status"],
            "objection_rounds": h["objection_rounds"], "messages": msgs, "transitions": trans}


def family(lab, hid: str) -> dict:
    """The multiple-testing context: every hypothesis and every recorded trial of the same family."""
    fam = lab.hypothesis(hid)["family"]
    hyps = [{"id": r["id"], "title": r["title"], "status": r["status"], "reject_stage": r["reject_stage"],
             "reject_code": r["reject_code"]} for r in lab.hypotheses() if r["family"] == fam]
    n = lab.con.execute("SELECT COUNT(*) FROM trials WHERE family = ?", (fam,)).fetchone()[0]
    return {"family": fam, "trials_recorded": n, "hypotheses": hyps}


def blocked(lab) -> list[dict]:
    """Hypotheses waiting for data, with what they need (the Archivist's brief)."""
    out = []
    for h in lab.hypotheses("BLOCKED_DATA"):
        card = lab.card(h["id"])
        out.append({"id": h["id"], "title": card["title"], "asset_classes": card.get("asset_classes"),
                    "signal": card.get("signal", {}).get("description"),
                    "data_requirements": card.get("data_requirements")})
    return out
