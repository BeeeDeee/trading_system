"""Agent definitions (`lab/agents/agents.yaml`) and the framework context copied into their workspaces.

The definition is framework data: model, prompt file, tools, allowed `lab` verbs, timeout, and the context
items an agent gets. Context items are produced here, never by an agent.
"""

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


def render_context(paths: LabPaths, ws: Path, items: tuple[str, ...]) -> None:
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
            case _:
                raise ValueError(f"unknown context item {item!r}")
