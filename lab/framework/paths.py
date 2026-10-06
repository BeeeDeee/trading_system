"""Where the lab keeps its state.

Two roots:
- the repo (`lab/`): code, gates.yaml, catalog, hypothesis cards and strategy code, all in git,
- the runtime home (`LAB_HOME`, default `<repo>/var/lab`): lab.db, agent workspaces, ingest data.
  In production this is `/srv/research-lab`, owned by the labcore user and unreadable for agents.

Tests and the demo build a `LabPaths` under a temporary directory, so they never touch the real state.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[1]
REPO = LAB_DIR.parent


@dataclass(frozen=True)
class LabPaths:
    home: Path          # runtime root (lab.db, workspaces)
    lab: Path           # the `lab/` tree with cards, strategies, catalog and gates.yaml

    @property
    def db(self) -> Path:
        return self.home / "lab.db"

    @property
    def workspaces(self) -> Path:
        """Agent workspaces. In production outside LAB_HOME (`LAB_WORKSPACES`), because the agent user may
        write there but must not even list LAB_HOME (lab.db, data, transcripts)."""
        env = os.environ.get("LAB_WORKSPACES")
        return Path(env) if env and self.home == default_paths().home else self.home / "workspaces"

    @property
    def kill_switch(self) -> Path:
        return self.home / "KILL"

    @property
    def gates(self) -> Path:
        return self.lab / "gates.yaml"

    @property
    def catalog(self) -> Path:
        return self.lab / "data" / "catalog.yaml"

    @property
    def hypotheses(self) -> Path:
        return self.lab / "hypotheses"

    @property
    def strategies(self) -> Path:
        return self.lab / "strategies"

    def card(self, hypothesis_id: str) -> Path:
        return self.hypotheses / f"{hypothesis_id}.yaml"


def default_paths() -> LabPaths:
    return LabPaths(home=Path(os.environ.get("LAB_HOME", REPO / "var" / "lab")), lab=LAB_DIR)


def sandbox_paths(root: Path) -> LabPaths:
    """A throwaway lab under `root`: copies gates.yaml and the catalog, empty cards and strategies."""
    lab = root / "lab"
    (lab / "data").mkdir(parents=True, exist_ok=True)
    (lab / "hypotheses").mkdir(exist_ok=True)
    (lab / "strategies").mkdir(exist_ok=True)
    shutil.copy(LAB_DIR / "gates.yaml", lab / "gates.yaml")
    shutil.copy(LAB_DIR / "data" / "catalog.yaml", lab / "data" / "catalog.yaml")
    return LabPaths(home=root / "home", lab=lab)
