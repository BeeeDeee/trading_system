import copy

import pytest

from lab import stubs
from lab.framework.blackboard import Lab
from lab.framework.gates import StubEvaluator
from lab.framework.paths import sandbox_paths
from lab.framework.tick import tick


@pytest.fixture
def lab(tmp_path) -> Lab:
    return Lab.open(sandbox_paths(tmp_path))


@pytest.fixture
def card() -> dict:
    """A complete card whose data is in the catalog (goes straight to DATA_READY)."""
    return copy.deepcopy(stubs.load_example("etf_sector_momentum.yaml"))


@pytest.fixture
def blocked_card() -> dict:
    """A complete card that needs a dataset not in the catalog."""
    return copy.deepcopy(stubs.load_example("btc_dvol_vrp.yaml"))


class Evaluator(StubEvaluator):
    def __init__(self, script=None, paper_ready=False):
        super().__init__(script, allow=True)
        self._paper = paper_ready

    def paper_ready(self, lab, hid):
        return self._paper


@pytest.fixture
def evaluator():
    return Evaluator()


def submit(lab: Lab, card: dict, actor: str = "human", evaluator=None) -> str:
    """Send NEW_HYPOTHESIS and let the framework react; returns the new id."""
    lab.send("NEW_HYPOTHESIS", actor, "system", None, {"card": card})
    tick(lab, evaluator)
    return lab.hypotheses()[-1]["id"]
