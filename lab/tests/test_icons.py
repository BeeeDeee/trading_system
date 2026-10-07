"""Icons: well-formed SVG for every agent and mood, and the pipeline shows who is an LLM and who is code."""

import xml.etree.ElementTree as ET

import pytest

from lab.framework import icons, pipeline_view
from lab.framework.states import LLM_AGENTS

MOODS = ("work", "wait", "done", "fail", "idle", "sleep")


@pytest.mark.parametrize("mood", MOODS)
def test_every_icon_is_wellformed_svg(mood):
    for agent in LLM_AGENTS:
        for kind in ("llm", "mixed"):
            ET.fromstring(icons.svg(kind, agent, mood))
    ET.fromstring(icons.svg("code", None, mood))


def test_stage_table_says_which_stages_use_an_llm():
    kinds = {key: kind for key, _, _, kind, _ in pipeline_view.STAGES}
    assert kinds["idea"] == "llm" and kinds["skeptic"] == "llm"
    assert kinds["g1"] == kinds["g2"] == kinds["g3"] == kinds["g4"] == "code"
    assert kinds["build"] == kinds["data"] == "mixed"
    assert pipeline_view.model_label("scout") == "Opus 5.5" and pipeline_view.model_label("chair") == "Sonnet 5.5"
    assert pipeline_view.model_label(None) == ""
