"""Stub agents for the walking skeleton: plain Python in place of the LLM agents of step 3.

They do exactly what a real agent may do: read their workspace and stage messages and files. That makes
the whole pipeline testable without spending a single headless Claude run.
"""

import json
from pathlib import Path

import yaml

from lab.framework.blackboard import Lab
from lab.framework.invocations import Invocation, stage

STRATEGY_TEMPLATE = '''"""Stub strategy for {hid} (walking skeleton). The real interface arrives in step 2."""

PARAMS = {params}


def target_weights(data, params=PARAMS):
    raise NotImplementedError("stub")
'''


def _inbox(inv: Invocation) -> list[dict]:
    return json.loads((inv.workspace / "inbox.json").read_text())


def scout(lab: Lab, inv: Invocation) -> int:
    """Answers objections and spec questions with a REVISION that addresses them."""
    for m in _inbox(inv):
        if m["type"] in ("OBJECTION", "QUESTION") and m["hypothesis_id"] == inv.hypothesis_id:
            card = _card_content(inv)
            card["notes"] = (card.get("notes", "") + f" Revised after message {m['id']}.").strip()
            stage(inv.workspace, "REVISION", "system", inv.hypothesis_id,
                  {"card": card, "reason": f"answer to message {m['id']} ({m['type']})"})
            return 0
    return 0


def _card_content(inv: Invocation) -> dict:
    card = yaml.safe_load((inv.workspace / "card.yaml").read_text())
    drop = {"id", "version", "status", "history", "terminal", "author_agent"}
    return {k: v for k, v in card.items() if k not in drop}


def archivist(lab: Lab, inv: Invocation) -> int:
    """Finds every requested dataset (in the stub: always) and proposes a catalog entry + fetcher."""
    seen = set()
    for m in _inbox(inv):
        if m["type"] != "DATA_REQUEST" or m["payload"]["dataset"] in seen:
            continue
        ds = m["payload"]["dataset"]
        seen.add(ds)
        src = inv.workspace / "sources" / f"{ds}.py"
        src.parent.mkdir(exist_ok=True)
        src.write_text(f'"""Fetcher for {ds} (stub)."""\n\n\ndef fetch(out_dir):\n    raise NotImplementedError\n')
        entry = DEMO_DATASETS.get(ds) or {
            "id": ds, "asset_class": "alt", "instruments": "stub", "frequency": m["payload"]["frequency"],
            "range": ["2015-01-01", "2026-09-30"], "clock": "next_morning", "source": "stub",
            "quality": "stub", "holdout_from": "2023-01-01", "known_biases": ["stub dataset"],
            "forward_source": None}
        stage(inv.workspace, "DATA_READY", "system", None,
              {"dataset": ds, "catalog_entry": entry, "source_file": f"sources/{ds}.py"})
    return 0


def builder(lab: Lab, inv: Invocation) -> int:
    hid = inv.hypothesis_id
    card = _card_content(inv)
    params = {k: v["value"] for k, v in card["signal"]["params"].items()}
    fixes = [m for m in _inbox(inv) if m["type"] in ("OBJECTION", "GATE_RESULT")]
    (inv.workspace / "strategy" / "strategy.py").write_text(STRATEGY_TEMPLATE.format(hid=hid, params=params))
    (inv.workspace / "strategy" / "test_strategy.py").write_text(
        "def test_stub():\n    assert True\n")
    summary = "initial implementation" if not fixes else f"addresses messages {[m['id'] for m in fixes]}"
    stage(inv.workspace, "IMPL_DONE", "gatekeeper", hid,
          {"files": ["strategy/strategy.py", "strategy/test_strategy.py"], "summary": summary,
           "tests_passed": True})
    return 0


def make_skeptic(objections: dict[str, int]):
    """A Skeptic that objects `objections[hid]` times (to the builder), then has no objection."""
    raised: dict[str, int] = {}

    def skeptic(lab: Lab, inv: Invocation) -> int:
        hid = inv.hypothesis_id
        if raised.get(hid, 0) < objections.get(hid, 0):
            raised[hid] = raised.get(hid, 0) + 1
            stage(inv.workspace, "OBJECTION", "builder", hid, {"return_to": "builder", "items": [
                {"check": "costs_capacity", "finding": "funding cost of the perp leg is not charged",
                 "evidence": "strategy.py never reads binance_funding"}]})
            return 0
        checklist = yaml.safe_load(lab.paths.gates.read_text())["skeptic"]["checklist"]
        stage(inv.workspace, "VERDICT", "system", hid, {
            "decision": "no_objection", "reason": "checklist complete, no open issue (stub)",
            "checklist": {c: {"status": "ok", "evidence": f"stub check of {c} found nothing"} for c in checklist}})
        return 0

    return skeptic


def librarian(lab: Lab, inv: Invocation) -> int:
    return 0  # consumes its inbox; lessons learned come in step 3


DEMO_DATASETS = {
    "deribit_dvol_1d": {
        "id": "deribit_dvol_1d", "asset_class": "crypto_spot",
        "instruments": "Deribit DVOL index (30-day implied volatility) for BTC and ETH",
        "frequency": "1d", "range": ["2021-03-24", "2026-10-03"], "clock": "crypto",
        "source": "Deribit public API get_volatility_index_data (free)", "quality": "stub ingest (demo)",
        "holdout_from": "2024-10-01", "cost_model": None,
        "known_biases": ["Starts 2021-03: short history, dev covers one full cycle only.",
                         "Single venue (Deribit dominates BTC options, but not all of them)."],
        "forward_source": "deribit_public_api"},
}


def load_example(name: str) -> dict:
    return yaml.safe_load((Path(__file__).parent / "examples" / name).read_text())
