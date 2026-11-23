"""Family merges (Chair/owner): trials count over all merged names, never split, never escape the penalty."""

import json

import pytest

from lab.framework import agents, invocations
from lab.framework.blackboard import LabError, card_families
from lab.framework.invocations import stage
from lab.framework.tick import tick

from .conftest import submit


def two_families(lab, card):
    a = submit(lab, card)
    b = submit(lab, dict(card, title="Another flow idea on the same pair", family="month-end-flows",
                         signal={**card["signal"], "description": "A completely different rule text for the "
                                 "second card so the duplicate check does not fire."}))
    return a, b


def test_merge_counts_trials_together_and_is_permanent(lab, card):
    a, b = two_families(lab, card)
    lab.con.execute("INSERT INTO trials (family, hypothesis_id, version, run_id, config_sha256, sharpe, n_obs, ts) "
                    "VALUES ('month-end-flows', ?, 1, 1, 'x', 0.1, 100, 'now')", (b,))
    assert lab.family_trials("sector-momentum") == 0
    lab.merge_family("month-end-flows", "sector-momentum", "chair", "same counterparty and same effect, really")
    assert lab.family_trials("sector-momentum") == 1 and lab.family_trials("month-end-flows") == 1
    assert lab.canonical_family("month-end-flows") == "sector-momentum"
    with pytest.raises(LabError, match="already merged"):
        lab.merge_family("month-end-flows", "sector-momentum", "chair", "again, which is not allowed at all")
    with pytest.raises(LabError, match="itself"):
        lab.merge_family("sector-momentum", "month-end-flows", "chair", "a cycle would split the family again")
    with pytest.raises(LabError, match="cannot merge"):
        lab.merge_family("sector-momentum", "month-end-flows", "scout", "agents other than the chair may not")


def test_chair_merges_through_a_message(lab, card):
    two_families(lab, card)
    inv = invocations.start(lab, "chair", None, task="backlog")
    fams = {f["family"] for f in __import__("json").loads((inv.workspace / "families.json").read_text())}
    assert fams == {"sector-momentum", "month-end-flows"}
    stage(inv.workspace, "FAMILY_MERGE", "system", None, {"from_family": "month-end-flows",
          "into_family": "sector-momentum", "reason": "same mechanism: both trade the same rebalancing flow"})
    assert invocations.finish(lab, inv.id) == "applied"
    tick(lab, None)
    assert lab.family_members("sector-momentum") == ["month-end-flows", "sector-momentum"]
    assert [f["family"] for f in agents.families(lab)] == ["sector-momentum"]


def test_prior_study_trials_count_for_the_family_and_only_the_owner_adds_them(lab, card):
    a, b = two_families(lab, card)
    assert lab.family_trials("sector-momentum") == 0
    n = lab.add_prior_trials("sector-momentum", 5, "research X grid", "human", "five configurations of the same rule")
    assert n == 5 and lab.family_trial_sharpes("sector-momentum") == []         # no Sharpe: variance is untouched
    lab.merge_family("month-end-flows", "sector-momentum", "chair", "same counterparty and same effect, really")
    assert lab.family_trials("month-end-flows") == 5                            # counted through the merge
    lab.add_prior_trials("month-end-flows", 2, "research Y", "human", "two more configurations of the rule")
    assert lab.family_trials("sector-momentum") == 7                            # stored under the canonical family
    for actor in ("chair", "scout", "system"):
        with pytest.raises(LabError, match="owner only"):
            lab.add_prior_trials("sector-momentum", 1, "research Z", actor, "agents cannot inflate the counter")
    with pytest.raises(LabError, match="unknown family"):
        lab.add_prior_trials("no-such-family", 1, "research Z grid", "human", "a family that does not exist here")
    with pytest.raises(LabError, match="between 1 and 5000"):
        lab.add_prior_trials("sector-momentum", 0, "research Z grid", "human", "zero is not a count of trials")
    with pytest.raises(LabError, match="say where"):
        lab.add_prior_trials("sector-momentum", 1, "x", "human", "short")


def test_related_families_add_their_trials_and_holdout_attempts_without_receiving_any(lab, card):
    a, b = two_families(lab, card)
    lab.add_prior_trials("sector-momentum", 4, "research X grid", "human", "four configurations of the same rule")
    lab.add_prior_trials("month-end-flows", 3, "research Y grid", "human", "three configurations of the rule")
    assert lab.family_trials(["sector-momentum", "month-end-flows"]) == 7
    assert lab.family_trials(["sector-momentum", "sector-momentum"]) == 4         # each family once
    combo = submit(lab, dict(card, title="Combination of both flow ideas", family="flow-combination",
                             related_families=["sector-momentum", "month-end-flows"],
                             signal={**card["signal"], "description": "Equal-risk blend of the two earlier rules, "
                                     "weights fixed before any run, nothing else changed."}))
    assert lab.hypothesis(combo)["status"] != "IDEA"
    assert lab.family_trials(card_families(lab.card(combo))) == 7
    assert lab.family_trials("flow-combination") == 0                            # its own trials go to its own family
    view = agents.family(lab, combo)
    assert view["related_families"] == ["sector-momentum", "month-end-flows"] and view["trials_recorded"] == 7
    assert {h["id"] for h in view["hypotheses"]} == {a, b, combo}


def test_related_families_must_exist_and_differ_from_the_own_family(lab, card):
    submit(lab, card)
    bad = submit(lab, dict(card, title="Variant naming an unknown parent", related_families=["no-such-family"],
                           signal={**card["signal"], "description": "Another rule text, long enough to differ."}))
    assert lab.hypothesis(bad)["status"] == "IDEA"
    same = submit(lab, dict(card, title="Variant naming its own family", related_families=[card["family"]],
                            signal={**card["signal"], "description": "Yet another rule text, also different."}))
    assert lab.hypothesis(same)["status"] == "IDEA"
    questions = [json.loads(r["payload_json"])["question"] for r in lab.con.execute(
        "SELECT payload_json FROM messages WHERE type = 'QUESTION' ORDER BY id")]
    assert any("is not a family of any hypothesis" in q for q in questions)
    assert any("is the card's own family" in q for q in questions)


def test_owner_queue_entry_fixes_the_families_its_card_must_carry(lab, card, tmp_path, monkeypatch):
    from lab.framework import briefs
    q = tmp_path / "queue.yaml"
    q.write_text("- id: V-1\n  title: variant\n  text: x\n  families: [sector-momentum]\n"
                 "- id: V-2\n  title: combo\n  text: x\n  families: [combo-flows, sector-momentum, month-end-flows]\n")
    monkeypatch.setattr(briefs, "OWNER_QUEUE", q)
    two_families(lab, card)
    rules = {"a": "Buy the laggard after the flow.", "b": "Short the leader into the month end, cover after.",
             "c": "Blend: half weight each rule, fixed.", "d": "Equal-risk mix of both books, rebalanced weekly."}
    text = lambda s: {**card["signal"], "description": rules[s]}  # noqa: E731
    escaped = submit(lab, dict(card, title="Variant under a fresh name", family="fresh-name", signal=text("a"),
                               notes="owner_seed: V-1"))
    assert lab.hypothesis(escaped)["status"] == "IDEA"
    kept = submit(lab, dict(card, title="Variant in the parent family", signal=text("b"), notes="owner_seed: V-1"))
    assert lab.hypothesis(kept)["status"] != "IDEA"
    half = submit(lab, dict(card, title="Combination missing a parent", family="combo-flows", signal=text("c"),
                            related_families=["sector-momentum"], notes="owner_seed: V-2"))
    assert lab.hypothesis(half)["status"] == "IDEA"
    full = submit(lab, dict(card, title="Combination with both parents", family="combo-flows", signal=text("d"),
                            related_families=["sector-momentum", "month-end-flows"], notes="owner_seed: V-2"))
    assert lab.hypothesis(full)["status"] != "IDEA"
