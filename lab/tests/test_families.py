"""Family merges (Chair/owner): trials count over all merged names, never split, never escape the penalty."""

import pytest

from lab.framework import agents, invocations
from lab.framework.blackboard import LabError
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
