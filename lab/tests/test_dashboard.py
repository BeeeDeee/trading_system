"""Static dashboard: builds from any lab state, escapes agent text, switches releases atomically."""

from lab.framework import dashboard

from .conftest import submit


def test_dashboard_builds_escapes_and_publishes(lab, card, tmp_path):
    hid = submit(lab, dict(card, title="Momentum <script>alert(1)</script> & co"))
    lab.send("QUESTION", "system", "human", hid, {"question": "Is <b>this</b> escaped?"})
    page = dashboard.build(lab)
    assert "<script>alert(1)" not in page and "&lt;script&gt;alert(1)" in page and "&lt;b&gt;this" in page   # agent text is escaped
    assert page.count("<script>") == 1                                                                       # only our own tab/picker script
    assert hid in page and "Coverage" in page and "Knowledge base" in page
    web = tmp_path / "web"
    first = dashboard.publish(lab, web)
    second = dashboard.publish(lab, web)
    assert (web / "current").resolve() == second.parent.resolve() and first.exists()
    for _ in range(dashboard.KEEP_RELEASES + 2):
        dashboard.publish(lab, web)
    assert len(list((web / "releases").iterdir())) <= dashboard.KEEP_RELEASES


def test_minimal_markdown():
    out = dashboard.md("# T\n\nA *b* `c` **d**\n\n- x\n- <y>")
    assert "<h3>T</h3>" in out and "<i>b</i>" in out and "<code>c</code>" in out and "&lt;y&gt;" in out


def test_pipeline_tab_covers_every_state(lab, card, evaluator, tmp_path):
    """The pipeline view builds for hypotheses in every kind of state and shows where each one is stuck."""
    import re

    from lab import stubs
    from lab.framework import invocations, pipeline_view
    from lab.framework.invocations import stage
    from lab.framework.states import S
    from lab.framework.tick import tick

    from .conftest import submit

    done = submit(lab, card)                                       # runs the whole stub pipeline
    inv = invocations.start(lab, "builder", done)
    stubs.builder(lab, inv)
    invocations.finish(lab, inv.id)
    tick(lab, evaluator)
    waiting = submit(lab, dict(card, title="Another idea waiting for the builder", family="other-family",
                               signal={**card["signal"], "description": "A different rule text, long enough to pass the duplicate check."}))
    rejected = submit(lab, dict(card, title="An idea that will be rejected", family="third-family",
                                signal={**card["signal"], "description": "Yet another rule text for the third hypothesis here."}))
    lab.transition(rejected, S.REJECTED, "human", "test", reason_code="g1_sharpe_excess", stage="G1")
    page = dashboard.build(lab)
    assert 'id="t-pipeline"' in page.replace("'", '"') and "Stuck or waiting" in page
    for hid in (done, waiting, rejected):
        assert f"id='h-{hid}'" in page
    assert "Builder has not implemented it yet" in page             # the waiting one says who must act
    assert "g1_sharpe_excess" in page and "died at" in page
    assert re.search(r"<svg viewBox=\"0 0 1120 \d+\" role=\"img\" aria-label=\"pipeline flow\"", page)
    st = pipeline_view.story(lab, lab.hypothesis(rejected))
    assert st["stages"]["g1"] == "failed" and st["stages"]["g2"] == "pending" and st["stages"]["data"] == "skipped"
    st = pipeline_view.story(lab, lab.hypothesis(waiting))
    assert st["stages"]["idea"] == "done" and st["stages"]["build"] == "active"
