"""Static dashboard: builds from any lab state, escapes agent text, switches releases atomically."""

from lab.framework import dashboard

from .conftest import submit


def test_dashboard_builds_escapes_and_publishes(lab, card, tmp_path):
    hid = submit(lab, dict(card, title="Momentum <script>alert(1)</script> & co"))
    lab.send("QUESTION", "system", "human", hid, {"question": "Is <b>this</b> escaped?"})
    page = dashboard.build(lab)
    assert "<script>" not in page and "&lt;script&gt;" in page and "&lt;b&gt;this" in page
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
