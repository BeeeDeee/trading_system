"""Paper trading (G5) end to end on the positive control: forward days appended once, Sentinel, G5."""

from datetime import date

import numpy as np
import pytest

from lab.framework import canaries as C
from lab.framework import paper
from lab.framework.evaluator import QlabEvaluator
from lab.framework.states import S
from lab.framework.tick import tick


@pytest.fixture
def lab_in_paper(tmp_path, monkeypatch):
    lab = C._lab(tmp_path)
    ev = QlabEvaluator(C.QUICK, require_canaries=False)
    hid = C.submit_and_build(lab, C._card("positive"), C.CANARIES / "positive" / "strategy.py", ev)
    C.clean_review(lab, hid, ev)
    assert lab.hypothesis(hid)["status"] == S.PAPER
    # the synthetic market has no forward source: treat 2025 as "forward" (admitted at the end of 2024)
    real = paper.extended_context
    def ext(lab_, ev_, hid_):
        ctx, _ = real(lab_, ev_, hid_)
        return ctx, int(np.searchsorted(ctx.view.dates, np.datetime64("2025-01-01")))
    monkeypatch.setattr(paper, "extended_context", ext)
    monkeypatch.setattr(paper, "admitted_on", lambda lab_, hid_: date(2024, 12, 31))
    monkeypatch.setattr(QlabEvaluator, "paper_forward", lambda self, lab_: [])
    return lab, ev, hid


def test_paper_days_are_appended_once_and_g5_decides(lab_in_paper):
    lab, ev, hid = lab_in_paper
    log = tick(lab, ev)
    rec = paper.record(lab, hid)
    assert rec["first"].startswith("2025-01") and rec["days"] > 200
    assert paper.run(lab, ev, hid) == 0                      # nothing new: stored days are not recomputed
    g5 = lab.con.execute("SELECT passed FROM gate_results WHERE hypothesis_id = ? AND gate = 'G5'", (hid,)).fetchone()
    assert g5 is not None, log
    assert lab.hypothesis(hid)["status"] == (S.LIVE_CANDIDATE if g5[0] else S.REJECTED)


def test_sentinel_retires_on_drawdown(lab_in_paper, monkeypatch):
    lab, ev, hid = lab_in_paper
    monkeypatch.setattr(paper, "dev_max_dd", lambda lab_, hid_: 1e-4)   # any loss is "1.5 x the dev drawdown"
    tick(lab, ev)
    assert lab.hypothesis(hid)["status"] == S.RETIRED
    assert any("retired" in m["payload_json"] for m in lab.inbox("human"))


def test_forward_store_is_append_only_and_extends_a_view(tmp_path):
    from datetime import datetime, timezone

    from lab.framework import forward
    from lab.framework.data import DataView

    def day_ms(d):
        return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)
    calls = []

    def get(url):
        calls.append(url)
        start = int(url.split("startTime=")[1].split("&")[0])
        days = [date(2026, 9, 1 + i) for i in range(10) if day_ms(date(2026, 9, 1 + i)) >= start]
        return [[day_ms(d), "100", "0", "0", str(100 + d.day), "0", 0, "1000000"] for d in days]
    home = tmp_path
    n = forward.update(home, "binance_spot_1d", "BTCUSDT", after=date(2026, 8, 31), today=date(2026, 9, 6), get=get)
    assert n == 5                                            # 1..5 September: closed days only
    assert forward.update(home, "binance_spot_1d", "BTCUSDT", after=date(2026, 8, 31), today=date(2026, 9, 6),
                          get=get) == 0
    assert forward.update(home, "binance_spot_1d", "BTCUSDT", after=date(2026, 8, 31), today=date(2026, 9, 8),
                          get=get) == 2
    d = np.arange(np.datetime64("2026-08-25"), np.datetime64("2026-09-01"))
    T = len(d)
    z = np.zeros((T, 1))
    b = np.ones((T, 1), bool)
    v = DataView(d, ("BTCUSDT",), ("crypto_spot",), z, z, b, b, ~b, np.full((T, 1), 100.0), np.full((T, 1), 1e6),
                 cash_ret=np.full(T, 1e-4))
    x = forward.extend(v, home, "binance_spot_1d")
    assert str(x.dates[-1]) == "2026-09-07" and len(x.dates) == T + 7
    assert np.isclose(x.ret_oc[T, 0], 101 / 100 - 1) and np.isclose(x.ret_co[T + 1, 0], 100 / 101 - 1)
    assert x.cash_ret[-1] == 1e-4


def test_steward_gets_the_paper_record_and_is_scheduled_weekly(lab_in_paper):
    import json

    from lab.framework import invocations
    from lab.orchestrator import pending_tasks, steward_due
    lab, ev, hid = lab_in_paper
    tick(lab, ev)
    if lab.hypothesis(hid)["status"] == S.PAPER:
        assert steward_due(lab) and any(t.agent == "steward" for t in pending_tasks(lab))
    inv = invocations.start(lab, "steward", None, task="weekly paper report")
    summary = json.loads((inv.workspace / "paper.json").read_text())
    assert summary[0]["id"] == hid and summary[0]["days"] > 200
