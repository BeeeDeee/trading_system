"""Exec probe (rodina xp): determinismus, jen ve stavu flat, všechny mechanismy dávají platné úrovně, mimo statistiky,
jen na testnetu."""

import shutil
from decimal import Decimal

import pytest

from arena.config import DEFAULT_CONFIG_DIR, ConfigError, load_config
from arena.execution.sizing import plan_exits
from arena.strategies.base import BarContext
from arena.strategies.registry import build_strategies
from arena.strategies.xp import MECHANISMS
from tests.test_filters import _sym

BAR0 = 1_792_000_000_000 - 1_792_000_000_000 % 300_000


def _ctx(i, status="flat", slot="xp.v00.BTCUSDT"):
    return BarContext("BTCUSDT", BAR0 + i * 300_000, {"c": 80_000.0}, slot, status)


def _probe():
    return build_strategies(load_config(DEFAULT_CONFIG_DIR))["xp.v00"]


def test_deterministic_and_only_when_flat():
    s = _probe()
    a = [s.on_bar(_ctx(i)) for i in range(400)]
    assert a == [s.on_bar(_ctx(i)) for i in range(400)]
    assert 0.08 < sum(x is not None for x in a) / 400 < 0.22          # ≈ entry_prob 0.15
    assert all(s.on_bar(_ctx(i, "in_position")) is None for i in range(400))
    assert all(s.on_bar(_ctx(i, "cooldown")) is None for i in range(400))


def test_all_mechanisms_and_sides_appear_and_plan_is_valid():
    s, f = _probe(), _sym()
    seen, sides = set(), set()
    for i in range(3000):
        it = s.on_bar(_ctx(i))
        if it is None:
            continue
        seen.add(it.reason.split(":")[0]); sides.add(it.side)
        plan = plan_exits(it, f, 80_000.0, Decimal("0.003"))
        assert plan.max_hold_bars >= 2
        if it.side == "LONG":
            assert plan.hard_stop_px < Decimal(80_000)
            assert plan.tp_px is None or plan.tp_px > Decimal(80_000)
        else:
            assert plan.hard_stop_px > Decimal(80_000)
            assert plan.tp_px is None or plan.tp_px < Decimal(80_000)
        assert (plan.callback_rate is None) == (it.callback_pct is None)
        if it.reason.startswith("tp"):
            assert plan.tp_px is not None
        if it.reason.startswith("trail_act"):
            assert plan.activate_px is not None
    assert seen == set(MECHANISMS) and sides == {"LONG", "SHORT"}


def test_probe_rejected_on_mainnet(tmp_path):
    d = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, d)
    p = d / "arena.yaml"
    p.write_text(p.read_text().replace("env: testnet", "env: mainnet", 1))
    with pytest.raises(ConfigError, match="xp"):
        load_config(d)


def test_probe_excluded_from_stats(tmp_path):
    from arena.config import load_config as lc
    from arena.db import Database
    from arena.snapshot import SnapshotWriter

    cfg = lc(DEFAULT_CONFIG_DIR)
    db = Database(tmp_path / "t.db")
    cols = ("trade_id,slot_id,mode,side,qty,notional,entry_px,entry_time,exit_px,exit_time,exit_reason,gross_pnl,net_pnl,"
            "gross_bps,net_bps,entry_slip_bps,exit_slip_bps,bars_held,tainted,cost_model_version,config_hash,entry_fee,"
            "exit_fee,funding")
    for tid, slot in (("a", "xp.v00.BTCUSDT"), ("b", "brk.v00.BTCUSDT")):
        db.execute(f"INSERT INTO trades({cols}) VALUES ({','.join('?' * len(cols.split(',')))})",
                   (tid, slot, "paper", "LONG", 1, 1000, 1, 1, 1, 2, "time", 0, 0, 0, 5, 0, 0, 1, 0,
                    cfg.paper.cost_model_version, "h", 0, 0, 0))
    w = SnapshotWriter(cfg, db, tmp_path, lambda: {}, lambda: [])
    assert [t.variant for t in w._trades()] == ["brk.v00"]


def test_today_pnl_ignores_tainted_and_probe(tmp_path):
    from arena.config import load_config as lc
    from arena.db import Database
    from arena.snapshot import SnapshotWriter
    from arena.util import wall_ms

    cfg = lc(DEFAULT_CONFIG_DIR)
    db = Database(tmp_path / "t.db")
    cols = ("trade_id,slot_id,mode,side,qty,notional,entry_px,entry_time,exit_px,exit_time,exit_reason,gross_pnl,net_pnl,"
            "gross_bps,net_bps,entry_slip_bps,exit_slip_bps,bars_held,tainted,cost_model_version,config_hash,entry_fee,"
            "exit_fee,funding")
    now = wall_ms()
    for tid, slot, taint, pnl in (("a", "brk.v00.BTCUSDT", 1, -193.0), ("b", "xp.v00.BTCUSDT", 0, -5.0),
                                  ("c", "brk.v00.ETHUSDT", 0, 2.5)):
        db.execute(f"INSERT INTO trades({cols}) VALUES ({','.join('?' * len(cols.split(',')))})",
                   (tid, slot, "live", "LONG", 1, 1000, 1, now, 1, now, "time", 0, pnl, 0, 5, 0, 0, 1, taint,
                    cfg.paper.cost_model_version, "h", 0, 0, 0))
    w = SnapshotWriter(cfg, db, tmp_path, lambda: {}, lambda: [])
    w.write_snapshot()
    import json
    today = json.loads((tmp_path / "data" / "snapshot.json").read_text())["overview"]["today"]
    assert today["live"]["n"] == 1 and today["live"]["pnl"] == 2.5
    assert today["probe"]["n"] == 1 and today["probe"]["pnl"] == -5.0           # sonda zvlášť (hýbe účtem)
    assert today["live"]["net_bps"] == 5 and today["live"]["slots"] == 1 and today["live"]["notional"] == 1000


def test_equity_curves_per_range(tmp_path):
    from arena.config import load_config as lc
    from arena.db import Database
    from arena.snapshot import SnapshotWriter
    from arena.util import wall_ms
    import json

    cfg = lc(DEFAULT_CONFIG_DIR)
    db = Database(tmp_path / "t.db")
    now = wall_ms()
    for i in range(3 * 24 * 60):                                   # 3 dny po minutě
        db.execute("INSERT INTO equity VALUES (?,?,?,?)", (now - i * 60_000, 5000.0 - i * 0.01, 0.0, 0.0))
    SnapshotWriter(cfg, db, tmp_path, lambda: {}, lambda: []).write_snapshot()
    cur = json.loads((tmp_path / "data" / "snapshot.json").read_text())["equity_curves"]
    assert set(cur) == {"1h", "6h", "24h", "7d", "30d", "all"}
    assert all(len(c["t"]) <= 241 for c in cur.values())
    assert cur["1h"]["t"][0] >= now - 3_600_000 and 60 <= len(cur["1h"]["t"]) <= 61
    assert cur["all"]["t"][0] == now - (3 * 24 * 60 - 1) * 60_000 and cur["all"]["t"][-1] == now
    assert cur["24h"]["equity"][-1] == 5000.0
