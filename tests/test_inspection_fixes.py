"""Regrese k inspekci 2026-10-25: income klíč, statistiky bez nulového notional, pevné paper poplatky, snapshot."""

import sqlite3

import pytest

from arena import stats
from arena.config import load_config
from arena.db import Database


def test_income_keeps_commission_and_realized_pnl_with_same_tran_id():
    db = Database(":memory:")
    for typ, amt in (("REALIZED_PNL", 0.08868), ("COMMISSION", -0.0386)):
        db.execute("INSERT OR IGNORE INTO income VALUES (?,?,?,?,?,?,?)",
                   ("90031550798210", "BTCUSDT", typ, amt, "USDT", 1, "550798210"))
    assert db.execute("SELECT COUNT(*), SUM(amount) FROM income").fetchone()[:] == (2, pytest.approx(0.05008))


def test_income_migration_from_old_primary_key(tmp_path):
    path = tmp_path / "a.db"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE income(tran_id TEXT PRIMARY KEY, symbol TEXT, income_type TEXT, amount REAL, asset TEXT, "
              "ts INTEGER, trade_id TEXT)")
    c.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    c.execute("INSERT INTO income VALUES ('1','BTCUSDT','REALIZED_PNL',1.0,'USDT',1,'9')")
    c.execute("INSERT INTO meta VALUES ('income_since','123')")
    c.commit()
    c.close()
    db = Database(path)
    pk = [r["name"] for r in db.execute("PRAGMA table_info(income)").fetchall() if r["pk"]]
    assert pk == ["tran_id", "income_type"]
    assert db.execute("SELECT COUNT(*) FROM income").fetchone()[0] == 1
    assert db.meta_get("income_since") is None          # znovu stáhnout, doplní chybějící COMMISSION
    db.execute("INSERT OR IGNORE INTO income VALUES ('1','BTCUSDT','COMMISSION',-0.1,'USDT',1,'9')")
    assert db.execute("SELECT COUNT(*) FROM income").fetchone()[0] == 2


def _row(**kw):
    r = dict(trade_id="l-x-1", slot_id="brk.v00.BTCUSDT", side="SHORT", entry_time=0, exit_time=1000,
             exit_reason="reconciled", notional=0.0, net_pnl=-193.4, gross_bps=0.0, net_bps=0.0, entry_fee=0.077,
             exit_fee=0.077, entry_slip_bps=10000.0, exit_slip_bps=8.5, funding=0.0, mfe_bps=None, mae_bps=None,
             tainted=1)
    r.update(kw)
    return r


def test_stats_skip_trade_without_notional():
    assert stats.from_rows([_row()]) == []
    assert len(stats.from_rows([_row(notional=193.2)])) == 1


def test_paper_fees_pinned_in_config():
    cfg = load_config()
    assert cfg.paper.taker_fee == 0.0005 and cfg.paper.maker_fee == 0.0002


class _AggRest:
    def __init__(self, ids, page=3):
        self.trades = [{"a": i, "T": 1000 + i, "p": "1"} for i in ids]
        self.page = page

    async def agg_trades(self, symbol, start_ms=None, end_ms=None, from_id=None, limit=1000):
        return [t for t in self.trades if t["a"] >= from_id][: self.page]


async def test_fetch_agg_trades_from_complete_and_gappy():
    from arena.engine import fetch_agg_trades_from
    out, ok = await fetch_agg_trades_from(_AggRest(range(10, 30)), "X", 12, to_id=20)
    assert [t["a"] for t in out] == list(range(12, 21)) and ok
    out, ok = await fetch_agg_trades_from(_AggRest([10, 11, 12, 14, 15, 16]), "X", 11, to_id=16)
    assert not ok                                        # díra 13 i na burze → neúplné
    out, ok = await fetch_agg_trades_from(_AggRest(range(10, 15)), "X", 11, to_id=20)
    assert not ok                                        # data nedosáhla to_id
    out, ok = await fetch_agg_trades_from(_AggRest(range(10, 30)), "X", 12, end_ms=1015)
    assert [t["a"] for t in out] == [12, 13, 14, 15] and ok
