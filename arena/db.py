"""SQLite (WAL), jeden zapisující proces (D6). Tenká vrstva nad `sqlite3`, žádné ORM."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from arena.util import wall_ms

SCHEMA = Path(__file__).with_name("schema.sql")


@dataclass(frozen=True)
class Bar:
    symbol: str
    open_time: int
    o: float
    h: float
    l: float  # noqa: E741
    c: float
    v: float
    qv: float
    n: int
    tbv: float
    tbqv: float

    @classmethod
    def from_rest(cls, symbol: str, k: Sequence[Any]) -> Bar:
        # [openTime, o, h, l, c, v, closeTime, qv, n, tbv, tbqv, ignore]
        return cls(symbol, int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]),
                   float(k[7]), int(k[8]), float(k[9]), float(k[10]))

    @classmethod
    def from_ws(cls, k: dict[str, Any]) -> Bar:
        # kline event `k`: t, o, h, l, c, v, n, q, V, Q
        return cls(k["s"], int(k["t"]), float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]), float(k["v"]),
                   float(k["q"]), int(k["n"]), float(k["V"]), float(k["Q"]))

    def row(self) -> tuple[Any, ...]:
        return (self.symbol, self.open_time, self.o, self.h, self.l, self.c, self.v, self.qv, self.n, self.tbv,
                self.tbqv)


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA.read_text(encoding="utf-8"))

    def close(self) -> None:
        self.conn.close()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        self.conn.execute("BEGIN")
        try:
            self.conn.executemany(sql, rows)
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    # --- bary ---------------------------------------------------------------

    def upsert_bars(self, bars: Iterable[Bar]) -> None:
        self.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?)", (b.row() for b in bars))

    def last_bar_time(self, symbol: str) -> int | None:
        r = self.execute("SELECT MAX(open_time) FROM bars WHERE symbol=?", (symbol,)).fetchone()
        return r[0]

    def bars(self, symbol: str, start: int | None = None, end: int | None = None,
             limit: int | None = None) -> list[Bar]:
        """Bary v [start, end] seřazené vzestupně. S `limit` posledních `limit` barů."""
        q = "SELECT * FROM bars WHERE symbol=?"
        p: list[Any] = [symbol]
        if start is not None:
            q += " AND open_time>=?"
            p.append(start)
        if end is not None:
            q += " AND open_time<=?"
            p.append(end)
        if limit is not None:
            q = f"SELECT * FROM ({q} ORDER BY open_time DESC LIMIT ?) ORDER BY open_time"
            p.append(limit)
        else:
            q += " ORDER BY open_time"
        return [Bar(*tuple(r)) for r in self.execute(q, p).fetchall()]

    # --- kniha --------------------------------------------------------------

    def insert_book_bands(self, rows: Iterable[tuple[str, int, float, float, float, float, str]]) -> None:
        self.executemany("INSERT OR REPLACE INTO book_bands VALUES (?,?,?,?,?,?,?)", rows)

    def book_bands(self, symbol: str, start: int, end: int, source: str = "live") -> list[sqlite3.Row]:
        return self.execute("SELECT * FROM book_bands WHERE symbol=? AND ts>=? AND ts<? AND source=? ORDER BY ts, pct",
                            (symbol, start, end, source)).fetchall()

    def upsert_bar_book(self, rows: Iterable[tuple[str, int, float, float | None, int, int, str]],
                        replace: bool = True) -> None:
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        self.executemany(f"{verb} INTO bar_book VALUES (?,?,?,?,?,?,?)", rows)

    # --- události a konfigurace --------------------------------------------

    def event(self, level: str, component: str, msg: str, data: Any = None, ts: int | None = None) -> None:
        self.execute("INSERT INTO events(ts, level, component, msg, data_json) VALUES (?,?,?,?,?)",
                     (ts or wall_ms(), level, component, msg, json.dumps(data, default=str) if data is not None else None))

    def record_config(self, config_hash: str, text: str) -> None:
        self.execute("INSERT OR IGNORE INTO config_versions VALUES (?,?,?)", (config_hash, text, wall_ms()))

    def meta_get(self, key: str) -> str | None:
        r = self.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def meta_set(self, key: str, value: str) -> None:
        self.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
