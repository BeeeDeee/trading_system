"""Portfolio runs and statistics of research 23 shared by the dev and final scripts (prereg §5-9)."""

from dataclasses import dataclass

import numpy as np

from qlab.research15 import evaluate as E15
from qlab.research23 import engine as G, setups as SU, vwap as VW
from qlab.research23.data import PER_DAY, epoch_5m

N_RND = 500
stats, trade_ci, window, by_year = E15.stats, E15.trade_ci, E15.window, E15.by_year
REASONS = {G.STOP: "stop", G.TARGET: "target", G.END: "end"}


def prepare(d: dict) -> dict:
    """Raw arrays + session position, VWAP, sigma, ATR and the signal Market."""
    a = dict(d)
    a["pos"] = a["t"] % PER_DAY
    a["V"], a["S"] = VW.session_vwap(a["t"], VW.bar_price(a["h"], a["l"], a["c"], a["v"], a["qv"]), a["v"])
    a["atr"] = VW.atr(a["h"], a["l"], a["c"])
    a["mk"] = SU.Market.build(a["t"], a["h"], a["l"], a["c"], a["real"], a["V"], a["S"], a["atr"])
    return a


def start_index(assets: dict, d) -> int:
    t0 = next(iter(assets.values()))["t"][0]
    return max(int(epoch_5m(d) - t0), 0)


@dataclass
class Run:
    dates: np.ndarray
    daily: np.ndarray                  # 50/50 portfolio daily net returns (all days of the stage)
    books: dict                        # symbol -> engine.Book

    def trades(self) -> np.ndarray:
        return np.concatenate([b.ret for b in self.books.values()])


def _daily(assets: dict, books: dict, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    t = next(iter(assets.values()))["t"]
    d0, d1 = t[start] // PER_DAY, (t[end - 1] // PER_DAY) + 1
    out = np.zeros(d1 - d0)
    w = 1.0 / len(assets)
    for sym, b in books.items():
        np.add.at(out, assets[sym]["t"][b.e] // PER_DAY - d0, w * b.ret)
    dates = (np.arange(d0, d1) * 86400).astype("datetime64[s]").astype("datetime64[D]")
    return dates, out


def signals(assets: dict, v: SU.Variant) -> dict:
    return {s: SU.signals(a["mk"], v) for s, a in assets.items()}


def run(assets: dict, v: SU.Variant, sigs: dict, start: int, end: int, **kw) -> Run:
    """kw: delay, cost."""
    books = {s: G.trade_book(a, sigs[s], v.target, start, end, **kw) for s, a in assets.items()}
    return Run(*_daily(assets, books, start, end), books)


def null(assets: dict, r: Run, start: int, end: int, n: int = N_RND, cost: float = G.COST) -> dict:
    """B_RND on the pooled mean net trade: p = (1 + #{runs >= P}) / (1 + n)."""
    sums, cnts = np.zeros(n), np.zeros(n)
    for i, (s, a) in enumerate(assets.items()):
        x, k = G.null_sums(a, r.books[s], start, end, n, seed=1000 * i, cost=cost)
        sums += x
        cnts += k
    tr = r.trades()
    if not len(tr):
        return {"p": 1.0, "median": None, "p95": None, "placed": 0.0}
    mean = tr.mean()
    rnd = sums / np.maximum(cnts, 1)
    return {"p": float((1 + (rnd >= mean).sum()) / (1 + n)), "median": float(np.median(rnd)),
            "p95": float(np.quantile(rnd, 0.95)), "placed": float(cnts.mean() / len(tr))}


def full_stats(r: Run) -> dict:
    tr = r.trades()
    out = stats(r.daily, tr)
    why = np.concatenate([b.why for b in r.books.values()])
    rm = np.concatenate([b.ret / b.risk for b in r.books.values()])
    risk = np.concatenate([b.risk for b in r.books.values()])
    hold = np.concatenate([b.x - b.e for b in r.books.values()])
    out |= {"mean_R": float(rm.mean()) if len(rm) else None,
            "median_risk": float(np.median(risk)) if len(risk) else None,
            "median_hold_candles": float(np.median(hold)) if len(hold) else None,
            "trades_per_day": float(len(tr) / len(r.daily)),
            "reasons": {name: int((why == c).sum()) for c, name in REASONS.items()},
            "per_asset_mean_trade": {s: (float(b.ret.mean()) if len(b) else None) for s, b in r.books.items()},
            "per_asset_n": {s: len(b) for s, b in r.books.items()}}
    return out


def sides(r: Run) -> dict:
    out = {}
    for name, sd in (("long", 1), ("short", -1)):
        tr = np.concatenate([b.ret[b.side == sd] for b in r.books.values()])
        out[name] = {"n": int(len(tr)), "mean": float(tr.mean()) if len(tr) else None}
    return out


def gross_mean(r: Run, cost: float = G.COST) -> float | None:
    """Mean trade before costs (funding kept)."""
    tr = r.trades()
    return float(tr.mean() + 2 * cost) if len(tr) else None


def by_hour(assets: dict, r: Run) -> dict:
    out = {}
    for s, b in r.books.items():
        hr = (assets[s]["t"][b.e] % PER_DAY) // 12
        for h in range(24):
            m = hr == h
            x = out.setdefault(h, [0, 0.0])
            x[0] += int(m.sum())
            x[1] += float(b.ret[m].sum())
    return {h: {"n": n, "mean": (sm / n if n else None)} for h, (n, sm) in out.items()}


def by_vol_tercile(assets: dict, r: Run) -> dict:
    """Mean trade by tercile of the ATR / price at entry (terciles of the trades' own values)."""
    v = np.concatenate([assets[s]["atr"][b.e] / assets[s]["c"][b.e] for s, b in r.books.items()])
    tr = r.trades()
    if len(tr) < 3:
        return {}
    q = np.quantile(v, [1 / 3, 2 / 3])
    g = np.digitize(v, q)
    return {name: {"n": int((g == i).sum()), "mean": float(tr[g == i].mean())}
            for i, name in enumerate(("low", "mid", "high"))}
