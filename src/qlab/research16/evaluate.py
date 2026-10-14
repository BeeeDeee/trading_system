"""Portfolio runs and statistics of research 16 shared by the dev and final scripts (prereg §5-9)."""

import numpy as np

from qlab.research15 import evaluate as E15, strategy as S
from qlab.research16 import engine as G, families as F
from qlab.research16.setup import FUND_DEV_START
from qlab.research15.setup import epoch_15m

Run = E15.Run
stats, trade_ci, window, by_year, run_hold, run_4h = (E15.stats, E15.trade_ci, E15.window, E15.by_year,
                                                     E15.run_hold, E15.run_4h)
N_RND = 1000


def build_assets(spot: dict, perp: dict | None = None) -> dict[str, G.Asset]:
    """Spot prices with the funding of the matching perp (prereg §3, §5); `perp=None`: the data's own funding."""
    if perp is None:
        return {s: G.Asset.build(d) for s, d in spot.items()}
    return {s: G.Asset.build(spot[s], S.align_funding(spot[s]["t"], perp[s])) for s in spot}


def start_index(assets: dict, v: F.Variant, start_date) -> int:
    """First candle of the stage; FUND's development starts at 2020-04-01 (prereg §7)."""
    t0 = next(iter(assets.values())).t[0]
    if v.family == "FUND" and start_date < FUND_DEV_START:
        start_date = FUND_DEV_START
    return max(int(epoch_15m(start_date) - t0), 0)


def run(assets: dict, cands: dict, start: int, end: int, **kw) -> Run:
    """Equal-weight sleeves (an asset without candidates holds cash). kw: delay, cost, use_funding."""
    books = {s: G.trade_book(a, cands[s], start, end, **kw) for s, a in assets.items()}
    dates, d = E15._portfolio(assets, {s: b.net for s, b in books.items()}, start, end)
    return Run(dates, d, books)


def run_random(assets: dict, v: F.Variant, ref: Run, start: int, end: int, seed: int, **kw) -> Run:
    rng = np.random.default_rng(seed)
    books = {s: G.random_book(a, ref.books[s], F.context(a, v), start, end, rng, **kw) for s, a in assets.items()}
    dates, d = E15._portfolio(assets, {s: b.net for s, b in books.items()}, start, end)
    return Run(dates, d, books)


def null(assets: dict, v: F.Variant, ref: Run, start: int, end: int, n: int = N_RND) -> dict:
    """B_RND Sharpe distribution and p = (1 + #{runs >= P}) / (1 + n)."""
    sr = stats(ref.daily)["sharpe"]
    rnd = np.array([stats(run_random(assets, v, ref, start, end, seed).daily)["sharpe"] for seed in range(n)])
    return {"median": float(np.median(rnd)), "p95": float(np.quantile(rnd, 0.95)),
            "p": float((1 + (rnd >= sr).sum()) / (1 + n))}


def breakdown(r: Run) -> dict:
    """Mean R (trades with a stop only) and counts by exit reason."""
    why = np.concatenate([b.reason for b in r.books.values()])
    rm = np.concatenate([b.ret / b.risk for b in r.books.values()])
    rm = rm[np.isfinite(rm)]
    return {"mean_R": float(rm.mean()) if len(rm) else None,
            "reasons": {name: int((why == code).sum()) for code, name in E15.REASONS.items()}}


def full_stats(r: Run) -> dict:
    tr = r.all_trades()
    return stats(r.daily, tr) | breakdown(r) | {
        "per_asset_mean_trade": {s: (float(b.ret.mean()) if len(b.ret) else None) for s, b in r.books.items()}}


def sides(r: Run) -> dict:
    out = {}
    for name, sd in (("long", 1), ("short", -1)):
        tr = np.concatenate([b.ret[b.sides == sd] for b in r.books.values()])
        out[name] = {"n": int(len(tr)), "mean": float(tr.mean()) if len(tr) else None}
    return out


def run_basket(assets_iter, members: dict[str, list[str]], v: F.Variant, start_date, weight: float):
    """Secondary basket (prereg §9, as research 15): one sleeve per symbol with a fixed `weight`, entries only in
    member months. Returns (dates, daily returns, net return of each trade)."""
    days, trades = {}, []
    for sym, a in assets_iter:
        um, inv = np.unique(a.month, return_inverse=True)
        mem = np.array([sym in members.get(f"{1970 + m // 12}-{m % 12 + 1:02d}", []) for m in um])
        s0 = max(int(epoch_15m(start_date) - a.t[0]), 0)
        b = G.trade_book(a, F.signals({sym: a}, v)[sym], s0, len(a.t), allowed=mem[inv])
        trades.append(b.ret)
        for d, x in zip(*S.daily(a.t, weight * b.net, s0, len(a.t))):
            days[d] = days.get(d, 0.0) + x
    dates = np.array(sorted(days), dtype="datetime64[D]")
    return dates, np.array([days[d] for d in dates]), (np.concatenate(trades) if trades else np.zeros(0))
