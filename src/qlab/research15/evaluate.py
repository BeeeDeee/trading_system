"""Portfolio runs and statistics of research 15 shared by the dev and final scripts (prereg §5-9)."""

from dataclasses import dataclass

import numpy as np

from qlab.research15 import strategy as S
from qlab.research15.setup import BASKET_WEIGHT, epoch_15m
from qlab.validation.stats import bootstrap_ci, sharpe

ANN = np.sqrt(365)
REASONS = {S.STOP: "stop", S.TARGET: "target", S.TIME: "time", S.END: "end"}


@dataclass
class Run:
    dates: np.ndarray
    daily: np.ndarray                  # portfolio daily net returns
    books: dict                        # symbol -> Book

    def all_trades(self) -> np.ndarray:
        return np.concatenate([b.ret for b in self.books.values()])

    def all_r(self) -> np.ndarray:
        """Net result of each trade in units of its initial risk."""
        return np.concatenate([b.ret / b.risk for b in self.books.values()])


def build_assets(spot: dict, perp: dict) -> dict[str, S.Asset]:
    """Spot prices with the funding of the matching perp (prereg §3, §5)."""
    return {s: S.Asset.build(spot[s], S.align_funding(spot[s]["t"], perp[s])) for s in spot}


def _portfolio(assets: dict[str, S.Asset], nets: dict[str, np.ndarray], start: int, end: int):
    w = 1.0 / len(assets)
    t = next(iter(assets.values())).t
    return S.daily(t, sum(w * x for x in nets.values()), start, end)


def run(assets: dict[str, S.Asset], cfg: S.Config, start: int, end: int, **kw) -> Run:
    """Equal-weight sleeves. kw: cost, delay, use_funding, confirm, use_regime (see `strategy.trade_book`)."""
    books = {sym: S.trade_book(a, cfg, start, end, **kw) for sym, a in assets.items()}
    dates, d = _portfolio(assets, {s: b.net for s, b in books.items()}, start, end)
    return Run(dates, d, books)


def run_random(assets: dict[str, S.Asset], cfg: S.Config, ref: Run, start: int, end: int, seed: int, **kw) -> Run:
    rng = np.random.default_rng(seed)
    books = {sym: S.random_book(a, cfg, ref.books[sym], start, end, rng, **kw) for sym, a in assets.items()}
    dates, d = _portfolio(assets, {s: b.net for s, b in books.items()}, start, end)
    return Run(dates, d, books)


def run_4h(assets: dict[str, S.Asset], side: str, start: int, end: int, **kw) -> tuple[np.ndarray, np.ndarray]:
    nets = {}
    for sym, a in assets.items():
        pos = a.reg15.astype(float) if side == "LS" else np.maximum(a.reg15, 0).astype(float)
        pos[:start] = 0.0
        pos[end - 1:] = 0.0
        nets[sym] = S.position_returns(a, pos, **kw)
    return _portfolio(assets, nets, start, end)


def run_basket(assets, members: dict[str, list[str]], cfg: S.Config, start_date, weight: float = BASKET_WEIGHT,
               **kw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Secondary basket (prereg §9): `assets` yields (symbol, Asset) lazily; a symbol may open trades only in
    months where it is in `members` ('YYYY-MM' -> symbols); every sleeve has a fixed `weight` of capital.
    Returns (dates, daily returns, net return of each trade)."""
    days, trades = {}, []
    for sym, a in assets:
        um, inv = np.unique(a.month, return_inverse=True)
        mem = np.array([sym in members.get(f"{1970 + m // 12}-{m % 12 + 1:02d}", []) for m in um])
        s0 = max(int(epoch_15m(start_date) - a.t[0]), 0)
        b = S.trade_book(a, cfg, s0, len(a.t), allowed=mem[inv], **kw)
        trades.append(b.ret)
        for d, v in zip(*S.daily(a.t, weight * b.net, s0, len(a.t))):
            days[d] = days.get(d, 0.0) + v
    dates = np.array(sorted(days), dtype="datetime64[D]")
    return dates, np.array([days[d] for d in dates]), (np.concatenate(trades) if trades else np.zeros(0))


def run_hold(assets: dict[str, S.Asset], start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """50/50 rebalanced daily, no costs: daily return = mean of the assets' daily returns."""
    rets = []
    for a in assets.values():
        r = np.zeros(len(a.c))
        r[1:] = a.c[1:] / a.c[:-1] - 1.0
        dates, lr = S.daily(a.t, np.log1p(r), start, end)
        rets.append(np.expm1(lr))
    return dates, np.mean(rets, axis=0)


def stats(daily: np.ndarray, trades: np.ndarray | None = None) -> dict:
    eq = np.cumprod(1.0 + daily)
    out = {"sharpe": sharpe(daily) * ANN, "cagr": float(eq[-1] ** (365 / len(daily)) - 1),
           "max_dd": float(1 - (eq / np.maximum.accumulate(eq)).min()), "days": int(len(daily))}
    if trades is not None and len(trades):
        out |= {"n_trades": int(len(trades)), "mean_trade": float(trades.mean()),
                "median_trade": float(np.median(trades)), "hit_rate": float((trades > 0).mean())}
    return out


def trade_breakdown(r: Run) -> dict:
    """Mean R multiple and counts by exit reason (report only)."""
    why = np.concatenate([b.reason for b in r.books.values()])
    rm = r.all_r()
    return {"mean_R": float(rm.mean()) if len(rm) else None,
            "reasons": {name: int((why == code).sum()) for code, name in REASONS.items()}}


def trade_ci(trades: np.ndarray) -> tuple[float, float, float]:
    """Mean net trade return and 90 % iid bootstrap interval (2000 samples, seed 0)."""
    return bootstrap_ci(trades, np.mean, n_boot=2000, mean_block=1.0, alpha=0.10, seed=0)


def window(dates: np.ndarray, daily: np.ndarray, lo, hi) -> np.ndarray:
    m = (dates >= np.datetime64(lo)) & (dates <= np.datetime64(hi))
    return daily[m]


def by_year(dates: np.ndarray, daily: np.ndarray) -> dict:
    yrs = dates.astype("datetime64[Y]").astype(int) + 1970
    return {int(y): {"sharpe": sharpe(daily[yrs == y]) * ANN,
                     "ret": float(np.prod(1 + daily[yrs == y]) - 1)} for y in np.unique(yrs)}
