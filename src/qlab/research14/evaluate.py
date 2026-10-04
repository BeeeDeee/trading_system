"""Portfolio runs and statistics of research 14 shared by the dev and final scripts (prereg §5-9)."""

from dataclasses import dataclass

import numpy as np

from qlab.research14 import strategy as S
from qlab.validation.stats import bootstrap_ci, sharpe

ANN = np.sqrt(365)


@dataclass
class Asset:
    hours: np.ndarray
    close: np.ndarray
    real: np.ndarray
    fund: np.ndarray
    reg: np.ndarray
    sig: np.ndarray

    @classmethod
    def build(cls, d: dict) -> "Asset":
        return cls(d["hours"], d["close"], d["real"], S.funding_rate(d["hours"], d["funding"]),
                   S.regime(d["hours"], d["close"]), S.sigma(d["close"], d["real"]))


@dataclass
class Run:
    dates: np.ndarray
    daily: np.ndarray                  # portfolio daily net returns
    trades: dict                       # symbol -> (entries, sides, net returns)

    def all_trades(self) -> np.ndarray:
        return np.concatenate([t[2] for t in self.trades.values()])


def _portfolio(assets: dict[str, Asset], hourly: dict[str, np.ndarray], start: int, end: int):
    w = 1.0 / len(assets)
    h = next(iter(assets.values())).hours
    tot = sum(w * x for x in hourly.values())
    return S.daily(h, tot, start, end)


def run(assets: dict[str, Asset], cfg: S.Config, start: int, end: int, cost: float = S.COST,
        delay: int = 0, use_funding: bool = True, nohtf: bool = False, entries: dict | None = None) -> Run:
    """Equal-weight sleeves; `entries` overrides the signal (random-entry null)."""
    hourly, trades = {}, {}
    for sym, a in assets.items():
        if entries is not None:
            e, s = entries[sym]
        else:
            sig = (S.signals_nohtf(a.close, a.real, cfg, a.sig) if nohtf
                   else S.signals(a.hours, a.close, a.real, cfg, a.reg, a.sig))
            e, s = S.pick_entries(sig, cfg.H, start, end)
        net, tr = S.simulate(a.close, a.fund, e, s, cfg.H, end, cost, delay, use_funding)
        hourly[sym], trades[sym] = net, (e, s, tr[np.isfinite(tr)])
    dates, d = _portfolio(assets, hourly, start, end)
    return Run(dates, d, trades)


def run_random(assets: dict[str, Asset], cfg: S.Config, ref: Run, start: int, end: int, seed: int) -> Run:
    rng = np.random.default_rng(seed)
    ent = {}
    for sym, a in assets.items():
        ok = np.isfinite(S.zscore(a.close, a.real, cfg.L, a.sig))
        e, s, _ = ref.trades[sym]
        ent[sym] = S.random_entries(rng, a.reg, a.real, ok, a.hours, e, s, cfg.H, start, end)
    return run(assets, cfg, start, end, entries=ent)


def run_htf(assets: dict[str, Asset], side: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    hourly = {}
    for sym, a in assets.items():
        pos = a.reg.astype(float) if side == "LS" else np.maximum(a.reg, 0).astype(float)
        pos[:start] = 0.0
        pos[end - 1:] = 0.0
        hourly[sym] = S.simulate_positions(a.close, a.fund, pos)
    return _portfolio(assets, hourly, start, end)


def run_hold(assets: dict[str, Asset], start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """50/50 rebalanced daily, no costs: daily return = mean of the assets' daily returns."""
    h = next(iter(assets.values())).hours
    rets = []
    for a in assets.values():
        r = np.zeros(len(a.close))
        r[1:] = a.close[1:] / a.close[:-1] - 1.0
        dates, lr = S.daily(h, np.log1p(r), start, end)
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
