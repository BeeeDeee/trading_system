"""H2: RSI(2) mean reversion on index / sector ETFs with Connors-style exits at the close (prereg §4.3),
controls c1 (always long above SMA200) and c2 (random entries in the same assets), robustness variants r1-r7.

Point in time: entries are decided at the close of t from rows <= t and filled at that close (MOC); exits on the
close of t from the SMA5 / RSI of t and the time stop.
"""

from dataclasses import asdict, dataclass

import numpy as np

from qlab.data.panel import Panel
from qlab.engine.vector import Decisions, SimResult, simulate
from qlab.features.basic import tr_index
from qlab.research5.sim import Candidates, SimOutput, SimSpec
from qlab.research5.sim import simulate as event_simulate
from qlab.research19 import setup as S
from qlab.research19.etf_trend import etf_cost
from qlab.research19.signals import rsi_wilder, sma


@dataclass(frozen=True)
class Config:
    name: str = "h2"
    tickers: tuple[str, ...] = tuple(S.RSI_UNIVERSE)
    max_positions: int = S.MAX_POSITIONS
    max_hold: int = S.MAX_HOLD
    entry_rsi: float = S.RSI_ENTRY
    entry_rule: str = "rsi"        # rsi | sum2 (RSI2_t + RSI2_{t-1} < 35, r3)
    exit_rule: str = "sma5"        # sma5 | rsi70 (r2)
    regime_filter: bool = True     # P > SMA200 (False: r5)
    entry_delay: int = 0           # 1: r1
    exit_at_close: bool = True     # False: r1
    cost_mult: float = 1.0

    def as_dict(self) -> dict:
        return {**asdict(self), "study": "research19"}


@dataclass(frozen=True)
class Indicators:
    P: np.ndarray
    valid: np.ndarray
    rsi: np.ndarray
    sma200: np.ndarray
    sma5: np.ndarray


def indicators(p: Panel, cols: np.ndarray) -> Indicators:
    """Indicators for `cols`, scattered into full (T, N) matrices (NaN outside `cols`)."""
    P = tr_index(p)
    valid = np.asarray(p.listed & p.tradable)
    full = lambda f: _scatter(f(P[:, cols], valid[:, cols]), cols, p.shape)  # noqa: E731
    return Indicators(P, valid & _mask(cols, p.shape), full(lambda a, v: rsi_wilder(a, v, S.RSI_N)),
                      full(lambda a, v: sma(a, v, 200)), full(lambda a, v: sma(a, v, 5)))


def _scatter(m: np.ndarray, cols: np.ndarray, shape) -> np.ndarray:
    out = np.full(shape, np.nan)
    out[:, cols] = m
    return out


def _mask(cols: np.ndarray, shape) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[:, cols] = True
    return m


def entry_mask(ind: Indicators, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(entry condition, score) per (t, asset). Score: lower is better."""
    with np.errstate(invalid="ignore"):
        if cfg.entry_rule == "sum2":
            score = ind.rsi + np.vstack([np.full((1, ind.rsi.shape[1]), np.nan), ind.rsi[:-1]])
            cond = score < 35.0
        else:
            score = ind.rsi
            cond = ind.rsi < cfg.entry_rsi
        if cfg.regime_filter:
            cond &= ind.P > ind.sma200
    return cond & ind.valid & np.isfinite(score), score


def candidates(ind: Indicators, cfg: Config, tickers: list[str], start: int) -> Candidates:
    """Entry candidates sorted by (day, score, ticker)."""
    cond, score = entry_mask(ind, cfg)
    cond[:start] = False
    ti, ai = np.nonzero(cond)
    name_rank = np.argsort(np.argsort(np.array(tickers)))
    order = np.lexsort((name_rank[ai], score[ti, ai], ti))
    ti, ai = ti[order], ai[order]
    return Candidates(ti, ai, score[ti, ai])


def exit_signal(ind: Indicators, cfg: Config) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        x = (ind.rsi > 70.0) if cfg.exit_rule == "rsi70" else (ind.P > ind.sma5)
    return np.where(ind.valid & x, 1.0, 0.0)


def spec_for(cfg: Config) -> SimSpec:
    # adv_cap is switched off: with adv20 = inf the size is always NAV / max_positions
    return SimSpec(cfg.max_positions, cfg.max_hold, entry_delay=cfg.entry_delay, exit_at_close=cfg.exit_at_close)


def run_panel(p: Panel, tickers: list[str], cfg: Config, start: int, cash_ret: np.ndarray | None,
              ind: Indicators | None = None, cands: Candidates | None = None,
              exit_sig: np.ndarray | None = None, end: int | None = None) -> SimOutput:
    """Event simulation of H2 on days [start, end). Pass `cands` for control c2."""
    cols = np.array([tickers.index(t) for t in cfg.tickers], dtype=int)
    ind = ind or indicators(p, cols)
    cands = cands or candidates(ind, cfg, tickers, start)
    ex = exit_signal(ind, cfg) if exit_sig is None else exit_sig
    cost = etf_cost(p.dates, p.shape, cfg.cost_mult)
    fn = lambda t, a: cost[t, a]  # noqa: E731
    adv = np.full(p.shape, np.inf)
    return event_simulate(p, cands, spec_for(cfg), adv, start, p.shape[0] if end is None else end, fn, fn,
                          ex, None, cash_ret)


def run(cfg: Config) -> SimOutput:
    p, ex, tickers = S.load_etf()
    start = S.index_of(p, S.PERIOD_H2[0])
    return run_panel(p, tickers, cfg, start, np.asarray(ex["cash_ret"]))


# ---- controls -------------------------------------------------------------------------------------------

def c1_regime_long(p: Panel, tickers: list[str], cfg: Config, start: int, cash_ret: np.ndarray | None) -> SimResult:
    """Equal weights over the assets with P > SMA200 (daily, filled at the next open); cash when none."""
    cols = np.array([tickers.index(t) for t in cfg.tickers], dtype=int)
    ind = indicators(p, cols)
    with np.errstate(invalid="ignore"):
        on = (ind.P > ind.sma200) & ind.valid
    n = on.sum(axis=1, keepdims=True)
    w = np.where(on, 1.0 / np.maximum(n, 1), 0.0)
    days = np.arange(start, p.shape[0])
    return simulate(p, Decisions(days, w[days]), etf_cost(p.dates, p.shape, cfg.cost_mult), cash_ret)


def c2_candidates(trades_asset: np.ndarray, trades_held: np.ndarray, regime: np.ndarray, start: int,
                  end: int, rng: np.random.Generator) -> Candidates:
    """One random entry day per real trade in the SAME asset on a day with P > SMA200 (and valid), holding the
    same number of days. Score = uniform random (priority when several entries fall on one day)."""
    ts, As, hs, sc = [], [], [], []
    for a, h in zip(trades_asset, trades_held):
        days = np.flatnonzero(regime[start:end - 1, a]) + start
        if not len(days):
            continue
        ts.append(int(rng.choice(days))); As.append(int(a)); hs.append(int(h)); sc.append(rng.random())
    ts, As, hs, sc = map(np.array, (ts, As, hs, sc))
    order = np.lexsort((As, sc, ts))
    return Candidates(ts[order].astype(int), As[order].astype(int), sc[order],
                      hold=hs[order].astype(int))
