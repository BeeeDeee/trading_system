"""Compacted weekly panel for the research 22 evaluator.

Decisions are taken after the close of the first trading day of each week (`qlab.schedule.period_starts`) and
executed at the next day's open; a position is held until the open after the next decision. The universe is
`in_liq1000` at the decision day. Only ~1000 of the N panel columns are eligible each week, so every matrix is
stored as (decisions F, eligible columns K) with `idx` mapping a compact column to the panel column (ascending).

Terminals are computed by `daily_leaves`, the single point-in-time source of truth (row t uses data <= t).
`build_compact` calls it per block of columns (the panel is memory-mapped) and never reads rows after `last`.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from qlab.data.panel import Panel
from qlab.engine.costs import CostModel
from qlab.research22.kernels import portfolio
from qlab.schedule import period_starts

CLIP = -0.9999            # log1p guard; terminal returns enter the holding return exactly (product), not via logs
MIN_HISTORY = 250         # first decision row (the longest terminal looks back 250 days)
RET_LAGS = (1, 5, 20, 60, 120, 250)
VOL_LAGS = (20, 60)
Z_BASES = ("ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "vol_60")
Z_WINDOWS = (20, 60)      # z-score windows of 250 days would need > 1 year of warm-up in run A
RAW_NAMES = (tuple(f"ret_{n}" for n in RET_LAGS) + tuple(f"vol_{n}" for n in VOL_LAGS)
             + ("dvr_20_250", "dmax_250", "dmin_250"))
Z_NAMES = tuple(f"z{w}_{b}" for b in Z_BASES for w in Z_WINDOWS)
LEAF_NAMES = RAW_NAMES + Z_NAMES
STATE_NAMES = ("mkt_trend", "mkt_vol_high")


def _log_returns(ret_co: np.ndarray, ret_oc: np.ndarray) -> np.ndarray:
    co = np.clip(np.nan_to_num(np.asarray(ret_co, dtype=np.float64)), CLIP, None)
    oc = np.clip(np.nan_to_num(np.asarray(ret_oc, dtype=np.float64)), CLIP, None)
    return np.log1p(co) + np.log1p(oc)


def _lag_diff(c: np.ndarray, lag: int) -> np.ndarray:
    """c[t] - c[t - lag]; NaN where t < lag."""
    out = np.full(c.shape, np.nan)
    out[lag:] = c[lag:] - c[:-lag]
    return out


def _rolling_z(x: np.ndarray, w: int) -> np.ndarray:
    """z-score of x[t] against x[t-w+1..t]; NaN unless all w values are finite, 0 if the window is constant."""
    ok = np.isfinite(x)
    xz = np.where(ok, x, 0.0)

    def win(a):
        out = a.copy()
        out[w:] = a[w:] - a[:-w]
        return out

    s, s2, n = win(np.cumsum(xz, axis=0)), win(np.cumsum(xz * xz, axis=0)), win(np.cumsum(ok, axis=0))
    mean = s / w
    sd = np.sqrt(np.maximum(s2 / w - mean ** 2, 0.0))
    z = np.where(sd > 1e-9, (x - mean) / np.where(sd > 1e-9, sd, 1.0), 0.0)
    return np.where(n == w, z, np.nan)


def daily_leaves(ret_co: np.ndarray, ret_oc: np.ndarray, dollar_volume: np.ndarray,
                 at: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Terminals at rows `at` (all rows if None) as float32 (len(at), n_assets). Row t uses data <= t."""
    t_len = ret_co.shape[0]
    rows = np.arange(t_len) if at is None else np.asarray(at)
    lr = _log_returns(ret_co, ret_oc)
    s = np.cumsum(lr, axis=0)
    s2 = np.cumsum(lr * lr, axis=0)
    base: dict[str, np.ndarray] = {}
    for lag in RET_LAGS:
        base[f"ret_{lag}"] = _lag_diff(s, lag)
    for lag in VOL_LAGS:
        mean = _lag_diff(s, lag) / lag
        base[f"vol_{lag}"] = np.sqrt(np.maximum(_lag_diff(s2, lag) / lag - mean ** 2, 0.0))
    out = {name: base[name][rows] for name in base}

    dv = np.cumsum(np.nan_to_num(np.asarray(dollar_volume, dtype=np.float64)), axis=0)
    m20, m250 = _lag_diff(dv, 20) / 20, _lag_diff(dv, 250) / 250
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dvr_20_250"] = np.where((m20 > 0) & (m250 > 0), np.log(m20 / m250), np.nan)[rows]

    dmax = np.full((len(rows), s.shape[1]), np.nan)
    dmin = np.full_like(dmax, np.nan)
    for k, t in enumerate(rows):
        if t >= 249:
            win = s[t - 249: t + 1]
            dmax[k] = s[t] - win.max(axis=0)
            dmin[k] = s[t] - win.min(axis=0)
    out["dmax_250"], out["dmin_250"] = dmax, dmin

    for b in Z_BASES:
        for w in Z_WINDOWS:
            out[f"z{w}_{b}"] = _rolling_z(base[b], w)[rows]
    return {name: out[name].astype(np.float32) for name in LEAF_NAMES}


def market_states(r_ew: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mkt_trend, mkt_vol_high) per day from the equal-weight universe return r_ew[t] (causal).

    trend: EW index above its 200-day mean; vol_high: 20-day volatility above the median of the last 250
    values. Both are False while their history is incomplete.
    """
    n = len(r_ew)
    index = np.cumprod(1.0 + r_ew)
    c0 = np.concatenate(([0.0], np.cumsum(index)))
    sma = np.full(n, np.nan)
    if n >= 200:
        sma[199:] = (c0[200:] - c0[:-200]) / 200
    trend = (index > sma) & np.isfinite(sma)

    vol = np.full(n, np.nan)
    if n > 20:
        cr, cr2 = np.concatenate(([0.0], np.cumsum(r_ew))), np.concatenate(([0.0], np.cumsum(r_ew ** 2)))
        mean = (cr[21:] - cr[1:-20]) / 20
        vol[20:] = np.sqrt(np.maximum((cr2[21:] - cr2[1:-20]) / 20 - mean ** 2, 0.0))
    high = np.zeros(n, dtype=bool)
    if n >= 270:
        med = np.median(sliding_window_view(vol[20:], 250), axis=1)   # median of vol[t-249..t] for t >= 269
        high[269:] = vol[269:] > med
    return trend, high


@dataclass
class Weekly:
    gross: np.ndarray         # return of the invested positions over NAV at the open
    net: np.ndarray           # NAV at the next open / NAV at this open - 1, costs included
    turnover: np.ndarray      # buys + sells over NAV per week
    n_positions: np.ndarray   # positions held after the trade (stuck ones included)
    nav_trade: np.ndarray     # NAV after the trade, in units of the first open


@dataclass
class Compact:
    dec_rows: np.ndarray            # (F,) panel rows of the decisions
    dates: np.ndarray               # (F,) decision dates
    idx: np.ndarray                 # (F, K) panel column of each compact column, -1 = padding
    valid: np.ndarray               # (F, K) bool
    wk_all: np.ndarray              # (F, N) float64 holding return by asset (any asset can be held, also a stuck one)
    dead_all: np.ndarray            # (F, N) bool, paid out by a delisting during the holding week
    trad: np.ndarray                # (F, N) bool, a usable open exists on the execution day
    rate: np.ndarray                # (F, N) float64 cost per side at 1x, by asset
    leaves: dict[str, np.ndarray]   # name -> (F, K) float32
    states: dict[str, np.ndarray]   # name -> (F, 1) bool
    n_assets: int
    n_folds: int = 8                # dev folds of the fitness (set by `setup.load_compact`: A 8, B 7)
    _bench: dict[float, Weekly] = field(default_factory=dict, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.idx.shape

    def _gather(self, full: np.ndarray, fill):
        rows = np.arange(self.idx.shape[0])[:, None]
        return np.where(self.valid, full[rows, np.where(self.valid, self.idx, 0)], fill)

    @cached_property
    def wk(self) -> np.ndarray:
        """(F, K) holding return of the eligible names (NaN in padding)."""
        return self._gather(self.wk_all, np.nan)

    @cached_property
    def dead(self) -> np.ndarray:
        """(F, K) paid out by a delisting during the holding week (False in padding)."""
        return self._gather(self.dead_all, False)

    def weekly(self, score: np.ndarray, topk: int, cost_scale: float = 1.0) -> Weekly:
        net, gross, turn, npos, nav = portfolio(np.asarray(score, dtype=np.float32), self.wk_all, self.dead_all,
                                                self.idx, self.rate, self.trad, float(cost_scale), int(topk))
        return Weekly(gross, net, turn, npos, nav)

    def benchmark(self, cost_scale: float = 1.0) -> Weekly:
        """Equal-weight eligible universe, rebalanced every week with the same cost model."""
        if cost_scale not in self._bench:
            score = np.where(self.valid, np.float32(0.0), np.float32(np.nan))
            self._bench[cost_scale] = self.weekly(score, self.shape[1], cost_scale)
        return self._bench[cost_scale]


def decision_schedule(dates: np.ndarray, first: int, last: int, embargo: int) -> tuple[np.ndarray, np.ndarray]:
    """Decision rows and the next decision rows; every holding week lies entirely inside rows <= last."""
    starts = np.flatnonzero(period_starts(np.asarray(dates, dtype="datetime64[D]")[: last + 1], "W"))
    starts = starts[starts >= max(first + embargo, MIN_HISTORY)]
    dec, nxt = starts[:-1], starts[1:]
    keep = nxt + 1 <= last
    return dec[keep], nxt[keep]


def _scatter(out: np.ndarray, x: np.ndarray, pos: np.ndarray) -> None:
    r, c = np.nonzero(pos >= 0)
    out[r, pos[r, c]] = x[r, c]


def build_compact(panel: Panel, extra: dict[str, np.ndarray], first: int, last: int, embargo: int,
                  cost_model: CostModel, block: int = 500) -> Compact:
    """Compact weekly panel for decisions with the whole holding week inside rows first..last."""
    dates = np.asarray(panel.dates, dtype="datetime64[D]")
    dec, nxt = decision_schedule(dates, first, last, embargo)
    if len(dec) < 2:
        raise ValueError("stage contains fewer than two decisions")
    n_assets = panel.shape[1]
    elig = np.asarray(extra["in_liq1000"][dec]) & np.asarray(panel.tradable[dec + 1])
    k_max = int(elig.sum(axis=1).max())
    n_dec = len(dec)
    idx = np.full((n_dec, k_max), -1, dtype=np.int64)
    pos = np.full((n_dec, n_assets), -1, dtype=np.int32)
    for i in range(n_dec):
        ids = np.flatnonzero(elig[i])
        idx[i, : len(ids)] = ids
        pos[i, ids] = np.arange(len(ids))
    rank = np.asarray(extra["liq_rank"][dec], dtype=np.float64)
    rate = np.asarray(cost_model.rate(rank, dates[dec + 1]), dtype=np.float64)

    leaves = {name: np.full((n_dec, k_max), np.nan, dtype=np.float32) for name in LEAF_NAMES}
    wk_all = np.zeros((n_dec, n_assets))
    dead_all = np.zeros((n_dec, n_assets), dtype=bool)
    trad = np.asarray(panel.tradable[dec + 1], dtype=bool)
    num, cnt = np.zeros(last + 1), np.zeros(last + 1)
    in_liq, listed, delisting = extra["in_liq1000"], panel.listed, panel.delisting

    for c0 in range(0, n_assets, block):
        c1 = min(c0 + block, n_assets)
        rco = np.nan_to_num(np.asarray(panel.ret_co[: last + 1, c0:c1], dtype=np.float64))
        roc = np.nan_to_num(np.asarray(panel.ret_oc[: last + 1, c0:c1], dtype=np.float64))
        sub = pos[:, c0:c1]
        for name, x in daily_leaves(rco, roc, panel.dollar_volume[: last + 1, c0:c1], dec).items():
            _scatter(leaves[name], x, sub)
        week = np.empty((n_dec, c1 - c0))
        for i in range(n_dec):
            f1, g = dec[i] + 1, nxt[i]
            p = 1.0 + roc[f1]
            for d in range(f1 + 1, g + 1):
                p = p * (1.0 + rco[d]) * (1.0 + roc[d])
            week[i] = p * (1.0 + rco[g + 1]) - 1.0
        wk_all[:, c0:c1] = week
        d_cum = np.cumsum(np.asarray(delisting[: last + 1, c0:c1]), axis=0, dtype=np.int32)
        dead_all[:, c0:c1] = (d_cum[nxt + 1] - d_cum[dec + 1]) > 0
        member = np.asarray(in_liq[: last, c0:c1]) & np.asarray(listed[1: last + 1, c0:c1])
        num[1:] += (((1.0 + rco[1:]) * (1.0 + roc[1:]) - 1.0) * member).sum(axis=1)
        cnt[1:] += member.sum(axis=1)

    trend, high = market_states(num / np.maximum(cnt, 1.0))
    states = {"mkt_trend": trend[dec][:, None], "mkt_vol_high": high[dec][:, None]}
    return Compact(dec, dates[dec], idx, idx >= 0, wk_all, dead_all, trad, rate, leaves, states, n_assets)
