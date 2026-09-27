"""STR-TF configuration, signal cache, candidates and one-call runs (pre-registration §4–§9)."""

import itertools
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import polars as pl

from qlab.data.panel import Panel
from qlab.research5 import signals as sig
from qlab.research5.costs import DateCost, RankCost, period_cost
from qlab.research5.sim import Candidates, SimOutput, SimSpec, simulate
from qlab.research5.universe import universe_mask
from qlab.strategies.signals import market_trend


@dataclass(frozen=True)
class Config:
    entry_z: float = 2.0
    lookback_ret: int = 3
    trend_sma: int = 200
    exit_sma: int = 5
    max_hold: int = 5
    max_positions: int = 10
    min_adv: float = 20e6
    # variants (§9); defaults = base version
    signal: str = "z"              # z | rsi2 | ibs
    side: str = "long"             # long | short (trade level only)
    entry_delay: int = 1           # 1 open t+1, 2 open t+2, 0 close t (MOC)
    market_filter: bool = False    # SPY TR > SMA 200
    atr_stop: float | None = None  # stop distance in ATR14 multiples
    vol_target: float | None = None
    capital: float = 1e6
    cash: str = "zero"             # zero | tbill
    cost: str = "period"           # period | tiers
    cost_mult: float = 1.0
    exclude_earnings: bool = False  # 8-K Item 2.02 in [t-2, t]
    vix_gate: str | None = None     # research 6: "abs<level>" (VIX close > level) or "rel80"
    core: str | None = None         # research 7: "spy" = idle capital held in SPY
    core_cost_bps: float = 2.0      # per side, for every SPY trade that funds a position

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
GRID = {"entry_z": (1.5, 2.0, 2.5), "lookback_ret": (2, 3, 5), "trend_sma": (100, 200),
        "exit_sma": (3, 5, 10), "max_hold": (3, 5, 10), "max_positions": (10, 20),
        "min_adv": (5e6, 20e6, 100e6)}


def grid_configs() -> list[Config]:
    keys = list(GRID)
    return [Config(**dict(zip(keys, v))) for v in itertools.product(*GRID.values())]


def neighbors(cfg: Config) -> list[Config]:
    """Configs one grid step away in exactly one parameter."""
    out = []
    for k, values in GRID.items():
        i = values.index(getattr(cfg, k))
        out += [replace(cfg, **{k: values[j]}) for j in (i - 1, i + 1) if 0 <= j < len(values)]
    return out


@dataclass
class Context:
    """One stage (dev / validation / late): panel rows [0, end), simulated days [start, end)."""
    panel: Panel
    extra: dict
    start: int
    cache_dir: Path
    spy: int
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def end(self) -> int:
        return self.panel.shape[0]

    def feature(self, name: str) -> np.ndarray:
        """Point-in-time feature matrix (float32), computed by column chunks and memory-mapped."""
        if name not in self._cache:
            path = self.cache_dir / f"{name}.npy"
            if not path.exists():
                self._compute(name, path)
            self._cache[name] = np.load(path, mmap_mode="r")
        return self._cache[name]

    def _compute(self, name: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=self.panel.shape)
        p = self.panel
        for a in range(0, p.shape[1], 1000):
            b = min(a + 1000, p.shape[1])
            sub = Panel(p.dates, p.assets[a:b], *(np.asarray(getattr(p, f)[:, a:b]) for f in
                        ("ret_co", "ret_oc", "tradable", "listed", "delisting", "close_u",
                         "dollar_volume")))
            hi = np.asarray(self.extra["high_u"][:self.end, a:b])
            lo = np.asarray(self.extra["low_u"][:self.end, a:b])
            out[:, a:b] = _feature(name, sub, hi, lo)
        out.flush()
        del out
        tmp.rename(path)

    def row_mask(self, name: str) -> np.ndarray:
        return np.asarray(self.extra[name][:self.end])


def _feature(name: str, panel: Panel, hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    if name.startswith("z"):
        return sig.z_reversal(panel, int(name[1:]))
    if name.startswith("sma"):
        return sig.sma_gap(panel, int(name[3:]))
    if name == "rsi2":
        return sig.rsi(panel, 2)
    if name == "ibs":
        return sig.ibs(panel, hi, lo)
    if name == "r1":
        return sig.daily_return(panel)
    if name == "atr14":
        return sig.atr_fraction(panel, hi, lo, 14)
    raise ValueError(f"unknown feature {name!r}")


class _Signed:
    """Row access to a matrix times a sign (short side: exit when below the SMA)."""

    def __init__(self, m: np.ndarray, sign: float):
        self.m, self.sign = m, sign

    def __getitem__(self, t: int) -> np.ndarray:
        return self.sign * np.asarray(self.m[t])


def candidates(ctx: Context, cfg: Config, chunk: int = 256) -> Candidates:
    """Entry candidates on decision days [start, end), ranked best first within each day."""
    base_ok, adv20 = ctx.extra["base_ok"], ctx.extra["adv20"]
    trend = ctx.feature(f"sma{cfg.trend_sma}")
    if cfg.signal == "z":
        score_m = ctx.feature(f"z{cfg.lookback_ret}")
    elif cfg.signal == "rsi2":
        score_m = ctx.feature("rsi2")
    elif cfg.signal == "ibs":
        score_m, r1 = ctx.feature("ibs"), ctx.feature("r1")
    else:
        raise ValueError(f"unknown signal {cfg.signal!r}")
    earn = ctx.extra["earn8k"] if cfg.exclude_earnings else None
    atr = ctx.feature("atr14") if cfg.atr_stop is not None else None

    ts, as_, scores, stops = [], [], [], []
    for r0 in range(ctx.start, ctx.end, chunk):
        r1_ = min(r0 + chunk, ctx.end)
        with np.errstate(invalid="ignore"):
            uni = universe_mask(base_ok[r0:r1_], adv20[r0:r1_], cfg.min_adv)
            tr = np.asarray(trend[r0:r1_])
            s = np.asarray(score_m[r0:r1_], dtype=float)
            if cfg.signal == "z" and cfg.side == "long":
                m, score = (tr > 0) & (s <= -cfg.entry_z), s
            elif cfg.signal == "z" and cfg.side == "short":
                m, score = (tr < 0) & (s >= cfg.entry_z), -s
            elif cfg.signal == "rsi2":
                m, score = (tr > 0) & (s < 10.0), s
            else:
                m, score = (tr > 0) & (s < 0.2) & (np.asarray(r1[r0:r1_]) < 0), s
        m &= uni
        if earn is not None:  # 8-K Item 2.02 filed on t, t-1 or t-2
            lo = max(r0 - 2, 0)
            e = np.asarray(earn[lo:r1_], bool)
            for lag in range(3):
                idx = np.arange(r0, r1_) - lag - lo
                ok = idx >= 0
                m[ok] &= ~e[idx[ok]]
        ti, ai = np.nonzero(m)
        ts.append(ti + r0)
        as_.append(ai)
        scores.append(score[ti, ai])
        if atr is not None:
            stops.append(cfg.atr_stop * np.asarray(atr[r0:r1_])[ti, ai])
    t = np.concatenate(ts)
    a = np.concatenate(as_)
    score = np.concatenate(scores)
    order = np.lexsort((a, score, t))
    stop = np.concatenate(stops)[order] if atr is not None else None
    return Candidates(t[order], a[order], score[order], stop=stop)


def cost_fns(ctx: Context, cfg: Config):
    dates = ctx.panel.dates
    if cfg.cost == "period":
        open_c = DateCost(period_cost(dates, True, cfg.cost_mult))
        close_c = DateCost(period_cost(dates, False, cfg.cost_mult))
        entry = close_c if cfg.entry_delay == 0 else open_c
        return entry.at, open_c.at
    if cfg.cost == "tiers":
        from qlab.engine.costs import CostModel
        rc = RankCost(ctx.extra["liq_rank"], dates, CostModel(multiplier=cfg.cost_mult))
        return rc.at, rc.at
    raise ValueError(f"unknown cost model {cfg.cost!r}")


def run(ctx: Context, cfg: Config, cands: Candidates | None = None) -> SimOutput:
    cands = candidates(ctx, cfg) if cands is None else cands
    entry_cost, exit_cost = cost_fns(ctx, cfg)
    exit_m = _Signed(ctx.feature(f"sma{cfg.exit_sma}"), 1.0 if cfg.side == "long" else -1.0)
    allow = market_trend(ctx.panel, ctx.spy, 200) > 0 if cfg.market_filter else None
    if cfg.vix_gate is not None:
        gate = vix_gate(np.asarray(ctx.extra["vix"][:ctx.end]), cfg.vix_gate)
        allow = gate if allow is None else allow & gate
    spec = SimSpec(cfg.max_positions, cfg.max_hold, cfg.entry_delay, cfg.capital,
                   vol_target=cfg.vol_target)
    cash = np.asarray(ctx.extra["cash_ret"][:ctx.end]) if cfg.cash == "tbill" else None
    cash_oc = None
    if cfg.core == "spy":
        cash = np.asarray(ctx.panel.ret_co[:, ctx.spy])
        cash_oc = np.asarray(ctx.panel.ret_oc[:, ctx.spy])
        extra_cost = cfg.core_cost_bps * cfg.cost_mult / 1e4
        entry_cost = _plus(entry_cost, extra_cost)
        exit_cost = _plus(exit_cost, extra_cost)
    elif cfg.core is not None:
        raise ValueError(f"unknown core {cfg.core!r}")
    return simulate(ctx.panel, cands, spec, ctx.extra["adv20"], ctx.start, ctx.end, entry_cost,
                    exit_cost, exit_m, allow, cash, cash_oc)


def _plus(fn, extra: float):
    return lambda t, a: fn(t, a) + extra


def vix_gate(vix: np.ndarray, rule: str, window: int = 252) -> np.ndarray:
    """(T,) bool: new entries allowed after the close of t (research 6 §3), point in time.

    "abs25": VIX close of t > 25; "rel80": VIX close of t above the 80th percentile of the last
    `window` closes including t. Missing VIX blocks entries.
    """
    v = np.asarray(vix, dtype=float)
    if rule.startswith("abs"):
        with np.errstate(invalid="ignore"):
            return np.isfinite(v) & (v > float(rule[3:]))
    if rule.startswith("rel"):
        q = float(rule[3:]) / 100.0
        out = np.zeros(len(v), bool)
        for t in range(window - 1, len(v)):
            w = v[t - window + 1:t + 1]
            w = w[np.isfinite(w)]
            out[t] = np.isfinite(v[t]) and len(w) > window // 2 and v[t] > np.quantile(w, q)
        return out
    raise ValueError(f"unknown VIX gate {rule!r}")


def eligible_pool(ctx: Context, cfg: Config, t: int) -> np.ndarray:
    """Universe members with trend_ok on day t (random benchmark pool)."""
    with np.errstate(invalid="ignore"):
        uni = universe_mask(ctx.extra["base_ok"][t], ctx.extra["adv20"][t], cfg.min_adv)
        return np.flatnonzero(uni & (np.asarray(ctx.feature(f"sma{cfg.trend_sma}")[t]) > 0))


def random_benchmark(ctx: Context, cfg: Config, strategy: SimOutput, n_sims: int = 1000,
                     seed: int = 0) -> np.ndarray:
    """Daily-return Sharpe (annualized) of `n_sims` random replays of the strategy's trades.

    On each decision day with k strategy entries, k random members of the trend-filtered universe
    are bought (random order, same sizing, costs and slots); each gets the holding length of the
    corresponding strategy trade and is closed by the time stop only.
    """
    tr = strategy.trades.filter(pl.col("reason") != "end").sort("signal_day", "score")
    sig_days = tr["signal_day"].to_numpy()
    holds = np.maximum(tr["held"].to_numpy(), 1)
    days, first = np.unique(sig_days, return_index=True)
    counts = np.diff(np.append(first, len(sig_days)))
    pools = {int(d): eligible_pool(ctx, cfg, int(d)) for d in days}
    max_entries = np.zeros(ctx.end, dtype=int)
    max_entries[days] = counts
    rng = np.random.default_rng(seed)
    entry_cost, exit_cost = cost_fns(ctx, cfg)
    spec = SimSpec(cfg.max_positions, cfg.max_hold, cfg.entry_delay, cfg.capital)
    out = np.empty(n_sims)
    for s in range(n_sims):
        t_list, a_list, h_list = [], [], []
        for d, f, c in zip(days, first, counts):
            pool = pools[int(d)]
            k = min(len(pool), c + 2 * cfg.max_positions)  # spares for held / untradable picks
            pick = rng.choice(pool, size=k, replace=False) if k else np.empty(0, int)
            t_list.append(np.full(k, d))
            a_list.append(pick)
            h = holds[f:f + c]
            h_list.append(np.resize(h, k) if len(h) else np.empty(0, int))
        t_arr = np.concatenate(t_list)
        cands = Candidates(t_arr, np.concatenate(a_list), np.arange(len(t_arr), dtype=float),
                           hold=np.concatenate(h_list), max_entries=max_entries)
        res = simulate(ctx.panel, cands, spec, ctx.extra["adv20"], ctx.start, ctx.end, entry_cost,
                       exit_cost, None, None, None)
        r = res.returns
        out[s] = r.mean() / r.std(ddof=1) * np.sqrt(252) if r.std() > 0 else 0.0
    return out
