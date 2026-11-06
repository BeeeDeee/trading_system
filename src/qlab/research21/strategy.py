"""S0 (prereg §3) and its proxy-filter partition (prereg §4).

S0: RSI(2) < 10 and P > SMA200 on securities in the PIT universe (ADV20 >= adv_min, price >= 5, >= 252 bars), decided
after the close of t, bought at the open of t+1, sold at the open after the first close above SMA5 or after 10 days.
Equal slots NAV / 10, no ADV cap (large caps), ranking by lowest RSI(2), ties by ticker. Costs: research 5 period model.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
import polars as pl

from qlab.data.panel import Panel
from qlab.research5.costs import DateCost, period_cost
from qlab.research5.sim import Candidates, SimOutput, SimSpec, simulate
from qlab.research21 import setup as S
from qlab.research21.features import Features


@dataclass(frozen=True)
class Config:
    name: str = "s0"
    adv_min: float = S.ADV_MIN
    regime_filter: bool = True        # P > SMA200 (False: r5)
    entry_delay: int = 1              # 0: MOC upper bound (r1)
    cost_mult: float = 1.0
    allow: str = ""                   # "" = S0; otherwise a filter name (E, V, G, I, C): only ALLOWED entries
    vol_ratio: float = S.VOL_RATIO
    gap_z: float = S.GAP_Z
    idio_z: float = S.IDIO_Z
    e_window: str = "e1"              # e1 | e2 (r2)
    control: str = ""                 # "" | "c0": regime stocks in random order, no exit signal, same slots and hold

    def as_dict(self) -> dict:
        return {**asdict(self), "study": "research21"}


@dataclass
class Context:
    panel: Panel
    extra: dict
    feats: Features
    tickers: np.ndarray            # ticker of every panel column ("" if unknown)
    start: int
    end: int
    spy: int


def load_context(snapshot: str = S.SHARADAR) -> Context:
    p, ex = S.load_stock_panel(snapshot)
    t = (pl.read_parquet(f"data/parquet/{snapshot}/tickers.parquet").filter(pl.col("table") == "SEP")
         .select("permaticker", "ticker").unique("permaticker"))
    a = pl.DataFrame({"permaticker": p.assets}).with_columns(pl.col("permaticker").cast(t["permaticker"].dtype))
    tick = np.array(a.join(t, on="permaticker", how="left")["ticker"].fill_null("").to_list())
    sfp = (pl.read_parquet(f"data/parquet/{snapshot}/tickers.parquet")
           .filter((pl.col("table") == "SFP") & (pl.col("ticker") == "SPY"))["permaticker"].cast(pl.Int64).item())
    spy = int(np.flatnonzero(p.assets == sfp)[0])
    return Context(p, ex, Features(snapshot), tick, S.index_of(p.dates, S.PERIOD_START), p.shape[0], spy)


# --------------------------------------------------------------------------- blocking flags

def blocked_flags(f: Features, t: np.ndarray, a: np.ndarray, cfg: Config) -> dict[str, np.ndarray]:
    """Per (signal day t, panel column a): True where the filter classifies the entry as 'information' (blocked).
    A filter that cannot be computed (NaN) blocks, except E (no 8-K means no earnings). E and C are False/undefined
    before E_START; the caller restricts them to that period."""
    j = f.pos(a)
    ev = np.asarray(f.m[cfg.e_window])[t, j]
    with np.errstate(invalid="ignore"):
        vr = np.asarray(f.m["volratio"])[t, j]
        gz = np.asarray(f.m["gapz"])[t, j]
        iz = np.asarray(f.m["idioz"])[t, j]
        out = {"E": ev.astype(bool),
               "V": ~(vr <= cfg.vol_ratio),            # NaN -> not (NaN <= x) -> blocked
               "G": ~(gz >= cfg.gap_z),
               "I": ~(iz >= cfg.idio_z)}
    out["C"] = out["E"] | out["V"] | out["G"] | out["I"]
    return out


# --------------------------------------------------------------------------- candidates and simulation

def entry_mask(ctx: Context, cfg: Config) -> np.ndarray:
    """(T, n_cols) S0 entry condition (before any proxy filter)."""
    f, ex = ctx.feats, ctx.extra
    cols = f.cols
    adv = np.asarray(ex["adv20"][:, cols])
    ok = np.asarray(ex["base_ok"][:, cols])
    with np.errstate(invalid="ignore"):
        uni = ok & (adv >= cfg.adv_min) & np.asarray(f.m["valid"])
        if cfg.control == "c0":
            cond = uni & np.asarray(f.m["regime"])
        else:
            cond = uni & (np.asarray(f.m["rsi"]) < S.RSI_ENTRY)
            if cfg.regime_filter:
                cond &= np.asarray(f.m["regime"])
    cond[:ctx.start] = False
    cond[ctx.end - 1:] = False
    return cond


def candidates(ctx: Context, cfg: Config) -> Candidates:
    cond = entry_mask(ctx, cfg)
    ti, ji = np.nonzero(cond)
    a = ctx.feats.cols[ji]
    if cfg.allow:
        fl = blocked_flags(ctx.feats, ti, a, cfg)[cfg.allow]
        keep = ~fl
        if cfg.allow in ("E", "C"):
            keep &= ti >= S.index_of(ctx.panel.dates, S.E_START)
        ti, a = ti[keep], a[keep]
    if cfg.control == "c0":
        score = np.random.default_rng(S.SEED).random(len(ti))
    else:
        score = np.asarray(ctx.feats.m["rsi"])[ti, ctx.feats.pos(a)].astype(float)
    rank = np.argsort(np.argsort(ctx.tickers))
    order = np.lexsort((rank[a], score, ti))
    return Candidates(ti[order], a[order], score[order])


def run(ctx: Context, cfg: Config, cands: Candidates | None = None) -> SimOutput:
    cands = candidates(ctx, cfg) if cands is None else cands
    dates = ctx.panel.dates
    open_c = DateCost(period_cost(dates, True, cfg.cost_mult))
    close_c = DateCost(period_cost(dates, False, cfg.cost_mult))
    entry = close_c if cfg.entry_delay == 0 else open_c
    spec = SimSpec(S.MAX_POSITIONS, S.MAX_HOLD, cfg.entry_delay, adv_cap=np.inf)
    exit_sig = None
    if cfg.control != "c0":
        exit_sig = np.zeros(ctx.panel.shape, dtype=bool)
        exit_sig[:, ctx.feats.cols] = np.asarray(ctx.feats.m["exit"])
    adv = np.broadcast_to(np.inf, ctx.panel.shape)
    cash = np.asarray(ctx.extra["cash_ret"][:ctx.end])
    return simulate(ctx.panel, cands, spec, adv, ctx.start, ctx.end, entry.at, open_c.at, exit_sig, None, cash)


def tag_trades(ctx: Context, trades: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Trades of an S0 run with the four blocked flags (and C) at the signal day, the signal year and month key."""
    t = trades["signal_day"].to_numpy()
    a = trades["asset"].to_numpy()
    fl = blocked_flags(ctx.feats, t, a, cfg)
    d = ctx.panel.dates[t].astype("datetime64[M]")
    return trades.with_columns(
        [pl.Series(f"b_{k}", v) for k, v in fl.items()]
        + [pl.Series("year", ctx.panel.dates[t].astype("datetime64[Y]").astype(int) + 1970),
           pl.Series("month", d.astype(int))])
