"""Daily dates x symbols panel for research 8 (prereg §3-4).

Matrices (float64, NaN = no data):
  po, ph, pl, pc   perp open/high/low/close
  so, sc           spot open/close in PERP units (spot price x multiplier)
  qv               perp quote volume (USD) of the day
  mh               mark-price high of the day (liquidation, prereg v1.1); NaN where Binance has no mark kline
  fund_hold        sum of funding rates with ts in (D 00:00, D+1 00:00]  -> earned by a short held over day D
  fund_sig         sum of funding rates with ts in [D 00:00, D+1 00:00)  -> known at D+1 00:00 for decisions
  fund_n           number of funding events in the fund_sig bucket
  tbill            daily T-bill return (vector)

Point-in-time: a decision at the open of day t may use fund_sig/qv/closes of days <= t-1 only.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from . import data as D

FIELDS = ("po", "ph", "pl", "pc", "so", "sc", "qv", "mh", "fund_hold", "fund_sig", "fund_n")


@dataclass
class Panel:
    dates: np.ndarray            # datetime64[D]
    symbols: list[str]
    m: dict[str, np.ndarray]     # field -> T x N
    tbill: np.ndarray            # T

    def slice(self, start: date, end: date) -> "Panel":
        i0 = int(np.searchsorted(self.dates, np.datetime64(start, "D")))
        i1 = int(np.searchsorted(self.dates, np.datetime64(end, "D"), side="right"))
        return Panel(self.dates[i0:i1], self.symbols, {k: v[i0:i1] for k, v in self.m.items()}, self.tbill[i0:i1])


def _bucket(f: pl.DataFrame, closed_right: bool) -> pl.DataFrame:
    # (D, D+1]: shift by 1 ms back so an event exactly at D+1 00:00 lands in day D
    ts = pl.col("ts") - pl.duration(milliseconds=1) if closed_right else pl.col("ts")
    return f.with_columns(ts.dt.date().alias("date")).group_by("date").agg(
        pl.col("rate").sum().alias("rate"), pl.len().alias("n"))


def build(raw: Path, pairs: dict[str, tuple[str, float]], tbill: pl.DataFrame, start: date, end: date) -> Panel:
    days = np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + 1)
    T, syms = len(days), sorted(pairs)
    m = {k: np.full((T, len(syms)), np.nan) for k in FIELDS}
    idx = {d: i for i, d in enumerate(days.astype(object))}

    def put(field: str, j: int, dates, vals) -> None:
        for d, v in zip(dates, vals):
            i = idx.get(d)
            if i is not None:
                m[field][i, j] = v

    for j, p in enumerate(syms):
        spot, mult = pairs[p]
        k = D.read_klines(raw / "perp_1d" / p)
        for fld, col in (("po", "open"), ("ph", "high"), ("pl", "low"), ("pc", "close"), ("qv", "quote_volume")):
            put(fld, j, k["date"].to_list(), k[col].to_list())
        mk = D.read_klines(raw / "mark_1d" / p)
        put("mh", j, mk["date"].to_list(), mk["high"].to_list())
        s = D.read_klines(raw / "spot_1d" / spot)
        put("so", j, s["date"].to_list(), (s["open"] * mult).to_list())
        put("sc", j, s["date"].to_list(), (s["close"] * mult).to_list())
        f = D.read_funding(raw / "funding" / p)
        if f.height:
            h = _bucket(f, True)
            put("fund_hold", j, h["date"].to_list(), h["rate"].to_list())
            g = _bucket(f, False)
            put("fund_sig", j, g["date"].to_list(), g["rate"].to_list())
            put("fund_n", j, g["date"].to_list(), g["n"].to_list())
    # days inside the listed span without a funding event earn/know 0 (not NaN) once the perp trades
    live = ~np.isnan(m["pc"])
    for fld in ("fund_hold", "fund_sig", "fund_n"):
        m[fld] = np.where(live & np.isnan(m[fld]), 0.0, m[fld])
    tb = tbill_daily(tbill, days)
    return Panel(days, syms, m, tb)


def tbill_daily(fred: pl.DataFrame, days: np.ndarray) -> np.ndarray:
    """FRED DTB3 (percent, annual) -> daily return on a 365-day calendar, forward-filled."""
    s = (fred.rename({fred.columns[0]: "date", fred.columns[1]: "r"})
             .with_columns(pl.col("date").str.to_date(), pl.col("r").cast(pl.Float64, strict=False))
             .drop_nulls().sort("date"))
    full = pl.DataFrame({"date": [d.astype(object) for d in days]}).join(s, on="date", how="left")
    # seed with the last value before the window, then forward-fill
    prev = s.filter(pl.col("date") < days[0].astype(object))
    r = full["r"].to_numpy(allow_copy=True).astype(float)
    if np.isnan(r[0]) and prev.height:
        r[0] = prev["r"][-1]
    r = pl.Series(r).fill_nan(None).forward_fill().to_numpy()
    return (1 + r / 100) ** (1 / 365) - 1


def save(p: Panel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, dates=p.dates, symbols=np.array(p.symbols), tbill=p.tbill, **p.m)


def load(path: Path, last_day: date) -> Panel:
    """Load the panel truncated at `last_day` (the vault boundary outside a final evaluation)."""
    z = np.load(path, allow_pickle=False)
    p = Panel(z["dates"], [str(s) for s in z["symbols"]], {k: z[k] for k in FIELDS}, z["tbill"])
    return p.slice(date(2000, 1, 1), last_day)


PAIR_TOL = 1.2      # prereg log 2026-10-03: perp and spot closes within a factor 1.2, else the pair is invalid


def pair_ok(p: Panel) -> np.ndarray:
    """T x N: both closes known and perp/spot within PAIR_TOL (same instrument on both legs)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(np.log(p.m["pc"] / p.m["sc"]))
    return np.nan_to_num(r, nan=np.inf) <= np.log(PAIR_TOL)


def mondays(dates: np.ndarray) -> np.ndarray:
    """Indices of Mondays (rebalance days) in `dates`."""
    wd = (dates.astype("datetime64[D]").view("int64") - 4) % 7      # 1970-01-01 was a Thursday
    return np.flatnonzero(wd == 0)


def as_date(d: np.datetime64) -> date:
    return date(1970, 1, 1) + timedelta(days=int(d.astype("datetime64[D]").view("int64")))
