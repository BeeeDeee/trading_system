"""Synthetic market in Sharadar SEP semantics, with known ground truth.

Each security follows a one-factor model with overnight and intraday returns. Corporate actions are
applied the way they happen economically (splits and ex-dividend drops at the open) and are then
backward-adjusted exactly as Sharadar documents it. The generator also returns the true
total-return index, so the normalization and the engines can be tested against it.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from qlab.data.normalize import (ACQUISITION, BANKRUPTCY, PERFORMANCE, SPAC_LIQUIDATION, UNKNOWN,
                                 VOLUNTARY)
from qlab.data.schema import DELISTINGS_SCHEMA, PRICES_SCHEMA

SPLIT_RATIOS = (2.0, 3.0, 1.5, 0.5, 0.1)  # new shares per old share; < 1 is a reverse split


@dataclass(frozen=True)
class SyntheticMarket:
    calendar: list[date]
    prices: pl.DataFrame      # PRICES_SCHEMA
    delistings: pl.DataFrame  # DELISTINGS_SCHEMA; UNKNOWN delistings are deliberately absent
    truth: pl.DataFrame       # permaticker, date, tr_index, close_u, volume_u (price rows only)
    events: pl.DataFrame      # permaticker, date, event, value


def generate_market(n_assets: int = 40, start: str = "2010-01-01", end: str = "2013-12-31",
                    seed: int = 0, p_halt: float = 0.01, p_no_open: float = 0.005,
                    p_split: float = 0.002, p_delist: float = 0.4) -> SyntheticMarket:
    rng = np.random.default_rng(seed)
    days = np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + np.timedelta64(1, "D"))
    days = days[np.is_busday(days)]
    n_days = len(days)
    market_co = rng.normal(0.0001, 0.004, n_days)
    market_oc = rng.normal(0.0002, 0.008, n_days)

    price_rows, truth_rows, delist_rows, event_rows = [], [], [], []
    for i in range(n_assets):
        pt = 100_000 + i
        first = int(rng.integers(0, n_days // 2)) if rng.random() < 0.3 else 0
        delists = rng.random() < p_delist
        last = int(rng.integers(first + 60, n_days - 2)) if delists else n_days - 1
        beta, sigma = rng.uniform(0.5, 1.5), rng.uniform(0.01, 0.04)
        pays_dividend = rng.random() < 0.5
        halted = rng.random(n_days) < p_halt
        halted[[first, last]] = False

        u_prev, tr = rng.uniform(5, 100), 1.0
        rec = []  # (day, UO, H, L, U, vol_u, split, div_post, tr, halted)
        for t in range(first, last + 1):
            r_co = beta * market_co[t] + rng.normal(0, sigma * 0.5)
            r_oc = beta * market_oc[t] + rng.normal(0, sigma)
            event_day = t > first and not halted[t]
            s = rng.choice(SPLIT_RATIOS) if event_day and rng.random() < p_split else 1.0
            d_pre = 0.005 * u_prev if event_day and pays_dividend and (t - first) % 63 == 62 else 0.0
            uo_pre = u_prev * np.exp(r_co) - d_pre
            u_pre = uo_pre * np.exp(r_oc)
            tr *= (u_pre + d_pre) / u_prev
            uo, u = uo_pre / s, u_pre / s
            hi = max(uo, u) * (1 + abs(rng.normal(0, sigma / 2)))
            lo = min(uo, u) * (1 - abs(rng.normal(0, sigma / 2)))
            rec.append((t, uo, hi, lo, u, float(rng.integers(10_000, 5_000_000)), s, d_pre / s, tr,
                        halted[t]))
            u_prev = u

        arr = np.array(rec, dtype=float)
        rows = arr[arr[:, 9] == 0]  # drop halted days
        t_idx = rows[:, 0].astype(int)
        uo, hi, lo, u, vol_u, split, div, tr_idx = (rows[:, k] for k in range(1, 9))
        # Split factor applying to day t = product of splits strictly after t (backward adjustment).
        s_after = np.cumprod(split[::-1])[::-1] / split
        div_ratio = np.where(div > 0, (u + div) / u, 1.0)
        a_after = np.cumprod(div_ratio[::-1])[::-1] / div_ratio
        close = u / s_after
        opn = np.where(rng.random(len(rows)) < p_no_open, 0.0, uo / s_after)
        opn[0] = uo[0] / s_after[0]
        for k, t in enumerate(t_idx):
            price_rows.append((pt, days[t], opn[k], hi[k] / s_after[k], lo[k] / s_after[k],
                               close[k], vol_u[k] * s_after[k], close[k] / a_after[k], u[k]))
            truth_rows.append((pt, days[t], tr_idx[k], u[k], vol_u[k]))
            if split[k] != 1.0:
                event_rows.append((pt, days[t], "split", split[k]))
            if div[k] > 0:
                event_rows.append((pt, days[t], "dividend", div[k]))

        if delists:
            kind = rng.choice([ACQUISITION, ACQUISITION, BANKRUPTCY, SPAC_LIQUIDATION, PERFORMANCE,
                               VOLUNTARY, UNKNOWN])
            cash = u[-1] * rng.uniform(1.1, 1.4) if kind == ACQUISITION and rng.random() < 0.8 else None
            if kind != UNKNOWN:
                delist_rows.append((pt, days[t_idx[-1]], kind, cash))
            event_rows.append((pt, days[t_idx[-1]], f"delisted:{kind}", cash or float("nan")))

    to_date = lambda d: d.astype("datetime64[D]").astype(date)  # noqa: E731
    prices = pl.DataFrame([(r[0], to_date(r[1]), *r[2:]) for r in price_rows],
                          schema=PRICES_SCHEMA, orient="row")
    truth = pl.DataFrame([(r[0], to_date(r[1]), *r[2:]) for r in truth_rows], orient="row",
                         schema={"permaticker": pl.Int64, "date": pl.Date, "tr_index": pl.Float64,
                                 "close_u": pl.Float64, "volume_u": pl.Float64})
    delistings = pl.DataFrame([(r[0], to_date(r[1]), r[2], r[3]) for r in delist_rows],
                              schema=DELISTINGS_SCHEMA, orient="row")
    events = pl.DataFrame([(r[0], to_date(r[1]), r[2], r[3]) for r in event_rows], orient="row",
                          schema={"permaticker": pl.Int64, "date": pl.Date, "event": pl.Utf8,
                                  "value": pl.Float64})
    return SyntheticMarket([to_date(d) for d in days], prices, delistings, truth, events)
