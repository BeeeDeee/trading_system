"""Forward data for paper trading (G5): daily bars that did not exist when the lab's snapshots were taken.

Only framework code fetches them, in the ingest phase (network), never strategy code. Bars are stored
append-only per instrument under LAB_HOME/forward/<dataset>/<instrument>.parquet: a day is written once,
only after it closed (UTC), and never rewritten, so a later revision at the source cannot change history.
`extend()` appends the stored days to a historical DataView with the same semantics as the loaders.

Sources (catalog `forward_source`): `binance_public_api` for binance_spot_1d (api.binance.com) and
binance_perp_1d (fapi.binance.com, klines + funding). Datasets without a forward source cannot be paper traded
(the Sentinel parks such hypotheses, decision Q3).
"""

import json
import urllib.request
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from lab.framework import data
from lab.framework.data import DataView

SPOT = "https://api.binance.com/api/v3/klines"
PERP = "https://fapi.binance.com/fapi/v1/klines"
FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
SUPPORTED = {"binance_spot_1d", "binance_perp_1d"}


def _get(url: str) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "research-lab-forward/1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def store_dir(home: Path, dataset: str) -> Path:
    return home / "forward" / dataset


def stored(home: Path, dataset: str, instrument: str):
    import polars as pl
    f = store_dir(home, dataset) / f"{instrument}.parquet"
    return pl.read_parquet(f) if f.exists() else None


def fetch_days(dataset: str, instrument: str, start: date, end: date, get=_get) -> list[dict]:
    """Closed UTC days start..end (inclusive): open, close, quote volume (+ funding for perps)."""
    sym = instrument.removesuffix(data.PERP_SUFFIX)
    url = (PERP if dataset == "binance_perp_1d" else SPOT) + \
        f"?symbol={sym}&interval=1d&startTime={_ms(start)}&endTime={_ms(end + timedelta(days=1)) - 1}&limit=1000"
    rows = [{"date": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).date(), "open": float(k[1]),
             "close": float(k[4]), "quote_volume": float(k[7])} for k in get(url)]
    rows = [r for r in rows if start <= r["date"] <= end]
    if dataset == "binance_perp_1d" and rows:
        ev = get(f"{FUNDING}?symbol={sym}&startTime={_ms(start)}&endTime={_ms(end + timedelta(days=2))}&limit=1000")
        ts = [(int(e["fundingTime"]), float(e["fundingRate"])) for e in ev]
        for r in rows:
            d0 = _ms(r["date"])
            d1 = d0 + 86_400_000
            r["funding"] = sum(v for t, v in ts if d0 <= t < d1)          # [D, D+1): known at the close (signal)
            r["funding_paid"] = sum(v for t, v in ts if d0 < t <= d1)     # (D, D+1]: charged over day D
    return rows


def update(home: Path, dataset: str, instrument: str, after: date, today: date | None = None, get=_get) -> int:
    """Fetch and append the closed days after max(`after`, last stored day). Returns the number of new days."""
    import polars as pl
    if dataset not in SUPPORTED:
        raise ValueError(f"no forward source for {dataset}")
    today = today or datetime.now(timezone.utc).date()
    old = stored(home, dataset, instrument)
    start = max(after, old["date"].max() if old is not None and old.height else after) + timedelta(days=1)
    end = today - timedelta(days=1)                                       # only closed days
    if start > end:
        return 0
    rows = fetch_days(dataset, instrument, start, end, get)
    if not rows:
        return 0
    new = pl.DataFrame(rows)
    out = new if old is None else pl.concat([old, new.select(old.columns)], how="vertical")
    d = store_dir(home, dataset)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{instrument}.parquet.tmp"
    out.unique("date", keep="first").sort("date").write_parquet(tmp)
    tmp.replace(d / f"{instrument}.parquet")
    return len(rows)


def extend(view: DataView, home: Path, dataset: str) -> DataView:
    """Append the stored forward days of `dataset`'s instruments in `view` after its last date."""
    cols = [j for j, c in enumerate(view.asset_class)
            if (c == "crypto_perp") == (dataset == "binance_perp_1d") and c.startswith("crypto")]
    frames = {view.instruments[j]: stored(home, dataset, view.instruments[j]) for j in cols}
    days = sorted({d for f in frames.values() if f is not None for d in f["date"].to_list()
                   if np.datetime64(d) > view.dates[-1]})
    if not days:
        return view
    new_dates = np.array(days, dtype="datetime64[D]")
    T, N, K = len(view.dates), len(view.instruments), len(new_dates)
    add = {m: np.zeros((K, N), dtype=getattr(view, m).dtype) for m in data.MATRICES}
    add["close"][:] = np.nan
    add["dollar_volume"][:] = np.nan
    extras = {k: np.full((K, N), np.nan, dtype=v.dtype) for k, v in view.extras.items()}
    last_close = np.array(view.close[-1], dtype=float)
    for j in cols:
        f = frames[view.instruments[j]]
        if f is None:
            continue
        by = {d: r for d, r in zip(f["date"].to_list(), f.iter_rows(named=True))}
        prev = last_close[j]
        for i, d in enumerate(days):
            r = by.get(d)
            if r is None:
                continue
            add["ret_co"][i, j] = r["open"] / prev - 1.0 if np.isfinite(prev) and prev > 0 else 0.0
            add["ret_oc"][i, j] = r["close"] / r["open"] - 1.0
            add["tradable"][i, j] = add["listed"][i, j] = True
            add["close"][i, j], add["dollar_volume"][i, j] = r["close"], r["quote_volume"]
            for k in ("funding", "funding_paid"):
                if k in extras and k in r:
                    extras[k][i, j] = r[k]
            prev = r["close"]
    cat = lambda a, b: np.concatenate([a, b])  # noqa: E731
    return replace(view, dates=cat(view.dates, new_dates), **{m: cat(getattr(view, m), add[m]) for m in data.MATRICES},
                   universe=None if view.universe is None else cat(view.universe, add["listed"]),
                   extras={k: cat(v, extras[k]) for k, v in view.extras.items()},
                   series={k: cat(v, np.full(K, np.nan)) for k, v in view.series.items()},
                   # T-bill rates are not fetched forward: the last known daily cash return carries on
                   cash_ret=None if view.cash_ret is None else cat(view.cash_ret, np.full(K, view.cash_ret[-1])))
