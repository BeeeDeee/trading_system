"""Stock-attached datasets: SF1 fundamentals, insider filings (Form 4) and 13F holdings as daily (T, N) matrices
on the `sharadar_sep` stock columns (`data.extras[...]`), built point-in-time from the filing dates.

A card lists the dataset next to `sharadar_sep` with the `fields` it wants (at most FIELD_CAP in total, because
every field is a float32 matrix of about 70 MB and this machine has 3.9 GB):

    sf1_{art|arq}_{column}        SEC fundamentals as reported (ART trailing twelve months, ARQ quarter), e.g.
                                  sf1_art_roe, sf1_art_netinc, sf1_arq_revenue. Usable from the day after the
                                  filing date; a later filing of an older period never overrides a newer one;
                                  stale after MAX_AGE_SF1 days.
    ins_{buy|sell}_{value|n}_{91d|182d|365d}
                                  open-market insider purchases (code P) or sales (S) filed in the trailing
                                  window: USD value, or number of transactions (a row = one transaction, not one
                                  person). Usable from the day after the filing; NaN before the data starts
                                  (2008) plus one window, 0 where nobody filed.
    f13_{io|d_io|d_holders|d_breadth|putcall}
                                  institutional ownership (value held / market cap), its quarterly change, the
                                  change in the log number of holders, in breadth (holders / filers), and the
                                  put-call imbalance. A quarter ending D is usable from D + 46 days.

Ids are the stocks of the SEP view (`E<permaticker>`); tickers are mapped through the SEP ticker table (a
reused ticker can mix two companies in old data: a known bias).
"""

import re
from datetime import timedelta
from pathlib import Path

import numpy as np

FIELD_CAP = 6
MAX_AGE_SF1 = 400
MAX_AGE_F13 = 120 + 46
INSIDER_WINDOWS = {"91d": 91, "182d": 182, "365d": 365}
F13_FIELDS = ("io", "d_io", "d_holders", "d_breadth", "putcall")
SF1_COLUMNS = (
    "accoci assets assetsavg assetsc assetsnc assetturnover bvps capex cashneq cor consolinc currentratio de debt "
    "debtc debtnc deferredrev depamor deposits divyield dps ebit ebitda ebitdamargin ebt eps epsdil equity "
    "equityavg ev evebit evebitda fcf fcfps gp grossmargin intangibles intexp invcap invcapavg inventory "
    "investments investmentsc investmentsnc liabilities liabilitiesc liabilitiesnc marketcap ncf ncfbus ncfcommon "
    "ncfdebt ncfdiv ncff ncfi ncfinv ncfo ncfx netinc netinccmn netincdis netincnci netmargin opex opinc payables "
    "payoutratio pb pe pe1 ppnenet prefdivis price ps ps1 receivables retearn revenue rnd roa roe roic ros sbcomp "
    "sgna sharesbas shareswa shareswadil sps tangibles taxassets taxexp taxliabilities tbvps workingcapital").split()
DATASETS = ("sharadar_sf1", "sharadar_insiders", "sharadar_13f")
GRAMMAR = {
    "sharadar_sf1": "sf1_<art|arq>_<column>, e.g. sf1_art_roe (columns: " + ", ".join(SF1_COLUMNS[:12]) + ", ...)",
    "sharadar_insiders": "ins_<buy|sell>_<value|n>_<91d|182d|365d>, e.g. ins_buy_value_91d",
    "sharadar_13f": "f13_<" + "|".join(F13_FIELDS) + ">, e.g. f13_io",
}
_PATTERN = {
    "sharadar_sf1": re.compile(r"^sf1_(art|arq)_(\w+)$"),
    "sharadar_insiders": re.compile(r"^ins_(buy|sell)_(value|n)_(91d|182d|365d)$"),
    "sharadar_13f": re.compile(r"^f13_(\w+)$"),
}


def field_problem(dataset: str, field: str) -> str | None:
    m = _PATTERN[dataset].match(field)
    ok = bool(m) and (dataset != "sharadar_sf1" or m.group(2) in SF1_COLUMNS) \
        and (dataset != "sharadar_13f" or m.group(1) in F13_FIELDS)
    return None if ok else f"{field!r} is not a field of {dataset}; form: {GRAMMAR[dataset]}"


def requirement_problems(card: dict, catalog_entries: dict) -> list[str]:
    """Card-level problems of attached datasets (fields missing or invalid, too many, no stock universe)."""
    out, total = [], 0
    reqs = [r for r in card.get("data_requirements", []) if catalog_entries.get(r["dataset"], {}).get("loader") == "sep_attached"]
    if reqs and not any(r["dataset"] == "sharadar_sep" for r in card["data_requirements"]):
        out.append("sharadar_sf1 / sharadar_insiders / sharadar_13f attach to sharadar_sep: add it as a requirement "
                   "(universe kind liq_n or sp500)")
    for r in reqs:
        fields = r.get("fields") or []
        if not fields:
            out.append(f"{r['dataset']}: list the fields you need; form: {GRAMMAR[r['dataset']]}")
        out += [p for f in fields if (p := field_problem(r["dataset"], f))]
        total += len(fields)
    if total > FIELD_CAP:
        out.append(f"at most {FIELD_CAP} attached fields per card (RAM: 70 MB each), the card asks for {total}")
    return out


# ---------------------------------------------------------------------------- point-in-time building blocks

def asof_matrix(cols: np.ndarray, filed: np.ndarray, period: np.ndarray, value: np.ndarray, dates: np.ndarray,
                n_cols: int, max_age_days: int) -> np.ndarray:
    """(T, N) float32: for each column the value of the filing with the latest period among the filings with
    filing date < the row's date (visible from the next trading row), NaN when stale or unknown.
    A filing of an older period that arrives later (a correction) does not override a newer period."""
    T = len(dates)
    out = np.full((T, n_cols), np.nan, dtype=np.float32)
    ok = np.isfinite(value)
    cols, filed, period, value = cols[ok], filed[ok], period[ok], value[ok]
    order = np.lexsort((period, filed, cols))
    cols, filed, period, value = cols[order], filed[order], period[order], value[order]
    start = np.searchsorted(dates, filed, side="right")                    # first row with date > filing date
    stale = np.searchsorted(dates, filed + np.timedelta64(max_age_days, "D"), side="right")
    bounds = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        c, newest, keep = int(cols[a]), None, []
        for k in range(a, b):
            if newest is None or period[k] >= newest:
                newest = period[k]
                keep.append(k)
        for i, k in enumerate(keep):
            end = start[keep[i + 1]] if i + 1 < len(keep) else T
            end = min(end, stale[k])
            if start[k] < end:
                out[start[k]:end, c] = value[k]
    return out


def window_sum_matrix(cols: np.ndarray, filed: np.ndarray, value: np.ndarray, dates: np.ndarray, n_cols: int,
                      window_days: int, first_known: np.datetime64) -> np.ndarray:
    """(T, N) float32: sum of `value` of the events filed in the trailing `window_days` (events usable from the
    row after their filing date); 0 where nothing was filed; NaN before `first_known` (data start + window)."""
    T = len(dates)
    use = np.searchsorted(dates, filed, side="right")                      # first row that may see the event
    lo = np.searchsorted(dates, dates - np.timedelta64(window_days, "D"), side="right")
    out = np.empty((T, n_cols), dtype=np.float32)
    keep = use < T
    for a in range(0, n_cols, 400):                                        # column blocks keep float64 temporaries small
        b = min(a + 400, n_cols)
        sel = keep & (cols >= a) & (cols < b)
        daily = np.zeros((T + 1, b - a))
        np.add.at(daily, (use[sel] + 1, cols[sel] - a), value[sel])
        csum = np.cumsum(daily, axis=0)                                    # csum[t + 1] = events usable up to row t
        out[:, a:b] = (csum[np.arange(T) + 1] - csum[lo]).astype(np.float32)
    out[dates < first_known] = np.nan
    return out


# ---------------------------------------------------------------------------- loaders from the Sharadar parquet files

def _ticker_cols(src: Path, instruments: tuple[str, ...]):
    import polars as pl
    col_of = {int(n[1:]): j for j, n in enumerate(instruments) if n.startswith("E")}
    tick = (pl.read_parquet(src / "tickers.parquet").filter(pl.col("table") == "SEP")
            .select("ticker", pl.col("permaticker").cast(pl.Int64)).unique("ticker"))
    tick = tick.filter(pl.col("permaticker").is_in(list(col_of)))
    return tick.with_columns(col=pl.Series([col_of[p] for p in tick["permaticker"].to_list()], dtype=pl.Int64)
                             ).select("ticker", "col")


def attach(view, dataset: str, fields: list[str], src: Path):
    """Add the requested fields of an attached dataset to `view.extras` (view = the SEP view)."""
    from dataclasses import replace

    import polars as pl
    for f in fields:
        if (p := field_problem(dataset, f)):
            raise NotImplementedError(p)
    tick = _ticker_cols(src, view.instruments)
    dates, N = view.dates, view.shape[1]
    extras = dict(view.extras)
    if dataset == "sharadar_sf1":
        for dim in ("art", "arq"):
            want = [(f, m.group(2)) for f in fields if (m := _PATTERN[dataset].match(f)) and m.group(1) == dim]
            if not want:
                continue
            df = (pl.scan_parquet(src / "fundamentals.parquet").filter(pl.col("dimension") == dim.upper())
                  .select("ticker", pl.col("calendardate").alias("period"), pl.col("date").alias("filed"),
                          *sorted({c for _, c in want})).collect().join(tick, on="ticker"))
            for f, c in want:
                d = df.select("col", "filed", "period", pl.col(c).cast(pl.Float64).alias("v")).drop_nulls()
                extras[f] = asof_matrix(d["col"].to_numpy(), d["filed"].to_numpy().astype("datetime64[D]"),
                                        d["period"].to_numpy().astype("datetime64[D]"), d["v"].to_numpy(),
                                        dates, N, MAX_AGE_SF1)
    elif dataset == "sharadar_insiders":
        ev = (pl.scan_parquet(src / "insiders.parquet").filter(pl.col("transactioncode").is_in(["P", "S"]))
              .select("ticker", pl.col("date").alias("filed"), "transactioncode",
                      pl.coalesce(pl.col("transactionvalue"),
                                  pl.col("transactionshares") * pl.col("transactionpricepershare"))
                      .cast(pl.Float64).abs().alias("value")).collect().join(tick, on="ticker"))
        first = np.datetime64(ev["filed"].min(), "D")
        for f in fields:
            side, what, w = _PATTERN[dataset].match(f).groups()
            e = ev.filter(pl.col("transactioncode") == ("P" if side == "buy" else "S"))
            val = e["value"].fill_null(0.0).to_numpy() if what == "value" else np.ones(e.height)
            extras[f] = window_sum_matrix(e["col"].to_numpy(), e["filed"].to_numpy().astype("datetime64[D]"), val,
                                          dates, N, INSIDER_WINDOWS[w], first + np.timedelta64(INSIDER_WINDOWS[w], "D"))
    else:
        from qlab.research11.features import flow_table
        t = flow_table(src, tick)
        for f in fields:
            c = _PATTERN[dataset].match(f).group(1)
            d = t.select("col", "filed", pl.col(c).cast(pl.Float64).alias("v")).drop_nulls()
            # `filed` is quarter end + 45 days: usable from the 46th day; the quarter end is the period
            period = d["filed"].to_numpy().astype("datetime64[D]") - np.timedelta64(45, "D")
            extras[f] = asof_matrix(d["col"].to_numpy(), d["filed"].to_numpy().astype("datetime64[D]"), period,
                                    d["v"].to_numpy(), dates, N, MAX_AGE_F13)
    return replace(view, extras=extras)
