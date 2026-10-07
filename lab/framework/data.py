"""DataView: the only way a strategy sees data, and the adapters that build it from the catalog.

A DataView is a dense date x instrument block on one calendar (the union of the calendars of its datasets),
plus named per-instrument matrices (`extras`) and named time series (`series`). Semantics follow
`qlab.data.panel.Panel`:
- `ret_co` previous close -> open, `ret_oc` open -> close, total returns, 0 where there is no trading,
- `tradable` an order can fill at the open, `listed` a row exists, `delisting` the position is paid out at
  this open (the terminal return is in `ret_co`),
- point-in-time: row t contains only what is known after the last close of day t. Series whose clock is
  `next_morning` (FRED) are lagged by one row when the view is built.

Merging an equity calendar (Mon-Fri) with crypto (every day) is exact for returns: a weekend row of an ETF
has zero returns and is not tradable, and Monday's `ret_co` still carries Friday close -> Monday open. A
decision after the close of day t uses the US close of t and the crypto close of t (00:00 UTC t+1), and both
precede every open of day t+1.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import numpy as np

MATRICES = ("ret_co", "ret_oc", "tradable", "listed", "delisting", "close", "dollar_volume")
DATA_ROOT = Path(os.environ.get("LAB_DATA_ROOT", "/home/kapo/ccode/strategy_backtester_2026_sep/data"))
SHARADAR = "sharadar_2026-09-25"
BINANCE = "binance_2026-10-03"


@dataclass(frozen=True)
class DataView:
    dates: np.ndarray                  # datetime64[D] (T,)
    instruments: tuple[str, ...]       # (N,) e.g. "SPY", "BTCUSDT"
    asset_class: tuple[str, ...]       # (N,) catalog asset class of each instrument
    ret_co: np.ndarray
    ret_oc: np.ndarray
    tradable: np.ndarray
    listed: np.ndarray
    delisting: np.ndarray
    close: np.ndarray                  # unadjusted close, NaN without a price (filters only)
    dollar_volume: np.ndarray          # NaN without a price
    universe: np.ndarray | None = None  # (T, N) bool point-in-time membership; None = all listed
    extras: dict[str, np.ndarray] = field(default_factory=dict)   # (T, N)
    series: dict[str, np.ndarray] = field(default_factory=dict)   # (T,)
    cash_ret: np.ndarray | None = None  # (T,) cash return earned over day t

    @property
    def shape(self) -> tuple[int, int]:
        return self.ret_co.shape

    def col(self, name: str) -> int:
        return self.instruments.index(name)

    def slice(self, end: int) -> "DataView":
        """Rows [0, end): the world as seen after the close of day end-1."""
        cut = lambda a: None if a is None else a[:end]  # noqa: E731
        return replace(self, dates=self.dates[:end], **{m: getattr(self, m)[:end] for m in MATRICES},
                       universe=cut(self.universe), extras={k: v[:end] for k, v in self.extras.items()},
                       series={k: v[:end] for k, v in self.series.items()}, cash_ret=cut(self.cash_ret))

    def between(self, start: np.datetime64 | None, end: np.datetime64 | None) -> "DataView":
        lo = 0 if start is None else int(np.searchsorted(self.dates, start, side="left"))
        hi = len(self.dates) if end is None else int(np.searchsorted(self.dates, end, side="right"))
        sub = lambda a: None if a is None else a[lo:hi]  # noqa: E731
        return replace(self, dates=self.dates[lo:hi], **{m: getattr(self, m)[lo:hi] for m in MATRICES},
                       universe=sub(self.universe), extras={k: v[lo:hi] for k, v in self.extras.items()},
                       series={k: v[lo:hi] for k, v in self.series.items()}, cash_ret=sub(self.cash_ret))

    def columns(self, names: list[str]) -> "DataView":
        """The same view restricted to `names` (a strategy sees only its own instruments)."""
        if tuple(names) == self.instruments:
            return self                           # no copy of ~10 matrices of 70 MB each (single stocks)
        idx = [self.col(n) for n in names]
        return replace(self, instruments=tuple(names), asset_class=tuple(self.asset_class[i] for i in idx),
                       **{m: getattr(self, m)[:, idx] for m in MATRICES},
                       universe=None if self.universe is None else self.universe[:, idx],
                       extras={k: v[:, idx] for k, v in self.extras.items()})


def perturb_after(view: DataView, t: int, rng: np.random.Generator) -> DataView:
    """Copy of `view` with every row after t replaced by noise (row t kept). For the look-ahead test."""
    def noise(a):
        a = np.array(a, copy=True)
        if a.dtype == bool:
            a[t + 1:] = rng.random(a[t + 1:].shape) < 0.5
        else:
            tail = a[t + 1:]
            a[t + 1:] = np.where(np.isnan(tail), tail, tail * rng.uniform(0.5, 1.5, tail.shape)
                                 + rng.normal(0, 0.01, tail.shape))
        return a
    return replace(view, **{m: noise(getattr(view, m)) for m in MATRICES},
                   universe=None if view.universe is None else noise(view.universe),
                   extras={k: noise(v) for k, v in view.extras.items()},
                   series={k: noise(v) for k, v in view.series.items()},
                   cash_ret=None if view.cash_ret is None else noise(view.cash_ret))


def merge(views: list[DataView]) -> DataView:
    """Union calendar and concatenated instruments. Missing rows: zero returns, not tradable/listed."""
    if len(views) == 1:
        return views[0]
    dates = np.unique(np.concatenate([v.dates for v in views]))
    T = len(dates)
    # a float32 view (single stocks, RAM) keeps the merged floats in float32; otherwise float64
    ftype = np.float32 if any(v.ret_co.dtype == np.float32 for v in views) else np.float64
    out = {m: [] for m in MATRICES}
    universe, extras, series, cash = [], {}, {}, None
    for v in views:
        rows = np.searchsorted(dates, v.dates)
        N = v.shape[1]
        for m in MATRICES:
            a = getattr(v, m)
            fill = False if a.dtype == bool else (np.nan if m in ("close", "dollar_volume") else 0.0)
            full = np.full((T, N), fill, dtype=bool if a.dtype == bool else ftype)
            full[rows] = a
            out[m].append(full)
        u = np.zeros((T, N), dtype=bool)
        u[rows] = v.listed if v.universe is None else v.universe
        universe.append(u)
        for k, a in v.extras.items():
            full = np.full((T, N), np.nan, dtype=ftype)
            full[rows] = a
            extras.setdefault(k, []).append((len(out["ret_co"]) - 1, full))
        for k, s in v.series.items():
            full = np.full(T, np.nan)
            full[rows] = s
            series[k] = _ffill(full)
        if v.cash_ret is not None and cash is None:
            cash = v.cash_ret  # recomputed on the union calendar by the caller (load_view)
    widths = [v.shape[1] for v in views]
    ext = {}
    for k, parts in extras.items():
        mats = [np.full((T, w), np.nan, dtype=ftype) for w in widths]
        for i, full in parts:
            mats[i] = full
        ext[k] = np.hstack(mats)
    return DataView(dates, tuple(i for v in views for i in v.instruments),
                    tuple(c for v in views for c in v.asset_class),
                    **{m: np.hstack(out[m]) for m in MATRICES}, universe=np.hstack(universe),
                    extras=ext, series=series, cash_ret=None)


def _ffill(a: np.ndarray) -> np.ndarray:
    idx = np.where(np.isnan(a), 0, np.arange(len(a)))
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    out[np.isnan(a) & (np.arange(len(a)) < np.argmax(~np.isnan(a)))] = np.nan
    return out


# ---------------------------------------------------------------------------- adapters

def _cache_dir() -> Path:
    from lab.framework.paths import default_paths
    return default_paths().home / "cache"


SFP_LOADER_VERSION = 2      # 2: one-day closeadj spikes dropped (decision log 2026-10-05)
SPIKE_LOG = 0.4             # |log move| of the spike day
SPIKE_REVERT = 0.1          # |log move of the spike day + log move of the next day|


def drop_spikes(prices, key: str = "permaticker"):
    """Drop vendor errors: a day whose adjusted close moves by more than SPIKE_LOG (log) and returns within
    SPIKE_REVERT the next day (e.g. SSO 2014-06-24 x3.95, a split adjustment applied to one day only).
    The dropped day becomes a day without a bar (not tradable, zero return), and the next day's return runs
    from the last good close. The flag of day t uses the price of t+1: this repairs the data the strategy
    sees, it is not a signal, and no strategy can trade the spike that never happened in the market."""
    import polars as pl
    lr = (pl.col("closeadj") / pl.col("closeadj").shift(1).over(key)).log()
    return (prices.sort(key, "date").with_columns(_lr=lr)
            .with_columns(_spike=(pl.col("_lr").abs() > SPIKE_LOG)
                          & ((pl.col("_lr") + pl.col("_lr").shift(-1).over(key)).abs() < SPIKE_REVERT))
            .filter(~pl.col("_spike").fill_null(False)).drop("_lr", "_spike"))


def sharadar_sfp(tickers: list[str]) -> DataView:
    """US ETFs from Sharadar SFP (total return from closeadj, one-day spikes dropped), built on demand and
    cached per ticker set and loader version."""
    import polars as pl

    from qlab.data.normalize import normalize_prices
    from qlab.data.panel import load_panel, panel_from_bars, save_panel
    from qlab.data.schema import DELISTINGS_SCHEMA, PRICES_SCHEMA

    key = hashlib.sha256((",".join(sorted(tickers)) + f"|v{SFP_LOADER_VERSION}").encode()).hexdigest()[:12]
    out = _cache_dir() / f"sfp_{key}"
    if not (out / "tickers.json").exists():
        src = DATA_ROOT / "parquet" / SHARADAR
        calendar = (pl.scan_parquet(src / "funds.parquet").filter(pl.col("ticker") == "SPY")
                    .select("date").collect()["date"].sort().to_list())
        perm = (pl.read_parquet(src / "tickers.parquet")
                .filter((pl.col("table") == "SFP") & pl.col("ticker").is_in(tickers))
                .select("ticker", pl.col("permaticker").cast(pl.Int64)).unique("ticker"))
        missing = sorted(set(tickers) - set(perm["ticker"].to_list()))
        if missing:
            raise KeyError(f"not in Sharadar SFP: {missing}")
        prices = drop_spikes(pl.scan_parquet(src / "funds.parquet").filter(pl.col("ticker").is_in(tickers))
                             .collect().join(perm, on="ticker")
                             .select([pl.col(c).cast(t) for c, t in PRICES_SCHEMA.items()]))
        panel = panel_from_bars(normalize_prices(prices, pl.DataFrame(schema=DELISTINGS_SCHEMA), calendar),
                                calendar)
        save_panel(panel, out)
        by_perm = dict(zip(perm["permaticker"].to_list(), perm["ticker"].to_list()))
        (out / "tickers.json").write_text(json.dumps([by_perm[int(a)] for a in panel.assets]))
    p, _ = load_panel(out, mmap=False)
    names = json.loads((out / "tickers.json").read_text())
    order = [names.index(t) for t in tickers]
    return DataView(p.dates, tuple(tickers), ("us_etf",) * len(tickers),
                    *(np.asarray(getattr(p, m))[:, order] for m in ("ret_co", "ret_oc", "tradable", "listed",
                                                                     "delisting", "close_u", "dollar_volume")))


def binance_spot(symbols: list[str] | None = None) -> DataView:
    """Binance spot daily (research 9 panel). `symbols=None` = all 663 pairs (for top-N universes)."""
    from qlab.data.panel import load_panel

    d = DATA_ROOT / "derived" / BINANCE / "panel_r9"
    p, _ = load_panel(d)
    names = (d / "symbols.txt").read_text().split()
    idx = list(range(len(names))) if symbols is None else [names.index(s) for s in symbols]
    sel = lambda m: np.asarray(getattr(p, m)[:, idx])  # noqa: E731
    return DataView(np.asarray(p.dates), tuple(names[i] for i in idx), ("crypto_spot",) * len(idx),
                    sel("ret_co"), sel("ret_oc"), sel("tradable"), sel("listed"), sel("delisting"),
                    sel("close_u"), sel("dollar_volume"))


def tbill_rates() -> tuple[np.ndarray, np.ndarray]:
    import polars as pl
    rf = pl.read_csv(DATA_ROOT / "raw" / BINANCE / "fred_DTB3.csv", null_values=".",
                     schema_overrides={"DTB3": pl.Float64})
    return rf["observation_date"].str.to_date().to_numpy(), rf["DTB3"].to_numpy()


def with_cash(view: DataView, rates: tuple[np.ndarray, np.ndarray] | None) -> DataView:
    from qlab.engine.costs import cash_returns
    if rates is None:
        return replace(view, cash_ret=np.zeros(len(view.dates)))
    return replace(view, cash_ret=cash_returns(view.dates, *rates))


def crypto_top_n(view: DataView, n: int, window: int = 30) -> np.ndarray:
    """(T, N) membership: the n listed pairs with the highest trailing median quote volume up to day t."""
    from numpy.lib.stride_tricks import sliding_window_view
    dv = np.where(view.listed, np.nan_to_num(view.dollar_volume), 0.0)
    pad = np.vstack([np.zeros((window - 1, dv.shape[1])), dv])
    med = np.median(sliding_window_view(pad, window, axis=0), axis=-1)
    rank = (-med).argsort(axis=1).argsort(axis=1)
    return (rank < n) & view.listed & (med > 0)


# ---------------------------------------------------------------------------- synthetic market (canaries)

SYNTHETIC_DATASET = {
    "id": "synthetic_market", "loader": True, "asset_class": "synthetic", "instruments": "S00..S29 and MKT (equal-weight index)",
    "frequency": "1d", "range": ["2000-01-03", "2025-12-31"], "clock": "us_close", "source": "lab.framework.data",
    "quality": "generated, known truth", "holdout_from": "2020-01-01", "known_biases": ["synthetic"],
    "forward_source": "synthetic"}
EDGE_ASSETS = ("S00", "S01", "S02", "S03", "S04")


def synthetic(seed: int = 7, edge: float = 0.004, n_assets: int = 30) -> DataView:
    """One-factor market with a planted edge: `series['signal'][t]` (public at the close of t) adds
    `edge * signal[t]` to the next day's intraday return of EDGE_ASSETS. Five assets delist."""
    rng = np.random.default_rng(seed)
    days = np.arange(np.datetime64("2000-01-03"), np.datetime64("2026-01-01"))
    days = days[np.is_busday(days)]
    T, N = len(days), n_assets
    mkt_co, mkt_oc = rng.normal(0.0001, 0.004, T), rng.normal(0.0002, 0.008, T)
    beta = rng.uniform(0.6, 1.4, N)
    ret_co = beta * mkt_co[:, None] + rng.normal(0, 0.005, (T, N))
    ret_oc = beta * mkt_oc[:, None] + rng.normal(0, 0.012, (T, N))
    signal = rng.normal(0, 1, T)
    names = [f"S{i:02d}" for i in range(N)]
    for a in EDGE_ASSETS:
        ret_oc[1:, names.index(a)] += edge * signal[:-1]
    listed = np.ones((T, N), dtype=bool)
    delisting = np.zeros((T, N), dtype=bool)
    for j in range(N - 5, N):                        # S25..S29 delist (performance, -30 %)
        last = int(rng.integers(T // 4, 3 * T // 4))
        delisting[last + 1, j], ret_co[last + 1, j] = True, -0.30
        ret_co[last + 2:, j] = ret_oc[last + 1:, j] = 0.0
        listed[last + 2:, j] = False
    tradable = listed & ~delisting
    mkt = np.where(listed, (1 + ret_co) * (1 + ret_oc) - 1, np.nan)
    mkt_ret = np.nanmean(np.where(delisting, np.nan, mkt), axis=1)
    ret_co = np.hstack([ret_co, np.zeros((T, 1))])
    ret_oc = np.hstack([ret_oc, mkt_ret[:, None]])    # MKT: daily rebalanced index, traded at the open
    add = lambda a, v: np.hstack([a, np.full((T, 1), v)])  # noqa: E731
    close = np.cumprod((1 + ret_co) * (1 + ret_oc), axis=0) * 50
    return DataView(days, tuple(names + ["MKT"]), ("synthetic",) * (N + 1), ret_co, ret_oc,
                    add(tradable, True), add(listed, True), add(delisting, False), close,
                    np.full((T, N + 1), 5e7), series={"signal": signal})


# ---------------------------------------------------------------------------- generic (Archivist ingest)

CLOCK_LAG_DAYS = {"us_close": 0, "crypto": 0, "next_morning": 1}


def generic_series(home: Path, dataset: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """key -> (observation dates, values) of an ingested dataset (dev and holdout files)."""
    import polars as pl
    d = home / "data" / dataset
    df = pl.concat([pl.read_parquet(d / f) for f in ("dev.parquet", "holdout.parquet") if (d / f).exists()])
    out = {}
    for (key,), g in df.sort("date").group_by("key", maintain_order=True):
        out[key] = (g["date"].to_numpy().astype("datetime64[D]"), g["value"].to_numpy().astype(float))
    return out


def attach_generic(view: DataView, home: Path, dataset: str, entry: dict) -> DataView:
    """Add `series["<dataset>.<key>"]` for every key of the dataset (the catalog's `fields`; a card's
    `fields` are the Scout's guess from before the ingest and are not used to filter): on row t the last observation known after the close of day t.
    An observation of day d is known at the close of d (clock us_close, crypto) or of d+1 (next_morning,
    published the next morning). Before the first known observation: NaN."""
    lag = np.timedelta64(CLOCK_LAG_DAYS[entry["clock"]], "D")
    series = dict(view.series)
    for key, (d, v) in generic_series(home, dataset).items():
        idx = np.searchsorted(d + lag, view.dates, side="right") - 1
        series[f"{dataset}.{key}"] = np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan)
    return replace(view, series=series)


# ---------------------------------------------------------------------------- US single stocks (Sharadar SEP)

SEP_PANEL = "panel_liq1000"
MAX_LIQ_N = 500          # RAM: the union of LIQ-500 and S&P 500 members since 1998 is ~2 600 columns


def sharadar_sep(universe: dict) -> DataView:
    """US common stocks from the qlab LIQ1000 panel (delistings with terminal returns, PIT universes).

    Ticker-blind: instruments are opaque ids `E<permaticker>`, so a strategy cannot pick a stock it remembers.
    Columns are every stock that was ever in LIQ-n or the S&P 500 (the G2 alternative universe), the
    `universe` mask is the card's point-in-time membership: kind `liq_n` (n <= 500: liquidity rank <= n among
    the LIQ1000 stocks, price and listing filters of qlab) or `sp500`. extras: `liq_rank` (PIT liquidity rank,
    1 = most liquid; the cost tiers use it), `alt_universe` (1.0 where the other universe has the stock)."""
    from qlab.data.panel import load_panel

    kind, n = universe["kind"], int(universe.get("n") or 0)
    if kind == "liq_n" and not 1 <= n <= MAX_LIQ_N:
        raise NotImplementedError(f"liq_n with n={n}: the loader supports 1..{MAX_LIQ_N} (RAM of this machine)")
    if kind not in ("liq_n", "sp500"):
        raise NotImplementedError(f"universe kind {kind!r} is not available for sharadar_sep")
    n = n or MAX_LIQ_N
    d = DATA_ROOT / "derived" / SHARADAR / SEP_PANEL
    p, ex = load_panel(d, mmap=True)
    special = set(json.loads((d / "special_assets.json").read_text()).values())
    T, N = len(p.dates), len(p.assets)

    def mask(j0, j1):
        rank = np.asarray(ex["liq_rank"][:, j0:j1])
        liq = np.asarray(ex["in_liq1000"][:, j0:j1]).astype(bool) & (rank <= n)
        return liq, np.asarray(ex["in_sp500"][:, j0:j1]).astype(bool)

    keep = np.zeros(N, dtype=bool)
    for j0 in range(0, N, 500):                     # column chunks keep the peak small
        liq, sp = mask(j0, j0 + 500)
        keep[j0:j0 + 500] = (liq | sp).any(axis=0)
    keep &= ~np.isin(np.asarray(p.assets), list(special))
    cols = np.flatnonzero(keep)
    liq, sp = mask(0, N)
    liq, sp = liq[:, cols], sp[:, cols]
    main, alt = (liq, sp) if kind == "liq_n" else (sp, liq)
    sel = lambda a: np.asarray(a[:, cols])  # noqa: E731
    f32 = lambda a: np.asarray(a[:, cols], dtype=np.float32)  # noqa: E731  (RAM: half of float64)
    return DataView(np.asarray(p.dates), tuple(f"E{int(a)}" for a in np.asarray(p.assets)[cols]),
                    ("us_equity",) * len(cols), f32(p.ret_co), f32(p.ret_oc), sel(p.tradable), sel(p.listed),
                    sel(p.delisting), f32(p.close_u), f32(p.dollar_volume), universe=main,
                    extras={"liq_rank": f32(ex["liq_rank"]), "alt_universe": alt.astype(np.float32)})


# ---------------------------------------------------------------------------- Binance USD-M perpetuals

PERP_PANEL = "r8_panel.npz"
PERP_SUFFIX = ".P"
PERP_DELIST_RETURN = -0.02     # same policy as spot: a contract that stops trading is closed at last close -2 %


def binance_perp(symbols: list[str] | None = None) -> DataView:
    """Binance USD-M perpetuals (research 8 panel: perps with a spot pair, 2019-12 .. 2026-09).

    Instruments are `<SYMBOL>.P` (e.g. `BTCUSDT.P`), so they never collide with spot pairs. extras:
    `funding` = sum of the funding rates of day t (events in [t 00:00, t+1 00:00), known after the close of
    t: usable as a signal), `funding_paid` = funding charged to a position held over day t (events in
    (t 00:00, t+1 00:00]); the engine charges it to longs and pays it to shorts. `basis` = perp close / spot
    close - 1 (spot in perp units)."""
    z = np.load(DATA_ROOT / "derived" / BINANCE / PERP_PANEL, allow_pickle=False)
    names = [f"{s}{PERP_SUFFIX}" for s in z["symbols"]]
    idx = list(range(len(names))) if symbols is None else [names.index(s) for s in symbols]
    po, pc = z["po"][:, idx], z["pc"][:, idx]
    T, N = pc.shape
    listed = np.isfinite(pc)
    prev = np.vstack([np.full((1, N), np.nan), pc[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        ret_co = np.where(listed & np.isfinite(prev) & np.isfinite(po), po / prev - 1.0, 0.0)
        ret_oc = np.where(listed & np.isfinite(po), pc / po - 1.0, 0.0)
        basis = z["pc"][:, idx] / z["sc"][:, idx] - 1.0
    tradable = listed & np.isfinite(po)
    delisting = np.zeros((T, N), dtype=bool)
    last = np.where(listed.any(axis=0), T - 1 - np.argmax(listed[::-1], axis=0), -1)
    for j in np.flatnonzero((last >= 0) & (last < T - 8)):    # stopped trading more than a week before the end
        delisting[last[j] + 1, j], ret_co[last[j] + 1, j] = True, PERP_DELIST_RETURN
    return DataView(np.asarray(z["dates"]), tuple(names[i] for i in idx), ("crypto_perp",) * N, ret_co, ret_oc,
                    tradable, listed, delisting, pc, z["qv"][:, idx],
                    extras={"funding": np.where(listed, np.nan_to_num(z["fund_sig"][:, idx]), np.nan),
                            "funding_paid": np.where(listed, np.nan_to_num(z["fund_hold"][:, idx]), 0.0),
                            "basis": np.where(listed, basis, np.nan)})


def load(dataset: str, instruments: list[str] | None) -> DataView:
    match dataset:
        case "binance_perp_1d":
            return binance_perp(instruments)
        case "sharadar_sfp":
            return sharadar_sfp(instruments)
        case "binance_spot_1d":
            return binance_spot(instruments)
        case "synthetic_market":
            v = synthetic()
            return v if instruments is None else v.columns(instruments)
    raise NotImplementedError(f"no adapter for dataset {dataset!r} yet")


def as_date(d) -> np.datetime64:
    return np.datetime64(d if isinstance(d, (str, date)) else str(d), "D")
