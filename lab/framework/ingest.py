"""Ingest of a new dataset: run the Archivist's fetcher, validate, split dev/holdout, add the catalog entry.

The Archivist writes `sources/<id>.py` and a draft catalog entry; it never decides the holdout boundary, the
date range, the location or the loader. This module does, deterministically:

    fetch(get) -> rows          the fetcher, statically scanned, run with a framework HTTP client (https only,
                                request and byte caps, every URL logged) and a wall-clock limit
    validate(rows)              dates, duplicates, non-finite values, gaps, outliers, history length
    boundary(entry, first, last)  holdout start: the asset class's lab-wide boundary, else 70 % of the history
    store                       LAB_HOME/data/<id>/{dev,holdout}.parquet + report.json (separate files, so
                                step 4 can give the holdout file other permissions)
    catalog.add                 entry with `loader: generic`

Fetcher contract (`lab/data/sources/<id>.py`):

    def fetch(get):             # get(url, params=None, headers=None) -> bytes
        return [("2021-03-24", "BTC", 81.2), ...]    # (date YYYY-MM-DD, series key, value)

Only daily data for now. Each key becomes `series["<id>.<key>"]` in a strategy's DataView, aligned by the
dataset's clock (see `data.attach_generic`).
"""

import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from lab.framework import catalog, strategy

FETCHER_IMPORTS = {"json", "csv", "io", "math", "datetime", "re", "statistics", "gzip", "zipfile", "itertools",
                   "functools", "typing", "dataclasses", "__future__"}
CLOCKS = {"us_close", "crypto", "next_morning"}
DRAFT_FIELDS = ("id", "asset_class", "instruments", "frequency", "clock", "source", "known_biases", "forward_source")
# Lab-wide holdout boundaries by asset class (PLAN §6.1, decision Q6); other classes use the 70 % rule.
CLASS_HOLDOUT = {"us_equity": "2021-01-01", "us_etf": "2021-01-01", "macro": "2021-01-01", "fx": "2021-01-01",
                 "crypto_spot": "2023-01-01", "crypto_perp": "2023-01-01"}
MIN_DEV_DAYS = 365
MIN_HOLDOUT_DAYS = 730
MAX_REQUESTS = 500
MAX_BYTES = 200 * 1024 * 1024
REQUEST_TIMEOUT_S = 30
FETCH_LIMIT_S = 900
OUTLIER_Z = 10.0


class IngestError(ValueError):
    pass


@dataclass
class Http:
    """The only network access a fetcher gets."""
    urls: list[str] = field(default_factory=list)
    bytes: int = 0

    def get(self, url: str, params: dict | None = None, headers: dict | None = None) -> bytes:
        if not str(url).startswith("https://"):
            raise IngestError(f"only https URLs are allowed: {url}")
        if len(self.urls) >= MAX_REQUESTS:
            raise IngestError(f"more than {MAX_REQUESTS} requests")
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "research-lab-ingest/1", **(headers or {})})
        self.urls.append(url)
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
            body = r.read(MAX_BYTES - self.bytes + 1)
        self.bytes += len(body)
        if self.bytes > MAX_BYTES:
            raise IngestError(f"download above {MAX_BYTES // 2**20} MB")
        return body


def scan_fetcher(source: str) -> list[str]:
    return strategy.scan(source, set(), set(), imports=FETCHER_IMPORTS, check_dates=False)


def draft_errors(entry: dict, dataset: str) -> list[str]:
    errors = [f"catalog entry: missing {f!r}" for f in DRAFT_FIELDS if f not in entry]
    if entry.get("id") != dataset:
        errors.append(f"catalog entry id {entry.get('id')!r} must equal the dataset {dataset!r}")
    if entry.get("frequency") != "1d":
        errors.append("only daily data (frequency 1d) can be ingested so far")
    if entry.get("clock") not in CLOCKS:
        errors.append(f"clock must be one of {sorted(CLOCKS)} (when an observation of day d becomes known)")
    if not entry.get("known_biases"):
        errors.append("known_biases must list at least one bias")
    return errors


def fetch(source_path: Path, http: Http | None = None) -> tuple[list, Http]:
    problems = scan_fetcher(source_path.read_text())
    if problems:
        raise IngestError("fetcher breaks the static rules: " + "; ".join(problems))
    module = strategy.load(source_path, entry="fetch")
    http = http or Http()
    with strategy.time_limit(FETCH_LIMIT_S):
        rows = module.fetch(http.get)
    return list(rows), http


def validate(rows: list, today: date | None = None) -> tuple[dict, dict]:
    """-> (data {key: (dates datetime64[D], values float)}, report). Raises IngestError on a hard error."""
    today = today or date.today()
    errors, by_key = [], {}
    for i, row in enumerate(rows):
        try:
            d, k, v = row
            dd = date.fromisoformat(str(d)[:10])
            v = float(v)
        except (TypeError, ValueError) as e:
            errors.append(f"row {i}: {row!r:.80} is not (YYYY-MM-DD, key, number): {e}")
            if len(errors) > 20:
                break
            continue
        if dd > today:
            errors.append(f"row {i}: date {dd} is in the future")
        by_key.setdefault(str(k), []).append((dd, v))
    if not by_key and not errors:
        errors.append("no rows")
    data, keys = {}, {}
    for k, obs in by_key.items():
        obs.sort()
        ds = np.array([o[0] for o in obs], dtype="datetime64[D]")
        vs = np.array([o[1] for o in obs], dtype=float)
        dup = int((ds[1:] == ds[:-1]).sum())
        bad = int((~np.isfinite(vs)).sum())
        if dup:
            errors.append(f"{k}: {dup} duplicate dates")
        if bad > 0.01 * len(vs):
            errors.append(f"{k}: {bad} non-finite values (> 1 %)")
        ok = np.isfinite(vs)
        ds, vs = ds[ok], vs[ok]
        if len(ds) < 2:
            errors.append(f"{k}: fewer than 2 observations")
            continue
        gaps = np.diff(ds).astype(int)
        diff = np.diff(vs)
        mad = np.median(np.abs(diff - np.median(diff))) * 1.4826
        outliers = int((np.abs(diff - np.median(diff)) > OUTLIER_Z * mad).sum()) if mad > 0 else 0
        keys[k] = {"n": int(len(ds)), "first": str(ds[0]), "last": str(ds[-1]), "max_gap_days": int(gaps.max()),
                   "gaps_over_7_days": int((gaps > 7).sum()), "dropped_non_finite": bad,
                   "jumps_over_10_robust_sd": outliers, "constant": bool(np.all(vs == vs[0]))}
        data[k] = (ds, vs)
    if errors:
        raise IngestError("; ".join(errors[:20]))
    return data, {"keys": keys, "n_rows": sum(v["n"] for v in keys.values())}


def boundary(asset_class: str, first: date, last: date) -> date:
    fixed = CLASS_HOLDOUT.get(asset_class)
    if fixed:
        b = date.fromisoformat(fixed)
        if (b - first).days >= MIN_DEV_DAYS and (last - b).days >= MIN_HOLDOUT_DAYS // 2:
            return b
    b = first + timedelta(days=int((last - first).days * 0.7))
    b = min(b, last - timedelta(days=MIN_HOLDOUT_DAYS))
    b = date(b.year, b.month, 1)
    if (b - first).days < MIN_DEV_DAYS:
        raise IngestError(f"history {first}..{last} is too short for {MIN_DEV_DAYS} dev days and "
                          f"{MIN_HOLDOUT_DAYS} holdout days")
    return b


def plan(rows: list, draft: dict) -> tuple[dict, dict, dict]:
    """Validation + the framework's part of the catalog entry, without writing anything."""
    data, report = validate(rows)
    first = min(date.fromisoformat(v["first"]) for v in report["keys"].values())
    last = max(date.fromisoformat(v["last"]) for v in report["keys"].values())
    hold = boundary(draft.get("asset_class", ""), first, last)
    report |= {"range": [str(first), str(last)], "holdout_from": str(hold)}
    return data, report, {"range": [str(first), str(last)], "holdout_from": str(hold),
                          "fields": sorted(data), "loader": "generic"}


def ingest(home: Path, catalog_path: Path, dataset: str, draft: dict, source_path: Path) -> dict:
    """Fetch, validate, store, add to the catalog. Returns the final entry. Nothing is written on error."""
    errors = draft_errors(draft, dataset)
    if errors:
        raise IngestError("; ".join(errors))
    t0 = time.monotonic()
    rows, http = fetch(source_path)
    data, report, framework = plan(rows, draft)
    _store(home, dataset, data, framework["holdout_from"])
    out = home / "data" / dataset
    report |= {"urls": http.urls[:50], "n_requests": len(http.urls), "bytes": http.bytes,
               "seconds": round(time.monotonic() - t0, 1), "ingested": str(date.today())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    entry = {k: v for k, v in draft.items() if k not in ("range", "holdout_from", "loader", "location", "fields")}
    entry |= framework | {"location": f"LAB_HOME/data/{dataset}", "cost_model": None,
                          "quality": f"lab ingest {report['ingested']}: {report['n_rows']} rows, "
                                     f"{len(data)} series, see LAB_HOME/data/{dataset}/report.json"}
    entry = {k: entry[k] for k in catalog.ENTRY_REQUIRED + tuple(k for k in entry if k not in catalog.ENTRY_REQUIRED)}
    catalog.add(catalog_path, entry)
    return entry


def _store(home: Path, dataset: str, data: dict, holdout_from: str) -> None:
    import polars as pl
    out = home / "data" / dataset
    out.mkdir(parents=True, exist_ok=True)
    hold = np.datetime64(holdout_from)
    frames = [pl.DataFrame({"date": ds, "key": [k] * len(ds), "value": vs}) for k, (ds, vs) in data.items()]
    df = pl.concat(frames).sort("key", "date")
    df.filter(pl.col("date") < hold).write_parquet(out / "dev.parquet")
    df.filter(pl.col("date") >= hold).write_parquet(out / "holdout.parquet")


# ---------------------------------------------------------------------------- data already on disk

HOURLY_DIR = Path("/home/kapo/ccode/market_relations_2026_oct/data/raw/binance_1h/spot_1h")
SESSIONS = {"asia": (0, 8), "europe": (8, 16), "us": (16, 24)}     # UTC hours [start, end)
LOCAL_ENTRIES = {
    "binance_1h_features": {
        "id": "binance_1h_features", "asset_class": "crypto_spot",
        "instruments": "daily features from hourly Binance spot klines of 105 liquid pairs: <PAIR>.rv (realized "
                       "volatility of the day from hourly log returns), <PAIR>.ret_asia / .ret_europe / .ret_us "
                       "(log return of the UTC sessions 00-08, 08-16, 16-24), <PAIR>.vshare_us (share of the "
                       "day's quote volume traded 16-24 UTC)",
        "frequency": "1d", "clock": "crypto", "calendar": "24/7 (UTC days)",
        "source": "data.binance.vision hourly klines downloaded by market_relations (free)",
        "known_biases": ["Universe of 105 pairs chosen by liquidity at download time (survivorship).",
                         "A day with missing hours is computed from the hours present; days with < 20 hours are dropped.",
                         "Session returns are parts of the same daily return the price data has: not independent."],
        "forward_source": "binance_public_api", "per_instrument": True},
}
FRED_DIRS = {"fred_macro": Path("/home/kapo/ccode/market_relations_2026_oct/data/raw/fred"),
             "fred_dtb3": Path("/home/kapo/ccode/strategy_backtester_2026_sep/data/raw/binance_2026-10-03")}


def hourly_feature_rows() -> list:
    """Daily features from hourly klines (UTC days): realized vol, session returns, US-session volume share."""
    import io
    import zipfile

    import polars as pl
    rows = []
    for d in sorted(p for p in HOURLY_DIR.iterdir() if p.is_dir()):
        frames = []
        for f in sorted(d.glob("*.zip")):
            with zipfile.ZipFile(f) as z:
                raw = z.read(z.namelist()[0])
            if raw.strip():
                frames.append(pl.read_csv(io.BytesIO(raw), has_header=not raw[:1].isdigit(), new_columns=[
                    "t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "x"], infer_schema_length=0)
                    .select(pl.col("t").cast(pl.Float64), pl.col("o").cast(pl.Float64), pl.col("c").cast(pl.Float64),
                            pl.col("qv").cast(pl.Float64)))
        if not frames:
            continue
        ms = pl.when(pl.col("t") > 1e14).then(pl.col("t") / 1000).otherwise(pl.col("t")).cast(pl.Int64)
        k = (pl.concat(frames).with_columns(pl.from_epoch(ms, time_unit="ms").alias("ts"))
             .unique("ts", keep="first").sort("ts")
             .with_columns(pl.col("ts").dt.date().alias("date"), pl.col("ts").dt.hour().alias("hour"),
                           (pl.col("c") / pl.col("o")).log().alias("lr")))
        sess = [pl.col("lr").filter(pl.col("hour").is_between(a, b - 1)).sum().alias(f"ret_{name}")
                for name, (a, b) in SESSIONS.items()]
        day = (k.group_by("date").agg(pl.len().alias("n"), (pl.col("lr") ** 2).sum().sqrt().alias("rv"), *sess,
                                      (pl.col("qv").filter(pl.col("hour") >= 16).sum() / pl.col("qv").sum())
                                      .alias("vshare_us"))
               .filter(pl.col("n") >= 20).sort("date"))
        for r in day.iter_rows(named=True):
            for key in ("rv", "ret_asia", "ret_europe", "ret_us", "vshare_us"):
                v = r[key]
                if v is not None and math.isfinite(v):
                    rows.append((str(r["date"]), f"{d.name}.{key}", float(v)))
    return rows


def local_rows(dataset: str) -> list:
    """Data already on disk. FRED CSVs from the sibling projects (observation_date,<SERIES>; '.' = missing),
    or daily features of the hourly Binance klines."""
    if dataset == "binance_1h_features":
        return hourly_feature_rows()
    import csv
    files = sorted(FRED_DIRS[dataset].glob("fred_DTB3.csv" if dataset == "fred_dtb3" else "*.csv"))
    rows = []
    for f in files:
        with f.open() as fh:
            r = csv.reader(fh)
            header = next(r)
            for line in r:
                if len(line) == 2 and line[1] not in ("", "."):
                    rows.append((line[0], header[1], float(line[1])))
    return rows


def import_local(home: Path, catalog_path: Path, dataset: str) -> dict:
    """Make an existing catalog entry loadable as a generic signal dataset. The entry's holdout boundary is
    kept (it was fixed when the entry was written); only `loader`, `fields` and `location` change."""
    existing = catalog.load(catalog_path).get(dataset)
    if existing is None and dataset in LOCAL_ENTRIES:      # a new derived dataset: boundary by the usual rule
        draft = LOCAL_ENTRIES[dataset]
        data, report, framework = plan(local_rows(dataset), draft)
        holdout = framework["holdout_from"]
    else:
        data, report = validate(local_rows(dataset))
        holdout = existing["holdout_from"]
    _store(home, dataset, data, holdout)
    report |= {"holdout_from": holdout, "imported": str(date.today()),
               "source": str(HOURLY_DIR if dataset in LOCAL_ENTRIES else FRED_DIRS[dataset])}
    (home / "data" / dataset / "report.json").write_text(json.dumps(report, indent=2))
    if existing is None:
        fields = sorted({k.split(".", 1)[1] for k in data}) if dataset == "binance_1h_features" else sorted(data)
        catalog.add(catalog_path, {**draft, **framework, "fields": fields, "location": f"LAB_HOME/data/{dataset}",
                                   "cost_model": None, "quality": f"lab import {report['imported']}: {report['n_rows']} rows"})
    else:
        catalog.enable_generic(catalog_path, dataset, sorted(data), f"LAB_HOME/data/{dataset}")
    return report


def report_text(report: dict) -> list[str]:
    """Value-free summary for the Archivist (data-blind like everyone else)."""
    lines = [f"range {report['range'][0]} .. {report['range'][1]}, holdout from {report['holdout_from']} "
             f"(set by the framework), {report['n_rows']} rows"]
    for k, v in report["keys"].items():
        lines.append(f"  {k}: n={v['n']} {v['first']}..{v['last']} max_gap={v['max_gap_days']}d "
                     f"gaps>7d={v['gaps_over_7_days']} non_finite_dropped={v['dropped_non_finite']} "
                     f"big_jumps={v['jumps_over_10_robust_sd']}{' CONSTANT' if v['constant'] else ''}")
    return lines

