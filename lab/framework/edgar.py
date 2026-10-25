"""SEC EDGAR 8-K filings: Item 2.02 ("Results of Operations"), the real earnings-release dates (dataset
sec_8k_item202), and the non-earnings 8-Ks (dataset sec_8k_items, field nonearn_8k_reaction_day).

Framework-owned, not an Archivist fetcher: the CIK list comes from the Sharadar tickers table (which a sandboxed
fetcher cannot read) and one fetch needs thousands of requests, far above the 500 of `ingest.MAX_REQUESTS`.

    cik_map(src)             permaticker -> CIK, from the `secfilings` URL of the SEP rows in tickers.parquet
    fetch_cik(get, cik)      one company: first filing date + the form 8-K filings that list item 2.02, and the
                             primary 8-Ks that carry other news (EDGAR submissions API: `filings.recent` plus the
                             older-filings files it links)
    fetch_all(...)           every CIK, polite (RATE_PER_S), resumable (one cache file per CIK)
    build(...)               LAB_HOME/data/sec_8k_item202/{filings,first}.parquet (written once, then frozen) and
                             LAB_HOME/data/sec_8k_items/{filings,first}.parquet, each with a report.json (counts only)

The matrices `ea_reaction_day` and `nonearn_8k_reaction_day` (attached.py) are built from these files. Needs a descriptive User-Agent with a
contact address (SEC fair-access policy): set the environment variable EDGAR_CONTACT, it is only ever sent to sec.gov.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

DATASET = "sec_8k_item202"
DATASET_ITEMS = "sec_8k_items"
NEWS_IGNORED = frozenset({"2.02", "5.07", "9.01"})     # earnings release, shareholder-vote results, exhibits only
BASE = "https://data.sec.gov/submissions/"
RATE_PER_S = 8                  # the SEC allows 10 requests per second
TIMEOUT_S = 30
RETRIES = 4
CIK_RE = re.compile(r"CIK=0*(\d+)")


class EdgarError(RuntimeError):
    pass


def cik_map(src: Path) -> dict[int, int]:
    """permaticker -> CIK for the SEP stocks (rows without a CIK in their SEC URL are left out)."""
    import polars as pl
    t = (pl.read_parquet(src / "tickers.parquet", columns=["table", "permaticker", "secfilings"])
         .filter((pl.col("table") == "SEP") & pl.col("secfilings").is_not_null())
         .with_columns(pl.col("secfilings").str.extract(CIK_RE.pattern).cast(pl.Int64).alias("cik"))
         .drop_nulls("cik"))
    return {int(p): int(c) for p, c in zip(t["permaticker"].to_list(), t["cik"].to_list())}


def user_agent() -> str:
    contact = os.environ.get("EDGAR_CONTACT", "").strip()
    if "@" not in contact:
        raise EdgarError("set EDGAR_CONTACT to a contact e-mail address: the SEC requires a descriptive "
                         "User-Agent with contact details (it is sent only to sec.gov)")
    return f"research-lab {contact}"


class Client:
    """The one network door: https, throttled to RATE_PER_S, retries with backoff on 429/5xx."""

    def __init__(self, agent: str, rate: float = RATE_PER_S):
        self.agent, self.gap, self.last, self.n = agent, 1.0 / rate, 0.0, 0

    def get(self, url: str) -> bytes | None:
        """The body, or None for a 404 (a CIK EDGAR does not know)."""
        for attempt in range(RETRIES + 1):
            wait = self.last + self.gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.n += 1
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.agent, "Accept-Encoding": "identity"})
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code not in (429, 500, 502, 503, 504) or attempt == RETRIES:
                    raise EdgarError(f"{url}: HTTP {e.code}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == RETRIES:
                    raise EdgarError(f"{url}: {e}") from None
            time.sleep(2 ** attempt * 5)
        raise EdgarError(f"{url}: gave up")


def parse_block(block: dict) -> tuple[str | None, list[tuple[str, str]]]:
    """-> (earliest filing date, [(filing date, acceptance time UTC ISO)] of the form 8-K filings with item 2.02)."""
    forms, filed = block.get("form") or [], block.get("filingDate") or []
    accepted, items = block.get("acceptanceDateTime") or [], block.get("items") or []
    first = min(filed) if filed else None
    out = []
    for i, form in enumerate(forms):
        if form == "8-K" and "2.02" in str(items[i]).split(","):          # 8-K/A (amendments) are dropped
            out.append((filed[i], accepted[i]))
    return first, out


def parse_other(block: dict) -> list[tuple[str, str, str]]:
    """[(filing date, acceptance time UTC ISO, item list as filed)] of the primary 8-Ks with news that is not an
    earnings release: at least one item outside NEWS_IGNORED and no item 2.02. 8-K/A is dropped."""
    forms, filed = block.get("form") or [], block.get("filingDate") or []
    accepted, items = block.get("acceptanceDateTime") or [], block.get("items") or []
    out = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        got = {x.strip() for x in str(items[i]).split(",") if x.strip()}
        if "2.02" not in got and got - NEWS_IGNORED:
            out.append((filed[i], accepted[i], ",".join(sorted(got))))
    return out


def fetch_cik(get, cik: int) -> dict:
    """{"first": earliest filing date EDGAR lists for the company | None, "e202": [[filed, accepted], ...],
    "other": [[filed, accepted, items], ...]}"""
    body = get(f"{BASE}CIK{cik:010d}.json")
    if body is None:
        return {"first": None, "e202": [], "other": []}
    doc = json.loads(body)
    filings = doc.get("filings", {})
    blocks = [filings.get("recent", {})]
    for f in filings.get("files", []):
        older = get(f"{BASE}{f['name']}")
        if older is not None:
            blocks.append(json.loads(older))
    first, rows, other = None, [], []
    for b in blocks:
        b_first, b_rows = parse_block(b)
        first = min(x for x in (first, b_first) if x) if (first or b_first) else None
        rows += b_rows
        other += parse_other(b)
    return {"first": first, "e202": sorted({tuple(r) for r in rows}), "other": sorted({tuple(r) for r in other})}


def fetch_all(get, ciks: list[int], cache: Path, progress=None) -> dict[int, dict]:
    """Every CIK, resumable: finished CIKs are read back from `cache/<cik>.json`. A cache file from before the
    "other" list existed is fetched again (a CIK EDGAR does not know, 404, has nothing to add and is completed
    without a request); the item-2.02 list of such a file is kept as it was, so sec_8k_item202 does not move."""
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, cik in enumerate(sorted(ciks)):
        f = cache / f"{cik}.json"
        old = json.loads(f.read_text()) if f.exists() else None
        if old is not None and "other" in old:
            out[cik] = old
        elif old is not None and old["first"] is None:
            out[cik] = {**old, "other": []}
            f.write_text(json.dumps(out[cik]))
        else:
            out[cik] = fetch_cik(get, cik)
            if old is not None:
                out[cik]["e202"] = old["e202"]
            f.write_text(json.dumps(out[cik]))
        if progress and (i + 1) % 500 == 0:
            progress(i + 1, len(ciks))
    return out


def _write_first(data: dict, out: Path) -> None:
    import polars as pl
    rows = [(c, d["first"]) for c, d in data.items() if d["first"]]
    pl.DataFrame({"cik": [r[0] for r in rows], "first": [r[1] for r in rows]},
                 schema={"cik": pl.Int64, "first": pl.String}).with_columns(pl.col("first").str.to_date()
                 ).write_parquet(out / "first.parquet")


def _write_filings(data: dict, key: str, out: Path, with_items: bool) -> dict:
    """filings.parquet of one list ("e202" or "other") and its report (counts and dates only)."""
    import polars as pl
    ev = [(c, *r) for c, d in data.items() for r in d[key]]
    cols = {"cik": [e[0] for e in ev], "filed": [e[1] for e in ev], "accepted": [e[2] for e in ev]}
    schema = {"cik": pl.Int64, "filed": pl.String, "accepted": pl.String}
    if with_items:
        cols["items"], schema["items"] = [e[3] for e in ev], pl.String
    filings = pl.DataFrame(cols, schema=schema).with_columns(pl.col("filed").str.to_date())
    filings.write_parquet(out / "filings.parquet")
    return {"ciks": len(data), "ciks_with_filings": filings["cik"].n_unique(), "n_filings": filings.height,
            "first_filed": str(filings["filed"].min()), "last_filed": str(filings["filed"].max())}


def build(home: Path, src: Path, client: Client | None = None, limit: int | None = None, progress=None) -> dict:
    """Fetch all CIKs of the SEP stocks and store the tables. Returns the report of sec_8k_item202 (counts and
    dates only) with the one of sec_8k_items under "items". sec_8k_item202 is written once and then frozen (a
    study may already have run on it); sec_8k_items is always rewritten from the cache."""
    client = client or Client(user_agent())
    ciks = sorted(set(cik_map(src).values()))[:limit]
    t0 = time.monotonic()
    data = fetch_all(client.get, ciks, home / "cache" / "edgar", progress)
    extra = {"requests_this_run": client.n, "seconds": round(time.monotonic() - t0, 1), "fetched": str(date.today())}
    out = home / "data" / DATASET
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists() and not limit:
        report = json.loads((out / "report.json").read_text())      # frozen
    else:
        _write_first(data, out)
        report = {**_write_filings(data, "e202", out, False), **extra}
        (out / "report.json").write_text(json.dumps(report, indent=2))
    out_i = home / "data" / DATASET_ITEMS
    out_i.mkdir(parents=True, exist_ok=True)
    _write_first(data, out_i)
    items = {**_write_filings(data, "other", out_i, True), **extra}
    (out_i / "report.json").write_text(json.dumps(items, indent=2))
    return {**report, "items": items}


def catalog_entry(report: dict) -> dict:
    return {
        "id": DATASET, "loader": "sep_attached", "attaches_to": "sharadar_sep", "asset_class": "alt",
        "instruments": "earnings-release dates: form 8-K filings with Item 2.02 (SEC EDGAR), as the daily flag "
                       "ea_reaction_day on the stocks of sharadar_sep",
        "frequency": "event", "range": ["2004-08-23", report["last_filed"]], "clock": "us_close",
        "source": "SEC EDGAR submissions API (data.sec.gov, free, public), fetched by the framework; CIK from the "
                  "secfilings URL of the Sharadar tickers table",
        "location": f"LAB_HOME/data/{DATASET}",
        "quality": f"lab edgar-fetch {report['fetched']}: {report['n_filings']} filings of "
                   f"{report['ciks_with_filings']} of {report['ciks']} CIKs",
        "holdout_from": "2021-01-01", "cost_model": None, "forward_source": None,
        "known_biases": [
            "Attached to sharadar_sep as a daily (T, N) matrix: 1 on the reaction session of each primary 8-K with "
            "Item 2.02, 0 on other sessions, NaN before the company's first EDGAR filing. Item 2.02 exists since "
            "the August 2004 8-K reform. Amendments (8-K/A) are dropped.",
            "Reaction session = first XNYS session whose 16:00 ET close is at or after the EDGAR acceptance time; "
            "known at the close of that session, not before. Early-close days (13:00 ET) are treated as 16:00.",
            "CIKs come from the Sharadar tickers table (the current CIK of a security); companies that changed CIK "
            "or a reused CIK can miss or mix filings. A CIK EDGAR no longer serves (404) has no filings and no data.",
            "Some firms file earnings releases under item 7.01 or as exhibits to other items: they are missed. "
            "The EDGAR item list is as filed, never revised.",
        ],
    }


def catalog_entry_items(report: dict) -> dict:
    rep = report["items"]
    return {
        "id": DATASET_ITEMS, "loader": "sep_attached", "attaches_to": "sharadar_sep", "asset_class": "alt",
        "instruments": "non-earnings 8-K news: primary form 8-K filings with at least one item other than 2.02, 5.07 "
                       "and 9.01 and without item 2.02 (SEC EDGAR), as the daily flag nonearn_8k_reaction_day on the "
                       "stocks of sharadar_sep",
        "frequency": "event", "range": ["2004-08-23", rep["last_filed"]], "clock": "us_close",
        "source": "SEC EDGAR submissions API (data.sec.gov, free, public), fetched by the framework from the same "
                  "per-CIK files as sec_8k_item202; CIK from the secfilings URL of the Sharadar tickers table",
        "location": f"LAB_HOME/data/{DATASET_ITEMS}",
        "quality": f"lab edgar-fetch {rep['fetched']}: {rep['n_filings']} filings of "
                   f"{rep['ciks_with_filings']} of {rep['ciks']} CIKs",
        "holdout_from": "2021-01-01", "cost_model": None, "forward_source": None,
        "known_biases": [
            "Attached to sharadar_sep as a daily (T, N) matrix: 1 on the reaction session of each primary 8-K whose "
            "item list has at least one item outside {2.02, 5.07, 9.01} and does not contain 2.02, 0 on other "
            "sessions, NaN before the company's first EDGAR filing. Same reaction-session rule, same CIK mapping and "
            "same early-close simplification as sec_8k_item202. Amendments (8-K/A) are dropped.",
            "An 8-K that carries both an earnings release (2.02) and other news is excluded, so a non-earnings event "
            "filed together with 2.02 is missed. Earnings releases filed only under item 7.01 or 8.01 are counted "
            "as non-earnings news: the flag is contaminated with earnings reactions, so a card must also condition "
            "on ea_reaction_day (an event day itself can still carry such a release).",
            "CIKs come from the Sharadar tickers table (the current CIK of a security): companies that changed or "
            "reused a CIK can miss or mix filings, and a CIK EDGAR no longer serves (404) has no data, so delisted "
            "firms are under-covered and the event set leans to survivors (same as sec_8k_item202).",
            "The EDGAR item list is as filed, never revised. The flag is a daily indicator with no information on "
            "the news itself: several unrelated 8-Ks on one session count once.",
        ],
    }
