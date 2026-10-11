"""SEC EDGAR 8-K Item 2.02 ("Results of Operations") filings, the real earnings-release dates (dataset sec_8k_item202).

Framework-owned, not an Archivist fetcher: the CIK list comes from the Sharadar tickers table (which a sandboxed
fetcher cannot read) and one fetch needs thousands of requests, far above the 500 of `ingest.MAX_REQUESTS`.

    cik_map(src)             permaticker -> CIK, from the `secfilings` URL of the SEP rows in tickers.parquet
    fetch_cik(get, cik)      one company: first filing date + the form 8-K filings that list item 2.02
                             (EDGAR submissions API: `filings.recent` plus the older-filings files it links)
    fetch_all(...)           every CIK, polite (RATE_PER_S), resumable (one cache file per CIK)
    build(...)               LAB_HOME/data/sec_8k_item202/{filings,first}.parquet + report.json (counts only)

The matrix `ea_reaction_day` (attached.py) is built from these two files. Needs a descriptive User-Agent with a
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


def fetch_cik(get, cik: int) -> dict:
    """{"first": earliest filing date EDGAR lists for the company | None, "e202": [[filed, accepted], ...]}"""
    body = get(f"{BASE}CIK{cik:010d}.json")
    if body is None:
        return {"first": None, "e202": []}
    doc = json.loads(body)
    filings = doc.get("filings", {})
    first, rows = parse_block(filings.get("recent", {}))
    for f in filings.get("files", []):
        older = get(f"{BASE}{f['name']}")
        if older is None:
            continue
        o_first, o_rows = parse_block(json.loads(older))
        first = min(x for x in (first, o_first) if x) if (first or o_first) else None
        rows += o_rows
    return {"first": first, "e202": sorted({tuple(r) for r in rows})}


def fetch_all(get, ciks: list[int], cache: Path, progress=None) -> dict[int, dict]:
    """Every CIK, resumable: finished CIKs are read back from `cache/<cik>.json`."""
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, cik in enumerate(sorted(ciks)):
        f = cache / f"{cik}.json"
        if f.exists():
            out[cik] = json.loads(f.read_text())
        else:
            out[cik] = fetch_cik(get, cik)
            f.write_text(json.dumps(out[cik]))
        if progress and (i + 1) % 500 == 0:
            progress(i + 1, len(ciks))
    return out


def build(home: Path, src: Path, client: Client | None = None, limit: int | None = None, progress=None) -> dict:
    """Fetch all CIKs of the SEP stocks and store the tables. Returns the report (counts and dates only)."""
    import polars as pl
    client = client or Client(user_agent())
    ciks = sorted(set(cik_map(src).values()))[:limit]
    t0 = time.monotonic()
    data = fetch_all(client.get, ciks, home / "cache" / "edgar", progress)
    out = home / "data" / DATASET
    out.mkdir(parents=True, exist_ok=True)
    rows = [(c, d["first"]) for c, d in data.items() if d["first"]]
    pl.DataFrame({"cik": [r[0] for r in rows], "first": [r[1] for r in rows]},
                 schema={"cik": pl.Int64, "first": pl.String}).with_columns(pl.col("first").str.to_date()
                 ).write_parquet(out / "first.parquet")
    ev = [(c, f, a) for c, d in data.items() for f, a in d["e202"]]
    filings = pl.DataFrame({"cik": [e[0] for e in ev], "filed": [e[1] for e in ev], "accepted": [e[2] for e in ev]},
                           schema={"cik": pl.Int64, "filed": pl.String, "accepted": pl.String})
    filings = filings.with_columns(pl.col("filed").str.to_date())
    filings.write_parquet(out / "filings.parquet")
    report = {"ciks": len(ciks), "ciks_with_filings": filings["cik"].n_unique(), "n_filings": filings.height,
              "first_filed": str(filings["filed"].min()), "last_filed": str(filings["filed"].max()),
              "requests_this_run": client.n, "seconds": round(time.monotonic() - t0, 1),
              "fetched": str(date.today())}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


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
