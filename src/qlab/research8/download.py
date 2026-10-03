"""Binance public data (data.binance.vision): S3 listing, monthly ZIPs, CHECKSUM verification.

Symbols come from the S3 listing, not from today's exchangeInfo, so delisted contracts are included
(no survivorship bias). Every file is verified against its `.CHECKSUM` (sha256) and stored as-is;
parsing never modifies raw files.
"""

import hashlib
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
HTTP = "https://data.binance.vision"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

PREFIX = {
    "funding": "data/futures/um/monthly/fundingRate/{sym}/",
    "perp_1d": "data/futures/um/monthly/klines/{sym}/1d/",
    "spot_1d": "data/spot/monthly/klines/{sym}/1d/",
    "mark_1d": "data/futures/um/monthly/markPriceKlines/{sym}/1d/",
}
ROOT_PREFIX = {
    "funding": "data/futures/um/monthly/fundingRate/",
    "perp_1d": "data/futures/um/monthly/klines/",
    "spot_1d": "data/spot/monthly/klines/",
    "mark_1d": "data/futures/um/monthly/markPriceKlines/",
}
MONTH_RE = re.compile(r"-(\d{4}-\d{2})\.zip$")


def _get(url: str, tries: int = 5) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                raise
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def s3_list(prefix: str, delimiter: bool) -> tuple[list[str], list[str]]:
    """All (keys, common prefixes) under `prefix`, following pagination."""
    keys, prefixes, marker = [], [], ""
    while True:
        url = f"{S3}?prefix={prefix}" + ("&delimiter=/" if delimiter else "") + (f"&marker={marker}" if marker else "")
        root = ET.fromstring(_get(url))
        keys += [c.findtext("s3:Key", namespaces=NS) for c in root.findall("s3:Contents", NS)]
        prefixes += [c.findtext("s3:Prefix", namespaces=NS) for c in root.findall("s3:CommonPrefixes", NS)]
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return keys, prefixes
        marker = root.findtext("s3:NextMarker", namespaces=NS) or (keys or prefixes)[-1]


def list_symbols(kind: str) -> list[str]:
    _, prefixes = s3_list(ROOT_PREFIX[kind], delimiter=True)
    return sorted(p.rstrip("/").split("/")[-1] for p in prefixes)


def fetch_symbol(kind: str, sym: str, out: Path, last_month: str) -> dict:
    """Download every monthly ZIP of `sym` up to `last_month` (YYYY-MM), verify sha256. Idempotent."""
    keys, _ = s3_list(PREFIX[kind].format(sym=sym), delimiter=False)
    zips = [k for k in keys if k.endswith(".zip") and (m := MONTH_RE.search(k)) and m.group(1) <= last_month]
    d = out / kind / sym
    d.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for k in zips:
        f = d / k.rsplit("/", 1)[1]
        if f.exists() and f.with_suffix(".zip.ok").exists():
            continue
        blob = _get(f"{HTTP}/{k}")
        expected = _get(f"{HTTP}/{k}.CHECKSUM").decode().split()[0]
        got = hashlib.sha256(blob).hexdigest()
        if got != expected:
            raise ValueError(f"checksum mismatch {k}: {got} != {expected}")
        f.write_bytes(blob)
        f.with_suffix(".zip.ok").write_text(expected + "\n")
        n_new += 1
    return {"kind": kind, "symbol": sym, "files": len(zips), "new": n_new}


def fetch_many(kind: str, symbols: list[str], out: Path, last_month: str, workers: int = 4,
               log=print) -> list[dict]:
    def one(sym: str) -> dict:
        try:
            return fetch_symbol(kind, sym, out, last_month)
        except Exception as e:  # recorded and reported; a rerun retries only what is missing
            return {"kind": kind, "symbol": sym, "error": f"{type(e).__name__}: {e}"}

    res = []
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, symbols), 1):
            res.append(r)
            if i % 50 == 0 or i == len(symbols):
                log(f"{kind}: {i}/{len(symbols)} symbols")
    return res
