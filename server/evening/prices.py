"""Market data for the evening bot: Yahoo chart API (primary, OHLC) and Nasdaq historical API (independent cross-check).

Both are plain JSON endpoints that answer from a server IP (stockanalysis.com, the source of the cloud runs, sits behind a
Cloudflare challenge). Prices are the plain (unadjusted-by-dividends) Close, never Adj. Close.
"""
import datetime
import json
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "Mozilla/5.0"          # api.nasdaq.com silently drops requests with other user agents
ETFS = {"SPY", "QQQ", "IWM", "TLT", "GLD", "XLK", "XLF", "XLE", "XLV", "XLP", "XLU"}


class FetchError(Exception):
    pass


def _get(url, timeout=25, retries=2):
    last = None
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code} from {url}"
            if e.code in (400, 404):
                break
        except Exception as e:  # network error, timeout, bad JSON
            last = f"{type(e).__name__}: {e} ({url})"
        time.sleep(2 * (i + 1))
    raise FetchError(last)


def _r(x):
    return None if x is None else round(float(x), 4)


def _yahoo(ticker, rng, events=""):
    return _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}?range={rng}&interval=1d{events}")


def yahoo_rows(ticker, rng="1mo"):
    """-> {date: [open, high, low, close]} for the last trading days (exchange-local dates)."""
    j = _yahoo(ticker, rng)
    res = (j.get("chart") or {}).get("result")
    if not res:
        raise FetchError(f"{ticker}: Yahoo bez výsledku: {(j.get('chart') or {}).get('error')}")
    r = res[0]
    off = (r.get("meta") or {}).get("gmtoffset", 0)
    q = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    out = {}
    for i, ts in enumerate(r.get("timestamp") or []):
        row = [q.get(k, [None] * (i + 1))[i] for k in ("open", "high", "low", "close")]
        if any(v is None for v in row):
            continue                                   # holiday placeholder / in-progress bar
        d = datetime.datetime.utcfromtimestamp(ts + off).date().isoformat()
        out[d] = [_r(v) for v in row]
    return out


def nasdaq_rows(ticker, days=12):
    """-> {date: [open, high, low, close]} from api.nasdaq.com (independent of Yahoo)."""
    cls = "etf" if ticker in ETFS else "stocks"
    start = (datetime.date.today() - datetime.timedelta(days=days * 2)).isoformat()
    j = _get(f"https://api.nasdaq.com/api/quote/{ticker}/historical?assetclass={cls}&fromdate={start}&limit={days}")
    rows = (((j.get("data") or {}).get("tradesTable")) or {}).get("rows") or []
    out = {}
    for x in rows:
        m, d, y = x["date"].split("/")
        f = lambda s: float(str(s).replace("$", "").replace(",", ""))
        out[f"{y}-{m}-{d}"] = [f(x["open"]), f(x["high"]), f(x["low"]), f(x["close"])]
    if not out:
        raise FetchError(f"{ticker}: Nasdaq bez řádků")
    return out


def yahoo_splits(ticker, rng="1mo"):
    """-> {date: ratio} (new shares per old share, 2 for a 2:1 split, 0.1 for a 1:10 reverse split)."""
    j = _yahoo(ticker, rng, "&events=split")
    res = (j.get("chart") or {}).get("result")
    if not res:
        raise FetchError(f"{ticker}: Yahoo bez výsledku")
    off = (res[0].get("meta") or {}).get("gmtoffset", 0)
    out = {}
    for v in ((res[0].get("events") or {}).get("splits") or {}).values():
        d = datetime.datetime.utcfromtimestamp(v["date"] + off).date().isoformat()
        out[d] = float(v["numerator"]) / float(v["denominator"])
    return out


def fetch_all(universe, rng="6mo"):
    """-> ({ticker: rows}, errors). One request per ticker, Yahoo chart API."""
    rows, errs = {}, []
    for tk in universe:
        try:
            rows[tk] = yahoo_rows(tk, rng)
        except FetchError as e:
            errs.append(f"{tk}: {e}")
    return rows, errs


def _consistent(row):
    o, h, l, c = row                                   # same rule as the engine's data check
    return l <= min(o, c) + 1e-6 and h >= max(o, c) - 1e-6 and l > 0


def repair_ohlc(rows_all, universe, tol_pct=0.3):
    """Yahoo occasionally serves an internally inconsistent bar (e.g. open 1 cent above high). Replace such a row with
    Nasdaq's row for the same date when that one is consistent and its close agrees within `tol_pct`.
    Modifies rows_all in place. -> notes (what was replaced / could not be)."""
    notes = []
    for tk in universe:
        bad = [d for d, r in (rows_all.get(tk) or {}).items() if not _consistent(r)]
        if not bad:
            continue
        try:
            alt = nasdaq_rows(tk)
        except FetchError as e:
            notes.append(f"{tk}: nekonzistentní OHLC u Yahoo, Nasdaq nedostupný ({e})")
            continue
        for d in sorted(bad):
            y, n = rows_all[tk][d], alt.get(d)
            if n is None:
                notes.append(f"{tk} {d}: nekonzistentní OHLC u Yahoo, Nasdaq řádek ještě nemá")
            elif not _consistent(n) or abs(y[3] / n[3] - 1) * 100 > tol_pct:
                notes.append(f"{tk} {d}: nekonzistentní OHLC u Yahoo, Nasdaq {n} nesedí")
            else:
                rows_all[tk][d] = [_r(v) for v in n]
                notes.append(f"{tk} {d}: OHLC z Nasdaq (Yahoo {y} nekonzistentní)")
    return notes


def day_prices(rows_all, universe, date):
    """today.json for `date` from fetched rows; tickers without a row for that date are left out."""
    t = {"date": date, "open": {}, "high": {}, "low": {}, "close": {}}
    for tk in universe:
        r = (rows_all.get(tk) or {}).get(date)
        if r:
            t["open"][tk], t["high"][tk], t["low"][tk], t["close"][tk] = r
    return t
