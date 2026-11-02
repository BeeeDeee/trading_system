"""Forward ETF data (prereg §3): Yahoo chart API, total return from split-adjusted close and cash dividends.

Per day t with ex-dividend D_t (same fetch, so splits cancel):
    ret_oc = close_t / open_t - 1
    ret_co = open_t / close_{t-1} * (1 + D_t / close_t) - 1
which is `qlab.data.normalize` for closeadj = close + dividends: the dividend is attributed to the overnight part
(open_adj = open * closeadj / close) and (1 + ret_co)(1 + ret_oc) - 1 = (close_t + D_t) / close_{t-1} - 1 exactly.
"""

import datetime
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

UA = "Mozilla/5.0"


class FetchError(Exception):
    pass


def _get(url: str, timeout: int = 25, retries: int = 2) -> dict:
    last = None
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
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


def _get_text(url: str, timeout: int = 30, retries: int = 2) -> str:
    last = None
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})   # FRED stalls on a browser UA
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode()
        except Exception as e:
            last = f"{type(e).__name__}: {e} ({url})"
            time.sleep(2 * (i + 1))
    raise FetchError(last)


def fetch_yahoo(ticker: str, rng: str = "2y") -> dict:
    """Raw chart JSON with dividends and splits."""
    return _get("https://query1.finance.yahoo.com/v8/finance/chart/"
                f"{urllib.parse.quote(ticker)}?range={rng}&interval=1d&events=div,split")


def _day(ts: int, off: int) -> datetime.date:
    return datetime.datetime.fromtimestamp(ts + off, datetime.timezone.utc).date()


def parse_yahoo(j: dict) -> dict:
    """-> {'dates': datetime64[D] (T,), 'open','high','low','close': (T,), 'div': (T,) cash dividend on the ex-date,
    'split': (T,) new shares per old on the split date (1 = none)}. Rows with a missing OHLC value are dropped."""
    res = (j.get("chart") or {}).get("result")
    if not res:
        raise FetchError(f"Yahoo without result: {(j.get('chart') or {}).get('error')}")
    r = res[0]
    off = (r.get("meta") or {}).get("gmtoffset", 0)
    q = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    ts = r.get("timestamp") or []
    cols = {k: np.array([np.nan if v is None else v for v in q.get(k, [None] * len(ts))], dtype=float)
            for k in ("open", "high", "low", "close")}
    ok = np.all([np.isfinite(v) for v in cols.values()], axis=0)
    dates = np.array([_day(t, off) for t in ts], dtype="datetime64[D]")
    ev = r.get("events") or {}
    div, split = np.zeros(len(ts)), np.ones(len(ts))
    for v in (ev.get("dividends") or {}).values():
        k = np.flatnonzero(dates == np.datetime64(_day(v["date"], off)))
        if len(k):
            div[k[0]] += float(v["amount"])
    for v in (ev.get("splits") or {}).values():
        k = np.flatnonzero(dates == np.datetime64(_day(v["date"], off)))
        if len(k):
            split[k[0]] = float(v["numerator"]) / float(v["denominator"])
    return {"dates": dates[ok], **{k: v[ok] for k, v in cols.items()}, "div": div[ok], "split": split[ok]}


def tr_bars(p: dict) -> dict:
    """Daily ret_co / ret_oc (first row 0) and the close-to-close total return from a parsed fetch."""
    o, c, d = p["open"], p["close"], p["div"]
    ret_oc = c / o - 1.0
    ret_co = np.zeros(len(c))
    ret_co[1:] = o[1:] / c[:-1] * (1.0 + d[1:] / c[1:]) - 1.0
    return {"dates": p["dates"], "ret_co": ret_co, "ret_oc": ret_oc, "tr": (1 + ret_co) * (1 + ret_oc) - 1.0}


def nasdaq_closes(ticker: str, start: datetime.date, limit: int = 300, asset: str = "etf") -> dict:
    """-> {date: close} from api.nasdaq.com (independent of Yahoo; unadjusted for dividends)."""
    j = _get(f"https://api.nasdaq.com/api/quote/{ticker}/historical?assetclass={asset}&fromdate={start.isoformat()}&limit={limit}")
    rows = (((j.get("data") or {}).get("tradesTable")) or {}).get("rows") or []
    out = {}
    for x in rows:
        m, d, y = x["date"].split("/")
        out[np.datetime64(f"{y}-{m}-{d}")] = float(str(x["close"]).replace("$", "").replace(",", ""))
    if not out:
        raise FetchError(f"{ticker}: Nasdaq without rows")
    return out


def gate_stats(dates_y: np.ndarray, tr_y: np.ndarray, dates_s: np.ndarray, tr_s: np.ndarray) -> dict:
    """Compare two daily TR series on the common dates (basis points); also count the dates only one has.
    `max_abs_bps` (daily) is descriptive, the gate uses `max_abs2_bps` (largest difference not cancelled by an adjacent day)."""
    common, iy, is_ = np.intersect1d(dates_y, dates_s, return_indices=True)
    d = (tr_y[iy] - tr_s[is_]) * 1e4
    a = np.abs(d)
    cum = (np.prod(1 + tr_y[iy]) / np.prod(1 + tr_s[is_]) - 1.0) * 100.0
    # A dividend booked on adjacent days by the two sources shows as +x, -x on those days. A day counts only if no
    # two-day window containing it cancels the difference: m_t = min(|d_t|, |w_{t-1,t}|, |w_{t,t+1}|), w = compounded
    # two-day difference (prereg v1.0, §3.3). An isolated error stays m_t = |d_t|.
    two = np.abs(((1 + tr_y[iy][1:]) * (1 + tr_y[iy][:-1]) - (1 + tr_s[is_][1:]) * (1 + tr_s[is_][:-1])) * 1e4)
    inf = np.full(1, np.inf)
    m2 = np.minimum(a, np.minimum(np.r_[inf, two], np.r_[two, inf])) if len(a) else a
    return {"n_common": int(len(common)), "n_only_yahoo": int(len(dates_y) - len(common)),
            "n_only_sharadar": int(len(dates_s) - len(common)), "median_abs_bps": float(np.median(a)) if len(a) else None,
            "p99_abs_bps": float(np.percentile(a, 99)) if len(a) else None, "max_abs_bps": float(a.max()) if len(a) else None,
            "max_abs2_bps": float(m2.max()) if len(m2) else None,
            "cum_diff_pct": float(cum), "worst_dates": [str(common[k]) for k in np.argsort(-a)[:3]]}
