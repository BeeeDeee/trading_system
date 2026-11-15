import csv
import io
import re
import datetime

BASE = "https://raw.githubusercontent.com/gerrymanoim/exchange_calendars/master/"
MICS = ["XTKS", "XHKG", "XKRX", "XTAI", "XSES", "XKLS", "XASX", "XBKK", "XIDX", "XPHS", "XNZE", "XBOM"]
CAP = 10
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# XPHS typhoon days put back as sessions (same list as exchange_holidays_1d)
XPHS_WEATHER = ["2011-09-27", "2012-08-07", "2014-07-16", "2014-09-19", "2014-12-08", "2017-09-12"]


def _parse_sessions(body):
    text = body.decode("utf-8", errors="replace")
    out = set()
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if not row:
            continue
        m = DATE_RE.match(row[0].strip())
        if m:
            out.add(m.group(0))
    return out


def _xhkg_weather(get):
    """Weather closure dates the library encodes as holidays (labelled block in the XHKG source)."""
    body = get(BASE + "exchange_calendars/exchange_calendar_xhkg.py").decode("utf-8", errors="replace")
    low = body.lower()
    i = low.find("weather related closures")
    if i < 0:
        raise ValueError("XHKG weather block not found")
    lines = body[i:].split("\n")
    dates = []
    for line in lines:
        s = line.strip()
        if s.startswith("#"):
            # the block ends where the library starts the special (non-weather) holidays of 2022
            if "mid-autumn" in s.lower():
                break
            continue
        dates.extend(DATE_RE.findall(s.split("#")[0]))
    if len(dates) < 10:
        raise ValueError("XHKG weather block too short: %d" % len(dates))
    return dates


def _weekday(d):
    return datetime.date.fromisoformat(d).weekday() < 5


def fetch(get):
    sess = {}
    for mic in ["XNYS"] + MICS:
        body = get(BASE + "tests/resources/" + mic.lower() + ".csv")
        sess[mic] = _parse_sessions(body)
    for d in _xhkg_weather(get):
        if _weekday(d):
            sess["XHKG"].add(d)
    for d in XPHS_WEATHER:
        if _weekday(d):
            sess["XPHS"].add(d)

    for mic in sess:
        sess[mic] = set(d for d in sess[mic] if _weekday(d))

    xnys = sorted(d for d in sess["XNYS"] if d >= "2001-01-02")
    n = len(xnys)
    if n < 100:
        raise ValueError("XNYS session list too short: %d" % n)
    dates = [datetime.date.fromisoformat(d) for d in xnys]

    rows = []
    for mic in MICS:
        s = sess[mic]
        closed = []
        for i in range(n - 1):
            d0 = dates[i]
            d1 = dates[i + 1]
            flag = 1
            cur = d0 + datetime.timedelta(days=1)
            while cur <= d1:
                if cur.isoformat() in s:
                    flag = 0
                    break
                cur += datetime.timedelta(days=1)
            closed.append(flag)
        # closed has length n-1: closed_next exists on sessions 0..n-2 (the last session has no next XNYS session)
        m = len(closed)
        # nxt[i] = index of first session j >= i with closed[j] == 1, or None
        nxt = [None] * (m + 1)
        for i in range(m - 1, -1, -1):
            nxt[i] = i if closed[i] == 1 else nxt[i + 1]
        for i in range(m):
            j = nxt[i]
            if j is not None:
                v = j - i
                if v > CAP:
                    v = CAP
                rows.append((xnys[i], mic + ".sessions_to_closure", float(v)))
            elif i + CAP < m:
                rows.append((xnys[i], mic + ".sessions_to_closure", float(CAP)))
            # else: within CAP sessions of the end of the file, no value
    return rows
