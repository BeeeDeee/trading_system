import re
import datetime

# exchange_calendars (Apache-2.0): tests/resources/<mic>.csv holds the full list of regular sessions of each
# calendar as encoded by the library's holiday rules (one line per session, first field = session date).
BASE = "https://raw.githubusercontent.com/gerrymanoim/exchange_calendars/master/tests/resources/"
DATE = re.compile(r"^\s*\"?(\d{4}-\d{2}-\d{2})")

EXCH = ["XTKS", "XHKG", "XKRX", "XTAI", "XSES", "XKLS", "XASX", "XBKK", "XIDX", "XPHS", "XNZE", "XBOM"]

START = "2001-01-02"

# Closures that were NOT scheduled a day ahead (weather) but are encoded as holidays in the library.
# They are put back as sessions, so they never produce closed_next = 1 / a skipped anchor.
# XHKG: the "Weather related closures" block of exchange_calendar_xhkg.py (typhoon / black rainstorm days
#       announced on the day or the evening before), entries 2001 and later.
# XPHS: the "# Typhoon" block of exchange_calendar_xphs.py.
UNSCHEDULED = {
    "XHKG": [
        "2001-07-06", "2001-07-25", "2008-06-25", "2008-08-06", "2008-08-22", "2009-09-15",
        "2011-09-29", "2012-07-24", "2013-08-14", "2013-09-23", "2014-09-16", "2015-07-09",
        "2016-08-02", "2016-10-21", "2017-08-23", "2020-10-13",
    ],
    "XPHS": [
        "2011-09-27", "2012-08-07", "2014-07-16", "2014-09-19", "2014-12-08", "2017-09-12",
    ],
}


def _parse(get, mic):
    body = get(BASE + mic.lower() + ".csv").decode("utf-8", "replace")
    out = set()
    for line in body.splitlines():
        m = DATE.match(line)
        if m:
            out.add(m.group(1))
    return out


def _weekdays(dates):
    res = set()
    for s in dates:
        d = datetime.date(int(s[0:4]), int(s[5:7]), int(s[8:10]))
        if d.weekday() < 5:
            res.add(s)
    return res


def _next_day(s):
    d = datetime.date(int(s[0:4]), int(s[5:7]), int(s[8:10])) + datetime.timedelta(days=1)
    return d.isoformat()


def fetch(get):
    nyse_all = _parse(get, "XNYS")
    nyse = sorted(_weekdays(nyse_all))
    cal = {}
    for mic in EXCH:
        raw = _parse(get, mic)
        lo, hi = min(raw), max(raw)
        sess = _weekdays(raw)
        for s in UNSCHEDULED.get(mic, []):
            if lo <= s <= hi:
                sess.add(s)
        cal[mic] = (sess, lo, hi)

    rows = []
    for i in range(1, len(nyse) - 1):
        d, p, n = nyse[i], nyse[i - 1], nyse[i + 1]
        if d < START:
            continue
        for mic in EXCH:
            sess, lo, hi = cal[mic]
            if p < lo or n > hi:
                continue
            # closed_next: no session on any local date in (d, n]
            closed = 1.0
            x = _next_day(d)
            while x <= n:
                if x in sess:
                    closed = 0.0
                    break
                x = _next_day(x)
            # anchored: a session on some local date in (p, d]
            anch = 0.0
            x = _next_day(p)
            while x <= d:
                if x in sess:
                    anch = 1.0
                    break
                x = _next_day(x)
            rows.append((d, mic + ".closed_next", closed))
            rows.append((d, mic + ".anchored", anch))
    return rows
