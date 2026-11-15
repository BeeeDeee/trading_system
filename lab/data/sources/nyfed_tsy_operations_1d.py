import json
import re
import datetime

START = datetime.date(2008, 1, 2)
END = datetime.date(2020, 12, 31)
BUCKETS = ["b0_3", "b3_7", "b7_10", "b10_20", "b20_30", "tips"]
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
          "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def _nth_weekday(year, month, weekday, n):
    d = datetime.date(year, month, 1)
    d += datetime.timedelta(days=(weekday - d.weekday()) % 7)
    return d + datetime.timedelta(days=7 * (n - 1))


def _last_weekday(year, month, weekday):
    d = datetime.date(year + (month == 12), (month % 12) + 1, 1) - datetime.timedelta(days=1)
    return d - datetime.timedelta(days=(d.weekday() - weekday) % 7)


def _easter(y):
    a = y % 19
    b, c = divmod(y, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return datetime.date(y, month, day)


def _holidays(year):
    hs = set()
    # fixed-date holidays: Saturday -> no closure for New Year (previous Friday open), otherwise Friday; Sunday -> Monday
    for (m, d, is_newyear) in [(1, 1, True), (7, 4, False), (12, 25, False)]:
        x = datetime.date(year, m, d)
        if x.weekday() == 5:
            if not is_newyear:
                hs.add(x - datetime.timedelta(days=1))
        elif x.weekday() == 6:
            hs.add(x + datetime.timedelta(days=1))
        else:
            hs.add(x)
    if year >= 2022:
        x = datetime.date(year, 6, 19)
        if x.weekday() == 5:
            hs.add(x - datetime.timedelta(days=1))
        elif x.weekday() == 6:
            hs.add(x + datetime.timedelta(days=1))
        else:
            hs.add(x)
    hs.add(_nth_weekday(year, 1, 0, 3))      # MLK
    hs.add(_nth_weekday(year, 2, 0, 3))      # Presidents
    hs.add(_easter(year) - datetime.timedelta(days=2))   # Good Friday
    hs.add(_last_weekday(year, 5, 0))        # Memorial
    hs.add(_nth_weekday(year, 9, 0, 1))      # Labor
    hs.add(_nth_weekday(year, 11, 3, 4))     # Thanksgiving
    return hs


SPECIAL = {datetime.date(2012, 10, 29), datetime.date(2012, 10, 30), datetime.date(2018, 12, 5)}


def _sessions(start, end):
    hol = set()
    for y in range(start.year - 1, end.year + 2):
        hol |= _holidays(y)
    hol |= SPECIAL
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in hol:
            out.append(d)
        d += datetime.timedelta(days=1)
    return out


def _parse_periods(html):
    """(start, end) of each tentative-schedule period on the NY Fed schedule archive page."""
    rows = re.findall(r'<td class="somatdL">(.*?)</td>\s*<td class="somatdR">(.*?)</td>', html, re.S)
    pat = re.compile(r"([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?\s*-\s*([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})")
    periods = set()
    for left, right in rows:
        if "tot_operation_schedule_" not in right:
            continue
        txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", left)).strip()
        m = pat.search(txt)
        if not m:
            continue
        m1 = MONTHS.get(m.group(1)[:3].lower())
        m2 = MONTHS.get(m.group(4)[:3].lower())
        if m1 is None or m2 is None:
            continue
        y2 = int(m.group(6))
        e = datetime.date(y2, m2, int(m.group(5)))
        y1 = int(m.group(3)) if m.group(3) else y2
        s = datetime.date(y1, m1, int(m.group(2)))
        if s > e:
            s = datetime.date(y1 - 1, m1, int(m.group(2)))
        periods.add((s, e))
    return sorted(periods)


def _bucket(op_date, a, b):
    mid = a + (b - a) / 2
    m = (mid - op_date).days / 365.25
    if m < 0.75:
        return None
    if m < 3:
        return "b0_3"
    if m < 7:
        return "b3_7"
    if m < 10:
        return "b7_10"
    if m < 20:
        return "b10_20"
    return "b20_30"


def fetch(get):
    html = get("https://www.newyorkfed.org/markets/tot_operation_schedule.html").decode("utf-8", "replace")
    periods = _parse_periods(html)
    body = get("https://markets.newyorkfed.org/api/tsy/all/results/summary/search.json",
               params={"startDate": "2008-01-02", "endDate": "2020-12-31"})
    auctions = json.loads(body)["treasury"]["auctions"]

    sessions = _sessions(START, END)
    index = {d: i for i, d in enumerate(sessions)}

    def session_on_or_after(d):
        lo, hi = 0, len(sessions)
        while lo < hi:
            mid = (lo + hi) // 2
            if sessions[mid] < d:
                lo = mid + 1
            else:
                hi = mid
        return lo

    next_flag = {}   # (session index, bucket) -> 1
    today_par = {}   # (session index, bucket) -> par
    seen = set()
    for x in auctions:
        oid = x.get("operationId")
        if oid in seen:
            continue
        seen.add(oid)
        if x.get("operationDirection") != "P" or x.get("auctionStatus") != "Results":
            continue
        typ = x.get("operationType")
        if typ not in ("Outright Coupon Purchase", "Outright TIPS Purchase"):
            continue
        od = datetime.date.fromisoformat(x["operationDate"][:10])
        if od < START or od > END:
            continue
        if typ == "Outright TIPS Purchase":
            bucket = "tips"
        else:
            a = datetime.date.fromisoformat(x["maturityRangeStart"][:10])
            b = datetime.date.fromisoformat(x["maturityRangeEnd"][:10])
            bucket = _bucket(od, a, b)
            if bucket is None:
                continue
        t = session_on_or_after(od)
        if t >= len(sessions):
            continue
        try:
            par = float(x.get("totalParAmtAccepted") or 0)
        except ValueError:
            par = 0.0
        today_par[(t, bucket)] = today_par.get((t, bucket), 0.0) + par
        # announcement stamp: the operation counts as announced from the first day of the tentative
        # schedule period that contains it (a schedule is public before its first operation).
        # Operations outside every known schedule period are dropped from buy_next.
        ann = None
        for (s, e) in periods:
            if s <= od <= e:
                ann = s
                break
        if ann is None or t == 0:
            continue
        p = t - 1
        if sessions[p] >= ann:
            next_flag[(p, bucket)] = 1

    rows = []
    for i, d in enumerate(sessions):
        ds = d.isoformat()
        for bk in BUCKETS:
            rows.append((ds, "buy_next." + bk, float(next_flag.get((i, bk), 0))))
        for bk in BUCKETS:
            rows.append((ds, "buy_today_par." + bk, float(today_par.get((i, bk), 0.0))))
    if not periods:
        raise ValueError("no schedule periods parsed")
    return rows
