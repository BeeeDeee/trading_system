import json
import re
import datetime

URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
CAL_START = datetime.date(1998, 1, 1)

# NYSE full-day closures outside the regular holiday rules (all inside the calendar used here)
SPECIAL_CLOSURES = [
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",
    "2004-06-11", "2007-01-02", "2012-10-29", "2012-10-30",
    "2018-12-05", "2025-01-09",
]


def _easter(y):
    a = y % 19
    b = y // 100
    c = y % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime.date(y, month, day)


def _nth_weekday(y, month, weekday, n):
    d = datetime.date(y, month, 1)
    shift = (weekday - d.weekday()) % 7
    return d + datetime.timedelta(days=shift + 7 * (n - 1))


def _last_weekday(y, month, weekday):
    if month == 12:
        d = datetime.date(y + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        d = datetime.date(y, month + 1, 1) - datetime.timedelta(days=1)
    return d - datetime.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d):
    # Saturday -> Friday, Sunday -> Monday
    if d.weekday() == 5:
        return d - datetime.timedelta(days=1)
    if d.weekday() == 6:
        return d + datetime.timedelta(days=1)
    return d


def _holidays(y):
    h = set()
    ny = datetime.date(y, 1, 1)
    if ny.weekday() == 6:
        h.add(ny + datetime.timedelta(days=1))
    elif ny.weekday() != 5:  # a Saturday Jan 1 closes nothing (Dec 31 stays open)
        h.add(ny)
    h.add(_nth_weekday(y, 1, 0, 3))                      # Martin Luther King Jr. Day
    h.add(_nth_weekday(y, 2, 0, 3))                      # Washington's Birthday
    h.add(_easter(y) - datetime.timedelta(days=2))       # Good Friday
    h.add(_last_weekday(y, 5, 0))                        # Memorial Day
    if y >= 2022:
        h.add(_observed(datetime.date(y, 6, 19)))        # Juneteenth
    h.add(_observed(datetime.date(y, 7, 4)))             # Independence Day
    h.add(_nth_weekday(y, 9, 0, 1))                      # Labor Day
    h.add(_nth_weekday(y, 11, 3, 4))                     # Thanksgiving
    h.add(_observed(datetime.date(y, 12, 25)))           # Christmas
    return h


def _sessions(start, end):
    hol = set()
    for y in range(start.year, end.year + 1):
        hol |= _holidays(y)
    for s in SPECIAL_CLOSURES:
        hol.add(datetime.date.fromisoformat(s))
    out = []
    d = start
    one = datetime.timedelta(days=1)
    while d <= end:
        if d.weekday() < 5 and d not in hol:
            out.append(d)
        d += one
    return out


def _term_months(s):
    y = re.search(r"(\d+)-Year", s)
    m = re.search(r"(\d+)-Month", s)
    if y is None and m is None:
        raise ValueError("unparsable term " + s)
    return (int(y.group(1)) if y else 0) * 12 + (int(m.group(1)) if m else 0)


def _load(get):
    rows = []
    page = 1
    while True:
        body = get(URL, params={
            "filter": "security_type:in:(Note,Bond)",
            "fields": "auction_date,announcemt_date,original_security_term,offering_amt,"
                      "inflation_index_security,floating_rate,cusip",
            "sort": "auction_date,cusip",
            "page[size]": "10000",
            "page[number]": str(page),
        })
        d = json.loads(body)
        rows.extend(d["data"])
        if page >= d["meta"]["total-pages"]:
            break
        page += 1
    return rows


def fetch(get):
    raw = _load(get)

    # nominal fixed-coupon Notes/Bonds with original term >= 5 years, one record per (cusip, auction date)
    auctions = {}
    for r in raw:
        if r["inflation_index_security"] != "No" or r["floating_rate"] != "No":
            continue
        if _term_months(r["original_security_term"]) < 60:
            continue
        ann = r["announcemt_date"]
        auc = r["auction_date"]
        if ann in (None, "", "null") or auc in (None, "", "null"):
            continue
        if ann > auc:
            continue  # inconsistent record: never use an auction before its announcement
        auctions[(r["cusip"], auc)] = (auc, ann, float(r["offering_amt"]))

    recs = sorted(auctions.values())
    last_day = max(a[0] for a in recs)
    last_ann = max(a[1] for a in recs)
    end = datetime.date.fromisoformat(last_day) + datetime.timedelta(days=400)
    sess = _sessions(CAL_START, end)
    sess_iso = [s.isoformat() for s in sess]
    n = len(sess)

    # index of the session on which each calendar day falls (a closed day maps to the next session)
    nxt = {}
    j = n - 1
    d = end
    one = datetime.timedelta(days=1)
    while d >= CAL_START:
        if j >= 0 and sess[j] >= d:
            while j - 1 >= 0 and sess[j - 1] >= d:
                j -= 1
            nxt[d.isoformat()] = j
        d -= one

    # (session index of the auction, announcement date, offering)
    lst = []
    for auc, ann, amt in recs:
        if auc < CAL_START.isoformat():
            continue
        lst.append((nxt[auc], ann, amt))
    lst.sort()
    if not lst:
        raise ValueError("no qualifying auctions")

    first_vis = min(a[1] for a in lst)
    out = []
    p = 0
    last_pos = -1  # session index of the most recent auction with index <= i
    m = len(lst)
    for i in range(n):
        ds = sess_iso[i]
        if ds < first_vis or ds > last_ann:
            continue
        while p < m and lst[p][0] <= i:
            p += 1
        # most recent auction up to and including session i (all are announced by then)
        if p > 0:
            last_pos = lst[p - 1][0]
            out.append((ds, "last_coupon5_sessions", float(i - last_pos)))
        # next auction strictly after i that is announced on or before ds
        q = p
        while q < m and lst[q][1] > ds:
            q += 1
        if q < m:
            tgt = lst[q][0]
            tot = 0.0
            k = q
            while k < m and lst[k][0] == tgt:
                if lst[k][1] <= ds:
                    tot += lst[k][2]
                k += 1
            out.append((ds, "next_coupon5_sessions", float(tgt - i)))
            out.append((ds, "next_coupon5_offering_usd", tot))
    return out
