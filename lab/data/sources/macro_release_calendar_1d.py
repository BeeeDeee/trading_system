import re
import datetime

ALFRED = "https://alfred.stlouisfed.org/release/downloaddates"
FED = "https://www.federalreserve.gov/monetarypolicy/"

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7,
    "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}
MONTHS3 = {k[:3]: v for k, v in MONTHS.items()}
SPECIAL_CLOSURES = {
    datetime.date(2001, 9, 11), datetime.date(2001, 9, 12), datetime.date(2001, 9, 13),
    datetime.date(2001, 9, 14), datetime.date(2004, 6, 11), datetime.date(2007, 1, 2),
    datetime.date(2012, 10, 29), datetime.date(2012, 10, 30), datetime.date(2018, 12, 5),
    datetime.date(2025, 1, 9),
}


def _nth_weekday(y, m, wd, n):
    first = datetime.date(y, m, 1)
    off = (wd - first.weekday()) % 7
    return first + datetime.timedelta(days=off + 7 * (n - 1))


def _last_monday(y, m):
    d = datetime.date(y, m + 1, 1) - datetime.timedelta(days=1) if m < 12 else datetime.date(y, 12, 31)
    while d.weekday() != 0:
        d -= datetime.timedelta(days=1)
    return d


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


def _observed(d, saturday_back=True):
    if d.weekday() == 5:
        return d - datetime.timedelta(days=1) if saturday_back else None
    if d.weekday() == 6:
        return d + datetime.timedelta(days=1)
    return d


def _holidays(y):
    hs = set()
    hs.add(_observed(datetime.date(y, 1, 1), saturday_back=False))  # New Year: no Friday closure
    hs.add(_nth_weekday(y, 1, 0, 3))  # MLK
    hs.add(_nth_weekday(y, 2, 0, 3))  # Presidents
    hs.add(_easter(y) - datetime.timedelta(days=2))  # Good Friday
    hs.add(_last_monday(y, 5))  # Memorial
    if y >= 2022:
        hs.add(_observed(datetime.date(y, 6, 19)))  # Juneteenth
    hs.add(_observed(datetime.date(y, 7, 4)))
    hs.add(_nth_weekday(y, 9, 0, 1))  # Labor
    hs.add(_nth_weekday(y, 11, 3, 4))  # Thanksgiving
    hs.add(_observed(datetime.date(y, 12, 25)))
    hs.discard(None)
    return hs


def _sessions(y0, y1):
    hol = set(SPECIAL_CLOSURES)
    for y in range(y0, y1 + 1):
        hol |= _holidays(y)
    d = datetime.date(y0, 1, 1)
    end = datetime.date(y1, 12, 31)
    out = []
    while d <= end:
        if d.weekday() < 5 and d not in hol:
            out.append(d)
        d += datetime.timedelta(days=1)
    return out


def _alfred_dates(get, rid):
    body = get(ALFRED, params={"rid": str(rid), "ff": "txt"}).decode("utf-8", "replace")
    out = set()
    for line in body.splitlines():
        line = line.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", line):
            out.add(datetime.date.fromisoformat(line))
    return sorted(out)


def _text(html):
    t = html.replace("&nbsp;", " ").replace("&ndash;", "-").replace("&#8211;", "-").replace("&#160;", " ")
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t)


HEAD = re.compile(
    r"([A-Z][a-z]+) (\d{1,2})(?: ?- ?(?:([A-Z][a-z]+) )?(\d{1,2}))? ?(\([^)]*\))? ?Meeting - (\d{4})"
)


def _fomc_historical(get, year):
    html = get(FED + "fomchistorical%d.htm" % year).decode("utf-8", "replace")
    out = set()
    for m in HEAD.finditer(_text(html)):
        m1, d1, m2, d2, paren, yr = m.groups()
        if int(yr) != year:
            continue
        if paren and re.search(r"unscheduled|cancel|notation", paren, re.I):
            continue
        if paren:
            raise ValueError("unknown FOMC heading qualifier %r in %d" % (paren, year))
        if d2 is None:
            mon, day = m1, int(d1)
        else:
            mon, day = (m2 or m1), int(d2)
        out.add(datetime.date(year, MONTHS3[mon[:3]], day))
    return out


def _fomc_recent(get):
    html = get(FED + "fomccalendars.htm").decode("utf-8", "replace")
    heads = list(re.finditer(r"(\d{4}) FOMC Meetings", html))
    out = set()
    pair = re.compile(
        r"fomc-meeting__month[^>]*><strong>([^<]*)</strong></div>\s*<div class=\"fomc-meeting__date[^>]*>([^<]*)</div>"
    )
    for i, h in enumerate(heads):
        year = int(h.group(1))
        end = heads[i + 1].start() if i + 1 < len(heads) else len(html)
        for mon, dt in pair.findall(html[h.start():end]):
            dt = dt.strip()
            if re.search(r"notation|unscheduled|cancel", dt, re.I):
                continue
            mm = re.fullmatch(r"(\d{1,2})(?:-(\d{1,2}))?\*?", dt)
            if not mm:
                raise ValueError("unknown FOMC date cell %r" % dt)
            first, last = mm.group(1), mm.group(2)
            parts = [p.strip() for p in mon.split("/")]
            if last is None:
                month_name, day = parts[0], int(first)
            else:
                month_name, day = parts[-1], int(last)
            out.add(datetime.date(year, MONTHS3[month_name[:3]], day))
    return out


def _check_counts(dates, label, lo, hi, y0, y1):
    for y in range(y0, y1 + 1):
        n = len([d for d in dates if d.year == y])
        if n < lo or n > hi:
            raise ValueError("%s: %d dates in %d (expected %d..%d): %s" % (
                label, n, y, lo, hi, [d.isoformat() for d in dates if d.year == y]))


def fetch(get):
    cpi = [d for d in _alfred_dates(get, 10) if d.year >= 1997]
    emp = [d for d in _alfred_dates(get, 50) if d.year >= 1997]
    fomc = set()
    for y in range(1997, 2021):
        fomc |= _fomc_historical(get, y)
    fomc |= _fomc_recent(get)
    # 2003-09-15 is an extra meeting the Fed page lists next to the scheduled 2003-09-16 one
    fomc.discard(datetime.date(2003, 9, 15))
    fomc = sorted(fomc)

    _check_counts(cpi, "cpi", 9, 14, 1998, 2025)
    _check_counts(emp, "empsit", 9, 14, 1998, 2025)
    _check_counts(fomc, "fomc", 7, 8, 1998, 2026)

    sess = _sessions(1997, 2027)
    idx = {d: i for i, d in enumerate(sess)}

    def to_idx(d):
        # an event on a day the NYSE is closed is counted on the next XNYS session
        k = 0
        while d not in idx:
            d += datetime.timedelta(days=1)
            k += 1
            if k > 10:
                raise ValueError("no session near %s" % d)
        return idx[d]

    ev = {
        "next_cpi_sessions": sorted(set(to_idx(d) for d in cpi)),
        "next_empsit_sessions": sorted(set(to_idx(d) for d in emp)),
        "next_fomc_sessions": sorted(set(to_idx(d) for d in fomc)),
    }
    first = idx[datetime.date(1998, 1, 2)]
    # last session for which all three schedules still have a later date; no partial rows
    last = min(v[-1] for v in ev.values()) - 1

    rows = []
    ptr = {k: 0 for k in ev}
    for i in range(first, last + 1):
        vals = {}
        for k, lst in ev.items():
            p = ptr[k]
            while p < len(lst) and lst[p] <= i:
                p += 1
            ptr[k] = p
            if p < len(lst):
                vals[k] = lst[p] - i
        if len(vals) < 3:
            continue
        vals["next_release_sessions"] = min(vals.values())
        ds = sess[i].isoformat()
        for k, v in vals.items():
            rows.append((ds, k.replace("next_", "next_"), float(v)))
    return rows
