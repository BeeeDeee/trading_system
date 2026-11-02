import csv
import io
import datetime

COUNTRIES = ["US", "JP", "GB", "CA", "AU", "CH", "SE", "DE", "KR", "MX", "ZA", "CL", "IL", "PL", "NZ"]


def _easter(year):
    a = year % 19
    b = year // 100
    c = year % 100
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
    return datetime.date(year, month, day)


def _nth_weekday(year, month, weekday, n):
    d = datetime.date(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    return d + datetime.timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    if month == 12:
        d = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        d = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    return d - datetime.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d):
    if d.weekday() == 5:
        return d - datetime.timedelta(days=1)
    if d.weekday() == 6:
        return d + datetime.timedelta(days=1)
    return d


def _holidays_for_month_start(year, month):
    """Holidays of XNYS that can fall in the first days of a month (regular rules plus
    special closures). Only days 1-7 of each month matter here."""
    hol = set()
    if month == 1:
        ny = datetime.date(year, 1, 1)
        if ny.weekday() != 5:  # Saturday new year is not observed on Friday by NYSE
            hol.add(_observed(ny) if ny.weekday() == 6 else ny)
        if year == 2007:
            hol.add(datetime.date(2007, 1, 2))
        if year == 2025:
            hol.add(datetime.date(2025, 1, 9))
    if month == 7:
        hol.add(_observed(datetime.date(year, 7, 4)))
    if month == 9:
        hol.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    if month == 4:
        hol.add(_easter(year) - datetime.timedelta(days=2))  # Good Friday (may fall in days 1-7)
    if month == 3:
        hol.add(_easter(year) - datetime.timedelta(days=2))  # Good Friday (early Easter)
    if month == 5 and False:
        pass
    if month == 6:
        if year == 2004:
            hol.add(datetime.date(2004, 6, 11))
    if month == 12:
        if year == 2018:
            hol.add(datetime.date(2018, 12, 5))
    if month == 10 and year == 2012:
        hol.add(datetime.date(2012, 10, 29))
        hol.add(datetime.date(2012, 10, 30))
    if month == 9 and year == 2001:
        for dd in (11, 12, 13, 14):
            hol.add(datetime.date(2001, 9, dd))
    return hol


def _first_session(year, month):
    d = datetime.date(year, month, 1)
    hol = _holidays_for_month_start(year, month)
    while d.weekday() >= 5 or d in hol:
        d += datetime.timedelta(days=1)
    return d


def fetch(get):
    """OECD MEI 10y government bond yields (monthly average, percent) via FRED fredgraph.csv.
    The value of month m is stamped on the first XNYS session of month m+2 (conservative PIT rule).
    Each stamped date carries the new monthly value; the framework forward-fills it as a step series.
    One row per stamp date and key; missing months ('.') are dropped."""
    rows = []
    for cc in COUNTRIES:
        sid = "IRLTLT01" + cc + "M156N"
        body = get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid, "cosd": "1997-01-01"})
        text = body.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
        if len(header) != 2 or header[1].strip() != sid:
            raise ValueError("unexpected FRED header for " + sid + ": " + ",".join(header))
        seen = set()
        for rec in reader:
            if len(rec) < 2:
                continue
            ds = rec[0].strip()[:10]
            v = rec[1].strip()
            if v in ("", ".", "NA", "NaN"):
                continue
            y = int(ds[0:4])
            m = int(ds[5:7])
            if y < 1998:
                continue
            m2 = m + 2
            y2 = y
            if m2 > 12:
                m2 -= 12
                y2 += 1
            stamp = _first_session(y2, m2).isoformat()
            if stamp in seen:
                continue
            seen.add(stamp)
            rows.append((stamp, cc, float(v)))
    return rows
