import json
import datetime

# CFTC Commitments of Traders, legacy report, futures only (CFTC Public Reporting, Socrata dataset 6dca-aqww).
# Every report is stamped on its PUBLIC RELEASE DATE (3:30 pm ET), not on the as-of date it describes.
# Series keys: <cftc contract market code>.nc_long / .nc_short / .open_interest / .asof_date
#   asof_date = days since 1970-01-01 of the report's as-of date (a float32-safe number; Tuesday, or Monday
#   when the Tuesday was a federal holiday).
#
# Release-date rule (the source gives only the as-of date): the Friday of the as-of week; when a federal
# holiday (or a known executive-order closure) falls on Wednesday, Thursday or Friday of that week the
# release moves to the next business day after the Friday (the Monday). Holidays on Monday or Tuesday do
# not move it (a Tuesday holiday makes the as-of date the Monday). This was checked against the CFTC
# release schedule pages and announcements (e.g. Friday 2021-01-01 holiday -> Monday 2021-01-04;
# Thursday 2026-01-01 holiday -> Monday 2026-01-05; Monday MLK 2026-01-19 -> normal Friday 2026-01-23).
#
# Government shutdowns: the catch-up publication dates could not be established from primary documents
# (only plans were found), so every report whose release was delayed by a shutdown is DROPPED:
#   as-of 2013-10-01 .. 2013-11-05   (October 2013 shutdown and catch-up)
#   as-of 2018-12-24 .. 2019-03-05   (December 2018 - January 2019 shutdown and catch-up)
#   as-of >= 2025-09-30              (October-November 2025 shutdown and catch-up; series stops here)
#
# Contract codes: none changed in 2004-2025. Names changed only cosmetically within a code (exchange
# renames IMM->CME, CSCE/NYBOT->ICE Futures U.S., "WHEAT"->"WHEAT-SRW", "CRUDE OIL, LIGHT SWEET"->"WTI-PHYSICAL",
# "NAT GAS NYME"->"NATURAL GAS", "COPPER- #1"->"COPPER-GRADE #1", "GASOLINE BLENDSTOCK (RBOB)"->"GASOLINE RBOB").
# RBOB (111659) has reports only from 2006-02-14 (the code was created for RBOB; no chaining to the old
# NY Harbor unleaded contract, which has no data under this code).

URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

NAMES = {
    "088691": ("GOLD",),
    "084691": ("SILVER",),
    "067651": ("CRUDE OIL, LIGHT", "WTI-PHYSICAL"),
    "023651": ("NATURAL GAS", "NAT GAS NYME"),
    "111659": ("GASOLINE",),
    "002602": ("CORN",),
    "001602": ("WHEAT",),
    "005602": ("SOYBEANS",),
    "085692": ("COPPER",),
    "076651": ("PLATINUM",),
    "075651": ("PALLADIUM",),
    "080732": ("SUGAR NO. 11",),
    "099741": ("EURO FX", "EUROPEAN CURRENCY UNIT"),
    "097741": ("JAPANESE YEN",),
    "096742": ("BRITISH POUND", "POUND STERLING"),
    "232741": ("AUSTRALIAN DOLLAR",),
    "090741": ("CANADIAN DOLLAR",),
    "092741": ("SWISS FRANC",),
}

FIRST_ASOF = datetime.date(2003, 12, 30)   # released Friday 2004-01-02
LAST_ASOF = datetime.date(2025, 9, 23)     # last report released before the October 2025 shutdown
DROP_WINDOWS = [
    (datetime.date(2013, 10, 1), datetime.date(2013, 11, 5)),
    (datetime.date(2018, 12, 24), datetime.date(2019, 3, 5)),
]

# executive-order / ad hoc closures of federal offices (conservative: they can only delay a release)
CLOSURES = [
    datetime.date(2004, 6, 11), datetime.date(2007, 1, 2), datetime.date(2009, 12, 24),
    datetime.date(2012, 12, 24), datetime.date(2014, 12, 24), datetime.date(2014, 12, 26),
    datetime.date(2018, 12, 5), datetime.date(2019, 12, 24), datetime.date(2020, 12, 24),
    datetime.date(2025, 1, 9),
]

EPOCH = datetime.date(1970, 1, 1)
ONE = datetime.timedelta(days=1)


def _nth(y, m, wd, n):
    d = datetime.date(y, m, 1)
    if n > 0:
        return d + datetime.timedelta(days=(wd - d.weekday()) % 7 + 7 * (n - 1))
    nxt = datetime.date(y + 1, 1, 1) if m == 12 else datetime.date(y, m + 1, 1)
    last = nxt - ONE
    return last - datetime.timedelta(days=(last.weekday() - wd) % 7)


def _obs(d):
    if d.weekday() == 5:
        return d - ONE
    if d.weekday() == 6:
        return d + ONE
    return d


def _holidays():
    hol = set(CLOSURES)
    for y in range(2003, 2028):
        hol.add(_obs(datetime.date(y, 1, 1)))
        hol.add(_nth(y, 1, 0, 3))
        hol.add(_nth(y, 2, 0, 3))
        hol.add(_nth(y, 5, 0, -1))
        if y >= 2021:
            hol.add(_obs(datetime.date(y, 6, 19)))
        hol.add(_obs(datetime.date(y, 7, 4)))
        hol.add(_nth(y, 9, 0, 1))
        hol.add(_nth(y, 10, 0, 2))
        hol.add(_obs(datetime.date(y, 11, 11)))
        hol.add(_nth(y, 11, 3, 4))
        hol.add(_obs(datetime.date(y, 12, 25)))
    return hol


def _release(asof, hol):
    fri = asof + datetime.timedelta(days=4 - asof.weekday())
    # a Wednesday as-of date (only 2007-01-03, after the Ford funeral closure of Tuesday 2007-01-02) is
    # released conservatively like a delayed week: the Monday
    if asof.weekday() == 2 or (fri - ONE - ONE) in hol or (fri - ONE) in hol or fri in hol:
        rel = fri + datetime.timedelta(days=3)
        while rel in hol or rel.weekday() >= 5:
            rel = rel + ONE
        return rel
    return fri


def _num(x):
    if x is None:
        return None
    return float(x)


def fetch(get):
    hol = _holidays()
    raw = []
    offset = 0
    while True:
        body = get(URL, params={
            "$select": "report_date_as_yyyy_mm_dd,cftc_contract_market_code,market_and_exchange_names,"
                       "noncomm_positions_long_all,noncomm_positions_short_all,open_interest_all",
            "$where": "cftc_contract_market_code in(" + ",".join("'" + c + "'" for c in sorted(NAMES)) + ")",
            "$order": "report_date_as_yyyy_mm_dd,cftc_contract_market_code",
            "$limit": "10000",
            "$offset": str(offset),
        })
        page = json.loads(body)
        raw.extend(page)
        if len(page) < 10000:
            break
        offset += 10000

    rows = []
    seen = set()
    bad_weekday = []
    for r in raw:
        code = r["cftc_contract_market_code"]
        asof = datetime.date.fromisoformat(r["report_date_as_yyyy_mm_dd"][:10])
        if asof < FIRST_ASOF or asof > LAST_ASOF:
            continue
        if any(a <= asof <= b for a, b in DROP_WINDOWS):
            continue
        name = r["market_and_exchange_names"]
        if not name.startswith(NAMES[code]):
            raise ValueError("unexpected market name for code %s: %s" % (code, name))
        if asof.weekday() not in (0, 1, 2):
            bad_weekday.append(str(asof))
            continue
        nl = _num(r.get("noncomm_positions_long_all"))
        ns = _num(r.get("noncomm_positions_short_all"))
        oi = _num(r.get("open_interest_all"))
        if nl is None or ns is None or oi is None or nl < 0 or ns < 0 or oi <= 0:
            continue
        if (code, asof) in seen:
            raise ValueError("duplicate report %s %s" % (code, asof))
        seen.add((code, asof))
        rel = str(_release(asof, hol))
        rows.append((rel, code + ".nc_long", nl))
        rows.append((rel, code + ".nc_short", ns))
        rows.append((rel, code + ".open_interest", oi))
        rows.append((rel, code + ".asof_date", float((asof - EPOCH).days)))
    if bad_weekday:
        raise ValueError("as-of dates that are not Monday/Tuesday: " + ",".join(sorted(set(bad_weekday))[:40]))
    keys = set()
    for d, k, v in rows:
        if (d, k) in keys:
            raise ValueError("two reports released on the same day: %s %s" % (d, k))
        keys.add((d, k))
    return rows
