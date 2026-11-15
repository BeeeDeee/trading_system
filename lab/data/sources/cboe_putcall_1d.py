import re
import datetime

BASE = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/"

# (file, key prefix). Current files take precedence over archive files on the same date.
FILES = [
    ("equitypc.csv", "equity", False),
    ("equitypcarchive.csv", "equity", True),
    ("totalpc.csv", "total", False),
    ("totalpcarchive.csv", "total", True),
    ("indexpc.csv", "index", False),
    ("indexpcarchive.csv", "index", True),
]

DATE_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{2,4})\s*$")


def _num(s):
    s = s.strip().replace(",", "").replace('"', "")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse(text):
    out = {}
    for ln in text.splitlines():
        parts = ln.split(",")
        if len(parts) < 5:
            continue
        m = DATE_RE.match(parts[0])
        if not m:
            continue
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000 if y < 70 else 1900
        try:
            dt = datetime.date(y, mo, d)
        except ValueError:
            continue
        call = _num(parts[1])
        put = _num(parts[2])
        tot = _num(parts[3])
        ratio = _num(parts[4])
        if ratio is None and call and put is not None and call > 0:
            ratio = put / call
        out[dt.isoformat()] = (call, put, tot, ratio)
    return out


def fetch(get):
    merged = {}  # (prefix) -> {date: tuple}
    for name, prefix, archive in FILES:
        body = get(BASE + name)
        rows = _parse(body.decode("utf-8", "replace"))
        cur = merged.setdefault(prefix, {})
        for dt, v in rows.items():
            if archive and dt in cur:
                continue
            cur[dt] = v
    result = []
    for prefix, rows in merged.items():
        for dt, (call, put, tot, ratio) in rows.items():
            if ratio is not None and ratio > 0:
                result.append((dt, prefix + "_pc", ratio))
            if call is not None and call > 0:
                result.append((dt, prefix + "_call", call))
            if put is not None and put > 0:
                result.append((dt, prefix + "_put", put))
    result.sort()
    return result
