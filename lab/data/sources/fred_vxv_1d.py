import csv
import io


def fetch(get):
    """FRED VXVCLS (CBOE S&P 500 3-Month Volatility Index), daily close, full history."""
    body = get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": "VXVCLS"})
    text = body.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    if len(header) != 2 or header[1].strip() != "VXVCLS":
        raise ValueError("unexpected header: %r" % (header,))
    rows = []
    seen = set()
    for rec in reader:
        if len(rec) < 2:
            continue
        d = rec[0].strip()
        v = rec[1].strip()
        if not d or v in ("", ".", "NA", "NaN"):
            continue
        if len(d) != 10 or d[4] != "-" or d[7] != "-":
            raise ValueError("bad date: %r" % (d,))
        if d in seen:
            continue
        seen.add(d)
        x = float(v)
        if x <= 0:
            continue
        rows.append((d, "VXVCLS", x))
    return rows
