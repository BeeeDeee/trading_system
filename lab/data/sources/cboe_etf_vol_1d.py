import csv
import io

SERIES = ["RVXCLS", "VXEEMCLS", "VXEWZCLS", "VXFXICLS", "VXXLECLS", "VXGDXCLS", "VXSLVCLS", "EVZCLS"]


def fetch(get):
    """FRED fredgraph.csv, one request per series, full history, latest vintage.
    Missing values ('.' or empty) are dropped, not filled."""
    rows = []
    for sid in SERIES:
        body = get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid})
        text = body.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
        if len(header) != 2 or header[1].strip() != sid:
            raise ValueError("unexpected FRED header for " + sid + ": " + ",".join(header))
        seen = set()
        for rec in reader:
            if len(rec) < 2:
                continue
            d = rec[0].strip()[:10]
            v = rec[1].strip()
            if v in ("", ".", "NA", "NaN"):
                continue
            if d in seen:
                continue
            seen.add(d)
            rows.append((d, sid, float(v)))
    return rows
