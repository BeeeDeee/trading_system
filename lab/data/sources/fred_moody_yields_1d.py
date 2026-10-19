import csv
import io
import re


def fetch(get):
    """FRED daily Moody's seasoned Aaa (DAAA) and Baa (DBAA) corporate bond yields, percent."""
    rows = []
    for sid in ("DAAA", "DBAA"):
        body = get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": sid, "cosd": "1986-01-01"},
        )
        text = body.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
        if len(header) != 2 or header[1].strip() != sid:
            raise ValueError("unexpected header for %s: %r" % (sid, header))
        for rec in reader:
            if len(rec) < 2:
                continue
            d = rec[0].strip()
            v = rec[1].strip()
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                continue
            if v in ("", ".", "NA", "NaN"):
                continue  # FRED marks missing days (holidays) with '.'; not filled
            rows.append((d, sid, float(v)))
    return rows
