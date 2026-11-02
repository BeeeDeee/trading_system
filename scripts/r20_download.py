"""Research 20: daily raw snapshot of the 26 ETFs (prereg §3.1). Downloads only, computes nothing.

    uv run python scripts/r20_download.py            # one snapshot dir data/raw/yahoo_forward20/<UTC date>/

Stores the raw Yahoo chart JSON (1y, daily, dividends and splits; gzip) per ticker plus the Nasdaq closes of the last
days as an independent check, and a manifest with sha256 and the Yahoo-vs-Nasdaq close mismatches (> 0.3 %).
Idempotent per UTC day. Exit code 1 if any ticker failed on Yahoo.
"""

import gzip
import hashlib
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np

from qlab.research19 import setup as S19
from qlab.research20 import data as D
from qlab.research20 import setup as S

today = datetime.now(timezone.utc).date()
dest = S.RAW / today.isoformat()
if (dest / "manifest.json").exists():
    print(f"{dest} exists: nothing to do")
    sys.exit(0)
dest.mkdir(parents=True, exist_ok=True)
files, errors, mismatches, nasdaq_errors = {}, {}, {}, {}
for tk in S19.ALL_ETF:
    try:
        j = D.fetch_yahoo(tk, "1y")
        blob = gzip.compress(json.dumps(j, separators=(",", ":")).encode())
        (dest / f"{tk}.json.gz").write_bytes(blob)
        files[tk] = hashlib.sha256(blob).hexdigest()
        p = D.parse_yahoo(j)
    except D.FetchError as e:
        errors[tk] = str(e)
        continue
    try:
        nq = D.nasdaq_closes(tk, today - timedelta(days=25), limit=25)
        (dest / f"{tk}.nasdaq.json").write_text(json.dumps({str(k): v for k, v in nq.items()}))
        bad = [(str(d), float(c), nq[d]) for d, c in zip(p["dates"], p["close"])
               if d in nq and abs(c / nq[d] - 1) * 100 > S.NASDAQ_TOL_PCT]
        if bad:
            mismatches[tk] = bad
    except D.FetchError as e:
        nasdaq_errors[tk] = str(e)
    time.sleep(0.5)
(dest / "manifest.json").write_text(json.dumps({
    "created_at": datetime.now(timezone.utc).isoformat(), "sha256": files, "yahoo_errors": errors,
    "nasdaq_errors": nasdaq_errors, "close_mismatch_gt_tol": mismatches, "tolerance_pct": S.NASDAQ_TOL_PCT}, indent=1))
print(f"{dest}: {len(files)}/{len(S19.ALL_ETF)} tickers, yahoo errors {len(errors)}, nasdaq errors {len(nasdaq_errors)}, "
      f"close mismatches {sum(len(v) for v in mismatches.values())}")
sys.exit(1 if errors else 0)
