"""Research 21 / F2: features (RSI(2), regimes, volume ratio, gap z, idiosyncratic z, earnings windows, open index).

Usage: python scripts/r21_build_features.py [sharadar_YYYY-MM-DD]
Needs data/derived/<snapshot>/panel_r5 and panel_r19_etf. Output: data/derived/<snapshot>/r21/ (git-ignored).
"""

import sys
import time

from qlab.research21 import features
from qlab.research21 import setup as S

snapshot = sys.argv[1] if len(sys.argv) > 1 else S.SHARADAR
t0 = time.time()
out = features.build(snapshot)
print(f"features in {out} ({time.time() - t0:.0f}s)")
