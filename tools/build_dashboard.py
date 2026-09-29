#!/usr/bin/env python3
"""Build the publishable dashboard: inject demo data into dashboard.src.html.

  python tools/simulate.py            # writes dashboard/sample.json
  python tools/build_dashboard.py     # writes dashboard/dashboard.html (git-ignored)

The published artifact is dashboard.html; it shows the demo data only until the
first real run has written rows to the database.
"""
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
src = (ROOT / "dashboard" / "dashboard.src.html").read_text()
sample = (ROOT / "dashboard" / "sample.json").read_text().replace("</", "<\\/")
if "__SAMPLE__" not in src:
    sys.exit("placeholder __SAMPLE__ missing in dashboard.src.html")
(ROOT / "dashboard" / "dashboard.html").write_text(src.replace("__SAMPLE__", sample))
print("dashboard/dashboard.html written")
