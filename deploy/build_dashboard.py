#!/usr/bin/env python3
"""Složí dashboard do jednoho souboru (style.css a app.js inline): CSP webu povoluje jen inline skripty a styly
(`script-src 'unsafe-inline'; style-src 'unsafe-inline'`, stejně jako u ostatních dashboardů), externí soubory blokuje.

  build_dashboard.py DASHBOARD_DIR OUT.html
"""
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
html = (d / "index.html").read_text(encoding="utf-8")
css = (d / "style.css").read_text(encoding="utf-8").replace("</style", "<\\/style")
js = (d / "app.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
html, n1 = re.subn(r'<link rel="stylesheet" href="style\.css[^"]*">', lambda m: f"<style>\n{css}</style>", html)
html, n2 = re.subn(r'<script src="app\.js[^"]*" defer></script>\n?', "", html)
html, n3 = re.subn(r"</body>", lambda m: f"<script>\n{js}</script>\n</body>", html)
assert (n1, n2, n3) == (1, 1, 1), (n1, n2, n3)
Path(sys.argv[2]).write_text(html, encoding="utf-8")
