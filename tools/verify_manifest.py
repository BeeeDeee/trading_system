#!/usr/bin/env python3
"""Fail if pinned files differ from MANIFEST.sha256 (and, unless --no-pin, from ~/.config/cpb/pin.json)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
from cpb import runner  # noqa: E402

version, problems = runner.version_info(ROOT, require_pin="--no-pin" not in sys.argv)
if problems:
    sys.exit("\n".join(problems))
print(f"OK: {version['tag']} MANIFEST {version['manifest_sha256'][:16]} odpovídá souborům" +
      ("" if "--no-pin" in sys.argv else " i připnutí"))
