#!/usr/bin/env python3
"""Release pinning.

  python tools/pin.py v1.0.1        write VERSION and MANIFEST.sha256 from the working tree (then commit + tag)
  python tools/pin.py --install     pin the CURRENT repo release for the daily run: writes ~/.config/cpb/pin.json
                                    (outside the repo, so a run can never "re-pin" itself)

The daily run refuses to trade when the files differ from MANIFEST.sha256 or MANIFEST differs from the pin.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
from cpb import canon, runner  # noqa: E402

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
if sys.argv[1] == "--install":
    version, problems = runner.version_info(ROOT, require_pin=False)
    if problems:
        raise SystemExit("nelze připnout: " + "; ".join(problems))
    canon.write_json(runner.pin_file(), {"tag": version["tag"], "manifest_sha256": version["manifest_sha256"],
                                         "pinned_at": canon.ms_iso(int(__import__("time").time() * 1000))})
    print(f"připnuto {version['tag']} {version['manifest_sha256'][:16]} -> {runner.pin_file()}")
else:
    tag = sys.argv[1]
    if not tag.startswith("v"):
        raise SystemExit("tag ve tvaru vX.Y.Z")
    canon.write_text(os.path.join(ROOT, "VERSION"), tag[1:] + "\n")
    canon.write_text(os.path.join(ROOT, "MANIFEST.sha256"), runner.manifest_text(ROOT))
    print(open(os.path.join(ROOT, "MANIFEST.sha256")).read(), end="")
    print(f"VERSION {tag}. Dál: python tools/verify_manifest.py, commit, git tag {tag}, push, pak tools/pin.py --install")
