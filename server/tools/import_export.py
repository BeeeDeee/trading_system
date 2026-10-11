#!/usr/bin/env python3
"""Merge an export of the artifact database into the server data dir (cutover: bring over the days the cloud task
wrote after the first seed). Existing documents are never overwritten; differing ones are reported.

  import_export.py EXPORT_DIR DATA_DIR      EXPORT_DIR has days/ trades/ scores/ prices/ runs/ history/main.json
"""
import json
import os
import shutil
import sys


def main(src, dst):
    added = same = differ = 0
    for coll in ("days", "trades", "scores", "prices", "runs"):
        d = os.path.join(src, coll)
        if not os.path.isdir(d):
            continue
        os.makedirs(os.path.join(dst, coll), exist_ok=True)
        for f in sorted(os.listdir(d)):
            t = os.path.join(dst, coll, f)
            if not os.path.exists(t):
                shutil.copy(os.path.join(d, f), t)
                added += 1
            elif json.load(open(t)) == json.load(open(os.path.join(d, f))):
                same += 1
            else:
                differ += 1
                print(f"DIFFERS (kept server copy): {coll}/{f}")
    h = os.path.join(src, "history", "main.json")
    if os.path.exists(h):
        new = json.load(open(h))
        t = os.path.join(dst, "history.json")
        old = json.load(open(t)) if os.path.exists(t) else {"updated": "", "data": {}}
        if new.get("updated", "") > old.get("updated", ""):
            shutil.copy(h, t)
            print(f"history.json updated: {old.get('updated')} -> {new.get('updated')}")
    print(f"added {added}, identical {same}, differing {differ}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
