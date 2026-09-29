#!/usr/bin/env python3
"""Update the 'Pinned release' block of task/task_prompt.md for a new tag.

  python tools/pin_prompt.py v1.1.0      # hashes are taken from the working tree
  python tools/verify_manifest.py        # afterwards: MANIFEST.sha256 matches

Release procedure: edit -> tests -> CHANGELOG -> pin_prompt.py (writes VERSION, MANIFEST.sha256,
the pins in the prompt and build/db_mirror_engine.json) -> verify_manifest.py -> commit -> git tag
-> push -> upload build/db_mirror_engine.json to database engine/main -> paste task/task_prompt.md
into the scheduled task.
"""
import hashlib, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
tag = sys.argv[1]
sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
f = ROOT / "task" / "task_prompt.md"
s = f.read_text()
for key, val in (("TAG", tag), ("ENGINE_SHA256", sha("engine/engine.py")), ("CONFIG_SHA256", sha("config/config.json"))):
    s, n = re.subn(rf"^{key} = .*$", f"{key} = {val}", s, flags=re.M)
    assert n == 1, key
f.write_text(s)
(ROOT / "VERSION").write_text(tag.lstrip("v") + "\n")
lines = "".join(f"{sha(p)}  {p}\n" for p in ("engine/engine.py", "config/config.json"))
(ROOT / "MANIFEST.sha256").write_text(lines)
# Fallback copy for the scheduled task (used when `git clone` fails, e.g. private repo).
# Upload to the journal database with ArtifactData action "update" (merge): collection engine, doc main.
mirror = {"filename": "engine.py", "sha256": sha("engine/engine.py"),
          "source": (ROOT / "engine" / "engine.py").read_text()}
(ROOT / "build").mkdir(exist_ok=True)
(ROOT / "build" / "db_mirror_engine.json").write_text(json.dumps(mirror, ensure_ascii=False))
print(lines, end="")
print("pinned", tag)
print("NEXT: upload build/db_mirror_engine.json to database engine/main (update), else the db-mirror fallback fails the hash check")
