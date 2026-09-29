#!/usr/bin/env python3
"""Fail if engine/config differ from MANIFEST.sha256 or from the pins in the task prompt."""
import hashlib, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
bad = []
for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
    h, p = line.split("  ", 1)
    if sha(p) != h:
        bad.append(f"MANIFEST: {p} changed")
prompt = (ROOT / "task" / "task_prompt.md").read_text()
tag = re.search(r"^TAG = (.*)$", prompt, re.M).group(1)
for key, p in (("ENGINE_SHA256", "engine/engine.py"), ("CONFIG_SHA256", "config/config.json")):
    if re.search(rf"^{key} = (.*)$", prompt, re.M).group(1) != sha(p):
        bad.append(f"task prompt pin {key} does not match {p}")
if (ROOT / "VERSION").read_text().strip() != tag.lstrip("v"):
    bad.append("VERSION does not match TAG in task prompt")
sys.exit("\n".join(bad)) if bad else print(f"OK: {tag} manifest and prompt pins consistent")
