"""File store that mirrors the collections of the artifact database (one JSON file per document).

  days/<date>.json  trades/<variant>-<ticker>-<entry_date>.json  scores/<date>.json  prices/<date>.json
  runs/<date>.json  history.json ({"updated", "data"})  config.json
"""
import json
import os
import subprocess

COLLECTIONS = ("days", "trades", "scores", "prices", "runs")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


class Store:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def p(self, *a):
        return os.path.join(self.root, *a)

    def put(self, coll, doc_id, doc):
        write_json(self.p(coll, f"{doc_id}.json"), doc)

    def get(self, coll, doc_id):
        return read_json(self.p(coll, f"{doc_id}.json"))

    def ids(self, coll):
        d = self.p(coll)
        return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []

    def all(self, coll):
        return [self.get(coll, i) for i in self.ids(coll)]

    def latest_day(self):
        ids = self.ids("days")
        return self.get("days", ids[-1]) if ids else None

    def history(self):
        return (read_json(self.p("history.json"), {}) or {}).get("data") or {}

    def put_history(self, data, date):
        write_json(self.p("history.json"), {"updated": date, "data": data})

    def config(self):
        return read_json(self.p("config.json"))

    # -- git backup of the data directory (own repo; push only when a remote is configured)
    def commit(self, message):
        def git(*a):
            return subprocess.run(["git", "-C", self.root, *a], capture_output=True, text=True)
        if not os.path.isdir(self.p(".git")):
            git("init", "-q", "-b", "main")
        git("add", "-A")
        if git("diff", "--cached", "--quiet").returncode == 0:
            return "nothing to commit"
        r = git("-c", "user.name=evening-bot", "-c", "user.email=evening-bot@localhost", "commit", "-q", "-m", message)
        if r.returncode != 0:
            return "commit failed: " + r.stderr.strip()[:200]
        if git("remote").stdout.strip():
            p = git("push", "-q", "origin", "HEAD")
            if p.returncode != 0:
                return "push failed: " + p.stderr.strip()[:200]
        return "committed"
