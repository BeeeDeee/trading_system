"""python -m evening run|report|notify|replay-check  (run from the server/ directory)"""
import argparse
import os
import sys

from . import report, runner

DEFAULT_DATA = os.environ.get("EVENING_DATA") or os.path.join(runner.REPO, "data")


def main():
    ap = argparse.ArgumentParser(prog="evening")
    ap.add_argument("--data", default=DEFAULT_DATA)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="today's run (idempotent)")
    r.add_argument("--date")
    r.add_argument("--dry-run", action="store_true", help="work on a copy of the data dir, write nothing, no commit")
    r.add_argument("--force", action="store_true", help="run before 16:10 ET")
    r.add_argument("--scores", help="inject scores.json (no LLM)")
    r.add_argument("--free", help="inject free.json")
    r.add_argument("--model")
    r.add_argument("--wait", type=int, default=600, help="seconds to wait once for a late daily row")
    n = sub.add_parser("notify", help="text of the notification for a date")
    n.add_argument("--date", required=True)
    b = sub.add_parser("report", help="publish the dashboard")
    b.add_argument("--publish", required=True, help="web root, e.g. /srv/evening")
    a = ap.parse_args()
    if a.cmd == "run":
        rec = runner.run(a.data, date=a.date, dry_run=a.dry_run, force=a.force, scores_file=a.scores, free_file=a.free, model=a.model, wait_s=a.wait)
        print(runner.notification(a.data.rstrip("/") + (".dry" if a.dry_run else ""), rec["date"]) if rec.get("status") != "exists" else f"běh {rec['date']} už existuje")
        sys.exit(1 if rec.get("status") == "failed" else 0)
    if a.cmd == "notify":
        print(runner.notification(a.data, a.date))
    if a.cmd == "report":
        print(report.publish(a.data, a.publish))


main()
