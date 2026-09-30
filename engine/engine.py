#!/usr/bin/env python3
"""crypto-paper-bot engine CLI (stdlib only). All times UTC.

  run --date YYYY-MM-DD [--dry-run]        full daily run (fetch, check, settle, claude -p, lock, snapshot, fill)
  notify --date D                           notification text of the run record (first line OK/UPOZORNĚNÍ/CHYBA)
  report [--publish DIR]                    dashboard (public/index.html + public/data.json), optional atomic publish
  verify                                    MANIFEST vs pin + structural hash-chain check
  validate SCORES.json --date D             validation report of an LLM output file
  apply-redenomination COIN RATIO --effective-date D --evidence URL [--new-symbol X] [--reason TEXT]
  correction --date D --message TEXT [--refers-to PATH]

RATIO = new units per old unit (1000 for a 1000x re-base; price / 1000, quantity x 1000).
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from cpb import canon, chain, corrections, llm, pipeline, runner  # noqa: E402
from cpb.fetch import LiveMarket  # noqa: E402


def load_cfg(repo):
    return canon.read_json(os.path.join(repo, "config", "config.json"))


def cmd_run(a):
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    D = a.date or today
    if D > today:
        raise SystemExit("datum v budoucnosti")
    repo = REPO
    if a.dry_run:
        tmp = tempfile.mkdtemp(prefix="cpb-dry-")
        shutil.copytree(REPO, os.path.join(tmp, "repo"), ignore=shutil.ignore_patterns(".git", "public", "__pycache__"))
        repo = os.path.join(tmp, "repo")
        print(f"suchý běh v {repo} (nic se necommituje ani nepublikuje)")
    version, problems = runner.version_info(REPO, require_pin=not a.allow_unpinned)
    clock = pipeline.Clock()
    cfg = load_cfg(repo)
    if problems:
        rdir = os.path.join(repo, "runs", D, f"failed-{canon.ms_iso(clock.now_ms())[11:19].replace(':', '')}")
        os.makedirs(rdir, exist_ok=True)
        rec = runner._write_record(repo, rdir, {"type": "run", "date": D, "status": "failed", "version": version,
                                                "error": "připnutá verze nesedí: " + "; ".join(problems)})
        print(runner.notify_text(rec))
        return 2
    if D != today:
        raise SystemExit("rozhodovací běh jde jen pro dnešek (snímek musí být živý); zmeškané dny se doplní automaticky "
                         "při dalším běhu jako 'catchup'")
    key = os.environ.get("COINGECKO_DEMO_KEY") or None
    llm_runner = llm.run_claude
    if a.no_llm:
        if not a.dry_run:
            raise SystemExit("--no-llm jen se --dry-run")
        llm_runner = lambda *args: {"exit_code": None, "session_id": None, "timed_out": False, "skipped": True}
    with runner.Lock(repo):
        status, rec = runner.execute_run(repo, cfg, D, lambda rec, d: LiveMarket(rec, key), llm_runner, clock, version, a.dry_run)
    if status == "exists":
        print(f"běh {D} už existuje ({rec['status']}), nic se nezapisuje")
        return 0
    print(runner.notify_text(rec))
    return 0 if status in ("ok", "warning") else 1


def cmd_notify(a):
    p = os.path.join(REPO, "runs", a.date, "run.json")
    rec = canon.read_json(p)
    if rec is None:
        import glob
        fails = sorted(glob.glob(os.path.join(REPO, "runs", a.date, "failed-*", "run.json")))
        rec = canon.read_json(fails[-1]) if fails else {"status": "failed", "date": a.date, "error": "běh nezapsal žádný záznam"}
    print(runner.notify_text(rec))


def cmd_report(a):
    from cpb import report
    out = report.build(REPO, load_cfg(REPO), os.path.join(REPO, "public"))
    print(f"dashboard: {out}")
    if a.publish:
        report.publish(os.path.join(REPO, "public"), a.publish)
        print(f"publikováno do {a.publish}")


def cmd_verify(a):
    version, problems = runner.version_info(REPO, require_pin=not a.allow_unpinned)
    problems += chain.verify(REPO)
    print(json.dumps(version, indent=1))
    if problems:
        print("\n".join("CHYBA: " + p for p in problems))
        return 1
    print(f"OK: verze {version['tag']} připnuta, řetězec {len(chain.entries(REPO))} záznamů neporušen")
    return 0


def cmd_validate(a):
    cfg = load_cfg(REPO)
    u = pipeline.current_universe(REPO, a.date)
    raw = json.load(open(a.scores, encoding="utf-8"))
    clean, rep = llm.validate(raw, [c["coin"] for c in u["coins"]], a.date)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 1 if rep["errors"] else 0


def cmd_redenom(a):
    with runner.Lock(REPO):
        rec = corrections.redenomination(REPO, pipeline.Clock(), a.coin, float(a.ratio), a.effective_date, a.new_symbol, a.evidence, a.reason)
    print(json.dumps({k: rec[k] for k in ("kind", "params", "value_before", "value_after", "this_hash")}, ensure_ascii=False, indent=1))


def cmd_correction(a):
    with runner.Lock(REPO):
        rec = corrections.note(REPO, pipeline.Clock(), a.date, a.message, a.refers_to)
    print(rec["this_hash"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run"); p.add_argument("--date"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-unpinned", action="store_true", help="jen pro vývoj/suchý běh")
    p.add_argument("--no-llm", action="store_true", help="jen se --dry-run: přeskočí claude -p (test dat a plnění)")
    p.set_defaults(f=cmd_run)
    p = sp.add_parser("notify"); p.add_argument("--date", required=True); p.set_defaults(f=cmd_notify)
    p = sp.add_parser("report"); p.add_argument("--publish"); p.set_defaults(f=cmd_report)
    p = sp.add_parser("verify"); p.add_argument("--allow-unpinned", action="store_true"); p.set_defaults(f=cmd_verify)
    p = sp.add_parser("validate"); p.add_argument("scores"); p.add_argument("--date", required=True); p.set_defaults(f=cmd_validate)
    p = sp.add_parser("apply-redenomination"); p.add_argument("coin"); p.add_argument("ratio")
    p.add_argument("--effective-date", required=True); p.add_argument("--evidence", required=True)
    p.add_argument("--new-symbol"); p.add_argument("--reason", default=""); p.set_defaults(f=cmd_redenom)
    p = sp.add_parser("correction"); p.add_argument("--date", required=True); p.add_argument("--message", required=True)
    p.add_argument("--refers-to"); p.set_defaults(f=cmd_correction)
    a = ap.parse_args(argv)
    return a.f(a) or 0


if __name__ == "__main__":
    sys.exit(main())
