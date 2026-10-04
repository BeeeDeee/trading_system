"""`lab` command line.

Two modes:
- direct (the owner, the orchestrator, framework jobs): reads and writes lab.db.
- staged (inside an agent run, `LAB_WORKSPACE` is set): only `send`, `inbox` and `context` work, and `send`
  appends to the workspace outbox instead of touching lab.db. The framework applies the outbox after the
  run with the role recorded for that invocation, so setting environment variables gains an agent nothing.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

from lab.framework import catalog, invocations, report, tick
from lab.framework.blackboard import Lab, LabError
from lab.framework.paths import default_paths
from lab.framework.states import ACTORS, FUNNEL, HUMAN, S

STAGED_COMMANDS = {"send", "inbox", "context"}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ws = os.environ.get("LAB_WORKSPACE")
    try:
        if ws:
            if args.cmd not in STAGED_COMMANDS:
                raise LabError(f"`lab {args.cmd}` is not available inside an agent run "
                               f"(allowed: {', '.join(sorted(STAGED_COMMANDS))})")
            return _staged(args, Path(ws))
        return _direct(args)
    except LabError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lab", description="Research lab blackboard CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="create lab.db")
    sub.add_parser("status", help="funnel and rejection reasons")
    s = sub.add_parser("list", help="list hypotheses")
    s.add_argument("--status", choices=[x.value for x in FUNNEL])
    s = sub.add_parser("show", help="card, status and history of a hypothesis")
    s.add_argument("hid")
    s = sub.add_parser("trace", help="time-ordered conversation around a hypothesis")
    s.add_argument("hid")
    s = sub.add_parser("new", help="enter a hypothesis card by hand (as human)")
    s.add_argument("card", type=Path)
    s = sub.add_parser("move", help="owner override of a state (logged as human)")
    s.add_argument("hid")
    s.add_argument("to", choices=[x.value for x in FUNNEL])
    s.add_argument("--reason", required=True)
    s.add_argument("--code")
    s = sub.add_parser("send", help="send a typed message (staged inside an agent run)")
    s.add_argument("type")
    s.add_argument("--to", required=True)
    s.add_argument("--hyp")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--payload", help="JSON object")
    g.add_argument("--payload-file", type=Path, help="JSON or YAML file")
    g.add_argument("--card", type=Path, help="card YAML -> {card: ...} for NEW_HYPOTHESIS / REVISION")
    s.add_argument("--reason", help="with --card for a REVISION")
    s.add_argument("--as", dest="actor", default=HUMAN, choices=ACTORS, help="direct mode only")
    s = sub.add_parser("inbox", help="unread messages")
    s.add_argument("agent", nargs="?")
    sub.add_parser("context", help="the current agent run (staged mode)")
    s = sub.add_parser("tick", help="deterministic phase: react to messages, run due gates")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--no-gates", action="store_true", help="react to messages only")
    g.add_argument("--stub-gates", action="store_true", help="tests/demo only, needs LAB_ALLOW_STUB=1")
    s = sub.add_parser("canaries", help="test the judge; required after every change of the framework")
    s.add_argument("--quick", action="store_true", help="fewer bootstrap/random-entry samples")
    s = sub.add_parser("invocations", help="recent agent runs")
    s.add_argument("--limit", type=int, default=30)
    sub.add_parser("catalog", help="datasets in the catalog")
    s = sub.add_parser("demo", help="walking skeleton with stub agents in a throwaway lab")
    s.add_argument("--dir", type=Path)
    return p


def _payload(args, base: Path) -> dict:
    if args.card:
        card = invocations.read_card(base, str(args.card)) if base else yaml.safe_load(args.card.read_text())
        drop = {"id", "version", "status", "history", "terminal", "author_agent"}
        payload = {"card": {k: v for k, v in card.items() if k not in drop}}
        if args.reason:
            payload["reason"] = args.reason
        return payload
    if args.payload_file:
        return yaml.safe_load(args.payload_file.read_text())
    return json.loads(args.payload)


def _staged(args, ws: Path) -> int:
    if args.cmd == "context":
        print((ws / "context.json").read_text())
    elif args.cmd == "inbox":
        print((ws / "inbox.json").read_text())
    else:
        n = invocations.stage(ws, args.type, args.to, args.hyp, _payload(args, ws))
        print(f"staged message #{n} ({args.type} -> {args.to}); applied after the run")
    return 0


def _direct(args) -> int:
    if args.cmd == "demo":
        from lab.demo import print_report, run_demo
        root = args.dir or Path(tempfile.mkdtemp(prefix="lab-demo-"))
        print_report(run_demo(root))
        return 0
    lab = Lab.open(default_paths())
    match args.cmd:
        case "init":
            print(f"lab.db ready at {lab.paths.db}")
        case "status":
            for k, v in report.funnel(lab).items():
                print(f"{k:<16}{v}")
            for r in report.rejection_reasons(lab):
                print(f"{r['status']} at {r['reject_stage']}: {r['reject_code']} x{r['n']}")
        case "list":
            for h in lab.hypotheses(args.status):
                print(f"{h['id']}  v{h['version']}  {h['status']:<15} {h['family']:<28} {h['title']}")
        case "show":
            lab.hypothesis(args.hid)
            print(lab.paths.card(args.hid).read_text())
        case "trace":
            for e in report.trace(lab, args.hid):
                print(f"{e['ts'][:19]}  {e['kind']:<10} {e['actor']:<10} {e['text']}")
        case "new":
            mid = lab.send("NEW_HYPOTHESIS", HUMAN, "system", None, _payload(
                argparse.Namespace(card=args.card, reason=None, payload=None, payload_file=None), None))
            print(f"message {mid} queued; `lab tick` creates the hypothesis")
        case "move":
            lab.transition(args.hid, S(args.to), HUMAN, args.reason, reason_code=args.code)
            print(f"{args.hid} -> {args.to}")
        case "send":
            mid = lab.send(args.type, args.actor, args.to, args.hyp, _payload(args, None))
            print(f"message {mid} stored")
        case "inbox":
            for m in lab.inbox(args.agent or HUMAN):
                print(f"#{m['id']} {m['type']} from {m['from_agent']} [{m['hypothesis_id'] or '-'}]: "
                      f"{report._summary(m['type'], json.loads(m['payload_json']))}")
        case "tick":
            evaluator = None
            if args.stub_gates:
                from lab.framework.gates import StubEvaluator
                evaluator = StubEvaluator(allow=os.environ.get("LAB_ALLOW_STUB") == "1")
            elif not args.no_gates:
                from lab.framework import canaries
                from lab.framework.evaluator import QlabEvaluator
                if not canaries.passed_for_current(lab):
                    raise LabError("canaries have not passed for the current framework; run `lab canaries`")
                evaluator = QlabEvaluator()
            for line in tick.tick(lab, evaluator):
                print(line)
        case "canaries":
            from lab.framework import canaries
            results = canaries.run_all(quick=args.quick)
            for r in results:
                print(f"{'ok  ' if r.ok else 'FAIL'} {r.name:<28} {r.detail}")
            ok = canaries.record(lab, results)
            print("canaries passed" if ok else "CANARIES FAILED: the judge is broken, no gate will run")
            return 0 if ok else 1
        case "invocations":
            for r in report.invocations(lab, args.limit):
                print(f"{r['started_at'][:19]}  {r['agent']:<10} {r['hypothesis_id'] or '-':<7} "
                      f"{r['outcome'] or 'running':<16} {r['error'] or ''}")
        case "catalog":
            for d in catalog.load(lab.paths.catalog).values():
                print(f"{d['id']:<22} {d['asset_class']:<12} {d['frequency']:<6} {d['range'][0]}..{d['range'][1]}"
                      f"  holdout {d['holdout_from']}  forward: {d['forward_source'] or '-'}")
        case "context":
            raise LabError("`lab context` only exists inside an agent run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
