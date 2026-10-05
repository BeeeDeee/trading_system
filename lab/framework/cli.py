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
import shlex
import sys
import tempfile
from pathlib import Path

import yaml

from lab.framework import catalog, invocations, report, tick
from lab.framework.blackboard import Lab, LabError
from lab.framework.paths import default_paths
from lab.framework.states import ACTORS, FUNNEL, HUMAN, S

STAGED_COMMANDS = {"send", "inbox", "context", "check", "try", "fetch"}


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
    s = sub.add_parser("check", help="what the framework will say about a card (staged mode)")
    s.add_argument("card", type=Path)
    s = sub.add_parser("fetch", help="run a fetcher with network and validate (staged mode, Archivist); no values shown")
    s.add_argument("source", type=Path)
    s.add_argument("--entry", type=Path, required=True, help="draft catalog entry (YAML)")
    sub.add_parser("try", help="G0 + grid neighbors + own tests on a synthetic market (staged mode, Builder)")
    s = sub.add_parser("agent", help="run one headless agent now (claude -p), apply its outbox, tick")
    s.add_argument("agent")
    s.add_argument("--hyp", help="hypothesis to work on (leased for the run)")
    s.add_argument("--task", help="task text for the agent (default: propose / answer)")
    s.add_argument("--dry-run", action="store_true", help="render the workspace, print the command, run nothing")
    s = sub.add_parser("rerun-g0", help="re-run G0 on the current code (owner, after a framework fix; not a trial)")
    s.add_argument("hid")
    s.add_argument("--reason", required=True)
    s = sub.add_parser("import-local", help="owner: make fred_macro / fred_dtb3 (on disk) loadable as signal series")
    s.add_argument("dataset", choices=["fred_macro", "fred_dtb3"])
    s = sub.add_parser("void", help="owner: undo a gate rejection caused by a framework bug (-> DATA_READY)")
    s.add_argument("hid")
    s.add_argument("--reason", required=True)
    s = sub.add_parser("factsheets", help="build the descriptive dataset fact sheets for the Scout")
    s.add_argument("--rebuild", action="store_true")
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
    elif args.cmd == "check":
        errors, warnings = invocations.check_card(ws, str(args.card))
        for e in errors:
            print(f"error: {e}")
        for w in warnings:
            print(f"warning: {w}")
        print("ok" if not errors else f"{len(errors)} error(s): fix them before sending")
        return 1 if errors else 0
    elif args.cmd == "fetch":
        from lab.framework import ingest
        src = invocations._inside(ws, str(args.source))
        draft = yaml.safe_load(invocations._inside(ws, str(args.entry)).read_text())
        errors = ingest.draft_errors(draft, draft.get("id", "")) if isinstance(draft, dict) else ["entry is not a mapping"]
        try:
            rows, http = ingest.fetch(src)
            _, report, _ = ingest.plan(rows, draft if isinstance(draft, dict) else {})
            print(f"fetched {len(rows)} rows with {len(http.urls)} requests, {http.bytes // 1024} kB")
            print("\n".join(ingest.report_text(report)))
        except Exception as e:  # noqa: BLE001 - agent code; the message goes back to the agent
            errors.append(f"{type(e).__name__}: {e}")
        for e in errors:
            print(f"error: {e}")
        print("ok: ready for DATA_READY" if not errors else "NOT ready")
        return 1 if errors else 0
    elif args.cmd == "try":
        from lab.framework import dryrun
        ok, lines = dryrun.run(ws)
        print("\n".join(lines))
        print("ok: ready for IMPL_DONE" if ok else "NOT ready: fix the failures above")
        return 0 if ok else 1
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
        case "context" | "check" | "try" | "fetch":
            raise LabError(f"`lab {args.cmd}` only exists inside an agent run")
        case "agent":
            return _run_agent(lab, args)
        case "rerun-g0":
            from lab.framework import canaries
            from lab.framework.evaluator import QlabEvaluator
            from lab.framework.gates import run_gate
            if lab.hypothesis(args.hid)["status"] != S.DATA_READY:
                raise LabError(f"{args.hid} is not in DATA_READY")
            if not (lab.paths.strategies / args.hid / "strategy.py").exists():
                raise LabError(f"{args.hid} has no strategy yet")
            if not canaries.passed_for_current(lab):
                raise LabError("canaries have not passed for the current framework; run `lab canaries`")
            lab.send("ALERT", "system", "human", args.hid, {"severity": "info",
                     "text": f"G0 re-run by the owner: {args.reason}"})
            out = run_gate(lab, args.hid, "G0", QlabEvaluator())
            print(f"G0 {'passed' if out.passed else 'failed: ' + str(out.reason_code)}")
            for line in tick.tick(lab, QlabEvaluator()):
                print(line)
        case "import-local":
            from lab.framework import ingest
            rep = ingest.import_local(lab.paths.home, lab.paths.catalog, args.dataset)
            print(f"{args.dataset}: {len(rep['keys'])} series, {rep['n_rows']} rows, holdout from {rep['holdout_from']}")
        case "void":
            h = lab.hypothesis(args.hid)
            if h["status"] != S.REJECTED or h["reject_stage"] not in ("G0", "G1", "G2", "G3", "IMPLEMENTED", "GATE_1",
                                                                      "GATE_2", "GATE_3"):
                raise LabError(f"{args.hid} was not rejected by a gate (status {h['status']}, "
                               f"stage {h['reject_stage']})")
            lab.transition(args.hid, S.DATA_READY, HUMAN, f"void ({h['reject_stage']}: {h['reject_code']}): "
                           f"{args.reason}. Earlier trials stay counted in the family.")
            print(f"{args.hid} -> DATA_READY; run `lab rerun-g0 {args.hid}` or let the Builder fix it")
        case "factsheets":
            from lab.framework import factsheets
            print(factsheets.get(lab.paths.home, lab.paths.catalog, rebuild=args.rebuild))
    return 0


def _default_task(lab: Lab, agent: str, hid: str | None) -> str:
    if agent == "builder":
        return "fix" if hid and lab.inbox("builder", hid) else "implement"
    if agent == "skeptic":
        return "review"
    if agent == "archivist":
        return "data requests"
    if agent == "librarian":
        return "lessons"
    return "answer" if hid else "propose"


def _run_agent(lab: Lab, args) -> int:
    """One agent run by hand (step 3: every agent's first runs are supervised by the owner)."""
    from lab.framework import headless
    runner = headless.ClaudeRunner(args.agent)
    tick.tick(lab, None)
    task = args.task or _default_task(lab, args.agent, args.hyp)
    inv = invocations.start(lab, args.agent, args.hyp, model=runner.model, task=task)
    print(f"invocation {inv.id}\nworkspace  {inv.workspace}")
    if args.dry_run:
        cmd = headless.command(runner.spec, headless.task_prompt(inv))
        print(" ".join(shlex.quote(c) if i != cmd.index("--system-prompt") + 1 else "<prompt>"
                       for i, c in enumerate(cmd)))
        print(invocations.finish(lab, inv.id, 130, note="dry run"))
        return 0
    res = runner(lab, inv)
    outcome = invocations.finish(lab, inv.id, res.exit_code, transcript_path=res.transcript_path, usage=res.usage,
                                 violations=res.violations, timed_out=res.timed_out, note=res.note)
    u = res.usage
    print(f"outcome    {outcome}  (exit {res.exit_code}, {u.get('wall_s')} s, {u.get('n_turns')} turns, "
          f"tokens in/out {u.get('tokens_in')}/{u.get('tokens_out')}, model {u.get('model')})")
    for v in res.violations:
        print(f"VIOLATION  {v}")
    if res.note:
        print(f"note       {res.note}")
    print(f"transcript {res.transcript_path}")
    for line in tick.tick(lab, None):
        print(line)
    if u.get("result_text"):
        print("\n" + u["result_text"])
    return 0 if outcome == "applied" else 1


if __name__ == "__main__":
    sys.exit(main())
