"""The Pipeline tab of the dashboard: where every hypothesis is, where it got stuck, what each agent produced.

Three views, all static HTML + inline SVG (a few lines of inline JS only switch the visible hypothesis):
1. the lab flow: the stages in a row with how many hypotheses reached each, died there (and why), and the
   live ones sitting on their stage;
2. "stuck or waiting": every live hypothesis with the time in its state, what the system is waiting for, and the
   blockers of the whole lab right now (pause, rate limit, quiet window, canaries, budget);
3. one hypothesis: its own pipeline strip (done / failed / active / not reached per stage, with times and
   revision loops) and below it, per stage, the outputs of the agents and the judge: card, data, the Builder's
   summary, the gate checks with values and thresholds, the event study, the Skeptic's checklist, lessons,
   and for every agent run its model, time, tokens and its own closing summary (from the transcript).
"""

import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import yaml

from lab.framework import agents as agents_mod
from lab.framework import headless, icons
from lab.framework.blackboard import Lab
from lab.framework.states import S, TERMINAL

STAGES = [("idea", "Idea", "Scout", "llm", "scout"), ("data", "Data", "Archivist + ingest", "mixed", "archivist"),
          ("build", "Build + G0", "Builder + G0", "mixed", "builder"), ("g1", "G1 basic", "judge", "code", None),
          ("g2", "G2 robust", "judge", "code", None), ("g3", "G3 trials", "judge", "code", None),
          ("skeptic", "Skeptic", "Skeptic", "llm", "skeptic"), ("g4", "G4 holdout", "judge", "code", None),
          ("paper", "Paper · G5", "paper runner", "mixed", "steward")]
KIND_LABEL = {"llm": "LLM agent", "mixed": "LLM + code", "code": "code, no LLM"}
ROLE = {"scout": "proposes hypotheses (data-blind)", "archivist": "finds data sources, writes fetchers",
        "builder": "implements strategy and diagnostic", "skeptic": "reviews before the holdout",
        "librarian": "writes lessons and knowledge", "chair": "merges families, parks, reopens",
        "steward": "weekly paper-trading report"}
STAGE_IDX = {k: i for i, (k, *_ ) in enumerate(STAGES)}
REJECT_TO_STAGE = {"IDEA": "idea", "SPECIFIED": "idea", "BLOCKED_DATA": "data", "DATA_READY": "build", "G0": "build",
                   "IMPLEMENTED": "g1", "G1": "g1", "GATE_1": "g2", "G2": "g2", "GATE_2": "g3", "G3": "g3",
                   "GATE_3": "skeptic", "SKEPTIC_REVIEW": "skeptic", "G4": "g4", "HOLDOUT": "paper",
                   "PAPER": "paper", "G5": "paper"}
# the state that means "this stage is running now"
ACTIVE_STATE = {"idea": {"IDEA"}, "data": {"BLOCKED_DATA"}, "build": {"DATA_READY"}, "g1": {"IMPLEMENTED"},
                "g2": {"GATE_1"}, "g3": {"GATE_2"}, "skeptic": {"GATE_3", "SKEPTIC_REVIEW"}, "g4": set(),
                "paper": {"HOLDOUT", "PAPER"}}
# the first state that proves the stage is done
DONE_STATE = {"idea": "SPECIFIED", "data": "DATA_READY", "build": "IMPLEMENTED", "g1": "GATE_1", "g2": "GATE_2",
              "g3": "GATE_3", "skeptic": "HOLDOUT", "g4": "HOLDOUT", "paper": "LIVE_CANDIDATE"}
ENTRY_STATE = {"idea": "IDEA", "data": "BLOCKED_DATA", "build": "DATA_READY", "g1": "IMPLEMENTED", "g2": "GATE_1",
               "g3": "GATE_2", "skeptic": "GATE_3", "g4": "SKEPTIC_REVIEW", "paper": "HOLDOUT"}
ORDER = [s.value for s in S]


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def fmt(v, nd=3) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v == int(v) and abs(v) >= 2:
            return f"{int(v):,}"
        return f"{v:.{nd}f}" if abs(v) < 1000 else f"{v:,.0f}"
    return esc(v)


def ago(ts: str | None, now: datetime) -> str:
    if not ts:
        return ""
    sec = (now - datetime.fromisoformat(ts)).total_seconds()
    if sec < 3600:
        return f"{max(sec // 60, 0):.0f} min"
    if sec < 172800:
        return f"{sec / 3600:.1f} h"
    return f"{sec / 86400:.1f} d"


def model_label(agent: str | None) -> str:
    if not agent:
        return ""
    try:
        m = agents_mod.spec(agent).model
    except KeyError:
        return ""
    name = m.replace("claude-", "").split("-")
    return (name[0].capitalize() + " " + ".".join(name[1:])) if len(name) > 1 else m


def running_now(lab: Lab) -> set[tuple[str, str | None]]:
    return {(r["agent"], r["hypothesis_id"]) for r in
            lab.con.execute("SELECT agent, hypothesis_id FROM agent_invocations WHERE outcome IS NULL")}


def mood_of(state: str, kind: str, agent: str | None, hid: str | None, running: set) -> str:
    if state == "done":
        return "done"
    if state == "failed":
        return "fail"
    if state in ("pending", "skipped"):
        return "sleep"
    if state == "active":
        if kind != "code" and ((agent, hid) in running or (agent, None) in running):
            return "work"
        return "wait"
    return "idle"


# ---------------------------------------------------------------------------- agent runs (with transcript summary)

def run_summaries(lab: Lab) -> dict:
    """invocation id -> {result_text, tools, ...} from the transcripts, cached (a finished run never changes)."""
    f = lab.paths.home / "cache" / "run_summaries.json"
    try:
        cache = json.loads(f.read_text()) if f.exists() else {}
    except (OSError, ValueError):
        cache = {}
    dirty = False
    for r in lab.con.execute("SELECT id, transcript_path FROM agent_invocations WHERE outcome IS NOT NULL"):
        if r["id"] in cache or not r["transcript_path"]:
            continue
        try:
            evs = headless.events(__import__("pathlib").Path(r["transcript_path"]))
        except OSError:
            continue
        calls = Counter(n for n, _ in headless.tool_calls(evs))
        u = headless.usage(evs)
        cache[r["id"]] = {"result_text": (u.get("result_text") or "")[:3000], "tools": dict(calls),
                          "wall_s": None, "model": u.get("model")}
        dirty = True
    if dirty:
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(cache))
        except OSError:
            pass
    return cache


def run_box(lab: Lab, inv_id: str | None, sums: dict, md) -> str:
    if not inv_id:
        return ""
    r = lab.con.execute("SELECT * FROM agent_invocations WHERE id = ?", (inv_id,)).fetchone()
    if r is None:
        return ""
    dur = ""
    if r["started_at"] and r["ended_at"]:
        dur = f"{(datetime.fromisoformat(r['ended_at']) - datetime.fromisoformat(r['started_at'])).total_seconds() / 60:.1f} min"
    sm = sums.get(inv_id) or {}
    tone = "good" if r["outcome"] == "applied" else "bad" if r["outcome"] else "warn"
    tools = ", ".join(f"{k} {v}" for k, v in sorted((sm.get("tools") or {}).items()))
    head = (f"<span class='pill {tone}'>{esc(r['outcome'] or 'running')}</span> <span class=mono>{esc(r['agent'])} · "
            f"{esc(r['model'])} · {esc(r['n_turns'] or '?')} turns · {esc(dur)} · "
            f"{fmt((r['tokens_in'] or 0) / 1000, 0)}k in / {fmt((r['tokens_out'] or 0) / 1000, 0)}k out</span>")
    body = ""
    if sm.get("result_text"):
        body += f"<div class='kb agentsays'>{md(sm['result_text'])}</div>"
    if tools:
        body += f"<div class=sub>tool calls: {esc(tools)}</div>"
    if r["error"]:
        body += f"<div class='note'>{esc(r['error'][:600])}</div>"
    return (f"<details class=run><summary>{head}</summary>{body or '<div class=sub>no transcript summary</div>'}</details>")


# ---------------------------------------------------------------------------- the story of one hypothesis

def story(lab: Lab, h) -> dict:
    """Per stage: state (done/failed/active/pending/skipped/parked), entered/left timestamps, loop counts."""
    hid, status = h["id"], h["status"]
    trans = lab.con.execute("SELECT * FROM transitions WHERE hypothesis_id = ? ORDER BY id", (hid,)).fetchall()
    first_to: dict[str, str] = {}
    for t in trans:
        first_to.setdefault(t["to_status"], t["ts"])
    reached = set(first_to)
    dead_stage = REJECT_TO_STAGE.get(h["reject_stage"] or "", None) if status in ("REJECTED", "RETIRED") else None
    parked = status == "PARKED"
    if parked:
        # parked from some state: the last non-PARKED state it was in
        prev = [t["from_status"] for t in trans if t["to_status"] == "PARKED"]
        dead_stage = REJECT_TO_STAGE.get(prev[-1] if prev else "IDEA", "idea")
    msgs = lab.con.execute("SELECT type FROM messages WHERE hypothesis_id = ?", (hid,)).fetchall()
    n_impl = sum(1 for m in msgs if m["type"] == "IMPL_DONE")
    n_obj = sum(1 for m in msgs if m["type"] == "OBJECTION")
    n_rev = sum(1 for m in msgs if m["type"] == "REVISION")
    live = status not in {str(t) for t in TERMINAL} and not parked
    out = {}
    for key, *_ in STAGES:
        if dead_stage == key:
            out[key] = "parked" if parked else "failed"
        elif live and status in ACTIVE_STATE[key]:
            out[key] = "active"
        elif key == "data" and "BLOCKED_DATA" not in reached:
            out[key] = "skipped" if "DATA_READY" in reached else "pending"
        elif DONE_STATE[key] in reached or (key == "skeptic" and dead_stage == "g4"):
            out[key] = "done"
        else:
            out[key] = "pending"
    if dead_stage:                                   # nothing after the place of death was reached
        for key, *_ in STAGES[STAGE_IDX[dead_stage] + 1:]:
            out[key] = "pending"
    entered = {key: first_to.get(ENTRY_STATE[key]) if key in ENTRY_STATE else None for key, *_ in STAGES}
    entered["idea"] = first_to.get("IDEA")
    return {"stages": out, "entered": entered, "loops": {"build": max(n_impl - 1, 0), "skeptic": n_obj, "idea": n_rev},
            "reached": reached, "dead_stage": dead_stage}


# ---------------------------------------------------------------------------- what is it waiting for

def lab_blockers(lab: Lab, now: datetime) -> list[tuple[str, str]]:
    """(severity, text) for everything that stops agents or gates right now."""
    from lab.framework import canaries
    from lab.orchestrator import Policy, used_today
    out = []
    home = lab.paths.home
    if (home / "PAUSE").exists():
        out.append(("bad", "PAUSE file: agent runs are paused (" + (home / "PAUSE").read_text().strip()[:200] + ")"))
    rl = home / "RATE_LIMITED_UNTIL"
    if rl.exists() and datetime.fromisoformat(rl.read_text().strip()) > now:
        out.append(("bad", f"rate limited until {rl.read_text().strip()}"))
    try:
        pol = Policy.load(lab.paths.lab / "budget.yaml")
        lo, hi = pol.quiet_utc
        if lo <= now.hour < hi:
            out.append(("warn", f"quiet window {lo:02d}-{hi:02d} UTC: no agent runs"))
        used = used_today(lab, now.date().isoformat())
        if sum(used.values()) >= pol.total_daily:
            out.append(("warn", f"daily total of {pol.total_daily} agent runs used"))
        for a, n in pol.daily.items():
            if n and used.get(a, 0) >= n:
                out.append(("muted", f"{a}: daily budget {used.get(a, 0)}/{n} used"))
    except (OSError, KeyError):
        pass
    if not canaries.passed_for_current(lab):
        out.append(("bad", "canaries have not passed for the current framework: no gates and no ingest until `lab canaries` passes"))
    return out


def waiting_for(lab: Lab, h, now: datetime) -> tuple[str, str]:
    """(text, tone) for a live hypothesis: who has to act next."""
    hid, status = h["id"], h["status"]
    ask = lab.con.execute("SELECT from_agent, payload_json FROM messages WHERE type = 'QUESTION' AND to_agent = 'human' "
                          "AND handled_at IS NULL AND hypothesis_id = ? ORDER BY id DESC LIMIT 1", (hid,)).fetchone()
    if ask:
        return (f"waiting for the OWNER: {ask['from_agent']} asked: "
                f"{json.loads(ask['payload_json'])['question'][:200]}  (answer with `lab ack <id> --reply ...` or park it)"), "bad"
    lock = lab.con.execute("SELECT invocation_id, lease_until FROM locks WHERE hypothesis_id = ?", (hid,)).fetchone()
    if lock and lock["lease_until"] > now.isoformat():
        return f"an agent run ({lock['invocation_id'].split('-')[0]}) holds it until {lock['lease_until'][11:16]} UTC", "warn"
    if status == "IDEA":
        inbox = lab.inbox("scout", hid)
        if inbox:
            p = json.loads(inbox[0]["payload_json"])
            return f"Scout must answer {len(inbox)} message(s): {(p.get('question') or p.get('items') or inbox[0]['type'])}"[:260], "warn"
        return "Scout must revise the card", "warn"
    if status == "BLOCKED_DATA":
        req = [json.loads(m["payload_json"])["dataset"] for m in lab.inbox("archivist")
               if m["hypothesis_id"] in (hid, None) and m["type"] == "DATA_REQUEST"]
        return f"Archivist must ingest {', '.join(sorted(set(req))) or 'the requested dataset'} (or call it infeasible)", "warn"
    if status == "DATA_READY":
        d = lab.paths.strategies / hid
        if not (d / "strategy.py").exists():
            return "Builder has not implemented it yet", "warn"
        g0 = lab.con.execute("SELECT passed, reason_code FROM gate_results WHERE hypothesis_id = ? AND gate = 'G0' "
                             "ORDER BY id DESC LIMIT 1", (hid,)).fetchone()
        if g0 and not g0["passed"]:
            return f"Builder must fix a G0 failure ({g0['reason_code']})", "warn"
        return "G0 has not run yet: the judge runs it in the next cycle (needs passed canaries)", "warn"
    if status in ("IMPLEMENTED", "GATE_1", "GATE_2", "GATE_3"):
        nxt = {"IMPLEMENTED": "G1", "GATE_1": "G2", "GATE_2": "G3", "GATE_3": "the Skeptic hand-over"}[status]
        return f"the judge runs {nxt} in the next cycle (needs passed canaries)", "warn"
    if status == "SKEPTIC_REVIEW":
        rounds = h["objection_rounds"]
        return f"Skeptic review pending (objection round {rounds}); one agent run per cycle", "warn"
    if status == "HOLDOUT":
        return "Sentinel decides about paper admission (forward data, capacity, kill switch)", "warn"
    if status == "PAPER":
        from lab.framework import paper
        rec = paper.record(lab, hid)
        return f"collecting forward days: {rec['days']} so far; G5 needs months of data and 30 entries", "good"
    if status == "LIVE_CANDIDATE":
        return "passed everything: the owner decides", "good"
    return "", ""


# ---------------------------------------------------------------------------- SVG

TONE = {"done": "#1f7a4d", "failed": "#b3261e", "active": "#9a6700", "pending": "#8a8984", "skipped": "#8a8984",
        "parked": "#6b6a66"}


def node_svg(x: float, y: float, bw: int, key: str, label: str, who: str, kind: str, agent: str | None,
             state: str, mood: str, count: int | None = None, caption: str | None = None, count_tone: float = 0.0) -> str:
    """One stage: robot or gear (mood = what it is doing), name, the actor, and what kind of worker it is."""
    col = TONE[state] if state in TONE else TONE["pending"]
    live = state in ("done", "failed", "active", "parked")
    box_fill = "var(--accent)" if count is not None else col
    fill_op = (0.10 + 0.5 * count_tone) if count is not None else (0.18 if live else 0.05)
    dash = ' stroke-dasharray="4 3"' if (count is None and state in ("pending", "skipped")) else ""
    sw = 2.4 if (count is None and state in ("active", "failed")) else 1.2
    badge_col = icons.AGENT_COLOR.get(agent or "", icons.CODE_COLOR) if kind != "code" else icons.CODE_COLOR
    model = model_label(agent) if agent and kind != "code" else ""
    badge = {"code": "code, no LLM", "llm": f"LLM · {model}" if model else "LLM",
             "mixed": f"{model} + code" if model else "LLM + code"}[kind]
    return (f'<g><rect x="{x}" y="{y}" width="{bw}" height="106" rx="9" fill="{box_fill}" fill-opacity="{fill_op:.2f}" '
            f'stroke="{col if count is None else "var(--line)"}" stroke-width="{sw}"{dash}/>'
            f'<g transform="translate({x + bw / 2 - 17},{y + 5}) scale(.72)">{icons.icon(kind, agent, mood)}</g>'
            + (f'<text x="{x + bw - 8}" y="{y + 22}" text-anchor="end" font-size="17" font-weight="700" fill="var(--fg)">{count}</text>'
               if count is not None else "")
            + f'<text x="{x + bw / 2}" y="{y + 62}" text-anchor="middle" font-size="13" font-weight="600" fill="var(--fg)">{esc(label)}</text>'
            f'<text x="{x + bw / 2}" y="{y + 76}" text-anchor="middle" font-size="10.5" fill="{col if caption else "var(--muted)"}">'
            f'{esc(caption or who)}</text>'
            f'<rect x="{x + 6}" y="{y + 84}" width="{bw - 12}" height="15" rx="7.5" fill="{badge_col}" fill-opacity=".16" stroke="{badge_col}" stroke-opacity=".6"/>'
            f'<text x="{x + bw / 2}" y="{y + 94.5}" text-anchor="middle" font-size="9" fill="var(--fg)">{esc(badge)}</text></g>')


def legend_html() -> str:
    return ('<div class=legend><span>' + icons.svg("llm", "scout", "idle", 22) + ' LLM agent (model named under it)</span>'
            '<span>' + icons.svg("mixed", "builder", "idle", 22) + ' LLM writes, code checks</span>'
            '<span>' + icons.svg("code", None, "idle", 22) + ' deterministic code, no LLM: the judge decides</span>'
            '<span class=muted>mood: <b>animated</b> = running now · <b>…</b> = waiting its turn · '
            '<b>smile</b> = done · <b>✕</b> = ended here · <b>asleep</b> = not reached</span></div>')


def flow_svg(lab: Lab, hyps) -> str:
    """The lab at a glance: stages left to right, how many reached each, died there (reason), live ones as chips."""
    stories = {h["id"]: story(lab, h) for h in hyps}
    n = len(STAGES)
    w, bw, gap = 1120, 108, 14
    reached, died, live = Counter(), defaultdict(list), defaultdict(list)
    for h in hyps:
        st = stories[h["id"]]
        for key, *_ in STAGES:
            if st["stages"][key] in ("done", "failed", "active", "parked", "skipped"):
                reached[key] += 1
        if st["dead_stage"]:
            died[st["dead_stage"]].append(h["reject_code"] or h["status"])
        elif h["status"] not in {str(t) for t in TERMINAL}:
            for key, *_ in STAGES:
                if st["stages"][key] == "active":
                    live[key].append(h["id"])
    height = 132 + 14 * max([len(set(v)) for v in died.values()] + [1]) + 22 * max([len(v) for v in live.values()] + [0])
    out = [f'<div class=flowwrap><svg viewBox="0 0 {w} {height}" role="img" aria-label="pipeline flow" class=flow>'
           '<defs><marker id="ah" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">'
           '<path d="M0,0 L7,3.5 L0,7 z" fill="var(--muted)"/></marker></defs>']
    top = max(reached.values() or [1])
    running = running_now(lab)
    for i, (key, label, who, kind, agent) in enumerate(STAGES):
        x = 8 + i * (bw + gap)
        r = reached.get(key, 0)
        busy = bool(live.get(key)) and any((agent, hid) in running for hid in live[key])
        mood = "work" if busy else ("sleep" if r == 0 else "idle")
        out.append(node_svg(x, 6, bw, key, label, who, kind, agent, "done" if r else "pending", mood, count=r,
                            count_tone=(r / top if top else 0)))
        if i < n - 1:
            out.append(f'<path d="M{x + bw + 2},58 l{gap - 4},0" stroke="var(--muted)" stroke-width="1.5" marker-end="url(#ah)"/>')
        y = 128
        for reason, cnt in Counter(died.get(key, [])).most_common(6):
            short = reason.split("_", 1)[1] if reason[:2] in ("g1", "g2", "g3", "g4", "g5", "g0") else reason
            short = short if len(short) <= 17 else short[:16] + "…"
            out.append(f'<text x="{x + 4}" y="{y}" font-size="10.5" fill="var(--bad)"><title>{esc(reason)}</title>✕ {cnt}× {esc(short)}</text>')
            y += 14
        y = max(y, 128) + 10 if live.get(key) else y
        for hid in live.get(key, []):
            out.append(f'<a href="#h-{hid}" data-hyp="{hid}"><rect x="{x + 2}" y="{y - 11}" width="{bw - 4}" height="17" rx="8.5" '
                       f'fill="var(--warn)" fill-opacity=".22" stroke="var(--warn)"/><text x="{x + bw / 2}" y="{y + 1}" '
                       f'text-anchor="middle" font-size="11" fill="var(--fg)">{hid}</text></a>')
            y += 22
    out.append("</svg></div>")
    return "".join(out)


def strip_svg(lab: Lab, st: dict, hid: str, running: set) -> str:
    """The pipeline of one hypothesis: nine workers, their mood is the state of the stage; loops as badges."""
    w, bw, gap = 1120, 108, 14
    out = [f'<div class=flowwrap><svg viewBox="0 0 {w} 134" role="img" aria-label="hypothesis pipeline" class=flow>']
    for i, (key, label, who, kind, agent) in enumerate(STAGES):
        x = 8 + i * (bw + gap)
        state = st["stages"][key]
        mood = mood_of(state, kind, agent, hid, running)
        caption = {"skipped": "not needed", "active": "working" if mood == "work" else "its turn"}.get(state, state)
        out.append(node_svg(x, 6, bw, key, label, who, kind, agent, state, mood, caption=caption))
        loops = st["loops"].get(key, 0)
        if loops:
            out.append(f'<text x="{x + bw - 4}" y="126" text-anchor="end" font-size="11" fill="var(--warn)">↺ {loops}× back</text>')
        ts = st["entered"].get(key)
        if ts and state != "pending":
            out.append(f'<text x="{x + 4}" y="126" font-size="10.5" fill="var(--muted)">{esc(ts[5:16].replace("T", " "))}</text>')
        if i < len(STAGES) - 1:
            out.append(f'<path d="M{x + bw + 1},58 l{gap - 2},0" stroke="{TONE.get(state, TONE["pending"])}" stroke-width="1.5"/>')
    out.append("</svg></div>")
    return "".join(out)


def team_html(lab: Lab) -> str:
    """The workers: one card per LLM agent and one for the deterministic judge, with their activity."""
    running = {r["agent"] for r in lab.con.execute("SELECT agent FROM agent_invocations WHERE outcome IS NULL")}
    day = datetime.now(timezone.utc).date().isoformat()
    cards = []
    for a in ("scout", "archivist", "builder", "skeptic", "librarian", "chair", "steward"):
        r = lab.con.execute("SELECT COUNT(*) n, SUM(outcome = 'applied') ok, MAX(started_at) last, SUM(tokens_in + tokens_out) tok "
                            "FROM agent_invocations WHERE agent = ? AND COALESCE(error, '') != 'dry run'", (a,)).fetchone()
        today = lab.con.execute("SELECT COUNT(*) FROM agent_invocations WHERE agent = ? AND substr(started_at, 1, 10) = ?",
                                (a, day)).fetchone()[0]
        last = lab.con.execute("SELECT outcome FROM agent_invocations WHERE agent = ? ORDER BY started_at DESC LIMIT 1", (a,)).fetchone()
        mood = "work" if a in running else ("sleep" if not r["n"] else "fail" if last and last["outcome"] not in ("applied", None) else "idle")
        status = ("working now" if a in running else "never run" if not r["n"] else
                  "last run failed" if mood == "fail" else "idle")
        cards.append(f"<div class=agent><svg viewBox='0 0 48 56' width=52 height=61 role=img aria-label='{a}'>{icons.robot(a, mood)}</svg>"
                     f"<div><b>{a.capitalize()}</b> <span class=pill>{esc(status)}</span><div class=sub>{esc(ROLE[a])}</div>"
                     f"<div class=sub>{esc(model_label(a))} · {today} today · {r['n'] or 0} runs"
                     f"{(' · ' + format((r['tok'] or 0) / 1e6, '.1f') + ' M tokens') if r['tok'] else ''}"
                     f"{(' · last ' + ago(r['last'], datetime.now(timezone.utc))) if r['last'] else ''}</div></div></div>")
    g = lab.con.execute("SELECT COUNT(*) n, MAX(ts) last FROM gate_results").fetchone()
    cards.append("<div class='agent code'><svg viewBox='0 0 48 56' width=52 height=61 role=img aria-label='judge'>"
                 f"{icons.gear('idle')}</svg><div><b>The judge</b> <span class=pill>code</span><div class=sub>gates G0-G5, "
                 "ingest, paper runner, Sentinel: deterministic, no LLM</div>"
                 f"<div class=sub>{g['n']} gate results"
                 f"{(' · last ' + ago(g['last'], datetime.now(timezone.utc))) if g['last'] else ''}</div></div></div>")
    return "<div class=team>" + "".join(cards) + "</div>"


# ---------------------------------------------------------------------------- stage details

def checks_table(m: dict) -> str:
    rows = []
    for k, c in (m.get("checks") or {}).items():
        if c.get("not_applicable"):
            rows.append(f"<tr><td class=mono>{esc(k)}</td><td colspan=3 class=muted>n/a: {esc(c['not_applicable'])}</td></tr>")
            continue
        rows.append(f"<tr><td class=mono>{esc(k)}</td><td class=num>{fmt(c.get('value'))}</td>"
                    f"<td class=mono>{esc(c.get('op', ''))} {fmt(c.get('threshold'))}</td>"
                    f"<td class={'good' if c.get('pass') else 'bad'}>{'✓' if c.get('pass') else '✕'}</td></tr>")
    return ("<div class=scroll><table class=checks><tr><th>check</th><th class=num>value</th><th>needs</th><th></th></tr>"
            + "".join(rows) + "</table></div>")


def mechanism_block(mech: dict) -> str:
    bl = ", ".join(f"{b['events']} → {fmt((b['mean_abnormal'] or 0) * 1e4, 1)} bp" for b in mech.get("blocks", []))
    pl = mech.get("placebo") or {}
    return ("<div class=mech><b>Mechanism test (event study vs placebo)</b><br>"
            f"{mech.get('n_events')} events on {mech.get('n_event_dates')} dates · horizon {mech.get('horizon_days')} d · "
            f"mean abnormal return {fmt((mech.get('mean_abnormal') or 0) * 1e4, 1)} bp · placebo percentile "
            f"{fmt(mech.get('placebo_percentile'), 2)} (placebo mean {fmt((pl.get('mean') or 0) * 1e4, 1)} bp, "
            f"p95 {fmt((pl.get('p95') or 0) * 1e4, 1)} bp) · sub-periods: {esc(bl)} · cost ratio {fmt(mech.get('cost_ratio'), 2)}"
            "</div>")


def message_html(m, md) -> str:
    p = json.loads(m["payload_json"])
    t = m["type"]
    if t in ("NEW_HYPOTHESIS", "REVISION"):
        c = p["card"]
        mech = c.get("mechanism") or {}
        refs = "; ".join(r.get("cite", "") for r in c.get("references") or [])
        return (f"<p><b>{esc(c.get('title'))}</b> <span class=muted>(family {esc(c.get('family'))}"
                + (f", revision: {esc(p.get('reason'))}" if p.get("reason") else "") + ")</span></p>"
                f"<p><b>Mechanism.</b> {esc(mech.get('why'))}</p><p><b>Counterparty.</b> {esc(mech.get('counterparty'))}</p>"
                f"<p><b>Signal.</b> {esc((c.get('signal') or {}).get('description'))}</p>"
                + (f"<p><b>Mechanism test.</b> {esc(c['mechanism_test'].get('event'))} (horizon "
                   f"{esc(c['mechanism_test'].get('primary_horizon_days'))} d)</p>" if c.get("mechanism_test") else "")
                + (f"<p><b>References.</b> {esc(refs)}</p>" if refs else "")
                + "<p><b>Falsification.</b> " + esc("; ".join(c.get("falsification_criteria") or [])) + "</p>")
    if t == "IMPL_DONE":
        return f"<p>{esc(p.get('summary'))}</p><div class=sub>files: {esc(', '.join(p.get('files', [])))}</div>"
    if t == "OBJECTION":
        return (f"<p>returned to <b>{esc(p['return_to'])}</b></p><table><tr><th>check</th><th>finding</th><th>evidence</th></tr>"
                + "".join(f"<tr><td class=mono>{esc(i['check'])}</td><td>{esc(i['finding'])}</td><td>{esc(i['evidence'])}</td></tr>"
                          for i in p["items"]) + "</table>")
    if t == "VERDICT":
        cl = p.get("checklist") or {}
        return (f"<p><b>{esc(p['decision'])}</b>: {esc(p['reason'])}</p>"
                + ("<table><tr><th>check</th><th></th><th>evidence</th></tr>"
                   + "".join(f"<tr><td class=mono>{esc(k)}</td><td class={'good' if v['status'] == 'ok' else 'bad'}>{esc(v['status'])}"
                             f"</td><td>{esc(v['evidence'])}</td></tr>" for k, v in cl.items()) + "</table>" if cl else ""))
    if t == "LESSON":
        return (f"<p><b>{esc(p['outcome'])}</b></p><p>{esc(p['lesson'])}</p>"
                + (f"<p><b>Avoid.</b> {esc(p['avoid'])}</p>" if p.get("avoid") else "")
                + (f"<p><b>Open questions.</b> {esc(p['open_questions'])}</p>" if p.get("open_questions") else ""))
    if t in ("DATA_REQUEST", "DATA_READY"):
        return f"<p>{esc(p.get('dataset'))}: {esc(p.get('description') or (p.get('catalog_entry') or {}).get('instruments', ''))}</p>"
    if t == "QUESTION":
        return f"<p>{esc(p['question'])}</p>"
    if t == "ALERT":
        return f"<p>[{esc(p['severity'])}] {esc(p['text'])}</p>"
    if t == "KNOWLEDGE":
        return f"<p><b>{esc(p['title'])}</b> ({esc(p['kind'])}): {esc(p['statement'])}</p>"
    return f"<p class=mono>{esc(json.dumps(p)[:500])}</p>"


STAGE_OF_MESSAGE = {"NEW_HYPOTHESIS": "idea", "REVISION": "idea", "DATA_REQUEST": "data", "DATA_READY": "data",
                    "IMPL_DONE": "build", "OBJECTION": "skeptic", "VERDICT": "skeptic", "LESSON": "after",
                    "QUESTION": "idea", "ALERT": "after"}


def hypothesis_view(lab: Lab, h, now: datetime, sums: dict, md) -> str:
    hid = h["id"]
    st = story(lab, h)
    card = json.loads(h["card_json"])
    msgs = lab.con.execute("SELECT * FROM messages WHERE hypothesis_id = ? OR id IN (SELECT message_id FROM transitions "
                           "WHERE hypothesis_id = ? AND message_id IS NOT NULL) ORDER BY id", (hid, hid)).fetchall()
    gates = lab.con.execute("SELECT * FROM gate_results WHERE hypothesis_id = ? ORDER BY id", (hid,)).fetchall()
    by_stage: dict[str, list[str]] = defaultdict(list)
    for m in msgs:
        if m["type"] in ("GATE_RESULT", "ALERT") and m["from_agent"] in ("gatekeeper", "system", "sentinel") and m["type"] == "GATE_RESULT":
            continue
        key = STAGE_OF_MESSAGE.get(m["type"], "after")
        if m["type"] == "QUESTION" and m["from_agent"] == "system":
            key = "idea"
        if m["type"] == "VERDICT" and m["from_agent"] in ("archivist",):
            key = "data"
        if m["type"] == "VERDICT" and m["from_agent"] in ("chair", "sentinel"):
            key = "after"
        who = m["from_agent"]
        by_stage[key].append(
            f"<div class=ev><div class=evh><span class=mono>{esc(m['created_at'][5:16].replace('T', ' '))}</span> "
            f"<b>{esc(who)}</b> → {esc(m['to_agent'])} · <span class=pill>{esc(m['type'])}</span></div>"
            f"{message_html(m, md)}{run_box(lab, m['invocation_id'], sums, md)}</div>")
    for g in gates:
        key = {"G0": "build", "G1": "g1", "G2": "g2", "G3": "g3", "G4": "g4", "G5": "paper"}[g["gate"]]
        m = json.loads(g["metrics_json"])
        extra = ""
        if m.get("mechanism"):
            extra += mechanism_block(m["mechanism"])
        if m.get("problems"):
            extra += f"<div class=note>{esc('; '.join(m['problems']))}</div>"
        if m.get("error"):
            extra += f"<div class=note>{esc(m['error'])}</div>"
        head = " · ".join(f"{k} {fmt(m[k])}" for k in ("sharpe", "bench_sharpe", "cagr", "bench_cagr", "max_dd", "mean_exposure", "dsr")
                          if k in m)
        by_stage[key].append(
            f"<div class=ev><div class=evh><span class=mono>{esc(g['ts'][5:16].replace('T', ' '))}</span> <b>judge</b> · "
            f"<span class=pill>{g['gate']} v{g['version']}</span> <span class='{'good' if g['passed'] else 'bad'}'>"
            f"{'PASS' if g['passed'] else 'FAIL ' + esc(g['reason_code'])}</span></div>"
            f"{('<div class=sub>' + head + (' vs ' + esc(m.get('benchmark')) if m.get('benchmark') else '') + '</div>') if head else ''}"
            f"{extra}{checks_table(m) if m.get('checks') else ''}</div>")
    if h["status"] in ("PAPER", "LIVE_CANDIDATE", "RETIRED"):
        from lab.framework import paper
        rec = paper.record(lab, hid)
        if rec["days"]:
            by_stage["paper"].append(f"<div class=ev><b>Paper record</b>: {rec['days']} days since {esc(rec['first'])}, "
                                     f"{rec['entries']} entries, max drawdown {rec['max_dd']:.1%}</div>")
    running = running_now(lab)
    parts = [f"<section class=hyp id='h-{hid}' data-id='{hid}'><h3>{hid} · {esc(h['title'])} "
             f"<span class='pill'>{esc(h['status'])}</span></h3>"
             f"<div class=sub>family {esc(lab.canonical_family(h['family']))} · {esc(card.get('market_exposure'))} · "
             f"{esc(', '.join(card.get('asset_classes') or []))} · created {esc(h['created_at'][:10])}"
             + (f" · died at <b>{esc(h['reject_stage'])}</b>: <span class=mono>{esc(h['reject_code'])}</span>" if h["reject_code"] else "")
             + "</div>", strip_svg(lab, st, hid, running)]
    wait, tone = ("", "")
    if h["status"] not in {str(t) for t in TERMINAL}:
        wait, tone = waiting_for(lab, h, now)
        parts.append(f"<div class='note {tone}'>▶ <b>Now:</b> {esc(wait)}</div>")
    for key, label, who, kind, agent in STAGES:
        items = by_stage.get(key, [])
        state = st["stages"][key]
        if not items and state in ("pending", "skipped"):
            continue
        parts.append(f"<details class=stage {'open' if state in ('active', 'failed') else ''}><summary>"
                     f"{icons.svg(kind, agent, mood_of(state, kind, agent, hid, running), 22)}<b>{esc(label)}</b> <span class=muted>{esc(who)} · {state}"
                     f"{' · ' + str(len(items)) + ' record' + ('s' if len(items) != 1 else '') if items else ''}</span>"
                     f"</summary>{''.join(items) or '<div class=sub>nothing recorded</div>'}</details>")
    if by_stage.get("after"):
        parts.append(f"<details class=stage><summary><span class='dot done'></span><b>Afterwards</b> "
                     f"<span class=muted>Librarian, Chair, alerts</span></summary>{''.join(by_stage['after'])}</details>")
    parts.append("</section>")
    return "".join(parts)


# ---------------------------------------------------------------------------- the whole tab

def build(lab: Lab, md) -> str:
    now = datetime.now(timezone.utc)
    hyps = lab.hypotheses()
    sums = run_summaries(lab)
    live = [h for h in hyps if h["status"] not in {str(t) for t in TERMINAL} | {"PARKED"}]
    blockers = lab_blockers(lab, now)
    out = ["<h2>Pipeline</h2><div class=sub>Where every hypothesis is, where it is stuck and what each agent produced. "
           "Click a hypothesis to open its pipeline.</div>", team_html(lab), legend_html(), flow_svg(lab, hyps)]
    out.append("<h3>Stuck or waiting</h3>")
    if blockers:
        out.append("<div>" + "".join(f"<div class='note {t}'>⚠ {esc(x)}</div>" for t, x in blockers) + "</div>")
    else:
        out.append("<div class=sub>No lab-wide blockers: agents and gates may run.</div>")
    if live:
        out.append("<div class=scroll><table><tr><th>hypothesis</th><th>stage</th><th>in this state</th><th>waiting for</th></tr>")
        for h in live:
            last = lab.con.execute("SELECT ts FROM transitions WHERE hypothesis_id = ? ORDER BY id DESC LIMIT 1", (h["id"],)).fetchone()
            text, tone = waiting_for(lab, h, now)
            age = ago(last["ts"] if last else h["created_at"], now)
            stale = (now - datetime.fromisoformat(last["ts"] if last else h["created_at"])).total_seconds() > 6 * 3600
            out.append(f"<tr><td class=mono><a href='#h-{h['id']}' data-hyp='{h['id']}'>{h['id']}</a> {esc(h['title'][:60])}</td>"
                       f"<td>{esc(h['status'])}</td><td class='{'bad' if stale else ''}'>{esc(age)}{' ⚠ long' if stale else ''}</td>"
                       f"<td>{esc(text)}</td></tr>")
        out.append("</table></div>")
    else:
        out.append("<div class=sub>No live hypothesis: the pipeline is idle. The Scout proposes a new one when the "
                   "budget and the schedule allow.</div>")
    out.append("<h3>Hypothesis pipeline</h3><div class=picker><label for=hsel>Hypothesis</label> <select id=hsel>"
               + "".join(f"<option value='{h['id']}'>{h['id']} · {esc(h['status'])} · {esc(h['title'][:70])}</option>"
                         for h in hyps[::-1]) + "</select></div>")
    out.append("<div id=hyps>" + "".join(hypothesis_view(lab, h, now, sums, md) for h in hyps[::-1]) + "</div>")
    out.append("<h3>Latest agent outputs</h3>")
    for r in lab.con.execute("SELECT * FROM agent_invocations WHERE outcome IS NOT NULL ORDER BY started_at DESC LIMIT 12"):
        out.append(f"<div class=ev><div class=evh><span class=mono>{esc(r['started_at'][5:16].replace('T', ' '))}</span> "
                   f"<b>{esc(r['agent'])}</b> {('<a href=#h-' + r['hypothesis_id'] + ' data-hyp=' + r['hypothesis_id'] + '>' + r['hypothesis_id'] + '</a>') if r['hypothesis_id'] else ''}"
                   f"</div>{run_box(lab, r['id'], sums, md)}</div>")
    return "".join(out)
