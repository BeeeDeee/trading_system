"""Static dashboard of the lab (PLAN §11): one HTML file, no JavaScript, regenerated after every cycle.

Published like the cpb dashboard: a new release directory per build under `<web>/releases/`, then an atomic
switch of the `<web>/current` symlink, served read-only by Caddy behind basic auth. Sections: status (last
cycle, canaries, budget, pauses), open questions and alerts for the owner, funnel and causes of death, the
hypothesis table with a full trace per hypothesis, the paper book, agent runs, knowledge base, lessons.
"""

import html
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lab.framework import agents, briefs, icons, knowledge, lessons, pipeline_view, report
from lab.framework.blackboard import Lab
from lab.framework.states import FUNNEL, TERMINAL

KEEP_RELEASES = 5
STATUS_TONE = {"REJECTED": "bad", "RETIRED": "bad", "PARKED": "muted", "PAPER": "good", "LIVE_CANDIDATE": "good",
               "HOLDOUT": "good", "SKEPTIC_REVIEW": "warn", "BLOCKED_DATA": "warn"}

CSS = """
:root{--bg:#fbfaf8;--fg:#1d1d1b;--muted:#6b6a66;--line:#e4e2dd;--card:#ffffff;--good:#1f7a4d;--bad:#b3261e;
--warn:#9a6700;--accent:#3355aa;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe7;--muted:#a3a19b;--line:#2f2e2b;--card:#1c1c1a;
--good:#5cc28d;--bad:#ef7b72;--warn:#e0b04a;--accent:#8ea8ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,
"Segoe UI",sans-serif}main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:36px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:18px 0 6px}.sub{color:var(--muted);font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.tile span{color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px;background:var(--card);border:1px solid var(--line);border-radius:8px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px}td.num{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}.scroll{overflow-x:auto}
.pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;border:1px solid var(--line);white-space:nowrap}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}.muted{color:var(--muted)}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:8px 12px}
summary{cursor:pointer;font-weight:600}code,.mono{font-family:var(--mono);font-size:12px}
.bar{height:8px;background:var(--accent);border-radius:4px;opacity:.75}
.note{border-left:3px solid var(--warn);padding:4px 10px;margin:6px 0;background:var(--card)}
.kb p{margin:4px 0 10px}svg{display:block;max-width:100%}a{color:var(--accent)}
.team{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px;margin:12px 0}
.agent{display:flex;gap:10px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.agent b{font-size:15px}.agent .sub{margin-top:2px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;font-size:12.5px;margin:6px 0 2px}
.tabs{display:flex;flex-wrap:wrap;gap:4px;margin:22px 0 0;border-bottom:1px solid var(--line)}
.tabs label{padding:8px 14px;cursor:pointer;border:1px solid transparent;border-bottom:0;border-radius:8px 8px 0 0;color:var(--muted);font-weight:600}
input.tab{position:absolute;opacity:0;pointer-events:none}
.panel{display:none}
#t-overview:checked~.tabs label[for=t-overview],#t-pipeline:checked~.tabs label[for=t-pipeline],
#t-hypotheses:checked~.tabs label[for=t-hypotheses],#t-paper:checked~.tabs label[for=t-paper],
#t-ops:checked~.tabs label[for=t-ops],#t-knowledge:checked~.tabs label[for=t-knowledge]
{color:var(--fg);background:var(--card);border-color:var(--line)}
#t-overview:checked~.panels #p-overview,#t-pipeline:checked~.panels #p-pipeline,#t-hypotheses:checked~.panels #p-hypotheses,
#t-paper:checked~.panels #p-paper,#t-ops:checked~.panels #p-ops,#t-knowledge:checked~.panels #p-knowledge{display:block}
svg.flow{width:100%;min-width:760px;height:auto;margin:10px 0}.flowwrap{overflow-x:auto}svg.flow a{cursor:pointer}
.picker{margin:8px 0}.picker select{font:inherit;padding:5px 8px;max-width:100%;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
body.js section.hyp{display:none}body.js section.hyp.on{display:block}
section.hyp{margin:10px 0;padding:6px 14px 10px;background:var(--card);border:1px solid var(--line);border-radius:10px}
.note.warn{border-left-color:var(--warn)}.note.bad{border-left-color:var(--bad)}.note.good{border-left-color:var(--good)}
details.stage{border-left:3px solid var(--line)}details.stage>summary{display:flex;align-items:center;gap:8px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--muted)}
.dot.done{background:var(--good)}.dot.failed{background:var(--bad)}.dot.active{background:var(--warn)}.dot.parked{background:var(--muted)}
.ev{border-top:1px solid var(--line);padding:8px 0}.ev:first-child{border-top:0}.evh{font-size:13px;margin-bottom:4px}
details.run{margin:6px 0;padding:4px 10px;background:var(--bg)}details.run summary{font-weight:400;font-size:13px}
.agentsays{border-left:3px solid var(--accent);padding-left:10px;margin:6px 0;max-height:420px;overflow:auto}
.mech{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px 10px;margin:6px 0;font-size:13px}
table.checks{width:auto;min-width:380px}.pill.good{border-color:var(--good)}.pill.bad{border-color:var(--bad)}.pill.warn{border-color:var(--warn)}
""" + icons.CSS


JS = """
(function(){var b=document.body;b.classList.add('js');
function tab(n){var r=document.getElementById('t-'+n);if(r)r.checked=true;}
function show(id){var all=document.querySelectorAll('section.hyp'),f=null;
 all.forEach(function(x){var on=x.getAttribute('data-id')===id;x.classList.toggle('on',on);if(on)f=x;});
 var sel=document.getElementById('hsel');if(sel&&id)sel.value=id;return f;}
function route(){var h=location.hash.replace('#','');
 if(h.indexOf('h-')===0){tab('pipeline');show(h.slice(2));}
 else if(h.indexOf('tab-')===0){tab(h.slice(4));}}
var sel=document.getElementById('hsel');
if(sel){sel.addEventListener('change',function(){show(sel.value);});}
document.addEventListener('click',function(e){var a=e.target.closest&&e.target.closest('[data-hyp]');
 if(a){e.preventDefault();location.hash='h-'+a.getAttribute('data-hyp');}});
window.addEventListener('hashchange',route);
var first=document.querySelector('section.hyp');if(first&&!location.hash.startsWith('#h-')){show(first.getAttribute('data-id'));}
route();})();
"""


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def md(text: str) -> str:
    """Minimal Markdown (headings, lists, paragraphs, *italic*, `code`, **bold**) for the rendered lab files."""
    out, para, items = [], [], []

    def inline(s):
        s = esc(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        return re.sub(r"\*([^*]+)\*", r"<i>\1</i>", s)

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()
    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            level = min(len(m.group(1)) + 2, 6)
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
        elif line.startswith("- "):
            if para:
                flush()
            items.append(line[2:])
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
    flush()
    return "\n".join(out)


def pill(status: str) -> str:
    return f'<span class="pill {STATUS_TONE.get(status, "")}">{esc(status)}</span>'


def nav_svg(series: list[tuple[str, np.ndarray]], width=760, height=180) -> str:
    """Line chart of cumulative growth for one or more return series (same length)."""
    curves = [(name, np.cumprod(1 + np.nan_to_num(r))) for name, r in series if len(r)]
    if not curves:
        return ""
    lo = min(c.min() for _, c in curves)
    hi = max(c.max() for _, c in curves)
    span = hi - lo or 1.0
    colors = ["var(--accent)", "var(--muted)"]
    paths, legend = [], []
    for k, (name, c) in enumerate(curves):
        xs = np.linspace(40, width - 10, len(c))
        ys = height - 20 - (c - lo) / span * (height - 40)
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        paths.append(f'<path d="{d}" fill="none" stroke="{colors[k % 2]}" stroke-width="1.6"/>')
        legend.append(f'<tspan fill="{colors[k % 2]}">■</tspan> {esc(name)} {c[-1] - 1:+.1%}  ')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="cumulative growth">'
            f'<text x="40" y="12" font-size="11" fill="var(--muted)">{"".join(legend)}</text>'
            f'<text x="2" y="{height - 18}" font-size="10" fill="var(--muted)">{lo:.2f}</text>'
            f'<text x="2" y="26" font-size="10" fill="var(--muted)">{hi:.2f}</text>{"".join(paths)}</svg>')


# ---------------------------------------------------------------------------- data

def last_cycles(lab: Lab, n: int = 12) -> list[dict]:
    f = lab.paths.home / "cycles.jsonl"
    if not f.exists():
        return []
    return [json.loads(x) for x in f.read_text().splitlines()[-n:]][::-1]


def gate_summary(lab: Lab, hid: str) -> list[dict]:
    keys = ("sharpe", "bench_sharpe", "benchmark", "cagr", "bench_cagr", "max_dd", "mean_exposure", "dsr", "days")
    out = []
    for r in lab.con.execute("SELECT gate, version, passed, reason_code, metrics_json, ts FROM gate_results "
                             "WHERE hypothesis_id = ? ORDER BY id", (hid,)):
        m = json.loads(r["metrics_json"])
        failed = [k for k, c in (m.get("checks") or {}).items() if isinstance(c, dict) and c.get("pass") is False]
        metrics = {k: m[k] for k in keys if k in m}
        if m.get("mechanism"):
            mech = m["mechanism"]
            metrics |= {"mechanism_events": mech.get("n_events"), "mechanism_abnormal": mech.get("mean_abnormal"),
                        "mechanism_placebo_pct": mech.get("placebo_percentile"), "mechanism_cost_ratio": mech.get("cost_ratio")}
        out.append({"gate": r["gate"], "v": r["version"], "passed": bool(r["passed"]), "code": r["reason_code"],
                    "ts": r["ts"][:16], "metrics": metrics, "failed": failed,
                    "error": m.get("error"), "problems": m.get("problems")})
    return out


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"
    return esc(v)


# ---------------------------------------------------------------------------- page

def build(lab: Lab) -> str:
    from lab.framework import canaries
    now = datetime.now(timezone.utc)
    hyps = lab.hypotheses()
    funnel = report.funnel(lab)
    cycles = last_cycles(lab)
    day = now.date().isoformat()
    used = dict(lab.con.execute("SELECT agent, COUNT(*) FROM agent_invocations WHERE substr(started_at,1,10) = ? "
                                "AND COALESCE(error,'') != 'dry run' GROUP BY agent", (day,)).fetchall())
    try:
        import yaml
        budget = yaml.safe_load((lab.paths.lab / "budget.yaml").read_text())
    except OSError:
        budget = {"daily": {}, "total_daily": 0}
    import shutil
    du = shutil.disk_usage(lab.paths.home if lab.paths.home.exists() else ".")
    free_gb, disk_pct = du.free / 2**30, 100 * (1 - du.free / du.total)
    disk_tone = "bad" if free_gb < 2 else "warn" if free_gb < 4 else ""
    pauses = [p.name for p in (lab.paths.home / "PAUSE", lab.paths.home / "RATE_LIMITED_UNTIL") if p.exists()]
    canary_ok = canaries.passed_for_current(lab)
    open_q = [m for m in lab.inbox("human")]
    tokens = lab.con.execute("SELECT COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0) FROM agent_invocations "
                             "WHERE substr(started_at,1,10) = ?", (day,)).fetchone()

    s = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,"
         f"initial-scale=1'><meta name=robots content=noindex><title>Research Lab</title><style>{CSS}</style></head>"
         f"<body><main><h1>Research Lab</h1><div class=sub>generated {now:%Y-%m-%d %H:%M} UTC · "
         f"{len(hyps)} hypotheses · last cycle {esc(cycles[0]['ts'][:16] if cycles else '—')}</div>"]

    alive = sum(1 for h in hyps if h["status"] not in {str(t) for t in TERMINAL})
    s.append("<div class=tiles>"
             f"<div class=tile><b class='{'good' if canary_ok else 'bad'}'>{'ok' if canary_ok else 'FAILED'}</b>"
             "<span>canaries (judge self-test)</span></div>"
             f"<div class=tile><b>{sum(used.values())} / {budget.get('total_daily', 0)}</b><span>agent runs today</span></div>"
             f"<div class=tile><b>{(tokens[0] + tokens[1]) / 1e6:.1f} M</b><span>tokens today</span></div>"
             f"<div class=tile><b>{alive}</b><span>hypotheses in play</span></div>"
             f"<div class=tile><b>{funnel.get('PAPER', 0)}</b><span>paper trading</span></div>"
             f"<div class=tile><b class='{'warn' if pauses else ''}'>{esc(', '.join(pauses) or 'none')}</b>"
             "<span>pauses</span></div>"
             f"<div class=tile><b class='{disk_tone}'>{free_gb:.1f} GB</b><span>disk free ({disk_pct:.0f} % used)</span></div></div>")

    cut = {"overview": len(s)}
    # owner inbox
    s.append(f"<h2>For the owner ({len(open_q)})</h2>")
    if open_q:
        for m in open_q[-30:][::-1]:
            p = json.loads(m["payload_json"])
            s.append(f"<div class=note><span class=mono>#{m['id']} {esc(m['created_at'][:16])} {esc(m['type'])} from "
                     f"{esc(m['from_agent'])} {esc(m['hypothesis_id'] or '')}</span><br>"
                     f"{esc(report._summary(m['type'], p))}</div>")
        s.append("<div class=sub>Mark handled with <code>lab inbox</code> / answer with <code>lab send</code>.</div>")
    else:
        s.append("<div class=sub>Nothing waiting.</div>")

    # funnel and deaths
    s.append("<h2>Funnel</h2><table><tr><th>status</th><th class=num>n</th><th></th></tr>")
    top = max(funnel.values() or [1]) or 1
    for st in FUNNEL:
        n = funnel.get(st.value, 0)
        if n:
            s.append(f"<tr><td>{pill(st.value)}</td><td class=num>{n}</td>"
                     f"<td style='width:60%'><div class=bar style='width:{100 * n / top:.0f}%'></div></td></tr>")
    s.append("</table><h3>Causes of death</h3><table><tr><th>status</th><th>stage</th><th>reason</th><th class=num>n</th></tr>")
    for r in report.rejection_reasons(lab):
        s.append(f"<tr><td>{pill(r['status'])}</td><td>{esc(r['reject_stage'])}</td><td class=mono>{esc(r['reject_code'])}"
                 f"</td><td class=num>{r['n']}</td></tr>")
    s.append("</table>")

    # coverage of horizons x asset groups (the Scout briefs)
    cards = [json.loads(h["card_json"]) for h in hyps]
    cov = Counter((briefs.horizon(c), briefs.group(c)) for c in cards)
    s.append("<h3>Coverage: holding horizon × asset group</h3><div class=scroll><table><tr><th></th>"
             + "".join(f"<th>{esc(g)}</th>" for g in briefs.GROUPS) + "</tr>")
    for h, span, *_ in briefs.HORIZONS:
        s.append(f"<tr><td>{esc(h)} <span class=muted>({esc(span)})</span></td>"
                 + "".join(f"<td class=num>{cov.get((h, g), 0) or ''}</td>" for g in briefs.GROUPS) + "</tr>")
    mtf = sum(map(briefs.multi_timeframe, cards))
    s.append(f"</table></div><div class=sub>{mtf} of {len(cards)} cards combine timeframes.</div>")

    cut["hypotheses"] = len(s)
    # hypotheses
    s.append("<h2>Hypotheses</h2><div class=scroll><table><tr><th>id</th><th>title</th><th>family</th><th>horizon / group"
             "</th><th>status</th><th>died at</th><th>updated</th></tr>")
    for h, c in zip(hyps[::-1], cards[::-1]):
        tf = ", ".join((c.get("signal") or {}).get("timeframes") or [])
        s.append(f"<tr><td class=mono><a href='#h-{h['id']}' data-hyp='{h['id']}'>{h['id']}</a></td><td>{esc(h['title'])}</td>"
                 f"<td class=mono>{esc(lab.canonical_family(h['family']))}</td>"
                 f"<td>{esc(briefs.horizon(c))} / {esc(briefs.group(c))}{(' · ' + esc(tf)) if tf else ''}</td>"
                 f"<td>{pill(h['status'])}</td><td class=mono>{esc((h['reject_stage'] or '') + ' ' + (h['reject_code'] or ''))}"
                 f"</td><td class=mono>{esc(h['updated_at'][:10])}</td></tr>")
    s.append("</table></div>")
    cut["paper"] = len(s)
    # paper
    book = agents.paper_summary(lab)
    s.append(f"<h2>Paper book ({len(book)})</h2>")
    if not book:
        s.append("<div class=sub>Nothing has reached paper trading yet (it needs G0–G4 and the Skeptic).</div>")
    for b in book:
        from lab.framework import paper
        rec = paper.record(lab, b["id"])
        s.append(f"<h3>{b['id']} {pill(b['status'])} · {b['days']} days · entries {b['entries']}</h3>"
                 + nav_svg([("strategy", rec["ret"]), ("benchmark", rec["bench"])])
                 + f"<div class=sub>Sharpe {b['sharpe']:.2f} vs benchmark {b['bench_sharpe']:.2f} · max drawdown "
                   f"{b['max_dd']:.1%} (dev {fmt(b['dev_max_dd'])}) · G5 ready: {esc(b['g5_ready'])}</div>")

    cut["ops"] = len(s)
    # agent runs and cycles
    s.append("<h2>Agent runs</h2><div class=scroll><table><tr><th>started</th><th>agent</th><th>hyp.</th><th>outcome</th>"
             "<th class=num>turns</th><th class=num>tokens in/out</th><th>error</th></tr>")
    for r in lab.con.execute("SELECT * FROM agent_invocations ORDER BY started_at DESC LIMIT 40"):
        tone = "good" if r["outcome"] == "applied" else "bad" if r["outcome"] else "warn"
        s.append(f"<tr><td class=mono>{esc(r['started_at'][:16])}</td><td>{esc(r['agent'])}</td>"
                 f"<td class=mono>{esc(r['hypothesis_id'] or '')}</td><td class={tone}>{esc(r['outcome'] or 'running')}</td>"
                 f"<td class=num>{esc(r['n_turns'] or '')}</td><td class=num>{esc(r['tokens_in'] or '')} / "
                 f"{esc(r['tokens_out'] or '')}</td><td>{esc((r['error'] or '')[:200])}</td></tr>")
    s.append("</table></div><h3>Budget today</h3><table><tr><th>agent</th><th class=num>used</th><th class=num>daily</th></tr>")
    for a, n in (budget.get("daily") or {}).items():
        s.append(f"<tr><td>{esc(a)}</td><td class=num>{used.get(a, 0)}</td><td class=num>{n}</td></tr>")
    s.append("</table><h3>Recent cycles</h3><table><tr><th>ts</th><th>blocked</th><th>agents</th><th>tick</th></tr>")
    for c in cycles:
        ran = ", ".join(f"{a['agent']} {a.get('hypothesis') or ''} → {a.get('outcome')}" for a in c.get("agents", []))
        s.append(f"<tr><td class=mono>{esc(c['ts'][:16])}</td><td>{esc(c.get('blocked') or c.get('skipped') or c.get('warning') or '')}"
                 f"</td><td>{esc(ran)}</td><td class=mono>{esc('; '.join(c.get('tick') or [])[:400])}</td></tr>")
    s.append("</table>")

    cut["knowledge"] = len(s)
    # knowledge and lessons
    kb = knowledge.path(lab)
    s.append("<h2>Knowledge base</h2><div class=kb>" + (md(kb.read_text()) if kb.exists() else "<p>Empty.</p>") + "</div>")
    ls = lessons.path(lab)
    s.append("<h2>Lessons</h2><details><summary>per-hypothesis lessons (Librarian)</summary><div class=kb>"
             + (md(ls.read_text()) if ls.exists() else "<p>None yet.</p>") + "</div></details>")
    names = [("overview", "Overview"), ("pipeline", "Pipeline"), ("hypotheses", "Hypotheses"), ("paper", "Paper"),
             ("ops", "Operations"), ("knowledge", "Knowledge")]
    bounds = [cut[k] for k, _ in names if k in cut] + [len(s)]
    header = s[:cut["overview"]]
    panels = {}
    keys = [k for k, _ in names if k in cut]
    for i, k in enumerate(keys):
        panels[k] = "\n".join(s[bounds[i]:bounds[i + 1]])
    panels["pipeline"] = pipeline_view.build(lab, md)
    tabs = "".join(f"<input class=tab type=radio name=tab id='t-{k}'{' checked' if k == 'overview' else ''}>" for k, _ in names)
    labels = "<div class=tabs>" + "".join(f"<label for='t-{k}'>{esc(v)}</label>" for k, v in names) + "</div>"
    body = "<div class=panels>" + "".join(f"<div class=panel id='p-{k}'>{panels.get(k, '')}</div>" for k, _ in names) + "</div>"
    foot = ("<p class=sub>Static page generated by <code>lab dashboard</code>. Numbers are dev-period and paper "
            "results of the lab's own judge; nothing here is investment advice.</p>")
    return "\n".join(header) + tabs + labels + body + foot + f"<script>{JS}</script></main></body></html>"


def publish(lab: Lab, web: Path) -> Path:
    """Write a new release and switch `current` to it atomically; keep the last few releases."""
    rel = web / "releases" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rel.mkdir(parents=True, exist_ok=True)
    for d in (web, web / "releases"):     # the web server reads them; the cycle service runs with umask 0007
        os.chmod(d, 0o755)
    (rel / "index.html").write_text(build(lab))
    os.chmod(rel, 0o755)
    os.chmod(rel / "index.html", 0o644)
    tmp = web / "current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(rel)
    tmp.replace(web / "current")
    for old in sorted((web / "releases").iterdir())[:-KEEP_RELEASES]:
        shutil.rmtree(old, ignore_errors=True)
    return rel / "index.html"
