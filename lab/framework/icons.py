"""Inline SVG icons for the dashboard: a robot per LLM agent, a gear for deterministic code.

Everything is drawn in a 48 x 56 box and returned as markup, so it works inside a bigger SVG (`<g transform>`) and
as a small standalone `<svg>`; no image files, nothing to fetch (the page's CSP only allows inline content).

Moods (what the stage or agent is doing):
    work   the run is in progress: antenna pulses, eyes blink (gear spins)
    wait   it is its turn but nothing is running: eyes open, thinking bubble
    done   finished well: happy eyes, smile
    fail   it ended here: X eyes, frown
    idle   neutral
    sleep  not reached yet: closed eyes, faded
CSS animations live in the dashboard stylesheet (classes rb-ant, eye, gear under `.work`) and are switched off for
`prefers-reduced-motion`.
"""

import math

AGENT_COLOR = {"scout": "#3b82f6", "archivist": "#0ea5a4", "builder": "#e07a1f", "skeptic": "#d6455d",
               "librarian": "#7c5cd6", "chair": "#b08900", "steward": "#2f9e63"}
CODE_COLOR = "#5f6b7a"
GLYPH = {  # chest symbols, drawn in white strokes inside the body (x 12..36, y 36..52)
    "scout": '<circle cx="22.5" cy="43" r="3.2"/><path d="M25 45.5l3 3"/>',
    "archivist": '<ellipse cx="24" cy="40.5" rx="5" ry="1.8"/><path d="M19 40.5v6.5M29 40.5v6.5M19 47a5 1.8 0 0 0 10 0"/>',
    "builder": '<path d="M20 48l8-8M26 38.5l4 4M24.5 40l3-3"/>',
    "skeptic": '<path d="M24 38.5l6 2v4.2c0 3-2.8 5-6 6.3-3.2-1.3-6-3.3-6-6.3v-4.2z"/>',
    "librarian": '<path d="M17.5 40h6v10h-6zM24.5 40h6v10h-6z"/>',
    "chair": '<path d="M18 49v-8l4 4 2-5 2 5 4-4v8z"/>',
    "steward": '<path d="M17 45q7-7 14 0q-7 7-14 0z"/><circle cx="24" cy="45" r="1.6"/>',
}


def _eyes(mood: str) -> str:
    if mood == "done":
        return ('<path d="M14.5 22.5q3.5-5 7 0M26.5 22.5q3.5-5 7 0" fill="none" stroke="#fff" stroke-width="2" '
                'stroke-linecap="round"/>')
    if mood == "fail":
        return ('<path d="M15 18.5l6 6M21 18.5l-6 6M27 18.5l6 6M33 18.5l-6 6" stroke="#fff" stroke-width="2" '
                'stroke-linecap="round"/>')
    if mood == "sleep":
        return '<path d="M15 22h6M27 22h6" stroke="#fff" stroke-width="2" stroke-linecap="round"/>'
    return ('<g class="eye"><circle cx="18" cy="21.5" r="3.6" fill="#fff"/><circle cx="18.6" cy="21.8" r="1.7" fill="#1d1d1b"/></g>'
            '<g class="eye"><circle cx="30" cy="21.5" r="3.6" fill="#fff"/><circle cx="30.6" cy="21.8" r="1.7" fill="#1d1d1b"/></g>')


def _mouth(mood: str) -> str:
    if mood in ("done", "idle"):
        return '<path d="M19 28.5q5 4 10 0" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>'
    if mood == "fail":
        return '<path d="M19 30.5q5-4 10 0" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>'
    if mood == "work":
        return '<circle cx="24" cy="29" r="2.2" fill="#fff"/>'
    return '<path d="M20 29h8" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>'


def robot(agent: str, mood: str = "idle") -> str:
    """The robot of an LLM agent (colour and chest symbol are the agent's), 48 x 56."""
    c = AGENT_COLOR.get(agent, "#6b6a66")
    faded = ' opacity=".45"' if mood == "sleep" else ""
    bubble = ('<g fill="var(--muted)"><circle cx="41" cy="6" r="1.6"/><circle cx="44.5" cy="3" r="1.2"/></g>'
              if mood == "wait" else "")
    return (f'<g class="robot {"work" if mood == "work" else ""}"{faded}>'
            f'<line x1="24" y1="10" x2="24" y2="4.5" stroke="{c}" stroke-width="2"/>'
            f'<circle class="rb-ant" cx="24" cy="3.5" r="2.8" fill="{"var(--warn)" if mood == "work" else c}"/>'
            f'<rect x="3" y="17" width="5" height="10" rx="2.5" fill="{c}" opacity=".7"/>'
            f'<rect x="40" y="17" width="5" height="10" rx="2.5" fill="{c}" opacity=".7"/>'
            f'<rect x="8" y="10" width="32" height="25" rx="8" fill="{c}"/>'
            f'<rect x="10.5" y="12.5" width="27" height="20" rx="6" fill="#000" opacity=".16"/>'
            f'{_eyes(mood)}{_mouth(mood)}'
            f'<rect x="12" y="36" width="24" height="16" rx="5" fill="{c}" opacity=".85"/>'
            f'<g fill="none" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
            f'{GLYPH.get(agent, "")}</g>{bubble}</g>')


def _gear_path(cx: float, cy: float, r_out: float, r_in: float, teeth: int) -> str:
    pts = []
    for i in range(teeth * 4):
        ang = 2 * math.pi * i / (teeth * 4) - math.pi / (teeth * 4)
        r = r_out if i % 4 in (1, 2) else r_in
        pts.append(f"{cx + r * math.cos(ang):.1f},{cy + r * math.sin(ang):.1f}")
    return "M" + " L".join(pts) + " Z"


def gear(mood: str = "idle", color: str = CODE_COLOR) -> str:
    """The deterministic code (the judge, ingest, the paper runner): a gear, 48 x 56."""
    faded = ' opacity=".45"' if mood == "sleep" else ""
    c = {"fail": "var(--bad)", "done": "var(--good)"}.get(mood, color)
    hole = f'M{24 + 7},29 a7,7 0 1,0 -14,0 a7,7 0 1,0 14,0 Z'
    return (f'<g class="{"work" if mood == "work" else ""}"{faded}><g class="gear">'
            f'<path d="{_gear_path(24, 29, 21, 16, 8)} {hole}" fill="{c}" fill-rule="evenodd"/></g>'
            f'<text x="24" y="52" text-anchor="middle" font-size="7.5" font-family="ui-monospace,monospace" '
            f'fill="var(--muted)">{{ }}</text></g>')


def duo(agent: str, mood: str = "idle") -> str:
    """An LLM writes, code verifies: a robot with a small gear at its feet."""
    return (f'<g transform="translate(-3,0) scale(.86)">{robot(agent, mood)}</g>'
            f'<g transform="translate(24,26) scale(.55)">{gear("work" if mood == "work" else "idle", AGENT_COLOR.get(agent, CODE_COLOR))}</g>')


def icon(kind: str, agent: str | None = None, mood: str = "idle") -> str:
    if kind == "llm":
        return robot(agent or "", mood)
    if kind == "mixed":
        return duo(agent or "", mood)
    return gear(mood)


def svg(kind: str, agent: str | None = None, mood: str = "idle", size: int = 24) -> str:
    """A standalone small icon for HTML (summaries, cards)."""
    return (f'<svg class="ico" viewBox="0 0 48 56" width="{size}" height="{size * 56 // 48}" role="img" '
            f'aria-label="{kind}">{icon(kind, agent, mood)}</svg>')


CSS = """
@keyframes ant{0%,100%{opacity:.25}50%{opacity:1}}
@keyframes blink{0%,90%,100%{transform:scaleY(1)}95%{transform:scaleY(.08)}}
@keyframes spin{to{transform:rotate(360deg)}}
.work .rb-ant{animation:ant 1s infinite}
.work .eye{animation:blink 3s infinite;transform-box:fill-box;transform-origin:center}
.work .gear{animation:spin 4s linear infinite;transform-box:fill-box;transform-origin:center}
@media (prefers-reduced-motion:reduce){.work .rb-ant,.work .eye,.work .gear{animation:none}}
svg.ico{display:inline-block;vertical-align:middle;flex:none}.agent svg{flex:none}.legend svg{display:inline-block;vertical-align:middle}
"""
