"""Static dashboard page of the research 17 forward test (one self-contained HTML, inline CSS/JS only).

Served by the portal as the tab "LEAD forward". Everything shown is descriptive; the only decision is the one of
docs/research17/FORWARD.md §4. All dynamic strings are HTML-escaped.
"""

from datetime import date, datetime
from html import escape

from qlab.research17 import ops, variants

CSS = """
:root{--bg:#f4f5f4;--card:#fbfcfb;--ink:#18211d;--muted:#5a6760;--rule:#d5dbd7;--ok:#2d7a4b;--warn:#b86e00;--bad:#b3392f;--accent:#2f6ba8}
@media (prefers-color-scheme:dark){:root{--bg:#101513;--card:#161d1a;--ink:#e3e9e5;--muted:#9aa8a1;--rule:#2a3430;--ok:#5fbf85;--warn:#e0a040;--bad:#e5766c;--accent:#4f8ad0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 8px}
.sub{color:var(--muted);margin-bottom:12px}.pill{display:inline-block;padding:2px 10px;border-radius:10px;font-weight:600;font-size:12px;border:1px solid currentColor}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:10px 12px}.card b{display:block;font-size:20px;margin-top:2px}
.card small{color:var(--muted)}.box{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:10px 12px;margin:10px 0}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--rule);border-radius:8px;overflow:hidden}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--rule)}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:0}svg{width:100%;height:auto;background:var(--card);border:1px solid var(--rule);border-radius:8px}
.bar{height:8px;background:var(--rule);border-radius:4px;overflow:hidden;margin-top:6px}.bar i{display:block;height:100%;background:var(--accent)}
ul{margin:4px 0;padding-left:20px}code{font-size:12px}
"""


def _pct(x, d=2):
    return "–" if x is None else f"{x * 100:+.{d}f} %"


def _num(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def equity_svg(dates: list[str], ret: list[float], trades: list[dict]) -> str:
    """Cumulative return of the paper portfolio, one point per day, trade entries as dots on the x axis."""
    w, h, pad = 640, 190, 34
    eq, e = [], 1.0
    for r in ret:
        e *= 1 + r
        eq.append(e - 1)
    if len(eq) < 2:
        return '<div class="muted box">Zatím málo dní na graf.</div>'
    lo, hi = min(0.0, min(eq)), max(0.0, max(eq))
    span = (hi - lo) or 0.01
    x = lambda i: pad + (w - 2 * pad) * i / (len(eq) - 1)
    y = lambda v: h - pad + (2 * pad - h) * (v - lo) / span
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(eq))
    idx = {d: i for i, d in enumerate(dates)}
    dots = "".join(f'<circle cx="{x(idx[t["entry"][:10]]):.1f}" cy="{h - pad + 8:.1f}" r="3" fill="var(--accent)"><title>'
                   f'{escape(t["sym"])} {escape(t["entry"])} {t["ret"] * 100:+.2f} %</title></circle>'
                   for t in trades if t["entry"][:10] in idx)
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Kumulativní výnos paper portfolia">'
            f'<line x1="{pad}" x2="{w - pad}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="var(--rule)"/>'
            f'<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="{pts}"/>{dots}'
            f'<text x="{pad}" y="14" fill="var(--muted)" font-size="11">{lo * 100:+.2f} % … {hi * 100:+.2f} %</text>'
            f'<text x="{w - pad}" y="{h - 4}" text-anchor="end" fill="var(--muted)" font-size="11">{escape(dates[-1])}</text>'
            f'<text x="{pad}" y="{h - 4}" fill="var(--muted)" font-size="11">{escape(dates[0])}</text></svg>')


def variants_table(vs: dict | None) -> str:
    """30 parameter variants next to the primary rule; descriptive only (FORWARD.md §10), no ranking, no selection."""
    if not vs:
        return ""
    rows = []
    for key, title in variants.GROUPS.items():
        items = [(n, v) for n, v in vs.items() if v["spec"]["group"] == key]
        if not items:
            continue
        rows.append(f"<tr><th colspan='6' style='text-align:left'>{escape(title)}</th></tr>")
        for n, v in items:
            cls = "" if not v["n_trades"] else (" class='ok'" if v["meets"] else " class='bad'")
            rows.append(f"<tr><td title='{escape(v['spec']['desc'])}'>{escape(n)}<br><small class='muted'>"
                        f"{escape(v['spec']['desc'])}</small></td><td>{v['n_events']}</td><td>{v['n_trades']}</td>"
                        f"<td{cls}>{_num(v['sharpe']) if v['n_trades'] else '–'}</td><td>{_pct(v['mean_trade'])}</td>"
                        f"<td>{_pct(v['total_return'])}<br><small class='muted'>DD {v['max_dd'] * 100:.1f} %</small></td></tr>")
    n_ok = sum(v["meets"] for v in vs.values())
    return ("<h2>Varianty parametrů (popisně)</h2><div class='sub'>30 variant téhož pravidla na týchž datech a členech. "
            "Nic nerozhoduje, nic se z nich nevybírá a netuní (FORWARD.md §10); jedna dobrá varianta ze 30 je očekávaná "
            f"náhoda. Kritérium (Sharpe &gt; 0 a průměr &gt; 0) zatím splňuje {n_ok} z {len(vs)}.</div>"
            "<table><tr><th>Varianta</th><th>Události</th><th>Obchodů</th><th>Sharpe</th><th>Průměr obchodu</th>"
            "<th>Výnos / propad</th></tr>" + "".join(rows) + "</table>")


def render(forward: dict | None, state: dict, now: datetime, start: date, end: date, min_events: int) -> str:
    probs = state.get("alerts", [])
    pill = ('<span class="pill bad">PROBLÉM</span>' if probs else '<span class="pill ok">OK</span>')
    parts = [f'<h1>LEAD forward test {pill}</h1>',
             f'<div class="sub">Výzkum 17: BTC skok ≥ 3σ, altcoin zaostává → long 4 h · paper, 10 bps/strana · '
             f'{start} → {end} · <a href="backtest.html">Historický backtest 30 variant (post hoc)</a>'
             ' · <a href="research18.html">Research 18: H1 a H2 (forward od 2026-10-20)</a></div>']
    if probs:
        parts.append('<div class="box bad"><b>Upozornění</b><ul>' + "".join(f"<li>{escape(p)}</li>" for p in probs)
                     + "</ul></div>")
    if forward is None:
        parts.append('<div class="box warn">Zatím není žádný výsledek statusu.</div>')
    else:
        f = forward
        sp, ci = f["P_stats"], f["mean_trade_ci90_events"]
        elapsed = (date.fromisoformat(f["asof"]) - start).days + 1
        total = (end - start).days + 1
        ev = f["n_events"]
        cum = 1.0
        for r in f["daily"]["ret"]:
            cum *= 1 + r
        cum -= 1
        parts.append(
            '<div class="grid">'
            f'<div class="card"><small>Dní</small><b>{elapsed} / {total}</b><div class="bar"><i style="width:{min(100, 100 * elapsed / total):.0f}%"></i></div></div>'
            f'<div class="card"><small>Události (min. {min_events})</small><b>{ev}</b><div class="bar"><i style="width:{min(100, 100 * ev / min_events):.0f}%"></i></div></div>'
            f'<div class="card"><small>Obchodů</small><b>{f["n_trades"]}</b></div>'
            f'<div class="card"><small>Sharpe (popisně)</small><b>{_num(sp["sharpe"])}</b></div>'
            f'<div class="card"><small>Průměr obchodu</small><b>{_pct(sp.get("mean_trade"))}</b>'
            f'<small>90 % CI {_pct(ci[1])} … {_pct(ci[2])}</small></div>'
            f'<div class="card"><small>Nula, p</small><b>{_num(f["null_p"], 3)}</b></div>'
            f'<div class="card"><small>Výnos portfolia</small><b>{_pct(cum)}</b>'
            f'<small>max. propad {sp["max_dd"] * 100:.1f} %</small></div></div>')
        pc = f.get("primary_criterion")
        if pc is None:
            verdict = '<span class="muted">Zatím žádná událost.</span>'
        elif not pc["enough_events"]:
            verdict = (f'<span class="warn">Neprůkazné: {ev} z {min_events} událostí.</span> Rozhodnutí až po {end} '
                       f'a ≥ {min_events} událostech (FORWARD.md §4).')
        else:
            verdict = ('<span class="ok">Kritérium zatím splněno</span>' if pc["meets"]
                       else '<span class="bad">Kritérium zatím nesplněno</span>') + f' (rozhoduje až stav k {end}).'
        parts.append(f'<div class="box">{verdict} Průběžné hodnoty jsou jen popisné a nic nezastavují.</div>')
        parts.append("<h2>Kumulativní výnos portfolia (10 rukávů po 1/10)</h2>")
        parts.append(equity_svg(f["daily"]["dates"], f["daily"]["ret"], f["trades"]))
        rows = "".join(f"<tr><td>{escape(t['entry'].replace('T', ' ')[:16])}</td><td>{escape(t['sym'])}</td>"
                       f"<td>{_pct(t['ret'])}</td></tr>" for t in reversed(f["trades"][-15:]))
        parts.append("<h2>Poslední obchody (UTC)</h2>"
                     + (f"<table><tr><th>Vstup</th><th>Symbol</th><th>Výnos</th></tr>{rows}</table>" if rows
                        else '<div class="muted box">Zatím žádný obchod.</div>'))
        cur = f"{date.fromisoformat(f['asof']):%Y-%m}"
        mem = "".join(f"<tr><td>{escape(m)}{' (aktuální)' if m == cur else ''}</td><td style='text-align:left'>"
                      f"{escape(', '.join(s.replace('USDT', '') for s in v))}</td></tr>"
                      for m, v in sorted(f["members_used"].items()))
        pend = ("".join(f"<tr><td>{escape(m)}</td><td style='text-align:left' class='warn'>členství zatím chybí</td></tr>"
                        for m in f["pending_months"]))
        parts.append(f"<h2>Členové (top 10 podle objemu)</h2><table><tr><th>Měsíc</th><th>Altcoiny</th></tr>{mem}{pend}</table>")
        sc = "".join(f"<tr><td>{escape(n)}</td><td>{_num(v['sharpe'])}</td><td>{v.get('n_trades', 0)}</td>"
                     f"<td>{_pct(v.get('mean_trade'))}</td></tr>" for n, v in f["scenarios"].items())
        parts.append("<h2>Citlivost (popisně)</h2><table><tr><th>Varianta</th><th>Sharpe</th><th>Obchodů</th>"
                     f"<th>Průměr</th></tr>{sc}</table><div class='sub'>cost75 = 7,5 bps, x2 = 20 bps, delay1 = plnění o "
                     "svíčku později, nolag = bez podmínky zaostávání.</div>")
        parts.append(variants_table(f.get("variants")))
        eth = f["eth_reference_weight1"]
        ps = "".join(f"<tr><td>{escape(s)}</td><td>{v['n']}</td><td>{_pct(v['mean'])}</td><td>{_pct(v['contribution'])}</td></tr>"
                     for s, v in sorted(f["per_symbol"].items(), key=lambda kv: -kv[1]["contribution"]))
        if ps:
            parts.append("<h2>Podle altcoinů</h2><table><tr><th>Symbol</th><th>Obchodů</th><th>Průměr</th>"
                         f"<th>Příspěvek</th></tr>{ps}</table>")
        parts.append(f'<div class="sub">ETH referenčně (váha 1): Sharpe {_num(eth["sharpe"])}, '
                     f'{eth.get("n_trades", 0)} obchodů.</div>')
    dl, st = state.get("download", {}), state.get("status", {})
    age = ops.run_age_days(state, now)
    stale = age is not None and age > ops.STALE_RUN_DAYS
    parts.append("<h2>Stav sběru dat</h2><div class='box'><ul>"
                 f"<li>Poslední týdenní běh: {escape(state.get('last_run', '–'))}"
                 + (f" <span class='bad'>(před {age:.0f} dny, cron neběží?)</span>" if stale else "") + "</li>"
                 f"<li>Poslední úspěšný běh bez problémů: {escape(state.get('last_ok', '–'))}</li>"
                 f"<li>Stahování: {'OK' if not dl.get('errors') and dl.get('exit', 0) == 0 else 'CHYBA'}, "
                 f"status: {'OK' if st.get('exit', 0) == 0 else 'CHYBA'}</li>"
                 + (f"<li>Data do: {escape(forward['asof'])}</li>" if forward else "")
                 + (f"<li><code>commit {escape(str(forward.get('commit', '')))}, pravidlo {escape(str(forward.get('rule_hash', ''))[:12])}, "
                    f"data {escape(str(forward.get('data_hash', ''))[:12])}</code></li>" if forward else "")
                 + "</ul></div>")
    parts.append('<div class="sub">Protokol: docs/research17/FORWARD.md · zdrojem je výsledek '
                 'results/forward_&lt;datum&gt;.json (commitováno).</div>')
    return ('<!doctype html><html lang="cs"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>LEAD forward test</title><style>{CSS}</style></head><body><main>' + "".join(parts) + "</main></body></html>")
