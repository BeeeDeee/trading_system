"""Static page of the research 18 forward test (H1 and H2), linked from the LEAD forward tab (`research18.html`).

Everything shown is descriptive; the only decisions are those of docs/research18/PREREGISTRATION.md §7 at the dates of
§6. All dynamic strings are HTML-escaped, no scripts (same contract as `qlab.research17.dashboard`).
"""

from datetime import date, datetime
from html import escape

from qlab.research17.dashboard import CSS, _num, _pct
from qlab.research18.setup import ALPHA, END, MIN_EVENTS, T0
from qlab.research18.paper_mixes import VARIANTS, START as MIX_START

EXTRA = """
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.hy h3{margin:0 0 6px;font-size:15px}.hy .grid{grid-template-columns:repeat(auto-fit,minmax(130px,1fr))}
li.ok::marker{color:var(--ok)}li.bad::marker{color:var(--bad)}li.warn::marker{color:var(--warn)}
.leg span{margin-right:14px}.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:4px;vertical-align:-1px}
.paper{border-left:3px solid var(--accent);padding-left:12px}.smallnote{font-size:12px;color:var(--muted)}
"""
NAMES = {"H1": "H1 · LEAD_LAG0", "H2": "H2 · LEAD_K3_BTC200"}
DESC = {"H1": "BTC skok ≥ 3σ a altcoin za hodinu klesl → long 4 h",
        "H2": "BTC skok ≥ 3σ a altcoin zaostává (< 50 %) → long 4 h, jen když BTC > MA200"}
SERIES = {"H1": "var(--accent)", "H2": "var(--warn)"}
PAPER_SERIES = {"mix25_75": "var(--accent)", "mix25_75_rv14": "var(--warn)", "mix25_75_dd10": "var(--ok)"}


def _p(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def curves_svg(dates: list[str], rets: dict[str, list[float]]) -> str:
    """Cumulative return of both portfolios (direct labels at the end, tooltip on every 5th day)."""
    w, h, pad = 640, 200, 36
    if len(dates) < 2:
        return '<div class="muted box">Zatím málo dní na graf.</div>'
    eq = {}
    for k, r in rets.items():
        e, out = 1.0, []
        for x in r:
            e *= 1 + x
            out.append(e - 1)
        eq[k] = out
    lo, hi = min(0.0, *(min(v) for v in eq.values())), max(0.0, *(max(v) for v in eq.values()))
    span = (hi - lo) or 0.01
    x = lambda i: pad + (w - 2 * pad) * i / (len(dates) - 1)
    y = lambda v: h - pad + (2 * pad - h) * (v - lo) / span
    s = (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Kumulativní výnos portfolií H1 a H2">'
         f'<line x1="{pad}" x2="{w - pad}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="var(--rule)"/>'
         f'<text x="{pad}" y="14" fill="var(--muted)" font-size="11">{lo * 100:+.2f} % … {hi * 100:+.2f} %</text>'
         f'<text x="{pad}" y="{h - 4}" fill="var(--muted)" font-size="11">{escape(dates[0])}</text>'
         f'<text x="{w - pad}" y="{h - 4}" text-anchor="end" fill="var(--muted)" font-size="11">{escape(dates[-1])}</text>')
    colors = SERIES | PAPER_SERIES
    for k, v in eq.items():
        pts = " ".join(f"{x(i):.1f},{y(c):.1f}" for i, c in enumerate(v))
        color = colors.get(k, "var(--muted)")
        s += (f'<polyline fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" points="{pts}"/>'
              f'<text x="{w - pad + 4}" y="{y(v[-1]) + 4:.1f}" fill="var(--ink)" font-size="11">{k} {v[-1] * 100:+.1f} %</text>')
        s += "".join(f'<circle cx="{x(i):.1f}" cy="{y(v[i]):.1f}" r="5" fill="{color}" fill-opacity="0"><title>'
                     f'{k} {escape(dates[i])} {v[i] * 100:+.2f} %</title></circle>' for i in range(0, len(dates), 5))
    return s + "</svg>"


def criteria(m: dict, p_thr: float, alive_other: bool) -> list[tuple[str, str, str]]:
    """The four success criteria of §7 as (class, text, detail) rows for the checklist."""
    rows = []
    ok = m["p"] is not None and m["p"] < p_thr
    rows.append(("ok" if ok else "warn", "p-hodnota (event bootstrap, jednostranně)",
                 f"{_p(m['p'])} vs práh {p_thr:.4f}" + ("" if alive_other else " (druhá hypotéza není ve hře)")))
    rows.append(("ok" if m["sharpe"] > 0 else "bad", "Sharpe při 10 bps > 0", _num(m["sharpe"])))
    rows.append(("ok" if m["sharpe_cost15"] > 0 else "bad", "Sharpe při 15 bps > 0", _num(m["sharpe_cost15"])))
    rows.append(("ok" if m["n_events"] >= m["min_events"] else "warn", f"událostí ≥ {m['min_events']}", str(m["n_events"])))
    return rows


def hypothesis_card(h: str, m: dict, thr: float, other_alive: bool) -> str:
    ci = m.get("ci90_events") or [None, None]
    ev, mn = m["n_events"], m["min_events"]
    items = "".join(f"<li class='{c}'>{escape(t)}: <b>{escape(d)}</b></li>" for c, t, d in criteria(m, thr, other_alive))
    return (f'<div class="box hy"><h3><span class="dot" style="background:{SERIES[h]}"></span>{escape(NAMES[h])}</h3>'
            f'<div class="sub">{escape(DESC[h])}</div><div class="grid">'
            f'<div class="card"><small>Události (min. {mn})</small><b>{ev}</b><div class="bar"><i style="width:{min(100, 100 * ev / mn):.0f}%"></i></div></div>'
            f'<div class="card"><small>Obchodů</small><b>{m["n_trades"]}</b></div>'
            f'<div class="card"><small>Sharpe</small><b>{_num(m["sharpe"])}</b><small>při 15 bps {_num(m["sharpe_cost15"])}</small></div>'
            f'<div class="card"><small>Průměr obchodu</small><b>{_pct(m["mean_trade"])}</b>'
            f'<small>90 % CI {_pct(ci[0])} … {_pct(ci[1])}</small></div></div>'
            f'<ul style="margin-top:8px">{items}</ul></div>')


def render(status: dict | None, state: dict, now: datetime, mix_status: dict | None = None,
           mix_state: dict | None = None) -> str:
    """`status` = latest results/status_<datum>.json (None before the first one), `state` = {'waiting': bool,
    'error': str | None, 'last_run': iso}."""
    end1 = END[1]
    pill = ('<span class="pill warn">ČEKÁ NA ZAČÁTEK</span>' if status is None and not state.get("error")
            else '<span class="pill bad">CHYBA</span>' if state.get("error") else '<span class="pill ok">BĚŽÍ</span>')
    parts = [f'<h1>Research 18 · H1 a H2 {pill}</h1>',
             f'<div class="sub">Forward od {T0} · 1. rozhodnutí {end1} (12 měsíců), případně 2. {END[2]} · '
             f'<a href="index.html">zpět na LEAD forward</a> · <a href="backtest.html">historický backtest</a></div>']
    if state.get("error"):
        parts.append(f'<div class="box bad"><b>Výpočet statusu selhal</b><br>{escape(str(state["error"])[:300])}</div>')
    if status is None:
        parts.append(f'<div class="box warn">Zatím není žádný výsledek: test začíná {T0} a status se počítá z dat '
                     'stažených po tomto dni.</div>')
    else:
        m, d = status["metrics"], status["descriptive"]
        elapsed = (date.fromisoformat(status["asof"]) - T0).days + 1
        total = (end1 - T0).days + 1
        parts.append(f'<div class="grid"><div class="card"><small>Dní do 1. rozhodnutí</small><b>{elapsed} / {total}</b>'
                     f'<div class="bar"><i style="width:{min(100, 100 * elapsed / total):.0f}%"></i></div></div>'
                     f'<div class="card"><small>Data do</small><b>{escape(status["asof"])}</b></div>'
                     f'<div class="card"><small>BTC nad MA200</small><b>{_pct(d.get("btc_above_ma200_share"), 0)}</b>'
                     f'<small>podíl 15m svíček</small></div>'
                     f'<div class="card"><small>Obchodů zahodil filtr</small><b>{d.get("h2_trades_dropped_by_gate_vs_v01", 0)}</b>'
                     f'<small>H2 oproti V01_base</small></div></div>')
        parts.append(f'<div class="box">Průběžné hodnoty jsou jen <b>popisné</b>; nic nezastavují ani neprodlužují. Rozhodnutí '
                     f'vznikne jednou, po {end1}: Holmova korekce přes H1 a H2 při α = {ALPHA} (prahy {ALPHA / 2:.4f} a {ALPHA:.3f}), '
                     f'Sharpe &gt; 0 při 10 i 15 bps a minimum událostí ({MIN_EVENTS["H1"]} / {MIN_EVENTS["H2"]}). '
                     f'Síla testu je malá (H1 ~20–30 %, H2 ~10–15 % za 12 měsíců), nejčastější výsledek bude „neprůkazné“.</div>')
        ps = {h: m[h]["p"] for h in m if m[h]["p"] is not None}
        thr = {h: (ALPHA / 2 if len(ps) == 2 and v == min(ps.values()) else ALPHA) for h, v in ps.items()}
        parts.append('<h2>Stav kritérií (kdyby se rozhodovalo dnes; nerozhoduje)</h2><div class="cols">'
                     + "".join(hypothesis_card(h, m[h], thr.get(h, ALPHA), len(ps) == 2) for h in ("H1", "H2")) + "</div>")
        parts.append("<h2>Kumulativní výnos portfolií (10 rukávů po 1/10, každé své)</h2>")
        parts.append(curves_svg(status["daily"]["dates"], {h: status["daily"][h] for h in ("H1", "H2")}))
        parts.append('<div class="leg sub"><span><i class="dot" style="background:var(--accent)"></i>H1</span>'
                     '<span><i class="dot" style="background:var(--warn)"></i>H2</span></div>')
        rows = ""
        for name, v in (("V01_base (H2 bez filtru)", d["V01_base_no_gate"]), ("H1 + filtr MA200", d["H1_plus_gate"])):
            rows += (f"<tr><td>{escape(name)}</td><td>{_num(v['sharpe'])}</td><td>{v['n_trades']}</td><td>{v['n_events']}</td>"
                     f"<td>{_pct(v['mean_trade'])}</td></tr>")
        for name, v in d["scenarios"].items():
            rows += (f"<tr><td>{escape(name)}</td><td>{_num(v['sharpe'])}</td><td>{v['n_trades']}</td><td>{v['n_events']}</td>"
                     f"<td>{_pct(v['mean_trade'])}</td></tr>")
        diff = d.get("H2_minus_V01", {})
        parts.append("<h2>Popisně (§9)</h2><table><tr><th>Řada</th><th>Sharpe</th><th>Obchodů</th><th>Událostí</th>"
                     f"<th>Průměr</th></tr>{rows}</table><div class='sub'>Rozdíl Sharpe H2 − V01_base: {_num(diff.get('diff'))} "
                     f"(90 % CI {_num(diff.get('lo'))} … {_num(diff.get('hi'))}, párový bootstrap dní). "
                     "cost75 / cost200 = 7,5 / 20 bps na stranu, delay1 / delay2 = plnění o 1 / 2 svíčky později.</div>")
        for h in ("H1", "H2"):
            tr = status["trades"].get(h, [])
            rows = "".join(f"<tr><td>{escape(t['entry'].replace('T', ' ')[:16])}</td><td>{escape(t['sym'])}</td>"
                           f"<td>{_pct(t['ret'])}</td></tr>" for t in reversed(tr[-10:]))
            parts.append(f"<h2>Poslední obchody {h} (UTC)</h2>"
                         + (f"<table><tr><th>Vstup</th><th>Symbol</th><th>Výnos</th></tr>{rows}</table>" if rows
                            else '<div class="muted box">Zatím žádný obchod.</div>'))
        parts.append(f'<div class="sub"><code>commit {escape(str(status.get("commit", "")))}, pravidlo '
                     f'{escape(str(status.get("rule_hash", ""))[:12])}, data {escape(str(status.get("data_hash", ""))[:12])}</code></div>')
    parts.append(f'<div class="sub">Poslední běh: {escape(str(state.get("last_run", "–")))} · protokol: '
                 'docs/research18/PREREGISTRATION.md · zdrojem je results/status_&lt;datum&gt;.json (commitovat ručně).</div>')
    parts.append("<h2>Paper portfolio · mix BTC/H2</h2>")
    parts.append('<div class="box paper">Popisné paper sledování od '
                 f'<b>{MIX_START}</b>; bez příkazů a bez vlivu na rozhodnutí H1/H2. H2 má váhu 75 %, BTC 25 % před případnou brzdou; '
                 'signály platí od dalšího UTC dne a změna vah stojí 10 bps za dolar obratu.</div>')
    mix_state = mix_state or {}
    if mix_state.get("error"):
        parts.append(f'<div class="box bad">Výpočet mixů selhal: {escape(str(mix_state["error"])[:300])}</div>')
    if not mix_status:
        parts.append(f'<div class="muted box">{"Paper sledování začne po " + str(MIX_START) + "." if now.date() < MIX_START else "Zatím nejsou uložené výsledky mixů."}</div>')
    else:
        rr = mix_status["results"]
        rows = "".join(
            f'<tr><td>{escape(VARIANTS[k]["label"])}</td><td>{_pct(v["total_return"])}</td>'
            f'<td>{_pct(v["cagr"])}</td><td>{_pct(v["max_dd"])}</td><td>{_num(v["sharpe"])}</td>'
            f'<td>{_pct(v["avg_btc_multiplier"])}</td></tr>'
            for k, v in rr.items())
        parts.append(f'<div class="sub">Data do {escape(mix_status["asof"])} · {mix_status["n_days"]} dní paper běhu</div>')
        parts.append('<table><tr><th>Varianta</th><th>Celkem</th><th>CAGR*</th><th>Max. DD</th><th>Sharpe*</th>'
                     f'<th>Průměrná BTC váha / cíl</th></tr>{rows}</table>')
        parts.append(curves_svg(mix_status["daily"]["dates"],
                                {k: mix_status["daily"][k] for k in VARIANTS}))
        parts.append('<div class="leg sub">' + "".join(
            f'<span><i class="dot" style="background:{PAPER_SERIES[k]}"></i>{escape(VARIANTS[k]["label"])} · '
            f'{escape(VARIANTS[k]["description"])}</span>' for k in VARIANTS) +
            '</div><div class="smallnote">* Anualizované hodnoty jsou při krátkém paper vzorku velmi nestabilní. '
            'Předchozí data slouží jen k výpočtu indikátorů; výnos paper řady začíná v den zahájení.</div>')
    return ('<!doctype html><html lang="cs"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Research 18 – H1 a H2</title><style>{CSS}{EXTRA}</style></head><body><main>' + "".join(parts)
            + "</main></body></html>")
