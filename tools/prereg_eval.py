#!/usr/bin/env python3
"""Pre-registered confirmatory evaluation (PREREGISTRATION.md). Written and tested on synthetic data BEFORE any
result was looked at; the document pins this file's sha256. Do not change it after sealing: any change is a new,
clearly labelled post-hoc analysis.

  python tools/prereg_eval.py export.json                     # pre-registered window
  python tools/prereg_eval.py export.json --from D1 --to D2   # other window: labelled NOT PRE-REGISTERED
  python tools/prereg_eval.py --demo                          # synthetic market (tool check only)

export.json = the journal database: {"days": [day docs], "scores": [{date, scores}], "prices": [{date, close}],
"history": {ticker: [[date, o, h, l, c], ...]}}  (same collections as tools/analyze.py and random_null.py).

H1  composite carries information: mean daily 1-day rank IC of composite vs the equal-weight universe > 0
H2  the LLM adds something over plain momentum: mean daily [IC(composite) - IC(mom20)] at 1 day > 0
    (mom20 = 20-trading-day return from the same close, computed here from prices: no LLM)
H3  this shows up in money: mean daily return difference zaklad - equal-weight universe > 0 (after costs)
Each: two-sided Student t test, alpha 0.05, plus the sign of the mean must agree in both halves of the window.
"""
import argparse
import datetime
import json
import math
import pathlib
import sys

PREREG_FROM, PREREG_TO, EVAL_NOT_BEFORE = "2026-09-29", "2026-11-25", "2026-12-04"
PREREG_TAGS = ("v1.1.0", "v1.1.1", "v1.1.2")
MIN_SCORE_DAYS = 32
ALPHA = 0.05
LENSES = ("trend", "mr", "news", "conviction")
MOM_LOOKBACK = 20


# ---------------------------------------------------------------- statistics (stdlib only)

def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1 + aa / c
            c = c if abs(c) > 1e-300 else 1e-300
            h *= d * c
        if abs(d * c - 1) < 1e-12:
            break
    return h


def betainc(a, b, x):
    if x <= 0 or x >= 1:
        return 0.0 if x <= 0 else 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def t_test(vals):
    n = len(vals)
    if n < 3:
        return {"n": n, "mean": None, "t": None, "p": None}
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    if sd == 0:
        return {"n": n, "mean": m, "t": None, "p": None}
    t = m / (sd / math.sqrt(n))
    df = n - 1
    half = n // 2
    m1, m2 = sum(vals[:half]) / half, sum(vals[half:]) / (n - half)
    return {"n": n, "mean": m, "t": t, "p": betainc(df / 2, 0.5, df / (df + t * t)),
            "first_half": m1, "second_half": m2, "halves_agree": (m1 > 0) == (m2 > 0)}


def block_means(vals, block):
    return [sum(vals[i:i + block]) / block for i in range(0, len(vals) - block + 1, block)]


def holm(ps):
    items = sorted((p, k) for k, p in ps.items() if p is not None)
    out, run, m = {}, 0.0, len(items)
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb) if va > 0 and vb > 0 else None


# ---------------------------------------------------------------- data

def composite(s):
    """Same as engine.composite (v1.1.x)."""
    return sum((s.get(k) or 0) for k in ("trend", "mr", "news")) + 0.5 * ((s.get("conviction") or 3) - 3)


def close_table(exp):
    """{date: {ticker: close}} from history (pre-start) + prices (journal); journal wins on overlap."""
    px = {}
    for t, rows in (exp.get("history") or {}).items():
        for r in rows:
            px.setdefault(r[0], {})[t] = r[4]
    for p in exp["prices"]:
        px.setdefault(p["date"], {}).update({t: v for t, v in p["close"].items() if v})
    return px


def daily_ic(scores, px, lens, h):
    dates = sorted(px)
    idx = {d: i for i, d in enumerate(dates)}
    out = []
    for s in scores:
        d = s["date"]
        if d not in idx or idx[d] + h >= len(dates):
            continue
        d2 = dates[idx[d] + h]
        rets = {t: px[d2][t] / px[d][t] - 1 for t in s["scores"] if px[d].get(t) and px[d2].get(t)}
        if len(rets) < 6:
            continue
        mkt = sum(rets.values()) / len(rets)
        xs, ys = [], []
        for t, r in rets.items():
            if lens == "mom20":
                if idx[d] < MOM_LOOKBACK:
                    continue
                p0 = px[dates[idx[d] - MOM_LOOKBACK]].get(t)
                v = px[d][t] / p0 - 1 if p0 else None
            elif lens == "composite":
                v = composite(s["scores"][t])
            else:
                v = s["scores"][t].get(lens)
            if v is None:
                continue
            xs.append(v)
            ys.append(r - mkt)
        if len(xs) >= 6:
            ic = corr(ranks(xs), ranks(ys))
            if ic is not None:
                out.append((d, ic))
    return out


def diff_series(a, b):
    db = dict(b)
    return [v - db[d] for d, v in a if d in db]


def money_diff(days, vid):
    """Daily return of variant minus the equal-weight universe benchmark, from consecutive journal days."""
    out = []
    for prev, cur in zip(days, days[1:]):
        try:
            a = cur["variants"][vid]["equity"] / prev["variants"][vid]["equity"] - 1
            b = cur["universe_equity"] / prev["universe_equity"] - 1
        except (KeyError, TypeError, ZeroDivisionError):
            continue
        out.append(a - b)
    return out


# ---------------------------------------------------------------- report

def verdict(r):
    if r["p"] is None:
        return "málo dat"
    if r["p"] < ALPHA and r["halves_agree"]:
        return "POTVRZENO" if r["mean"] > 0 else "OPAČNÝ EFEKT"
    return "nepotvrzeno"


def fmt(r, scale=1.0, unit=""):
    if r["p"] is None:
        return f"n={r['n']}  (málo dat)"
    return (f"n={r['n']:>3}  průměr {r['mean'] * scale:+.4f}{unit}  t={r['t']:+.2f}  p={r['p']:.4f}  "
            f"poloviny {r['first_half'] * scale:+.4f} / {r['second_half'] * scale:+.4f}")


def tag_of(day):
    return ((day.get("code") or {}).get("tag")) or day.get("tag")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", nargs="?")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--from", dest="d_from")
    ap.add_argument("--to", dest="d_to")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.demo:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import simulate
        exp = simulate.run(json.load(open(simulate.ROOT / "config" / "config.json")), days=45, seed=5)
        a.d_from, a.d_to = a.d_from or "0000-00-00", a.d_to or "9999-99-99"
    elif a.export:
        exp = json.load(open(a.export))
    else:
        raise SystemExit(__doc__)
    prereg = a.d_from is None and a.d_to is None
    d_from, d_to = a.d_from or PREREG_FROM, a.d_to or PREREG_TO
    print("=" * 100)
    if prereg:
        print(f"PRE-REGISTROVANÉ VYHODNOCENÍ  okno {d_from} … {d_to}")
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        if today < EVAL_NOT_BEFORE:
            print(f"POZOR: dnes je {today}, vyhodnocení je platné až od {EVAL_NOT_BEFORE}")
    else:
        print(f"NENÍ PRE-REGISTROVANÉ OKNO ({d_from} … {d_to}) – jen test nástroje / post-hoc, nikdy důkaz")
    print("=" * 100)

    days, excluded = [], []
    for d in sorted(exp["days"], key=lambda x: x["date"]):
        if not (d_from <= d["date"] <= d_to):
            continue
        tag = tag_of(d)
        if prereg and tag not in PREREG_TAGS:
            excluded.append((d["date"], f"verze {tag}"))
            continue
        days.append(d)
    ok_dates = {d["date"] for d in days}
    scores = [s for s in exp["scores"] if s["date"] in ok_dates and s.get("scores")]
    px = close_table(exp)
    print(f"dní v okně: {len(days)}, se skóre: {len(scores)}, vyřazeno: {len(excluded)}")
    for D, why in excluded:
        print(f"  vyřazeno {D}: {why}")
    valid = len(scores) >= MIN_SCORE_DAYS
    if not valid:
        print(f"NEPLATNÉ: méně než {MIN_SCORE_DAYS} dní se skóre – neprůkazné kvůli datům")
    res = {"window": [d_from, d_to], "preregistered": prereg, "days": len(days), "score_days": len(scores),
           "valid": valid, "excluded": excluded}

    ic_c = daily_ic(scores, px, "composite", 1)
    ic_m = daily_ic(scores, px, "mom20", 1)
    h1, h2, h3 = t_test([v for _, v in ic_c]), t_test(diff_series(ic_c, ic_m)), t_test(money_diff(days, "zaklad"))
    v1, v2, v3 = verdict(h1), verdict(h2), verdict(h3)
    print("\nPRIMÁRNÍ HYPOTÉZY (každá α = 0,05, dvoustranně, + shoda znaménka v obou polovinách)")
    print(f"  H1 IC composite 1 d               {fmt(h1)}  → {v1}")
    print(f"  H2 IC composite − IC mom20 1 d    {fmt(h2)}  → {v2}")
    print(f"  H3 zaklad − rovné váhy denně      {fmt(h3, 1e4, ' bps')}  → {v3}")
    adds = valid and v1 == "POTVRZENO" and v2 == "POTVRZENO"
    print(f"\n  ZÁVĚR: LLM {'PŘIDÁVÁ informaci nad momentum' if adds else 'neprokázal přidanou hodnotu nad momentum'}"
          f"{'; projevilo se to i v penězích (H3)' if adds and v3 == 'POTVRZENO' else ''}")
    res["primary"] = {"H1": {**h1, "verdict": v1}, "H2": {**h2, "verdict": v2}, "H3": {**h3, "verdict": v3},
                      "llm_adds_information": adds}

    print("\nSEKUNDÁRNÍ A – horizont 5 dní (nepřekrývající se 10denní bloky, Holm přes 2)")
    c5, m5 = daily_ic(scores, px, "composite", 5), daily_ic(scores, px, "mom20", 5)
    sa = {"IC composite 5 d": t_test(block_means([v for _, v in c5], 10)),
          "IC composite − mom20 5 d": t_test(block_means(diff_series(c5, m5), 10))}
    adj = holm({k: r["p"] for k, r in sa.items()})
    for k, r in sa.items():
        print(f"  {k:<28} {fmt(r)}  p_holm={adj.get(k, float('nan')):.4f}")
    res["secondary_5d"] = {k: {**r, "p_holm": adj.get(k)} for k, r in sa.items()}

    print("\nSEKUNDÁRNÍ B – jednotlivé pohledy, IC 1 d (Holm přes 4)")
    sb = {lens: t_test([v for _, v in daily_ic(scores, px, lens, 1)]) for lens in LENSES}
    adj = holm({k: r["p"] for k, r in sb.items()})
    for k, r in sb.items():
        sig = adj.get(k) is not None and adj[k] < ALPHA and r["halves_agree"]
        print(f"  {k:<28} {fmt(r)}  p_holm={adj.get(k, float('nan')):.4f}  {'EDGE' if sig else ''}")
    res["secondary_lenses"] = {k: {**r, "p_holm": adj.get(k)} for k, r in sb.items()}

    vids = sorted({v for d in days for v in d["variants"]} - {"zaklad"})
    print(f"\nSEKUNDÁRNÍ C – ostatní varianty vs rovné váhy, denní rozdíl (Holm přes {len(vids)})")
    sc = {v: t_test(money_diff(days, v)) for v in vids}
    adj = holm({k: r["p"] for k, r in sc.items()})
    for k, r in sc.items():
        sig = adj.get(k) is not None and adj[k] < ALPHA and r["mean"] > 0 and r["halves_agree"]
        print(f"  {k:<18} {fmt(r, 1e4, ' bps')}  p_holm={adj.get(k, float('nan')):.4f}  {'EDGE' if sig else ''}")
    res["secondary_variants"] = {k: {**r, "p_holm": adj.get(k)} for k, r in sc.items()}

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)
        print(f"uloženo: {a.json}")


if __name__ == "__main__":
    main()
