#!/usr/bin/env python3
"""Pre-registered confirmatory evaluation (docs/PREREGISTRATION.md). Written and tested on synthetic data
BEFORE any result was looked at; the document pins this file's sha256. Do not change it after sealing:
any change is a new, clearly labelled post-hoc analysis.

  python tools/prereg_eval.py                       # pre-registered window, run on/after the evaluation date
  python tools/prereg_eval.py --from D1 --to D2     # other window: output is labelled NOT PRE-REGISTERED (tests)

H1  composite carries information: mean daily 1-day rank IC (vs BTC) > 0
H2  the LLM adds something over plain momentum: mean daily [IC(composite) - IC(rel30)] at 1 day > 0
H3  this shows up in money: mean daily return difference zaklad - mech_momentum > 0 (net), and also > 0 under stress
Each: two-sided Student t test, alpha 0.05, plus the sign of the mean must agree in both halves of the window.
Secondary tests are reported with Holm (confirmatory families) or Benjamini-Hochberg (exploratory) adjustment.
"""
import argparse
import datetime
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
from cpb import analytics as A, canon, pipeline  # noqa: E402

PREREG_FROM, PREREG_TO, EVAL_NOT_BEFORE = "2026-10-01", "2026-11-30", "2026-12-15"
PREREG_VERSIONS = ("v1.0.0", "v1.0.1", "v1.0.2", "v1.0.3", "v1.0.4")   # patch releases without rule changes
MODEL = "claude-opus-5-5"
MIN_SCORE_DAYS = 45
ALPHA = 0.05
SECONDARY_LENSES = ("trend", "news", "mr", "conviction", "p_outperform_btc_7d", "expected_move_7d_pct")
DAILY_VARIANTS = ("zaklad", "rovne", "inv_vol", "skore_vol", "bez_brany", "btc_filtr", "vol_target", "top3", "tydenni",
                  "stop", "jen_trend", "jen_zpravy", "jen_mr", "pravdepodobnost", "claude_volne", "kontrarian")


# ---------------------------------------------------------------- Student t without scipy

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
    """Two-sided one-sample t test of mean 0 -> dict."""
    n = len(vals)
    if n < 3:
        return {"n": n, "mean": None, "t": None, "p": None}
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    if sd == 0:
        return {"n": n, "mean": m, "t": None, "p": None}
    t = m / (sd / math.sqrt(n))
    df = n - 1
    p = betainc(df / 2, 0.5, df / (df + t * t))
    half = n // 2
    m1, m2 = sum(vals[:half]) / half, sum(vals[half:]) / (n - half)
    return {"n": n, "mean": m, "t": t, "p": p, "first_half": m1, "second_half": m2, "halves_agree": (m1 > 0) == (m2 > 0)}


def block_means(vals, block):
    """Non-overlapping block means: overlapping h-day windows are autocorrelated, block = 2h keeps the test honest."""
    return [sum(vals[i:i + block]) / block for i in range(0, len(vals) - block + 1, block)]


def holm(ps):
    """{key: p} -> {key: adjusted p}."""
    items = sorted((p, k) for k, p in ps.items() if p is not None)
    out, run, m = {}, 0.0, len(items)
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def bh(ps):
    items = sorted((p, k) for k, p in ps.items() if p is not None)
    m, out, run = len(items), {}, 1.0
    for i in range(m - 1, -1, -1):
        p, k = items[i]
        run = min(run, p * m / (i + 1))
        out[k] = run
    return out


# ---------------------------------------------------------------- data

def load(repo, d_from, d_to, check_segment=True):
    days, excluded = [], []
    for p in sorted(glob.glob(os.path.join(repo, "runs", "*", "run.json"))):
        rec = canon.read_json(p)
        D = rec["date"]
        if not (d_from <= D <= d_to):
            continue
        tag = (rec.get("version") or {}).get("tag")
        why = None
        if rec["status"] not in ("ok", "warning", "catchup"):
            why = f"status {rec['status']}"
        elif check_segment and tag not in PREREG_VERSIONS:
            why = f"verze {tag} není v primárním segmentu"
        elif check_segment and rec.get("llm_model_pinned") != MODEL:
            why = f"model {rec.get('llm_model_pinned')}"
        if why:
            excluded.append((D, why))
            continue
        d = os.path.dirname(p)
        inp = canon.read_json(os.path.join(d, "inputs.json"))
        days.append({"date": D, "asof": inp["asof"], "scores": inp.get("scores"),
                     "features": canon.read_json(os.path.join(d, "features.json")) if os.path.exists(os.path.join(d, "features.json")) else None,
                     "ledger": canon.read_json(os.path.join(d, "ledger.json"))})
    return days, excluded


def daily_ic_pairs(days, px, rules, lens, h):
    """[(date, ic)] with lens from scores, or 'rel30' from features (the mechanical momentum control)."""
    from cpb.portfolio import composite
    out = []
    for d in days:
        sc, ft = d["scores"], d["features"]
        if not sc or not ft:
            continue
        xs, ys = [], []
        for c, s in sc["coins"].items():
            if c == "BTC":
                continue
            if lens == "composite":
                v = composite(s, rules)
            elif lens == "rel30":
                v = (ft.get(c) or {}).get("rel30")
            else:
                v = s.get(lens)
            _, rel = A.fwd_rel(px, c, d["asof"], h)
            if v is None or rel is None:
                continue
            xs.append(v)
            ys.append(rel)
        if len(xs) >= 6:
            ic = A.corr(A.ranks(xs), A.ranks(ys))
            if ic is not None:
                out.append((d["date"], ic))
    return out


def diff_series(a, b):
    db = dict(b)
    return [v - db[D] for D, v in a if D in db]


def ret_diff(days, p1, p2, scn):
    """Daily return difference p1 - p2 from the 00:00 close marks (same moment for both)."""
    out = []
    for prev, cur in zip(days, days[1:]):
        try:
            a0, a1 = prev["ledger"]["close"][p1][scn]["equity"], cur["ledger"]["close"][p1][scn]["equity"]
            b0, b1 = prev["ledger"]["close"][p2][scn]["equity"], cur["ledger"]["close"][p2][scn]["equity"]
        except (KeyError, TypeError):
            continue
        out.append((a1 / a0 - 1) - (b1 / b0 - 1))
    return out


# ---------------------------------------------------------------- report

def verdict(r):
    if r["p"] is None:
        return "málo dat"
    if r["p"] < ALPHA and r["mean"] > 0 and r["halves_agree"]:
        return "POTVRZENO"
    if r["p"] < ALPHA and r["mean"] < 0 and r["halves_agree"]:
        return "OPAČNÝ EFEKT"
    return "nepotvrzeno"


def fmt(r, scale=1.0, unit=""):
    if r["p"] is None:
        return f"n={r['n']}  (málo dat)"
    return (f"n={r['n']:>3}  průměr {r['mean'] * scale:+.4f}{unit}  t={r['t']:+.2f}  p={r['p']:.4f}  "
            f"poloviny {r['first_half'] * scale:+.4f} / {r['second_half'] * scale:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=ROOT)
    ap.add_argument("--from", dest="d_from")
    ap.add_argument("--to", dest="d_to")
    ap.add_argument("--json", help="uložit výsledek jako JSON")
    a = ap.parse_args()
    prereg = a.d_from is None and a.d_to is None
    d_from, d_to = a.d_from or PREREG_FROM, a.d_to or PREREG_TO
    print("=" * 100)
    if prereg:
        print(f"PRE-REGISTROVANÉ VYHODNOCENÍ  okno {d_from} … {d_to}")
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        if today < EVAL_NOT_BEFORE:
            print(f"POZOR: dnes je {today}, vyhodnocení je platné až od {EVAL_NOT_BEFORE} (forward okna nejsou uzavřená)")
    else:
        print(f"NENÍ PRE-REGISTROVANÉ OKNO ({d_from} … {d_to}) – jen test nástroje / post-hoc, nikdy důkaz")
    print("=" * 100)

    cfg = canon.read_json(os.path.join(a.repo, "config", "config.json"))
    rules = cfg["rules"]
    days, excluded = load(a.repo, d_from, d_to, check_segment=prereg)
    px = A.closes_by_date(pipeline.load_history(a.repo))
    n_scores = sum(1 for d in days if d["scores"])
    print(f"dní v okně: {len(days)}, se skóre LLM: {n_scores}, vyřazeno: {len(excluded)}")
    for D, why in excluded:
        print(f"  vyřazeno {D}: {why}")
    valid = n_scores >= MIN_SCORE_DAYS
    if not valid:
        print(f"NEPLATNÉ: méně než {MIN_SCORE_DAYS} dní se skóre – výsledek je neprůkazný kvůli datům, ne kvůli edge")

    res = {"window": [d_from, d_to], "preregistered": prereg, "days": len(days), "score_days": n_scores,
           "valid": valid, "excluded": excluded}

    # ---- primary
    ic_comp = daily_ic_pairs(days, px, rules, "composite", 1)
    ic_mom = daily_ic_pairs(days, px, rules, "rel30", 1)
    h1 = t_test([v for _, v in ic_comp])
    h2 = t_test(diff_series(ic_comp, ic_mom))
    h3 = t_test(ret_diff(days, "zaklad", "mech_momentum", "base"))
    h3s = t_test(ret_diff(days, "zaklad", "mech_momentum", "stress"))
    v1, v2 = verdict(h1), verdict(h2)
    v3 = verdict(h3) if not (verdict(h3) == "POTVRZENO" and (h3s["mean"] or 0) <= 0) else "nepotvrzeno (stres)"
    print("\nPRIMÁRNÍ HYPOTÉZY (každá α = 0,05, dvoustranně, + shoda znaménka v obou polovinách)")
    print(f"  H1 IC composite 1 d              {fmt(h1)}  → {v1}")
    print(f"  H2 IC composite − IC rel30 1 d   {fmt(h2)}  → {v2}")
    print(f"  H3 zaklad − mech_momentum denně  {fmt(h3, 1e4, ' bps')}  → {v3}")
    print(f"     (stres)                       {fmt(h3s, 1e4, ' bps')}")
    llm_adds = valid and v1 == "POTVRZENO" and v2 == "POTVRZENO"
    print(f"\n  ZÁVĚR: LLM {'PŘIDÁVÁ informaci nad momentum' if llm_adds else 'neprokázal přidanou hodnotu nad momentum'}"
          f"{'; projevilo se to i v penězích (H3)' if llm_adds and v3 == 'POTVRZENO' else ''}")
    res["primary"] = {"H1": {**h1, "verdict": v1}, "H2": {**h2, "verdict": v2}, "H3": {**h3, "verdict": v3, "stress": h3s},
                      "llm_adds_information": llm_adds}

    # ---- secondary, confirmatory family A: 7-day horizon (non-overlapping 14-day blocks)
    print("\nSEKUNDÁRNÍ A – horizont 7 dní (bloky 14 dní, Holm přes 2 testy)")
    c7 = daily_ic_pairs(days, px, rules, "composite", 7)
    m7 = daily_ic_pairs(days, px, rules, "rel30", 7)
    sa = {"IC composite 7 d": t_test(block_means([v for _, v in c7], 14)),
          "IC composite − rel30 7 d": t_test(block_means(diff_series(c7, m7), 14))}
    adj = holm({k: r["p"] for k, r in sa.items()})
    for k, r in sa.items():
        print(f"  {k:<28} {fmt(r)}  p_holm={adj.get(k, float('nan')):.4f}")
    res["secondary_7d"] = {k: {**r, "p_holm": adj.get(k)} for k, r in sa.items()}

    # ---- secondary, confirmatory family B: individual lenses at 1 day
    print("\nSEKUNDÁRNÍ B – jednotlivé pohledy, IC 1 d (Holm přes 6 testů)")
    sb = {lens: t_test([v for _, v in daily_ic_pairs(days, px, rules, lens, 1)]) for lens in SECONDARY_LENSES}
    adj = holm({k: r["p"] for k, r in sb.items()})
    for k, r in sb.items():
        sig = adj.get(k) is not None and adj[k] < ALPHA and r["halves_agree"]
        print(f"  {k:<28} {fmt(r)}  p_holm={adj.get(k, float('nan')):.4f}  {'EDGE' if sig else ''}")
    res["secondary_lenses"] = {k: {**r, "p_holm": adj.get(k)} for k, r in sb.items()}

    # ---- secondary, family C: variants vs equal-weight universe (separates selection from alt beta), Holm over 16
    print("\nSEKUNDÁRNÍ C – varianty vs rovné váhy univerza (b_ew20), denní rozdíl, Holm přes 16")
    sc = {v: t_test(ret_diff(days, v, "b_ew20", "base")) for v in DAILY_VARIANTS}
    adj = holm({k: r["p"] for k, r in sc.items()})
    for k, r in sc.items():
        sig = adj.get(k) is not None and adj[k] < ALPHA and r["mean"] > 0 and r["halves_agree"]
        print(f"  {k:<18} {fmt(r, 1e4, ' bps')}  p_holm={adj.get(k, float('nan')):.4f}  {'EDGE' if sig else ''}")
    res["secondary_variants"] = {k: {**r, "p_holm": adj.get(k)} for k, r in sc.items()}

    # ---- calibration (descriptive)
    sd = [(d["date"], d["asof"], d["scores"]) for d in days if d["scores"]]
    c = A.calibration(sd, px, "p_outperform_btc_7d")
    if c["n"]:
        print(f"\nKALIBRACE p_outperform_btc_7d (popisně): n={c['n']}, Brier {c['brier']:.4f} vs klimatologie {c['brier_climatology']:.4f}"
              f" → {'lepší než konstanta' if c['brier'] < c['brier_climatology'] else 'není lepší než konstanta'}")
        res["calibration"] = {k: c[k] for k in ("n", "brier", "brier_climatology", "base_rate")}

    print("\nHodinové signály a varianty jsou explorativní: tools/analyze.py --hourly, BH FDR 10 %, nikdy důkaz.")
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)
        print(f"uloženo: {a.json}")


if __name__ == "__main__":
    main()
