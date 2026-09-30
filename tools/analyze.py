#!/usr/bin/env python3
"""Do the LLM lenses carry information? Rank IC per lens and horizon with block-bootstrap CIs, stability in the first
vs second half, calibration of the probabilities (Brier, reliability).

  python tools/analyze.py [--repo DIR] [--version vX.Y.Z] [--model MODEL]

Decision rule written in README before the first results: a lens has an edge only when the 95 % CI of its 7-day
rank IC excludes 0 AND the sign is the same in both halves of the experiment. Never mix versions or models:
use --version / --model to select one segment.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
from cpb import analytics as A, canon, pipeline  # noqa: E402


def score_days(repo, version=None, model=None):
    out = []
    for p in sorted(os.listdir(os.path.join(repo, "runs"))):
        rec = canon.read_json(os.path.join(repo, "runs", p, "run.json"))
        if not rec or rec["status"] not in ("ok", "warning"):
            continue
        if version and (rec.get("version") or {}).get("tag") != version:
            continue
        if model and rec.get("llm_model_pinned") != model:
            continue
        inp = canon.read_json(os.path.join(repo, "runs", p, "inputs.json"))
        if inp.get("scores"):
            out.append((inp["date"], inp["asof"], inp["scores"]))
    return out


def f(x, nd=3):
    return "   –  " if x is None else f"{x:+.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=ROOT)
    ap.add_argument("--version")
    ap.add_argument("--model")
    a = ap.parse_args()
    cfg = canon.read_json(os.path.join(a.repo, "config", "config.json"))
    sd = score_days(a.repo, a.version, a.model)
    px = A.closes_by_date(pipeline.load_history(a.repo))
    tags = sorted({(canon.read_json(os.path.join(a.repo, "runs", d, "run.json")).get("version") or {}).get("tag") for d, _, _ in sd})
    print(f"dní se skóre: {len(sd)}; verze: {', '.join(t or '?' for t in tags)}")
    if len(tags) > 1:
        print("POZOR: mícháte více verzí – použijte --version")
    print(f"{'h':>3} {'pohled':<22}{'dní':>5} {'rank IC':>8} {'95% CI':>20} {'1. pol.':>8} {'2. pol.':>8}  verdikt")
    for r in A.ic_table(sd, px, cfg["rules"]):
        if not r["days"]:
            print(f"{r['h']:>3} {r['lens']:<22}{0:>5}   (málo dat)")
            continue
        verdict = "EDGE" if r["edge"] else "nelze odlišit od šumu"
        print(f"{r['h']:>3} {r['lens']:<22}{r['days']:>5} {f(r['ic']):>8} [{f(r['ci'][0])},{f(r['ci'][1])}] {f(r['first_half']):>8} {f(r['second_half']):>8}  {verdict}")
    for fld in ("p_outperform_btc_7d", "p_up_7d"):
        c = A.calibration(sd, px, fld)
        if not c["n"]:
            continue
        print(f"\nKalibrace {fld}: n={c['n']}, Brier {c['brier']:.4f} vs {c['brier_climatology']:.4f} (konstantní základní míra {c['base_rate']:.2f})")
        for b in c["bins"]:
            print(f"  {b['bin'][0]:.1f}–{b['bin'][1]:.1f}: odhad {b['p_mean']:.2f}  skutečnost {b['freq']:.2f}  (n={b['n']})")


if __name__ == "__main__":
    main()
