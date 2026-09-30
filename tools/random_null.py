#!/usr/bin/env python3
"""Skill or luck? For each variant its own null distribution computed with the real engine (cpb/null.py):
the same days, number of coins, weight profile, number of swaps, rules (band, stops, max hold) and costs as the
variant, but random coins from that day's tradable universe.

  python tools/random_null.py [--repo DIR] [--paths 2000]

A variant has an edge only above the 95th percentile of its own null AND above BTC HODL AND under stress costs.
With 18 variants expect ~1 above the 95th percentile by chance.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
from cpb import analytics as A, canon, null, pipeline, report  # noqa: E402


def rng(r):
    return f"[{r['p05'] * 100:+.1f}, {r['p95'] * 100:+.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=ROOT)
    ap.add_argument("--paths", type=int, default=2000)
    a = ap.parse_args()
    cfg = canon.read_json(os.path.join(a.repo, "config", "config.json"))
    good = [r for r in report._runs(a.repo) if r["status"] in ("ok", "warning", "catchup")]
    ledgers = [canon.read_json(os.path.join(r["_dir"], "ledger.json")) for r in good]
    hist = pipeline.load_history(a.repo)
    res = null.null_percentiles(null.load_days(a.repo, [(r, r["_dir"]) for r in good]), hist, cfg["variants"], n_paths=a.paths)
    met = A.metrics(ledgers, cfg["start_capital"])
    print(f"{'varianta':<16}{'skutečně':>10}{'nula medián':>13}{'nula 5–95 %':>22}{'percentil':>11}{'vs BTC':>9}{'stres':>9}  edge?")
    for vid, r in res.items():
        m = met[vid]
        edge = r["percentile"] > 95 and (m["vs_btc"] or 0) > 0 and m["ret_stress"] > 0
        print(f"{vid:<16}{r['actual'] * 100:>+9.1f}%{r['p50'] * 100:>+12.1f}%{rng(r):>22}"
              f"{r['percentile']:>10.0f}%{m['vs_btc'] * 100:>+8.1f}%{m['ret_stress'] * 100:>+8.1f}%  {'ANO' if edge else 'ne'}")


if __name__ == "__main__":
    main()
