#!/usr/bin/env python3
"""How large an edge can the pre-registered IC rule detect? Monte Carlo on SYNTHETIC data only (never reads runs/).

  python tools/power.py [--days 61] [--coins 19] [--sims 400]

Model: every coin has a score s that changes slowly (AR(1), phi = persistence of daily scores) and tomorrow's return
vs BTC is r = b * s + e with fat-tailed noise (Student t, df 3). b is set so the 1-day rank IC equals the target;
7- and 14-day ICs follow from overlapping windows. Two detection rules side by side, each "positive edge found":
  README  cpb.analytics.ic_table: block-bootstrap 95 % CI excludes 0, same sign in both halves, >= 20 days
  PREREG  tools/prereg_eval.py: two-sided t test p < 0.05 (1 d on daily ICs, 7/14 d on non-overlapping 2h-day
          block means), same sign in both halves
Row b = 0 is the false-positive rate (one-sided, nominal 2.5 %).
"""
import argparse
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
from cpb import analytics as A  # noqa: E402
from prereg_eval import block_means, t_test  # noqa: E402


def t3(rng):
    z = rng.gauss(0, 1)
    c = sum(rng.gauss(0, 1) ** 2 for _ in range(3))
    return z / math.sqrt(c / 3) / math.sqrt(3)      # unit variance


def one_sim(rng, days, coins, b, phi, horizons, boot):
    T = days + max(horizons) + 1
    s = [[rng.gauss(0, 1) for _ in range(coins)]]
    for _ in range(T):
        s.append([phi * x + math.sqrt(1 - phi * phi) * rng.gauss(0, 1) for x in s[-1]])
    r = [[b * s[t][i] + t3(rng) for i in range(coins)] for t in range(T)]   # r[t] = return over day t+1
    out = {}
    for h in horizons:
        vals = []
        for t in range(days):                       # evaluation waits until every window is closed
            fwd = [sum(r[t + k][i] for k in range(h)) for i in range(coins)]
            vals.append(A.corr(A.ranks(s[t]), A.ranks(fwd)))
        half = len(vals) // 2
        lo, hi = A.block_bootstrap_mean(vals, block=h, n=boot, seed=rng.randrange(1 << 30))
        m1, m2 = sum(vals[:half]) / half, sum(vals[half:]) / (len(vals) - half)
        edge = (lo > 0 or hi < 0) and (m1 > 0) == (m2 > 0) and len(vals) >= 20
        tt = t_test(vals if h == 1 else block_means(vals, 2 * h))
        pre = tt["p"] is not None and tt["p"] < 0.05 and tt["mean"] > 0 and tt["halves_agree"]
        out[h] = (sum(vals) / len(vals), edge and sum(vals) > 0, pre)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=61, help="rozhodovacích dní v primárním okně")
    ap.add_argument("--coins", type=int, default=19)
    ap.add_argument("--phi", type=float, default=0.9, help="denní autokorelace skóre")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    hz = (1, 7, 14)
    print(f"dní {a.days}, coinů {a.coins}, autokorelace skóre {a.phi}, simulací {a.sims}")
    print(f"{'b':>6} " + " ".join(f"{'IC ' + str(h) + 'd':>8} {'README':>7} {'PREREG':>7}" for h in hz))
    for b in (0.0, 0.03, 0.05, 0.08, 0.12, 0.2):
        res = [one_sim(rng, a.days, a.coins, b, a.phi, hz, a.boot) for _ in range(a.sims)]
        cells = []
        for h in hz:
            ic = sum(x[h][0] for x in res) / len(res)
            pw = sum(x[h][1] for x in res) / len(res)
            pp = sum(x[h][2] for x in res) / len(res)
            cells.append(f"{ic:>+8.3f} {pw * 100:>6.1f}% {pp * 100:>6.1f}%")
        print(f"{b:>6.2f} " + " ".join(cells))


if __name__ == "__main__":
    main()
