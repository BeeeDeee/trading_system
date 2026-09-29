#!/usr/bin/env python3
"""Do the lenses carry information? Rank IC and score spreads with day-block bootstrap CIs.

The experiment's real question is not "which variant made most money" (19 correlated variants
over ~40 days is mostly noise) but whether trend / mr / news / conviction predict forward
returns. Every evening gives 20 scores per lens, so this is where the statistical power is.

  python tools/analyze.py export.json            # {"scores": [{date, scores}], "prices": [{date, close}]}
  python tools/analyze.py --demo                 # synthetic data: edge is planted in trend/news/conviction, none in mr (tool sanity check)

Forward return = close(d) -> close(d+h), minus the equal-weighted universe mean of the same window
(removes the market). Slightly optimistic vs. the traded open(d+1); fine for ranking lenses.
Days are resampled as blocks (scores of one day are correlated), so CIs are honest about ~N days.
"""
import json, random, statistics as st, sys

LENSES = ("trend", "mr", "news", "conviction")
HORIZONS = (1, 5)


def fwd_table(scores, prices, h):
    """{date: {ticker: (score_dict, excess_forward_return_pct)}}"""
    px = {p["date"]: p["close"] for p in prices}
    dates = sorted(px)
    out = {}
    for s in scores:
        d = s["date"]
        if d not in px or dates.index(d) + h >= len(dates):
            continue
        d2 = dates[dates.index(d) + h]
        rets = {t: (px[d2][t] / px[d][t] - 1) * 100 for t in s["scores"] if px[d].get(t) and px[d2].get(t)}
        if len(rets) < 5:
            continue
        m = sum(rets.values()) / len(rets)
        out[d] = {t: (s["scores"][t], r - m) for t, r in rets.items()}
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
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5 if va and vb else None


def day_stats(day, lens):
    rows = [(s.get(lens), r) for s, r in day.values() if s.get(lens) is not None]
    if len(rows) < 5:
        return None
    ic = corr(ranks([x for x, _ in rows]), ranks([r for _, r in rows]))
    up = [r for x, r in rows if x >= (1 if lens != "conviction" else 4)]
    dn = [r for x, r in rows if x <= (-1 if lens != "conviction" else 2)]
    return {"ic": ic, "up": up, "dn": dn}


def boot(days, f, n=2000, seed=1):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        v = f([rng.choice(days) for _ in days])
        if v is not None:
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if len(vals) > 20 else (None, None)


def analyse(scores, prices):
    res = []
    for h in HORIZONS:
        tab = fwd_table(scores, prices, h)
        for lens in LENSES:
            days = [x for x in (day_stats(d, lens) for d in tab.values()) if x]
            if len(days) < 5:
                res.append((h, lens, len(days), None, (None, None), None, (None, None), None))
                continue
            f_ic = lambda ds: (lambda v: sum(v) / len(v) if v else None)([d["ic"] for d in ds if d["ic"] is not None])
            def f_sp(ds):
                up = [r for d in ds for r in d["up"]]
                dn = [r for d in ds for r in d["dn"]]
                return sum(up) / len(up) - sum(dn) / len(dn) if up and dn else None
            up = [r for d in days for r in d["up"]]
            hit = sum(r > 0 for r in up) / len(up) * 100 if up else None
            res.append((h, lens, len(days), f_ic(days), boot(days, f_ic), f_sp(days), boot(days, f_sp), hit))
    return res


def fmt(x, nd=3):
    return "  n/a " if x is None else f"{x:+.{nd}f}"


def report(res):
    print(f"{'h':>2} {'lens':<11}{'days':>5} {'rankIC':>8} {'95% CI':>18} {'spread %':>9} {'95% CI':>18} {'hit % up':>9}  verdict")
    for h, lens, n, ic, ci, sp, sci, hit in res:
        if ic is None:
            print(f"{h:>2} {lens:<11}{n:>5}   (málo dat)")
            continue
        sig = ci[0] is not None and (ci[0] > 0 or ci[1] < 0)
        print(f"{h:>2} {lens:<11}{n:>5} {fmt(ic):>8} [{fmt(ci[0])},{fmt(ci[1])}] {fmt(sp, 2):>9} [{fmt(sci[0], 2)},{fmt(sci[1], 2)}] "
              f"{'' if hit is None else f'{hit:.0f}':>9}  {'SIGNÁL' if sig else 'nelze odlišit od šumu'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--demo":
        import pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import simulate
        sim = simulate.run(json.load(open(simulate.ROOT / "config" / "config.json")), days=40, seed=5)
    else:
        sim = json.load(open(sys.argv[1]))
    report(analyse(sim["scores"], sim["prices"]))
