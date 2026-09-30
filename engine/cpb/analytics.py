"""Evaluation: portfolio metrics, rank IC of every lens with bootstrap CIs, calibration, per-variant null.

Rank IC: per run date D, Spearman correlation across the universe between a lens score (made with data up to
the close of D-1) and the forward h-day return vs BTC from that close. Days are resampled in blocks
(overlapping 7-day windows are autocorrelated), so the CI is honest about the real number of independent days.
"""
import math
import random

from . import canon

LENSES = ("trend", "mr", "news", "conviction", "composite", "p_outperform_btc_7d", "expected_move_7d_pct")
HORIZONS = (1, 3, 7, 14)


# ---------------------------------------------------------------- portfolio metrics

def series_from_ledgers(ledgers):
    """ledgers = [{date, close:{pid:{scn:mark}}, after:{...}|None}] sorted -> {pid: {scn: [(date, equity)]}}."""
    out = {}
    for L in ledgers:
        for pid, scns in L["close"].items():
            for scn, m in scns.items():
                out.setdefault(pid, {}).setdefault(scn, []).append((L["date"], m["equity"]))
    return out


def max_drawdown(eq):
    peak, mdd = -1, 0.0
    for x in eq:
        peak = max(peak, x)
        mdd = min(mdd, x / peak - 1)
    return mdd


def metrics(ledgers, start_capital, btc_id="b_btc", btc_ret=None):
    S = series_from_ledgers(ledgers)
    last = ledgers[-1]
    if btc_ret is None:
        btc = S.get(btc_id, {}).get("base", [])
        btc_ret = btc[-1][1] / start_capital - 1 if btc else None
    out = {}
    for pid, scns in S.items():
        eq = [e for _, e in scns["base"]]
        cur = (last.get("after") or last["close"])[pid]
        final = {s: cur[s]["equity"] for s in cur}
        rets = [b / a - 1 for a, b in zip(eq, eq[1:]) if a > 0]
        mu = sum(rets) / len(rets) if rets else 0
        sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)) if len(rets) > 1 else 0
        exps = [L["after"][pid]["base"]["exposure"] for L in ledgers if L.get("after") and pid in L["after"]]
        avg_eq = sum(eq) / len(eq) if eq else start_capital
        b = cur["base"]
        out[pid] = {
            "ret": final["base"] / start_capital - 1,
            "ret_gross": final["gross"] / start_capital - 1,
            "ret_stress": final["stress"] / start_capital - 1,
            "vs_btc": (final["base"] / start_capital - 1) - btc_ret if btc_ret is not None else None,
            "max_dd": max_drawdown(eq + [final["base"]]),
            "vol": sd * math.sqrt(365) if sd and len(rets) >= 10 else None,
            "sharpe": mu / sd * math.sqrt(365) if sd and len(rets) >= 10 else None,     # annualized from < 10 days is noise
            "avg_exposure": sum(exps) / len(exps) if exps else 0.0,
            "turnover": b["turnover"] / avg_eq if avg_eq else 0.0,
            "costs_usd": b["fees"] + b["slippage"],
            "n_trades": b["n_trades"],
            "days": len(eq),
        }
    return out


# ---------------------------------------------------------------- forward returns and IC

def closes_by_date(hist):
    return {c: {r[0]: r[4] for r in rows} for c, rows in hist.items()}


def fwd_rel(px, coin, asof, h, btc="BTC"):
    """Forward h-day return of coin from close(asof), minus BTC's; None if not yet known."""
    end = canon.add_days(asof, h)
    a, b = px.get(coin, {}).get(asof), px.get(coin, {}).get(end)
    ba, bb = px.get(btc, {}).get(asof), px.get(btc, {}).get(end)
    if None in (a, b, ba, bb):
        return None, None
    return b / a - 1, (b / a - 1) - (bb / ba - 1)


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


def daily_ic(score_days, px, lens, h, rules):
    """[(date, ic)] for dates whose forward window is complete. score_days = [(date, asof, scores)]."""
    from .portfolio import composite
    out = []
    for D, asof, sc in score_days:
        xs, ys = [], []
        for c, s in sc["coins"].items():
            v = composite(s, rules) if lens == "composite" else s.get(lens)
            _, rel = fwd_rel(px, c, asof, h)
            if v is None or rel is None or c == "BTC":
                continue
            xs.append(v)
            ys.append(rel)
        if len(xs) >= 6:
            ic = corr(ranks(xs), ranks(ys))
            if ic is not None:
                out.append((D, ic))
    return out


def block_bootstrap_mean(vals, block=7, n=2000, seed=1):
    if len(vals) < 5:
        return None, None
    rng = random.Random(seed)
    L = len(vals)
    b = min(block, max(1, L // 3))
    means = []
    for _ in range(n):
        s = []
        while len(s) < L:
            i = rng.randrange(0, L - b + 1)
            s += vals[i:i + b]
        s = s[:L]
        means.append(sum(s) / L)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def ic_table(score_days, px, rules, horizons=HORIZONS, lenses=LENSES):
    rows = []
    for h in horizons:
        for lens in lenses:
            d = daily_ic(score_days, px, lens, h, rules)
            vals = [v for _, v in d]
            if not vals:
                rows.append({"lens": lens, "h": h, "days": 0})
                continue
            half = len(vals) // 2
            lo, hi = block_bootstrap_mean(vals, block=h)
            m1 = sum(vals[:half]) / half if half else None
            m2 = sum(vals[half:]) / (len(vals) - half) if len(vals) - half else None
            sig = lo is not None and (lo > 0 or hi < 0)
            stable = m1 is not None and m2 is not None and (m1 > 0) == (m2 > 0)
            rows.append({"lens": lens, "h": h, "days": len(vals), "ic": sum(vals) / len(vals), "ci": [lo, hi],
                         "first_half": m1, "second_half": m2, "edge": bool(sig and stable and len(vals) >= 20),
                         "series": [[D, round(v, 4)] for D, v in d]})
    return rows


def calibration(score_days, px, field="p_outperform_btc_7d", h=7, bins=10):
    pts = []
    for D, asof, sc in score_days:
        for c, s in sc["coins"].items():
            if c == "BTC" or s.get(field) is None:
                continue
            ab, rel = fwd_rel(px, c, asof, h)
            if rel is None:
                continue
            y = (rel > 0) if field == "p_outperform_btc_7d" else (ab > 0)
            pts.append((s[field], 1.0 if y else 0.0))
    if not pts:
        return {"field": field, "n": 0}
    brier = sum((p - y) ** 2 for p, y in pts) / len(pts)
    base = sum(y for _, y in pts) / len(pts)
    brier_ref = sum((base - y) ** 2 for _, y in pts) / len(pts)
    rel = []
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        b = [(p, y) for p, y in pts if lo <= p < hi or (k == bins - 1 and p == 1.0)]
        if b:
            rel.append({"bin": [lo, hi], "n": len(b), "p_mean": sum(p for p, _ in b) / len(b), "freq": sum(y for _, y in b) / len(b)})
    return {"field": field, "h": h, "n": len(pts), "brier": brier, "brier_climatology": brier_ref, "base_rate": base, "bins": rel}


# ---------------------------------------------------------------- hourly signals

HOURLY_SIGNALS = ("rel4h", "ret1h", "z1h", "vol_ratio", "taker_ratio", "book_imbalance", "funding", "oi_chg_1h", "ls_ratio", "rsi14h")
HOURLY_HORIZONS = (1, 4, 24)


def hourly_closes(hh):
    """{coin: {close_time_ms: close}} (close time = candle open + 1 h)."""
    return {c: {r[0] + 3_600_000: r[4] for r in rows} for c, rows in hh.items()}


def fwd_rel_h(pxh, coin, t, h, btc="BTC"):
    t2 = t + h * 3_600_000
    a, b = pxh.get(coin, {}).get(t), pxh.get(coin, {}).get(t2)
    ba, bb = pxh.get(btc, {}).get(t), pxh.get(btc, {}).get(t2)
    if None in (a, b, ba, bb):
        return None, None
    return b / a - 1, (b / a - 1) - (bb / ba - 1)


def hourly_ic_table(records, pxh, signals=HOURLY_SIGNALS, horizons=HOURLY_HORIZONS):
    """records = [(date, t_end, features)]. Rank IC per hourly run, averaged per day, day-block bootstrap."""
    rows = []
    for h in horizons:
        for sig in signals:
            per_day = {}
            for D, t, feats in records:
                xs, ys = [], []
                for c, f in feats.items():
                    if not f or c == "BTC" or f.get(sig) is None:
                        continue
                    _, rel = fwd_rel_h(pxh, c, t, h)
                    if rel is None:
                        continue
                    xs.append(f[sig])
                    ys.append(rel)
                if len(xs) >= 6:
                    ic = corr(ranks(xs), ranks(ys))
                    if ic is not None:
                        per_day.setdefault(D, []).append(ic)
            days = sorted(per_day)
            vals = [sum(per_day[d]) / len(per_day[d]) for d in days]
            if not vals:
                rows.append({"signal": sig, "h": h, "days": 0})
                continue
            half = len(vals) // 2
            lo, hi = block_bootstrap_mean(vals, block=max(1, h // 24 + 1))
            m1 = sum(vals[:half]) / half if half else None
            m2 = sum(vals[half:]) / (len(vals) - half)
            sig_ok = lo is not None and (lo > 0 or hi < 0)
            rows.append({"signal": sig, "h": h, "days": len(vals), "runs": sum(len(v) for v in per_day.values()),
                         "ic": sum(vals) / len(vals), "ci": [lo, hi], "first_half": m1, "second_half": m2,
                         "edge": bool(sig_ok and m1 is not None and (m1 > 0) == (m2 > 0) and len(vals) >= 20)})
    return rows
