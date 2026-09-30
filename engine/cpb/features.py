"""Indicators per coin as of the last closed daily candle (stdlib only, deterministic)."""
import math

from . import canon

FIELDS = ("close", "ret1", "ret3", "ret7", "ret30", "rel7", "rel30", "sma20_dist", "sma50_dist", "sma100_dist",
          "rsi14", "atr14_pct", "vol30", "dist_high30", "dist_low30", "volchg_7_30", "corr30_btc", "beta30_btc",
          "quote_volume")


def _ret(c, n):
    return c[-1] / c[-1 - n] - 1 if len(c) > n else None


def _sma(c, n):
    return sum(c[-n:]) / n if len(c) >= n else None


def rsi(c, n=14):
    if len(c) < n + 1:
        return None
    d = [c[i] - c[i - 1] for i in range(1, len(c))]
    ag = sum(max(x, 0) for x in d[:n]) / n
    al = sum(max(-x, 0) for x in d[:n]) / n
    for x in d[n:]:
        ag = (ag * (n - 1) + max(x, 0)) / n
        al = (al * (n - 1) + max(-x, 0)) / n
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    return 100 - 100 / (1 + ag / al)


def atr(rows, n=14):
    if len(rows) < n + 1:
        return None
    tr = [max(r[2] - r[3], abs(r[2] - p[4]), abs(r[3] - p[4])) for p, r in zip(rows, rows[1:])]
    a = sum(tr[:n]) / n
    for x in tr[n:]:
        a = (a * (n - 1) + x) / n
    return a


def logrets(c):
    return [math.log(c[i] / c[i - 1]) for i in range(1, len(c))]


def stdev(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def cov(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) - 1)


def aligned_returns(hist, coins, asof, n):
    """{coin: [n daily log returns ending asof]} only for coins with full, gap-free data."""
    out = {}
    dates = canon.date_range(canon.add_days(asof, -n), asof)
    for c in coins:
        rows = {r[0]: r[4] for r in hist.get(c, [])}
        if all(d in rows for d in dates):
            px = [rows[d] for d in dates]
            out[c] = logrets(px)
    return out


def compute(hist, coins, asof, btc="BTC"):
    """hist = {coin: [[date,o,h,l,c,vb,vq], ...] sorted}; only rows <= asof are used."""
    rows_of = {c: [r for r in hist.get(c, []) if r[0] <= asof] for c in set(coins) | {btc}}
    b = [r[4] for r in rows_of[btc]]
    br30 = aligned_returns(hist, [btc], asof, 30).get(btc)
    out = {}
    for c in sorted(coins):
        rows = rows_of[c]
        if not rows or rows[-1][0] != asof:
            out[c] = None
            continue
        cl = [r[4] for r in rows]
        f = {"close": cl[-1], "quote_volume": rows[-1][6], "history_days": len(rows)}
        for n in (1, 3, 7, 30):
            f[f"ret{n}"] = _ret(cl, n)
        for n in (7, 30):
            rb = _ret(b, n)
            f[f"rel{n}"] = None if f[f"ret{n}"] is None or rb is None else f[f"ret{n}"] - rb
        for n in (20, 50, 100):
            s = _sma(cl, n)
            f[f"sma{n}"] = s
            f[f"sma{n}_dist"] = None if s is None else cl[-1] / s - 1
        f["rsi14"] = rsi(cl[-120:])
        a = atr(rows[-120:])
        f["atr14_pct"] = None if a is None else a / cl[-1]
        lr = logrets(cl[-31:])
        f["vol30"] = stdev(lr) * math.sqrt(365) if len(lr) == 30 else None
        if len(rows) >= 30:
            f["dist_high30"] = cl[-1] / max(r[2] for r in rows[-30:]) - 1
            f["dist_low30"] = cl[-1] / min(r[3] for r in rows[-30:]) - 1
        else:
            f["dist_high30"] = f["dist_low30"] = None
        qv = [r[6] for r in rows]
        f["volchg_7_30"] = (sum(qv[-7:]) / 7) / (sum(qv[-30:]) / 30) - 1 if len(qv) >= 30 and sum(qv[-30:]) > 0 else None
        cr = aligned_returns(hist, [c], asof, 30).get(c)
        if cr and br30:
            vb = cov(br30, br30)
            sa, sb = stdev(cr), stdev(br30)
            f["beta30_btc"] = cov(cr, br30) / vb if vb else None
            f["corr30_btc"] = cov(cr, br30) / (sa * sb) if sa and sb else None
        else:
            f["beta30_btc"] = f["corr30_btc"] = None
        out[c] = {k: (round(v, 8) if isinstance(v, float) else v) for k, v in f.items()}
    return out


def gate(f, rules):
    """Indicator gate: close > SMA20 and RSI14 < rsi_max (risk_flag is checked by the caller)."""
    if not f or f.get("sma20") is None or f.get("rsi14") is None:
        return False
    return f["close"] > f["sma20"] and f["rsi14"] < rules["gate"]["rsi_max"]


def table_md(feats, universe_coins, asof):
    """Readable table for the LLM (it must read indicators, never compute them)."""
    def p(x, pct=True, nd=1):
        if x is None:
            return "–"
        return f"{x * 100:+.{nd}f}" if pct else f"{x:.{nd}f}"
    lines = [f"# Indikátory k uzavření denní svíčky {asof} (00:00 UTC následujícího dne)", "",
             "Procenta jsou v %. rel = výnos minus výnos BTC. dist = vzdálenost od SMA / 30d max / 30d min. "
             "ATR% a vol30 (roční) v %. volchg = průměrný objem 7 d vs 30 d.", "",
             "| coin | close | ret1 | ret3 | ret7 | ret30 | rel7 | rel30 | SMA20 | SMA50 | SMA100 | RSI14 | ATR% | vol30 | od 30d max | od 30d min | volchg | corr BTC | beta BTC | objem $M |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in universe_coins:
        f = feats.get(c)
        if not f:
            lines.append(f"| {c} | chybí data |" + " |" * 18)
            continue
        lines.append("| " + " | ".join([
            c, f"{f['close']:.6g}", p(f["ret1"]), p(f["ret3"]), p(f["ret7"]), p(f["ret30"]), p(f["rel7"]), p(f["rel30"]),
            p(f["sma20_dist"]), p(f["sma50_dist"]), p(f["sma100_dist"]), p(f["rsi14"], False), p(f["atr14_pct"]),
            p(f["vol30"], True, 0), p(f["dist_high30"]), p(f["dist_low30"]), p(f["volchg_7_30"], True, 0),
            p(f["corr30_btc"], False, 2), p(f["beta30_btc"], False, 2), f"{f['quote_volume'] / 1e6:.0f}"]) + " |")
    return "\n".join(lines) + "\n"
