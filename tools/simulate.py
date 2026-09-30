#!/usr/bin/env python3
"""Synthetic market through the REAL engine (runner.execute_run), for tests and dashboard sample data.

  python tools/simulate.py [--days 60] [--seed 7] [--out DIR]     -> DIR/repo with runs/, data/, public/

The synthetic market is a BTC factor model (alts = beta * BTC + idiosyncratic noise), with 5-minute paths
generated deterministically per (coin, day). The fake LLM plants a small edge in `news` and `trend`
(it peeks at the simulated future, which is the point: the analytics must be able to find it) and none in `mr`.
Day 25 is skipped (tests catch-up), day 12 has a failing LLM step, one coin is delisted on day 40.
"""
import argparse
import math
import os
import random
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

from cpb import canon, llm, runner  # noqa: E402
from cpb.http import FetchError  # noqa: E402

COINS = ["BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "DOGE", "ADA", "LINK", "HYPE", "XLM", "BCH", "SUI", "AVAX", "LTC",
         "HBAR", "NEAR", "UNI", "ZEC", "TAO", "DOT", "AAVE"]
STABLES = ["USDT", "USDC", "WBTC", "STETH"]
DAY_MS = 86_400_000
BAR = 300_000
N_BARS = 288


class SimClock:
    simulated = True

    def __init__(self, ms):
        self.ms = ms

    def now_ms(self):
        self.ms += 1500          # every call advances time a little (keeps lock < snapshot strictly ordered)
        return self.ms

    def set(self, ms):
        self.ms = ms


class World:
    """Deterministic synthetic market from start_date - 200 days to start_date + days + 10."""

    def __init__(self, start_date, days, seed=7):
        self.seed = seed
        self.t0 = canon.add_days(start_date, -200)
        self.dates = canon.date_range(self.t0, canon.add_days(start_date, days + 10))
        rng = random.Random(seed)
        self.beta = {c: (1.0 if c == "BTC" else rng.uniform(0.7, 1.6)) for c in COINS}
        self.idvol = {c: (0.0 if c == "BTC" else rng.uniform(0.02, 0.05)) for c in COINS}
        self.p0 = {c: 10 ** rng.uniform(-1, 4) if c != "BTC" else 60000.0 for c in COINS}
        self.mcap0 = {c: 1e12 / (i + 1) ** 1.3 for i, c in enumerate(COINS)}
        self.vol_usd = {c: 2e9 / (i + 1) ** 0.9 for i, c in enumerate(COINS)}
        self.delist = {}                          # coin -> first date without trading
        self._paths = {}
        self.ret = {}
        btc = [rng.gauss(0.0004, 0.03) for _ in self.dates]
        for c in COINS:
            r = []
            for i, d in enumerate(self.dates):
                r.append(btc[i] if c == "BTC" else self.beta[c] * btc[i] + rng.gauss(0, self.idvol[c]))
            self.ret[c] = r
        self.idx = {d: i for i, d in enumerate(self.dates)}
        self.opens = {}
        for c in COINS:
            px, o = self.p0[c], {}
            for i, d in enumerate(self.dates):
                o[d] = px
                px = px * math.exp(self.ret[c][i])
            self.opens[c] = o

    def path(self, c, d):
        """288 five-minute bars of day d: Brownian bridge from open(d) to open(d+1)."""
        key = (c, d)
        if key not in self._paths:
            self._paths[key] = self._path(c, d)
        return self._paths[key]

    def _path(self, c, d):
        i = self.idx[d]
        o = self.opens[c][d]
        cl = o * math.exp(self.ret[c][i])
        rng = random.Random(f"{self.seed}|{c}|{d}")
        sig = (0.03 * self.beta[c] + self.idvol[c]) / math.sqrt(N_BARS)
        w = [0.0]
        for _ in range(N_BARS):
            w.append(w[-1] + rng.gauss(0, sig))
        lo, lc = math.log(o), math.log(cl)
        pts = [math.exp(lo + w[k] - k / N_BARS * w[-1] + k / N_BARS * (lc - lo)) for k in range(N_BARS + 1)]
        start = canon.date_ms(d)
        bars = []
        for k in range(N_BARS):
            a, b = pts[k], pts[k + 1]
            wig = abs(rng.gauss(0, sig)) * 0.5
            bars.append([start + k * BAR, a, max(a, b) * (1 + wig), min(a, b) * (1 - wig), b])
        return bars

    def daily(self, c, d):
        b = self.path(c, d)
        v = self.vol_usd[c] * math.exp(random.Random(f"v|{c}|{d}").gauss(0, 0.3))
        return [d, b[0][1], max(x[2] for x in b), min(x[3] for x in b), b[-1][4], v / b[-1][4], v]

    def price_at(self, c, ms):
        d = canon.ms_date(ms)
        k = min(int((ms - canon.date_ms(d)) // BAR), N_BARS - 1)
        return self.path(c, d)[k][1]

    def fwd_rel(self, c, d, h=7):
        i = self.idx[d]
        rc = sum(self.ret[c][i:i + h])
        rb = sum(self.ret["BTC"][i:i + h])
        return rc - rb


class SimMarket:
    """Implements the LiveMarket interface over the World; records synthetic 'raw' bodies like the real one."""
    simulated = True

    def __init__(self, world, rec, clock):
        self.w, self.rec, self.clock = world, rec, clock

    def _raw(self, name, kind, body, meta=None):
        import json
        data = json.dumps(body).encode()
        seq = len(self.rec.entries) + 1
        fn = f"{seq:04d}_{name}.json"
        canon.write_bytes(os.path.join(self.rec.raw_dir, fn), data)
        self.rec.entries.append({"seq": seq, "file": fn, "url": f"sim://{name}", "kind": kind, "meta": meta or {},
                                 "fetched_at": canon.ms_iso(self.clock.now_ms()), "status": 200, "error": None,
                                 "sha256": canon.sha256_bytes(data), "bytes": len(data), "gzip": False})

    def _trading(self, c, day):
        dl = self.w.delist.get(c)
        return dl is None or day < dl

    def exchange_status(self, pairs=None):
        day = canon.ms_date(self.clock.ms)
        st = {c + "USDT": ("TRADING" if self._trading(c, day) else "BREAK") for c in COINS}
        if pairs:
            st = {p: st.get(p, "MISSING") for p in pairs}
        self._raw("exchangeinfo", "exchange_info", st)
        return st

    def binance_quote_volume_24h(self, pairs):
        out = {p: self.w.vol_usd[p[:-4]] for p in pairs if p[:-4] in COINS}
        self._raw("24hr", "volume_24h", out)
        return out

    def market_caps(self, n):
        rows = [{"id": c.lower(), "symbol": c, "name": c, "market_cap": self.w.mcap0[c], "rank": i + 1} for i, c in enumerate(COINS)]
        rows = rows[:3] + [{"id": s.lower(), "symbol": s, "name": s + (" Wrapped" if s == "WBTC" else ""), "market_cap": 5e10, "rank": 3}
                           for s in STABLES] + rows[3:]
        self._raw("markets", "market_caps", rows)
        return rows[:n], "coingecko"

    def daily_candles(self, coin, pair, start, end):
        rows = [self.w.daily(coin, d) for d in canon.date_range(max(start, self.w.t0), end)
                if self._trading(coin, canon.add_days(d, 1))]
        if not rows:
            raise FetchError(f"{coin}: no rows")
        self._raw(f"1d_{coin}", "daily", rows, {"coin": coin})
        return rows, "binance"

    def intraday(self, coin, pair, start_ms, end_ms, interval="5m"):
        out = []
        for d in canon.date_range(canon.ms_date(start_ms), canon.ms_date(end_ms - 1)):
            out += [b for b in self.w.path(coin, d) if start_ms <= b[0] and b[0] + BAR <= end_ms]
        self._raw(f"5m_{coin}", "intraday", out, {"coin": coin})
        return out, "binance"

    def ticker_prices(self, pairs):
        t = self.clock.now_ms()
        out = {p: self.w.price_at(p[:-4], t) for p in pairs}
        self._raw("snapshot", "snapshot", out)
        return out, t + 5, "binance"

    def ticker_price_fallback(self, coin):
        return self.w.price_at(coin, self.clock.now_ms()), "okx"

    def second_close(self, coin, date):
        return self.w.daily(coin, date)[4], "okx"

    def okx_close(self, coin, date):
        return self.w.daily(coin, date)[4], "okx"

    def reference_price(self, coin, cg_id):
        return self.w.price_at(coin, self.clock.ms) * 1.0005, "coingecko"


def fake_llm(world, fail_dates=()):
    """Returns a runner compatible with llm.run_claude: writes scores.json into workdir."""
    def run(cfg, workdir, prompt, timeout_s, transcript_name, resume):
        import json
        D = prompt.split("Date of this run: ")[1][:10] if "Date of this run: " in prompt else None
        meta = {"model_init": cfg["llm"]["model"], "models_used": [cfg["llm"]["model"]], "session_id": "sim-session",
                "exit_code": 0, "timed_out": False, "wall_s": 1.0, "web_search_requests": 12, "web_fetch_requests": 3,
                "result": {"num_turns": 20}, "stderr_tail": ""}
        canon.write_text(os.path.join(workdir, transcript_name), '{"type":"system","subtype":"init","model":"sim"}\n')
        if resume:
            return meta
        if D in fail_dates:
            meta.update(exit_code=1, result={"is_error": True})
            return meta
        feats_rows = {}
        for ln in open(os.path.join(workdir, "features.md"), encoding="utf-8"):
            parts = [p.strip() for p in ln.strip().strip("|").split("|")]
            if len(parts) > 10 and parts[0] in COINS:
                feats_rows[parts[0]] = parts
        rng = random.Random(f"llm|{D}")
        coins = {}
        for c in sorted(feats_rows):
            fr = world.fwd_rel(c, D)
            rel30 = feats_rows[c][7]
            rel30 = float(rel30) if rel30 not in ("–", "") else 0.0
            trend = max(-2, min(2, round(rel30 / 10 + (1.0 if fr > 0.02 else -1.0 if fr < -0.02 else 0) * (rng.random() < 0.3) + rng.gauss(0, 0.6))))
            news = max(-2, min(2, round((1.5 if fr > 0 else -1.5) * (rng.random() < 0.35) + rng.gauss(0, 0.7))))
            mr = max(-2, min(2, round(rng.gauss(0, 0.9))))
            conv = max(1, min(5, round(3 + 0.5 * trend + 0.5 * news + rng.gauss(0, 0.5))))
            p = min(0.95, max(0.05, 0.5 + 0.08 * news + 0.05 * trend + rng.gauss(0, 0.05)))
            if c == "BTC":
                p = 0.5
            coins[c] = {"trend": trend, "mr": mr, "news": news, "conviction": conv, "p_outperform_btc_7d": round(p, 3),
                        "p_up_7d": round(min(0.95, max(0.05, p + rng.gauss(0, 0.05))), 3), "expected_move_7d_pct": round(10 * (p - 0.5) * 4, 2),
                        "event": rng.random() < 0.1, "event_type": "none", "risk_flag": rng.random() < 0.02,
                        "note": f"Syntetická poznámka k {c}.", "sources": ["https://example.com/sim"]}
        comp = {c: s["trend"] + s["news"] + 0.5 * s["mr"] + s["conviction"] - 3 for c, s in coins.items()}
        top = sorted(coins, key=lambda c: (-comp[c], c))[:10]
        free = [{"coin": c, "weight_pct": 15, "reason": "Nejvyšší složené skóre.", "invalidation": "Pokles pod SMA20."}
                for c in top[:5] if comp[c] > 1]
        out = {"date": D, "regime": {"label": "Simulace", "summary": "Syntetický trh pro test enginu a dashboardu."},
               "events_7d": [], "coins": coins, "top10": [{"coin": c, "reason": "Syntetické pořadí."} for c in top],
               "claude_volne": {"positions": free, "comment": "Simulace."}}
        with open(os.path.join(workdir, "scores.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        return meta
    return run


def make_repo(out):
    repo = os.path.join(out, "repo")
    if os.path.exists(repo):
        shutil.rmtree(repo)
    os.makedirs(repo)
    for d in ("engine", "config", "task", "dashboard"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(repo, d), ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("run_daily.sh", "VERSION"):
        if os.path.exists(os.path.join(ROOT, f)):
            shutil.copy(os.path.join(ROOT, f), os.path.join(repo, f))
    return repo


def run(out, days=60, seed=7, start="2026-06-01", skip=(25,), llm_fail=(12,), delist_day=40, delist_coin="NEAR", report=True):
    repo = make_repo(out)
    cfg = canon.read_json(os.path.join(repo, "config", "config.json"))
    world = World(start, days, seed)
    if delist_day is not None:
        world.delist[delist_coin] = canon.add_days(start, delist_day)
    clock = SimClock(canon.date_ms(start))
    dates = canon.date_range(start, canon.add_days(start, days - 1))
    fail_dates = {dates[i] for i in llm_fail if i < len(dates)}
    version = {"tag": "sim", "code_commit": None, "manifest_sha256": "sim", "pin_ok": True}
    llm_run = fake_llm(world, fail_dates)
    results = []
    for i, D in enumerate(dates):
        if i in skip:
            continue
        clock.set(canon.date_ms(D) + 20 * 60_000)
        factory = lambda rec, d, _c=clock: SimMarket(world, rec, _c)
        status, rec = runner.execute_run(repo, cfg, D, factory, llm_run, clock, version)
        results.append((D, status, rec.get("error")))
    if report:
        from cpb import report as rp
        rp.build(repo, cfg, os.path.join(repo, "public"), label="SIMULACE – syntetická data")
    return repo, results, world


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "sim"))
    a = ap.parse_args()
    repo, res, _ = run(a.out, a.days, a.seed)
    bad = [r for r in res if r[1] not in ("ok", "warning")]
    print(f"{len(res)} běhů, {len(bad)} selhalo; repo: {repo}")
    for r in bad:
        print("  ", r)
    print("dashboard:", os.path.join(repo, "public", "index.html"))
