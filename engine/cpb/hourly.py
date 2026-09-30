"""Hourly run: data collection every hour + 7 hourly strategies (no LLM call; one of them uses the daily LLM picks).

Run for hour H of day D starts at H:02 UTC and uses only candles closed by H:00:
  fetch      closed 1h candles (+ 35 days on the first run), order book summary (top 100 levels), perp futures sentiment
             (funding, open-interest change, long/short ratio); verbatim raw bodies + manifest
  check      per-coin hourly check; bad data of a coin nobody holds (and not BTC) = quarantine for the hour
  phase_h    history update, trailing stops on the new candles, delisting, marking at H:00, hourly features
  plan       target weights of the 7 hourly variants  -> decisions.json (locked BEFORE the snapshot)
  snapshot   best bid/ask + server time (must be after the lock), BTC mid cross-checked against OKX
  execute    ledger.rebalance with the same costs and three scenarios as the daily variants

State lives in data/hourly/state.json and data/hourly/history/<COIN>.json (separate from the daily run, own lock).
Missed hours are not caught up: an hourly strategy never decides in hindsight; standing trailing stops are evaluated
on all candles since the last check, whenever the next run happens.
"""
import fcntl
import glob
from concurrent.futures import ThreadPoolExecutor
import math
import os
import shutil
import traceback

from . import canon, check as chk, ledger as L
from .features import rsi, stdev
from .runner import _write_record
from .http import FetchError, Recorder
from .pipeline import SCENARIOS, FailRun, current_universe

HOUR_MS = 3_600_000


def label_of(D, H):
    return f"{D}T{H:02d}"


def hvariants(cfg):
    return cfg["hourly"]["variants"]


# ============================================================ pure: history and features

def merge_hourly(hh, candles, keep_hours):
    out = dict(hh)
    for c, rows in candles.items():
        m = {r[0]: r for r in out.get(c, [])}
        for r in rows:
            m[r[0]] = list(r)
        out[c] = [m[t] for t in sorted(m)][-keep_hours:]
    return out


def hourly_features(hh, coins, t_end, D, book, derivs, btc):
    """Indicators as of t_end (the close of the last hourly candle)."""
    rows_of = {c: [r for r in hh.get(c, []) if r[0] < t_end] for c in set(coins) | {btc}}
    b = rows_of.get(btc, [])
    bok = bool(b) and b[-1][0] == t_end - HOUR_MS

    def ret(rows, n):
        return rows[-1][4] / rows[-1 - n][4] - 1 if len(rows) > n else None

    out = {}
    day_open_t = canon.date_ms(D)
    for c in sorted(coins):
        rows = rows_of[c]
        if not rows or rows[-1][0] != t_end - HOUR_MS:
            out[c] = None
            continue
        cl = [r[4] for r in rows]
        f = {"close": cl[-1], "open_last": rows[-1][1]}
        for n in (1, 4, 24):
            f[f"ret{n}h"] = ret(rows, n)
        f["btc_ret1h"] = ret(b, 1) if bok else None
        f["rel4h"] = f["ret4h"] - ret(b, 4) if bok and f["ret4h"] is not None and ret(b, 4) is not None else None
        lr = [math.log(cl[i] / cl[i - 1]) for i in range(max(1, len(cl) - 720), len(cl))]
        sd = stdev(lr[:-1]) if len(lr) >= 73 else None
        f["z1h"] = lr[-1] / sd if sd else None
        prev = rows[-25:-1]
        f["vol_ratio"] = rows[-1][5] / (sum(r[5] for r in prev) / len(prev)) if len(prev) == 24 and sum(r[5] for r in prev) > 0 else None
        f["high24_prev"] = max(r[2] for r in prev) if len(prev) == 24 else None
        f["breakout"] = f["high24_prev"] is not None and cl[-1] > f["high24_prev"]
        f["rsi14h"] = rsi(cl[-100:])
        f["taker_ratio"] = rows[-1][6] / rows[-1][5] if rows[-1][6] is not None and rows[-1][5] > 0 else None
        f["vol24_usd"] = sum(r[5] for r in rows[-24:])
        dopen = [r[1] for r in rows if r[0] == day_open_t]
        f["day_open"] = dopen[0] if dopen else None
        bk = (book or {}).get(c) or {}
        f["book_imbalance"], f["spread_bps"] = bk.get("imbalance"), bk.get("spread_bps")
        dv = (derivs or {}).get(c) or {}
        f["funding"], f["oi_chg_1h"], f["ls_ratio"] = dv.get("funding"), dv.get("oi_chg_1h"), dv.get("ls_ratio")
        out[c] = {k: (round(v, 10) if isinstance(v, float) else v) for k, v in f.items()}
    return out


# ============================================================ pure: phases

def held(state):
    return sorted({c for scns in state.get("portfolios", {}).values() for led in scns.values() for c in led["positions"]})


def phase_h(cfg, state, hh, inp):
    """History, trailing stops, delisting, marking at H:00, features. Returns (state, hh, trades, marks, feats)."""
    # hh itself is already canonical (read from disk or built from canonical inputs); only small objects round-trip
    state, inp = canon.roundtrip(state), canon.roundtrip(inp)
    hc = cfg["hourly"]
    lab, t_end = inp["label"], inp["t_end"]
    hh = merge_hourly(hh, inp["candles"], hc["history_hours"])
    if not state.get("portfolios"):
        state = {"portfolios": {v["id"]: {s: dict(L.new_ledger(cfg["start_capital"]), started=lab) for s in SCENARIOS}
                                for v in hvariants(cfg)}, "meta": {v["id"]: {} for v in hvariants(cfg)}, "started": lab}
    trades = []
    vol = {c: sum(r[5] for r in hh.get(c, [])[-24:]) for c in set(held(state)) | set(inp["universe"])}
    stops = {v["id"]: {"trail_pct": v["trail_pct"]} for v in hvariants(cfg) if v.get("trail_pct")}
    bars = {c: [r[:5] for r in hh.get(c, []) if r[0] < t_end] for c in held(state)}
    delisted = sorted(c for c in held(state) if inp["status"].get(c, "TRADING") != "TRADING")
    for vid in sorted(state["portfolios"]):
        for scn in SCENARIOS:
            led = state["portfolios"][vid][scn]
            led["blocked"] = {c: b for c, b in led["blocked"].items() if b >= lab}
            if vid in stops:
                L.apply_stops(led, vid, scn, cfg, stops[vid], bars, vol, inp["date"], lab, trades, "binance 1h")
            for c in delisted:
                last = [r for r in hh.get(c, []) if r[0] < t_end][-1]
                L.force_sell(led, vid, scn, cfg, c, last[4], vol.get(c), inp["date"], last[0] + HOUR_MS, "delisting",
                             cfg["costs"]["delist_extra_bps"], trades, "poslední platná hodinová close")
    closes = {c: [r for r in hh[c] if r[0] < t_end][-1][4] for c in held(state)}
    marks = {vid: {scn: L.mark(state["portfolios"][vid][scn], closes) for scn in SCENARIOS} for vid in sorted(state["portfolios"])}
    feats = canon.roundtrip(hourly_features(hh, sorted(set(inp["universe"]) | set(held(state))), t_end, inp["date"],
                                            inp.get("book"), inp.get("derivs"), cfg["btc"]))
    state["last_label"] = lab
    return canon.roundtrip(state), hh, canon.roundtrip(trades), canon.roundtrip(marks), feats


def _age_h(meta, c, t_end):
    m = meta.get(c)
    return (t_end - m["entry_t"]) / HOUR_MS if m else 1e9


def plan_hourly(cfg, state, feats, inp):
    """-> {vid: {"targets": {coin: w} | None, "keep": [coins], "exits": {coin: reason}, "info": {...}}}"""
    H, t_end = inp["hour"], inp["t_end"]
    tradable = sorted(c for c in inp["universe"] if feats.get(c) and inp["status"].get(c) == "TRADING")
    f = lambda c, k: (feats.get(c) or {}).get(k)
    tie = lambda c: canon.hash_rank(inp["label"], c)
    out = {}
    for v in hvariants(cfg):
        vid, kind = v["id"], v["kind"]
        led = state["portfolios"][vid]["base"]
        meta = state["meta"].get(vid, {})
        hold = sorted(led["positions"])
        blocked = {c for c, b in led["blocked"].items() if b >= inp["label"]}
        pool = [c for c in tradable if c not in blocked]
        keep, exits, targets, info = [], {}, {}, {}
        if kind == "momentum":
            cands = sorted((c for c in pool if (f(c, "rel4h") or 0) > 0), key=lambda c: (-f(c, "rel4h"), tie(c)))[: v["n"]]
            keep = [c for c in hold if _age_h(meta, c, t_end) < v["min_hold_h"] or c in cands]
            slots = max(0, v["n"] - len(keep))
            targets = {c: 1 / v["n"] for c in [c for c in cands if c not in hold][:slots]}
            info["candidates"] = cands
        elif kind == "reversal":
            for c in hold:
                if _age_h(meta, c, t_end) >= v["max_hold_h"]:
                    exits[c] = "max_hold"
                elif f(c, "close") is not None and f(c, "close") > meta.get(c, {}).get("ref_open", float("inf")):
                    exits[c] = "signal"
                else:
                    keep.append(c)
            sig = [c for c in pool if c not in hold and f(c, "z1h") is not None and f(c, "z1h") < v["z"]
                   and (f(c, "btc_ret1h") or 0) > v["btc_floor"]]
            sig = sorted(sig, key=lambda c: (f(c, "z1h"), tie(c)))[: max(0, v["max_positions"] - len(keep))]
            targets = {c: v["weight"] for c in sig}
            info["signals"] = sig
        elif kind == "breakout":
            for c in hold:
                if _age_h(meta, c, t_end) >= v["max_hold_h"]:
                    exits[c] = "max_hold"
                else:
                    keep.append(c)
            sig = [c for c in pool if c not in hold and f(c, "breakout") and (f(c, "vol_ratio") or 0) >= v["vol_mult"]]
            sig = sorted(sig, key=lambda c: (-f(c, "vol_ratio"), tie(c)))[: max(0, v["max_positions"] - len(keep))]
            targets = {c: v["weight"] for c in sig}
            info["signals"] = sig
        elif kind == "book":
            ok = lambda c: (f(c, "book_imbalance") or -1) > v["imbalance"] and (f(c, "taker_ratio") or 0) > v["taker"]
            for c in hold:
                if _age_h(meta, c, t_end) >= v["hold_h"] and not ok(c):
                    exits[c] = "signal"
                else:
                    keep.append(c)
            sig = sorted((c for c in pool if c not in hold and ok(c)), key=lambda c: (-f(c, "book_imbalance"), tie(c)))
            sig = sig[: max(0, v["n"] - len(keep))]
            targets = {c: v["weight"] for c in sig}
            info["signals"] = sig
        elif kind == "funding":
            if H not in v["hours"]:
                out[vid] = {"targets": None, "keep": [], "exits": {}, "info": {"hold": "rebalanc jen v 00, 08, 16 UTC"}}
                continue
            fc = sorted((c for c in pool if f(c, "funding") is not None), key=lambda c: (f(c, "funding"), tie(c)))
            cut = len(fc) - int(len(fc) * v["exclude_top_pct"])
            sel = [c for c in fc[:cut] if f(c, "funding") < v["max_funding"]][: v["n"]]
            targets = {c: 1 / v["n"] for c in sel}
            info["selected"] = sel
        elif kind == "llm_timing":
            T = inp.get("daily_targets")
            if T is None:
                out[vid] = {"targets": None, "keep": [], "exits": {}, "info": {"hold": "denní výběr LLM není k dispozici"}}
                continue
            for c in hold:
                if c in T:
                    targets[c] = T[c]
            for c in sorted(T):
                if c in hold or c not in pool:
                    continue
                r, dopen, cl = f(c, "rsi14h"), f(c, "day_open"), f(c, "close")
                trig = (r is not None and r < v["rsi_max"]) or (dopen and cl <= dopen * (1 - v["dip"])) or H == v["force_hour"]
                if trig:
                    targets[c] = T[c]
            info["waiting"] = sorted(c for c in T if c not in targets)
        elif kind == "session":
            if v["from_h"] <= H < v["to_h"]:
                mc = inp["mcaps"]
                top = sorted((c for c in pool if mc.get(c)), key=lambda c: (-mc[c], c))[: v["n"]]
                targets = {c: 1 / v["n"] for c in top}
            info["in_session"] = v["from_h"] <= H < v["to_h"]
        else:
            raise ValueError(kind)
        targets = {c: w for c, w in sorted(targets.items()) if w >= cfg["rules"]["min_weight"]}
        for c in keep:
            targets.pop(c, None)
        out[vid] = {"targets": targets, "keep": sorted(keep), "exits": dict(sorted(exits.items())), "info": info}
    return canon.roundtrip(out)


def execute_h(cfg, state, hh, decisions, snapshot, inp):
    state, decisions, snapshot = canon.roundtrip(state), canon.roundtrip(decisions), canon.roundtrip(snapshot)
    lab, t_end, D = inp["label"], inp["t_end"], inp["date"]
    px, ts = snapshot["prices"], snapshot["fetched_ms"]
    vol = {c: sum(r[5] for r in hh.get(c, [])[-24:]) for c in px}
    trades = []
    for v in hvariants(cfg):
        vid = v["id"]
        dec = decisions[vid]
        for scn in SCENARIOS:
            led = state["portfolios"][vid][scn]
            targets = dec["targets"]
            if targets is not None:
                keep = set(dec["keep"]) & set(led["positions"])
                targets = {c: w for c, w in targets.items() if c in px and c not in led["blocked"]}
                # a strategy that adds to kept positions: kept coins keep their weight, the rest is sized on equity
                L.rebalance(led, vid, scn, cfg, targets, dec["exits"], px, vol, D, ts, snapshot["source"], trades,
                            snapshot.get("quotes"), keep)
            else:
                L.rebalance(led, vid, scn, cfg, None, dec["exits"], px, vol, D, ts, snapshot["source"], trades, snapshot.get("quotes"))
            for c, why in dec["exits"].items():
                if why == "max_hold":
                    led["blocked"][c] = lab
            if v.get("trail_pct"):
                for p in led["positions"].values():
                    p["stop"] = L.stop_level(p, {"trail_pct": v["trail_pct"]})
        # entry meta (base ledger): entry time and the open of the signal candle (reversal exit rule)
        meta = state["meta"].setdefault(vid, {})
        base = state["portfolios"][vid]["base"]["positions"]
        for c in list(meta):
            if c not in base:
                del meta[c]
        for c in base:
            if c not in meta:
                meta[c] = {"entry_t": t_end, "ref_open": [r for r in hh[c] if r[0] < t_end][-1][1]}
    marks = {vid: {scn: L.mark(state["portfolios"][vid][scn], {c: px[c] for c in held(state)}) for scn in SCENARIOS}
             for vid in sorted(state["portfolios"])}
    return canon.roundtrip(state), canon.roundtrip(trades), canon.roundtrip(marks)


# ============================================================ repo helpers

def hdir(repo):
    return os.path.join(repo, "data", "hourly")


def load_hh(repo, days=36):
    """data/hourly/history/<COIN>/<YYYY-MM-DD>.json (24 rows per day file; closed days never change)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(hdir(repo), "history", "*"))):
        rows = []
        for p in sorted(glob.glob(os.path.join(d, "*.json")))[-days:]:
            rows += canon.read_json(p)
        out[os.path.basename(d)] = rows
    return out


def save_hh(repo, hh, touched):
    """Write only the day files that received new candles (touched = {coin: [open_ms, ...]})."""
    for c, ts in touched.items():
        for day in sorted({canon.ms_date(t) for t in ts}):
            lo = canon.date_ms(day)
            rows = [r for r in hh.get(c, []) if lo <= r[0] < lo + 24 * HOUR_MS]
            canon.write_bytes(os.path.join(hdir(repo), "history", c, f"{day}.json"), (canon.dumps(rows) + "\n").encode())


def load_hstate(repo):
    return canon.read_json(os.path.join(hdir(repo), "state.json"), {})


def latest_daily_targets(repo, D, source):
    """Latest daily decision (<= D) of the `source` variant, as it was locked. (targets, date) or (None, None)."""
    for p in sorted(glob.glob(os.path.join(repo, "runs", "*", "decisions.json")), reverse=True):
        d = os.path.basename(os.path.dirname(p))
        if d > D:
            continue
        dec = canon.read_json(p)
        return dec["portfolios"][source]["targets"], d
    return None, None


def latest_sentiment(repo, D):
    """Derivatives/book summary of the latest successful hourly run on or before day D (for the daily LLM table)."""
    for p in sorted(glob.glob(os.path.join(repo, "runs", "*", "hourly", "[0-9][0-9]", "inputs.json")), reverse=True):
        inp = canon.read_json(p)
        if inp["date"] > D:
            continue
        rows = {}
        for c in inp["universe"]:
            dv, bk = (inp.get("derivs") or {}).get(c) or {}, (inp.get("book") or {}).get(c) or {}
            rows[c] = {"funding": dv.get("funding"), "oi_chg_1h": dv.get("oi_chg_1h"), "ls_ratio": dv.get("ls_ratio"),
                       "imbalance": bk.get("imbalance"), "spread_bps": bk.get("spread_bps")}
        return inp["label"], rows
    return None, {}


def sentiment_md(label, rows):
    if not label:
        return ""
    f = lambda x, m=1, nd=2, suf="": "–" if x is None else f"{x * m:+.{nd}f}{suf}"
    out = ["", f"## Deriváty a kniha (hodinový běh {label.replace('T', ' ')}:02 UTC)", "",
           "funding = poslední sazba perpetual futures za 8 h (0,010 % je výchozí); ΔOI = změna open interest za 1 h; "
           "L/S = poměr long/short účtů; nerovnováha = (bid − ask)/(bid + ask) v top 100 úrovních knihy.", "",
           "| coin | funding % | ΔOI 1h % | L/S | nerovnováha knihy | spread bps |", "|---|---|---|---|---|---|"]
    for c, r in sorted(rows.items()):
        ls = "–" if r["ls_ratio"] is None else f"{r['ls_ratio']:.2f}"
        sp = "–" if r["spread_bps"] is None else f"{r['spread_bps']:.1f}"
        out.append(f"| {c} | {f(r['funding'], 100, 4)} | {f(r['oi_chg_1h'], 100, 2)} | {ls} | {f(r['imbalance'], 1, 2)} | {sp} |")
    return "\n".join(out) + "\n"


class HLock:
    def __init__(self, repo):
        os.makedirs(hdir(repo), exist_ok=True)
        self.f = open(os.path.join(repo, "data", ".hourly.lock"), "w")

    def __enter__(self):
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("hodinový běh už probíhá")
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


# ============================================================ live run

def _workers(hc):
    """Parallel HTTP requests (network latency dominates). Simulations run single-threaded: their clock advances per call."""
    return int(os.environ.get("CPB_FETCH_WORKERS", hc.get("workers", 8)))


def fetch_hour(cfg, market, state, hh, U, D, t_end, warnings):
    hc = cfg["hourly"]
    hold = held(state)
    critical = set(hold) | {cfg["btc"]}
    coins = sorted(set(U["coins"]) | set(hold) | {cfg["btc"]})
    pairs = {c: U["pairs"].get(c, c + cfg["quote"]) for c in coins}
    st_p = market.exchange_status(sorted(pairs.values()))
    status = {c: st_p.get(pairs[c], "MISSING") for c in coins}
    candles, errors = {}, {}

    def confirm(coin, open_ms, close):
        try:
            o, _ = market.second_hourly_close(coin, open_ms)
            return abs(o / close - 1) <= cfg["data"]["move_confirm_tol"]
        except FetchError:
            return False

    def one(c):
        """Fetch + check one coin (runs in a thread pool; results are merged in sorted coin order)."""
        stored = hh.get(c, [])
        start = stored[-1][0] - (hc["overlap_hours"] - 1) * HOUR_MS if stored else t_end - hc["history_hours"] * HOUR_MS
        try:
            rows, src = market.hourly_candles(c, pairs[c], start, t_end)
        except FetchError as e:
            return c, None, [str(e)], []
        w = [f"{c}: hodinová data ze záložního zdroje {src}"] if src != "binance" else []
        e, w2 = chk.check_hourly(c, rows, stored, t_end, cfg, confirm)
        return c, rows, e, w + w2

    live = []
    for c in coins:
        if status[c] != "TRADING":
            warnings.append(f"{c}: pár má stav {status[c]}")
        else:
            live.append(c)
    with ThreadPoolExecutor(max_workers=_workers(hc)) as ex:
        results = list(ex.map(one, live))
    for c, rows, e, w in results:
        warnings += w
        if e:
            errors[c] = e
        else:
            candles[c] = rows
    fatal = [e for c in sorted(errors) if c in critical for e in errors[c]]
    quarantine = {c: errors[c] for c in sorted(errors) if c not in critical}
    for c, e in quarantine.items():
        status[c] = "QUARANTINE"
        warnings.append(f"{c}: hodinová karanténa: {e[0]}")
    if fatal:
        raise FailRun("hodinová kontrola dat selhala: " + "; ".join(fatal[:6]))
    ok = [c for c in coins if status[c] == "TRADING"]

    def extras(c):
        bk = dv = err = None
        try:
            bk = market.book(c, pairs[c], hc["book_levels"])
        except (FetchError, KeyError, ValueError) as e:
            err = f"{c}: kniha nedostupná ({str(e)[:80]})"
        try:
            dv = market.derivatives(c)
        except (FetchError, KeyError, ValueError, IndexError):
            dv = None
        return c, bk, dv, err

    with ThreadPoolExecutor(max_workers=_workers(hc)) as ex:
        res2 = list(ex.map(extras, ok))
    book, derivs = {}, {}
    for c, bk, dv, err in res2:
        book[c], derivs[c] = bk, dv
        if err:
            warnings.append(err)
    miss = [c for c in ok if derivs.get(c) is None]
    if miss:
        warnings.append("bez dat futures: " + ", ".join(miss))
    return {"candles": candles, "status": status, "pairs": pairs, "book": book, "derivs": derivs, "quarantine": quarantine}


def snapshot_h(cfg, market, clock, coins, pairs, locked_ms, required, warnings):
    quotes_p, server_ms, src = market.ticker_prices(sorted({pairs[c] for c in coins}))
    fetched_ms = clock.now_ms()
    quotes = {}
    for c in coins:
        q = quotes_p.get(pairs[c])
        if q and q[0] > 0 and q[1] >= q[0]:
            quotes[c] = [q[0], q[1]]
    miss = sorted(set(required) - set(quotes))
    if miss:
        raise FailRun("snímek bez platné ceny pro držené coiny: " + ", ".join(miss))
    if not (fetched_ms > locked_ms and server_ms > locked_ms):
        raise FailRun(f"lookahead: snímek {canon.ms_iso(fetched_ms)} není po zamčení {canon.ms_iso(locked_ms)}")
    prices = {c: (q[0] + q[1]) / 2 for c, q in quotes.items()}
    xc = {}
    try:
        ref, rsrc = market.okx_ticker_mid(cfg["btc"])
        xc = dict(chk.crosscheck(prices[cfg["btc"]], ref, cfg["data"]["crosscheck_tol"]), other_source=rsrc)
        if not xc["ok"]:
            raise FailRun(f"křížová kontrola snímku BTC: {xc['diff_pct']} % vs {rsrc}")
    except FetchError as e:
        warnings.append(f"křížová kontrola snímku BTC nedostupná ({e})")
    return {"prices": dict(sorted(prices.items())), "quotes": dict(sorted(quotes.items())), "fetched_ms": fetched_ms,
            "fetched_at": canon.ms_iso(fetched_ms), "server_ms": server_ms, "locked_at": canon.ms_iso(locked_ms),
            "source": src, "crosscheck": xc}


def run_hour(repo, cfg, D, H, market_factory, clock, version):
    """One hourly run. Returns (status, record). Failures are recorded, state is unchanged."""
    lab = label_of(D, H)
    base_dir = os.path.join(repo, "runs", D, "hourly")
    out_dir = os.path.join(base_dir, f"{H:02d}")
    if os.path.exists(os.path.join(out_dir, "run.json")):
        return "exists", canon.read_json(os.path.join(out_dir, "run.json"))
    stamp = canon.ms_iso(clock.now_ms())[11:19].replace(":", "")
    work = os.path.join(base_dir, f".work-{H:02d}-{stamp}")
    os.makedirs(work)
    t_end = canon.date_ms(D) + H * HOUR_MS
    warnings, timing = [], {"started_at": canon.ms_iso(clock.now_ms())}
    rec_base = {"type": "hourly", "date": D, "hour": H, "label": lab, "version": version}
    rec = Recorder(os.path.join(work, "raw"), clock)
    checks = {}
    try:
        market = market_factory(rec, D)
        u = current_universe(repo, D)
        if u is None:
            raise FailRun("univerzum zatím neexistuje (čeká se na první denní běh)")
        U = {"coins": [c["coin"] for c in u["coins"]], "pairs": {c["coin"]: c["pair"] for c in u["coins"]},
             "mcaps": {c["coin"]: c["market_cap"] for c in u["coins"]}}
        state, hh = load_hstate(repo), load_hh(repo)
        fm = fetch_hour(cfg, market, state, hh, U, D, t_end, warnings)
        checks = {"quarantine": fm["quarantine"], "status": fm["status"]}
        dt, dsrc = latest_daily_targets(repo, D, next(v["source"] for v in hvariants(cfg) if v["kind"] == "llm_timing"))
        inp = {"date": D, "hour": H, "label": lab, "t_end": t_end, "universe": U["coins"], "mcaps": U["mcaps"],
               "universe_date": u["date"], "candles": fm["candles"], "status": fm["status"], "book": fm["book"],
               "derivs": fm["derivs"], "daily_targets": dt, "daily_targets_date": dsrc}
        state, hh, trades_a, marks_close, feats = phase_h(cfg, state, hh, inp)
        decisions = plan_hourly(cfg, state, feats, inp)
        locked_ms = clock.now_ms()
        timing["locked_at"] = canon.ms_iso(locked_ms)
        canon.write_json(os.path.join(work, "decisions.json"), {"label": lab, "locked_at": canon.ms_iso(locked_ms), "locked_ms": locked_ms,
                                                                "portfolios": decisions})
        tradable = [c for c in U["coins"] if feats.get(c) and fm["status"].get(c) == "TRADING"]
        coins = sorted(set(tradable) | set(held(state)) | {c for d in decisions.values() for c in (d["targets"] or {})})
        snap = snapshot_h(cfg, market, clock, coins, fm["pairs"], locked_ms, held(state), warnings)
        timing["snapshot_at"] = snap["fetched_at"]
        state, trades_c, marks_after = execute_h(cfg, state, hh, decisions, snap, inp)
        inp.update({"locked_ms": locked_ms, "snapshot": {k: snap[k] for k in ("prices", "quotes", "fetched_ms", "source")}})
        for fn, obj in (("inputs.json", inp), ("features.json", feats), ("snapshot.json", snap), ("fills.json", trades_a + trades_c),
                        ("ledger.json", {"label": lab, "close": marks_close, "after": marks_after}),
                        ("raw_manifest.json", rec.manifest())):
            canon.write_json(os.path.join(work, fn), obj)
        shutil.copy(os.path.join(repo, "config", "config.json"), os.path.join(work, "config.json"))
    except Exception as e:
        err = str(e) if isinstance(e, FailRun) else f"{type(e).__name__}: {e}"
        canon.write_json(os.path.join(work, "raw_manifest.json"), rec.manifest())
        fdir = os.path.join(base_dir, f"failed-{H:02d}-{stamp}")
        os.rename(work, fdir)
        r = _write_record(repo, fdir, dict(rec_base, status="failed", error=err, warnings=warnings, checks=checks,
                                           traceback=None if isinstance(e, FailRun) else traceback.format_exc()[-2000:],
                                           timing=dict(timing, finished_at=canon.ms_iso(clock.now_ms()))))
        return "failed", r
    save_hh(repo, hh, {c: [r[0] for r in rows] for c, rows in inp["candles"].items()})
    canon.write_json(os.path.join(hdir(repo), "state.json"), state)
    os.rename(work, out_dir)
    from .runner import _write_record
    n_tr = sum(1 for t in trades_a + trades_c if t["scenario"] == "base")
    r = _write_record(repo, out_dir, dict(rec_base, status="warning" if warnings else "ok", warnings=warnings, checks=checks,
                                          timing=dict(timing, finished_at=canon.ms_iso(clock.now_ms())),
                                          summary={"n_trades_base": n_tr, "equity": {v: m["base"]["equity"] for v, m in marks_after.items()}}))
    return r["status"], r
