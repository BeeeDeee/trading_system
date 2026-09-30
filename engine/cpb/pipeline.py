"""The daily run. Pure phases (replayable from stored inputs) + the live orchestration around them.

  phase_a(state, hist, inputs)   history update, stops since last check, delisting, marking at 00:00 UTC, features
  plan_all(...)                  target weights of every portfolio  -> decisions.json (locked BEFORE the snapshot)
  execute(state, decisions, ...) stops up to the snapshot, exits, rebalancing at snapshot prices for 3 cost scenarios

Live run (run_day) = fetch -> check -> phase_a -> claude -p -> validate -> lock -> snapshot -> execute -> write.
Every value that enters the phases is stored in runs/D/inputs.json, so tools/verify_chain.py can replay the whole
ledger byte for byte.
"""
import copy
import glob
import json
import os
import shutil
import time

from . import canon, check as chk, features as fx, ledger as L, llm, portfolio as P, universe as uni
from .http import FetchError, Recorder

DAY_MS = 86_400_000
SCENARIOS = ("base", "stress", "gross")


class FailRun(Exception):
    pass


class Clock:
    simulated = False

    def now_ms(self):
        return int(time.time() * 1000)


# ============================================================ pure phases

def portfolio_defs(cfg):
    return [("variant", v) for v in cfg["variants"]] + [("benchmark", b) for b in cfg["benchmarks"]]


def merge_history(hist, candles):
    out = dict(hist)
    for c, rows in candles.items():
        m = {r[0]: r for r in out.get(c, [])}
        for r in rows:
            m[r[0]] = list(r)
        out[c] = [m[d] for d in sorted(m)]
    return out


def last_close(hist, coin, asof):
    rows = [r for r in hist.get(coin, []) if r[0] <= asof]
    return (rows[-1][4], rows[-1][6], rows[-1][0]) if rows else (None, None, None)


def held_coins(state):
    return sorted({c for scns in state.get("portfolios", {}).values() for led in scns.values() for c in led["positions"]})


def stop_cfgs(cfg):
    return {v["id"]: v["stops"] for v in cfg["variants"] if v.get("stops")}


def phase_a(cfg, state, hist, inp):
    """Returns (state, hist, trades, marks, feats, delisted). Does not mutate its arguments.

    All inputs and outputs pass through canonical JSON, so a live run and a replay from the stored files see
    exactly the same numbers."""
    state, hist, inp = canon.roundtrip(state), canon.roundtrip(hist), canon.roundtrip(inp)
    D, asof = inp["date"], inp["asof"]
    hist = merge_history(hist, inp["candles"])
    if inp["mode"] == "decision" and not state.get("portfolios"):
        state["portfolios"] = {v["id"]: {s: dict(L.new_ledger(cfg["start_capital"]), started=D) for s in SCENARIOS}
                               for _, v in portfolio_defs(cfg)}
    trades = []
    stops = stop_cfgs(cfg)
    vol = {c: last_close(hist, c, asof)[1] for c in set(held_coins(state)) | set(inp["universe"]["coins"])}
    delisted = sorted(c for c in held_coins(state) if inp["status"].get(c, "TRADING") != "TRADING")
    for pid in sorted(state.get("portfolios", {})):
        for scn in SCENARIOS:
            led = state["portfolios"][pid][scn]
            led["blocked"] = {c: d for c, d in led["blocked"].items() if d >= D}
            if pid in stops:
                L.apply_stops(led, pid, scn, cfg, stops[pid], inp["intraday_a"]["bars"], vol, D, D, trades, inp["intraday_a"]["source"])
            for c in delisted:
                px, _, d = last_close(hist, c, asof)
                L.force_sell(led, pid, scn, cfg, c, px, vol.get(c), D, canon.date_ms(d) + DAY_MS, "delisting",
                             cfg["costs"]["delist_extra_bps"], trades, f"poslední platná close {d}")
    closes = {c: last_close(hist, c, asof)[0] for c in held_coins(state)}
    marks = {pid: {scn: L.mark(state["portfolios"][pid][scn], closes) for scn in SCENARIOS} for pid in sorted(state.get("portfolios", {}))}
    feat_coins = sorted(set(inp["universe"]["coins"]) | set(held_coins(state)) | {cfg["btc"]})
    feats = canon.roundtrip(fx.compute(hist, feat_coins, asof, cfg["btc"]))
    state["last_date"] = D
    return canon.roundtrip(state), hist, canon.roundtrip(trades), canon.roundtrip(marks), feats, delisted


def plan_all(cfg, state, hist, feats, scores, universe, D, first_run, status):
    scores = canon.roundtrip(scores)
    feats = {c: (f if status.get(c, "TRADING") == "TRADING" else None) for c, f in feats.items()}
    base = {pid: scns["base"] for pid, scns in state["portfolios"].items()}
    mcaps = {c["coin"]: c["market_cap"] for c in universe["coins_full"]}
    ctx = P.Ctx(cfg, D, universe["coins"], feats, scores, hist, base, first_run, mcaps)
    out = {}
    for kind, d in portfolio_defs(cfg):
        pl = P.variant_plan(d, ctx) if kind == "variant" else P.benchmark_plan(d, ctx)
        out[d["id"]] = {"kind": kind, "targets": pl["targets"], "exits": pl["exits"], "info": pl["info"]}
    return canon.roundtrip(out)


def execute(cfg, state, hist, decisions, snapshot, intraday_c, asof, D):
    state, decisions, snapshot, intraday_c = (canon.roundtrip(x) for x in (state, decisions, snapshot, intraday_c))
    trades = []
    stops = stop_cfgs(cfg)
    px = snapshot["prices"]
    ts = snapshot["fetched_ms"]
    src = snapshot["source"]
    vol = {c: last_close(hist, c, asof)[1] for c in px}
    for pid in sorted(decisions):
        dec = decisions[pid]
        for scn in SCENARIOS:
            led = state["portfolios"][pid][scn]
            before = set(led["positions"])
            if pid in stops:
                L.apply_stops(led, pid, scn, cfg, stops[pid], intraday_c["bars"], vol, D, D, trades, intraday_c["source"])
            stopped = before - set(led["positions"])
            targets = dec["targets"]
            if targets is not None:
                targets = {c: w for c, w in targets.items() if c not in stopped and c not in led["blocked"]}
            L.rebalance(led, pid, scn, cfg, targets, dec["exits"], px, vol, D, ts, src, trades)
            for c, reason in dec["exits"].items():
                if reason == "max_hold":
                    led["blocked"][c] = D
            if dec["targets"] is not None and dec["kind"] == "benchmark":
                led["last_rebalance"] = D
            if pid in stops:
                for p in led["positions"].values():
                    p["stop"] = L.stop_level(p, stops[pid])
    marks = {pid: {scn: L.mark(state["portfolios"][pid][scn], {c: px[c] for c in held_coins(state)}) for scn in SCENARIOS}
             for pid in sorted(state["portfolios"])}
    state["last_fill_ms"] = ts
    return canon.roundtrip(state), canon.roundtrip(trades), canon.roundtrip(marks)


def needed_snapshot_coins(state, decisions):
    s = set(held_coins(state))
    for d in decisions.values():
        s |= set((d["targets"] or {}).keys())
    return sorted(s)


def intraday_window_start(cfg, state):
    stops = stop_cfgs(cfg)
    starts = {}
    for pid in stops:
        for led in state.get("portfolios", {}).get(pid, {}).values():
            for c, p in led["positions"].items():
                starts[c] = min(starts.get(c, p["stop_checked_ms"]), p["stop_checked_ms"])
    return starts


# ============================================================ repo helpers

def load_history(repo):
    h = {}
    for f in sorted(glob.glob(os.path.join(repo, "data", "history", "*.json"))):
        h[os.path.basename(f)[:-5]] = canon.read_json(f)
    return h


def save_history(repo, hist):
    for c, rows in hist.items():
        canon.write_json(os.path.join(repo, "data", "history", f"{c}.json"), rows)


def load_state(repo):
    return canon.read_json(os.path.join(repo, "data", "state.json"), {"portfolios": {}, "last_date": None, "last_fill_ms": None})


def current_universe(repo, D):
    files = sorted(f for f in glob.glob(os.path.join(repo, "universe", "*.json")) if os.path.basename(f)[:-5] <= D)
    return canon.read_json(files[-1]) if files else None


def universe_view(u):
    return {"date": u["date"], "coins": [c["coin"] for c in u["coins"]], "coins_full": u["coins"],
            "pairs": {c["coin"]: c["pair"] for c in u["coins"]}}


# ============================================================ live run

class RunCtx:
    def __init__(self, repo, cfg, D, market, llm_runner, clock, version, work, dry_run=False):
        self.repo, self.cfg, self.D, self.market, self.llm_runner, self.clock = repo, cfg, D, market, llm_runner, clock
        self.version, self.work, self.dry_run = version, work, dry_run
        self.asof = canon.add_days(D, -1)
        self.warnings, self.checks, self.timing, self.sources = [], {}, {}, {}
        self.llm_meta = None
        self.catchup_days = []


def pair_of(cfg, u, coin, extra_pairs):
    return u["pairs"].get(coin) or extra_pairs.get(coin) or coin + cfg["quote"]


def fetch_market(rc, state, hist, u, mode):
    """Fetch, check and cross-check daily data. Returns inputs for phase_a."""
    cfg, D, asof = rc.cfg, rc.D, rc.asof
    held = held_coins(state)
    coins = sorted(set(u["coins"]) | set(held) | {cfg["btc"], "ETH"})
    pairs = {c: pair_of(cfg, u, c, state.get("pairs", {})) for c in coins}
    status_pairs = rc.market.exchange_status(sorted(pairs.values()))
    status = {c: status_pairs.get(pairs[c], "MISSING") for c in coins}
    candles, sources, errors, warnings, suspects = {}, {}, [], [], []
    confirm_cache = {}

    def confirm(coin, date, close):
        """A > max_daily_move day counts only when a second exchange shows the same close (within move_confirm_tol)."""
        try:
            o, src = rc.market.second_close(coin, date)
        except FetchError as e:
            confirm_cache[(coin, date)] = str(e)
            return False
        confirm_cache[(coin, date)] = {"other": o, "source": src}
        return abs(o / close - 1) <= cfg["data"]["move_confirm_tol"]

    for c in coins:
        stored = hist.get(c, [])
        if stored:
            start = min(canon.add_days(asof, -cfg["data"]["overlap_days"]), canon.add_days(stored[-1][0], 1))
        else:
            start = canon.add_days(asof, -cfg["data"]["history_days"] + 1)
        start = max(start, state.get("fetch_from", {}).get(c, start))   # after a redenomination: new units only
        if status[c] != "TRADING":
            warnings.append(f"{c}: pár {pairs[c]} má stav {status[c]}" +
                            (" → nucený prodej za poslední platnou cenu" if c in held else " → dnes se neobchoduje"))
            continue
        try:
            rows, src = rc.market.daily_candles(c, pairs[c], start, asof)
        except FetchError as e:
            errors.append(str(e))
            continue
        sources[c] = src
        if src != "binance":
            warnings.append(f"{c}: denní data ze záložního zdroje {src}")
        e, w, s = chk.check_coin(c, rows, stored, asof, cfg, confirm)
        errors += e
        warnings += w
        if s:
            suspects.append(s)
        candles[c] = rows
    # cross-check closes: BTC + one rotating universe coin vs OKX
    others = [c for c in u["coins"] if c != cfg["btc"]]
    rot = others[canon.days_between("2026-01-01", D) % len(others)] if others else None
    xc = {}
    for c in [cfg["btc"], rot]:
        if not c or c not in candles:
            continue
        mine = [r for r in candles[c] if r[0] == asof]
        if not mine:
            continue
        res = None
        for attempt in range(2):
            try:
                o, osrc = rc.market.okx_close(c, asof)
            except FetchError as e:
                res = {"error": str(e)}
                break
            res = dict(chk.crosscheck(mine[0][4], o, cfg["data"]["crosscheck_tol"]), other_source=osrc)
            if res["ok"]:
                break
            if attempt == 0:   # re-fetch primary once
                rows, _ = rc.market.daily_candles(c, pairs[c], asof, asof)
                mine = rows
        xc[c] = res
        if res and "error" in res:
            warnings.append(f"křížová kontrola {c}: druhý zdroj nedostupný ({res['error']})")
        elif res and not res["ok"]:
            errors.append(f"křížová kontrola close {c}: rozdíl {res['diff_pct']} % proti OKX")
    rc.checks.update({"errors": errors, "warnings": warnings, "redenomination_suspects": suspects,
                      "close_crosscheck": xc, "sources": sources, "status": status,
                      "big_move_confirmations": {f"{c} {d}": v for (c, d), v in sorted(confirm_cache.items())}})
    rc.warnings += warnings
    if errors:
        raise FailRun("kontrola dat selhala: " + "; ".join(errors[:8]))
    return {"candles": candles, "status": status, "pairs": pairs}


def fetch_intraday(rc, starts, end_ms, pairs):
    bars, srcs = {}, set()
    for c in sorted(starts):
        if starts[c] >= end_ms:
            continue
        try:
            b, src = rc.market.intraday(c, pairs.get(c, c + rc.cfg["quote"]), starts[c] - starts[c] % 300_000, end_ms,
                                        rc.cfg["data"]["intraday_interval"])
        except FetchError as e:
            raise FailRun(f"intraday data pro stopy: {e}")
        bars[c] = b
        srcs.add(src)
    return {"bars": bars, "source": ",".join(sorted(srcs)) or "binance", "end_ms": end_ms}


def build_universe(rc):
    cfg = rc.cfg
    caps, src = rc.market.market_caps(60)
    cand = caps[: cfg["universe"]["candidates"]]
    pairs = sorted({cfg["universe"].get("symbol_overrides", {}).get(c["symbol"], c["symbol"]) + cfg["quote"] for c in cand})
    status = rc.market.exchange_status(None)
    live = [p for p in pairs if status.get(p) == "TRADING"]
    vol = rc.market.binance_quote_volume_24h(live) if live else {}
    u = uni.build(cfg, rc.D, caps, src, status, vol)
    if len(u["coins"]) < cfg["universe"]["size"]:
        rc.warnings.append(f"univerzum má jen {len(u['coins'])} coinů")
    if not any(c["coin"] == cfg["btc"] for c in u["coins"]):
        raise FailRun("BTC není v univerzu – nelze pokračovat")
    return u


def prepare_llm_dir(rc, u, feats, state):
    d = os.path.join(rc.work, "llm")
    os.makedirs(d, exist_ok=True)
    tpl = open(os.path.join(rc.repo, "task", "daily_prompt.md"), encoding="utf-8").read()
    prompt = (tpl.replace("{{DATE}}", rc.D).replace("{{ASOF}}", rc.asof).replace("{{N}}", str(len(u["coins"])))
              .replace("{{COINS}}", ", ".join(u["coins"])).replace("{{MAX_SEARCHES}}", str(rc.cfg["llm"]["max_searches"])))
    canon.write_text(os.path.join(d, "prompt.md"), prompt)
    canon.write_text(os.path.join(d, "features.md"), fx.table_md(feats, u["coins"], rc.asof))
    canon.write_json(os.path.join(d, "universe.json"), [{k: c[k] for k in ("coin", "name", "rank", "market_cap")} for c in u["coins_full"]])
    led = state["portfolios"]["claude_volne"]["base"]
    canon.write_json(os.path.join(d, "claude_volne_positions.json"),
                     {"cash_usd": led["cash"], "positions": {c: {"qty": p["qty"], "entry_date": p["entry_date"], "avg_price": p["avg_px"]}
                                                             for c, p in sorted(led["positions"].items())}})
    shutil.copy(os.path.join(rc.repo, "task", "scores.schema.json"), os.path.join(d, "scores.schema.json"))
    return d, prompt


def llm_step(rc, u, feats, state):
    """Returns validated scores or None (LLM variants then hold)."""
    cfg = rc.cfg
    d, prompt = prepare_llm_dir(rc, u, feats, state)
    t0 = rc.clock.now_ms()
    meta = rc.llm_runner(cfg, d, prompt, cfg["llm"]["timeout_min"] * 60, "transcript.jsonl", None)
    meta["started_at"] = canon.ms_iso(t0)
    raw = _read_scores(d)
    clean, rep = llm.validate(raw, u["coins"], rc.D) if raw is not None else (None, {"errors": ["scores.json chybí nebo není platný JSON"], "warnings": []})
    meta["repair"] = None
    if rep["errors"] and meta.get("session_id") and not meta.get("timed_out"):
        if raw is not None:
            shutil.copy(os.path.join(d, "scores.json"), os.path.join(d, "scores.v1.json"))
        msg = ("Engine validation of scores.json found these problems. Fix ONLY these in scores.json (score missing coins "
               "properly; do not invent defaults), write the file again and reply with one line:\n- " + "\n- ".join(rep["errors"][:30]))
        m2 = rc.llm_runner(cfg, d, msg, cfg["llm"]["repair_timeout_min"] * 60, "transcript_repair.jsonl", meta["session_id"])
        meta["repair"] = {k: m2.get(k) for k in ("exit_code", "timed_out", "wall_s", "result", "models_used", "model_init")}
        raw = _read_scores(d)
        clean, rep = llm.validate(raw, u["coins"], rc.D) if raw is not None else (None, {"errors": ["scores.json po opravě chybí"], "warnings": []})
    meta["finished_at"] = canon.ms_iso(rc.clock.now_ms())
    meta["model_ok"] = llm.model_ok(cfg, meta)
    rc.llm_meta = meta
    canon.write_json(os.path.join(d, "meta.json"), meta)
    canon.write_json(os.path.join(rc.work, "validate.json"), rep)
    if clean is not None and not meta["model_ok"]:
        rc.warnings.append(f"LLM běžel na jiném modelu ({meta.get('model_init')}, {meta.get('models_used')}) než připnutý {cfg['llm']['model']} – skóre se nepoužije")
        clean = None
    if clean is None:
        rc.warnings.append("krok LLM selhal – LLM varianty drží pozice beze změn: " + "; ".join(rep["errors"][:3]))
    elif rep["errors"]:
        rc.warnings.append(f"validace skóre: {len(rep['errors'])} chyb po opravě (vyřazeno: {', '.join(rep.get('dropped', [])) or '—'})")
    srch = meta.get("web_search_requests") or 0
    if srch > cfg["llm"]["max_searches"]:
        rc.warnings.append(f"LLM překročil rozpočet hledání ({srch} > {cfg['llm']['max_searches']})")
    if clean is not None:
        canon.write_json(os.path.join(rc.work, "scores.json"), clean)
    return clean


def _read_scores(d):
    p = os.path.join(d, "scores.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, UnicodeDecodeError):
        return None


def take_snapshot(rc, coins, pairs, locked_ms):
    cfg = rc.cfg
    prs = {c: pairs.get(c, c + cfg["quote"]) for c in coins}
    snap_src = {}
    for attempt in range(2):
        prices_p, server_ms, src = rc.market.ticker_prices(sorted(set(prs.values())))
        fetched_ms = rc.clock.now_ms()
        prices = {}
        for c in coins:
            if prs[c] in prices_p:
                prices[c] = prices_p[prs[c]]
                snap_src[c] = src
            else:
                p, s2 = rc.market.ticker_price_fallback(c)
                prices[c] = p
                snap_src[c] = s2
                rc.warnings.append(f"{c}: snímek ze záložního zdroje {s2}")
        if not (fetched_ms > locked_ms and server_ms > locked_ms):
            raise FailRun(f"lookahead: snímek {canon.ms_iso(fetched_ms)} / server {canon.ms_iso(server_ms)} není po zamčení {canon.ms_iso(locked_ms)}")
        if any(p <= 0 for p in prices.values()):
            raise FailRun("snímek obsahuje nekladnou cenu")
        # cross-check BTC + rotating coin against an aggregate reference price
        others = [c for c in coins if c != cfg["btc"]]
        rot = others[canon.days_between("2026-01-01", rc.D) % len(others)] if others else None
        xc, bad = {}, []
        for c in [cfg["btc"], rot]:
            if not c:
                continue
            try:
                ref, rsrc = rc.market.reference_price(c, rc.cg_ids.get(c))
                r = dict(chk.crosscheck(prices[c], ref, cfg["data"]["crosscheck_tol"]), other_source=rsrc)
                xc[c] = r
                if not r["ok"]:
                    bad.append(f"{c} {r['diff_pct']} % vs {rsrc}")
            except FetchError as e:
                xc[c] = {"error": str(e)}
                rc.warnings.append(f"křížová kontrola snímku {c}: referenční zdroj nedostupný")
        if not bad:
            break
        if attempt == 1:
            raise FailRun("křížová kontrola snímku selhala: " + ", ".join(bad))
    return {"prices": dict(sorted(prices.items())), "fetched_ms": fetched_ms, "fetched_at": canon.ms_iso(fetched_ms),
            "server_ms": server_ms, "server_time": canon.ms_iso(server_ms), "source": src, "sources": snap_src,
            "locked_at": canon.ms_iso(locked_ms), "crosscheck": xc}


def run_catchup(rc, d, state, hist, u):
    """Missed day d: settle and mark only. No decisions (that would be hindsight)."""
    sub = RunCtx(rc.repo, rc.cfg, d, rc.market, rc.llm_runner, rc.clock, rc.version, os.path.join(rc.work, f"catchup-{d}"), rc.dry_run)
    sub.cg_ids = rc.cg_ids
    rec = Recorder(os.path.join(sub.work, "raw"), rc.clock)
    sub.market = rc.market_factory(rec, d)
    u = universe_view(current_universe(rc.repo, d) or u) if current_universe(rc.repo, d) else u
    started = rc.clock.now_ms()
    fm = fetch_market(sub, state, hist, u, "catchup")
    starts = intraday_window_start(rc.cfg, state)
    intr = fetch_intraday(sub, starts, canon.date_ms(d), fm["pairs"])
    inp = {"date": d, "asof": sub.asof, "mode": "catchup", "universe": {"date": u["date"], "coins": u["coins"]},
           "candles": fm["candles"], "status": fm["status"], "intraday_a": intr}
    state2, hist2, tr, marks, feats, delisted = phase_a(rc.cfg, state, hist, inp)
    canon.write_json(os.path.join(sub.work, "raw_manifest.json"), rec.manifest())
    canon.write_json(os.path.join(sub.work, "inputs.json"), inp)
    canon.write_json(os.path.join(sub.work, "check.json"), sub.checks)
    canon.write_json(os.path.join(sub.work, "features.json"), feats)
    canon.write_json(os.path.join(sub.work, "fills.json"), tr)
    canon.write_json(os.path.join(sub.work, "ledger.json"), {"date": d, "close": marks, "after": None})
    shutil.copy(os.path.join(rc.repo, "config", "config.json"), os.path.join(sub.work, "config.json"))
    sub.warnings.insert(0, "Doplněno zpětně: jen vypořádání a ocenění, žádná nová rozhodnutí")
    return sub, state2, hist2, started, tr, marks


def run_day(rc):
    """Full decision run for rc.D. Writes everything into rc.work. Returns dict for run.json."""
    cfg, repo, D = rc.cfg, rc.repo, rc.D
    rc.timing["started_at"] = canon.ms_iso(rc.clock.now_ms())
    state, hist = load_state(repo), load_history(repo)
    rec = Recorder(os.path.join(rc.work, "raw"), rc.clock)
    rc.rec = rec                       # the runner writes raw_manifest.json even when the run fails
    rc.market = rc.market_factory(rec, D)
    # ---- universe
    u = current_universe(repo, D)
    new_universe = None
    if uni.needs_rebuild(cfg, D, u):
        u = build_universe(rc)
        new_universe = u
    uv = universe_view(u)
    rc.cg_ids = {c["coin"]: c["id"] for c in u["coins"]} if u["source"] == "coingecko" else {}
    canon.write_json(os.path.join(rc.work, "universe.json"), u)
    # ---- catch-up of missed days (settle + mark only)
    catchups = []
    if state.get("last_date") and state["last_date"] < canon.add_days(D, -1):
        for d in canon.date_range(canon.add_days(state["last_date"], 1), canon.add_days(D, -1)):
            if os.path.exists(os.path.join(repo, "runs", d, "run.json")):
                continue
            sub, state, hist, started, tr, marks = run_catchup(rc, d, state, hist, uv)
            catchups.append((d, sub, started, len(tr)))
            rc.catchup_days.append(d)
    # ---- A: fetch, check, settle, features
    fm = fetch_market(rc, state, hist, uv, "decision")
    state.setdefault("pairs", {}).update({c: p for c, p in fm["pairs"].items()})
    starts = intraday_window_start(cfg, state)
    intr_a = fetch_intraday(rc, starts, canon.date_ms(D), fm["pairs"])
    inp = {"date": D, "asof": rc.asof, "mode": "decision", "universe": {"date": u["date"], "coins": uv["coins"]},
           "candles": fm["candles"], "status": fm["status"], "intraday_a": intr_a}
    first_run = not state.get("portfolios")
    state, hist, trades_a, marks_close, feats, delisted = phase_a(cfg, state, hist, inp)
    canon.write_json(os.path.join(rc.work, "check.json"), rc.checks)
    canon.write_json(os.path.join(rc.work, "features.json"), feats)
    canon.write_text(os.path.join(rc.work, "features.md"), fx.table_md(feats, uv["coins"], rc.asof))
    # ---- B: LLM
    scores = llm_step(rc, uv, feats, state)
    # ---- C: lock, snapshot, execute
    decisions = plan_all(cfg, state, hist, feats, scores, uv, D, first_run, fm["status"])
    locked_ms = rc.clock.now_ms()
    rc.timing["locked_at"] = canon.ms_iso(locked_ms)
    dec_doc = {"date": D, "locked_at": canon.ms_iso(locked_ms), "locked_ms": locked_ms,
               "scores_sha256": canon.sha256_obj(scores) if scores else None, "llm_ok": scores is not None,
               "portfolios": decisions}
    canon.write_json(os.path.join(rc.work, "decisions.json"), dec_doc)
    coins = needed_snapshot_coins(state, decisions)
    snap = take_snapshot(rc, coins, fm["pairs"], locked_ms)
    rc.timing["snapshot_at"] = snap["fetched_at"]
    canon.write_json(os.path.join(rc.work, "snapshot.json"), snap)
    starts_c = {c: max(s, canon.date_ms(D)) for c, s in intraday_window_start(cfg, state).items()}
    intr_c = fetch_intraday(rc, starts_c, snap["fetched_ms"] - snap["fetched_ms"] % 300_000, fm["pairs"])
    state, trades_c, marks_after = execute(cfg, state, hist, decisions, snap, intr_c, rc.asof, D)
    inp.update({"scores": scores, "first_run": first_run, "universe_full": u["coins"], "locked_ms": locked_ms,
                "snapshot": {k: snap[k] for k in ("prices", "fetched_ms", "source")}, "intraday_c": intr_c})
    canon.write_json(os.path.join(rc.work, "inputs.json"), inp)
    canon.write_json(os.path.join(rc.work, "raw_manifest.json"), rec.manifest())
    canon.write_json(os.path.join(rc.work, "fills.json"), trades_a + trades_c)
    canon.write_json(os.path.join(rc.work, "ledger.json"), {"date": D, "close": marks_close, "after": marks_after})
    shutil.copy(os.path.join(repo, "config", "config.json"), os.path.join(rc.work, "config.json"))
    return {"state": state, "hist": hist, "new_universe": new_universe, "catchups": catchups,
            "trades": trades_a + trades_c, "marks_after": marks_after, "marks_close": marks_close, "scores": scores,
            "delisted": delisted}
