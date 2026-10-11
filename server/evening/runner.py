"""One evening run on the server: the orchestration of task/task_prompt.md (v1.1.x), steps 0-9, in Python.

The deterministic parts (engine.py, config.json) are untouched and hash-pinned. Only the research/scoring/free-decision
part (steps 4-6) is done by `claude -p`; its instructions are cut verbatim out of the pinned task prompt.
"""
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zoneinfo

from . import cal, llm, prices
from .store import Store, read_json, write_json

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(REPO, "engine", "engine.py")
CONFIG = os.path.join(REPO, "config", "config.json")
PROMPT = os.path.join(REPO, "task", "task_prompt.md")
NY = zoneinfo.ZoneInfo("America/New_York")
STOCKS_ROTATION = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "JPM"]
DEFAULT_MODEL = "claude-opus-5-5"

LLM_HEADER = """You are running the research and scoring part of ONE daily step of a PAPER-TRADING experiment (no real money, no broker). Each evening you score 20 US stocks/ETFs with three independent lenses; a fixed Python engine (not you) then trades 19 strategy variants from those scores (plus a random baseline). The prices, the checks, the settlement and the planning are already done; do only the steps below.

Write every human-readable text field (notes, regime label/summary, score notes, reasons, invalidation) in CZECH, short and concrete. Never invent or estimate prices.

Today (New York date) is {date}. Files in the current directory (read them with the Read tool):
- features.txt : the features table of step 2 (indicators for every ticker; use it for trend and mr, never compute indicators yourself)
- settled_volny.json : the settled state of the "Claude volně" variant (its open positions, cash, equity) for step 6
- universe: {universe}
You have no shell: write the two result files with the Write tool, exactly in the formats described below, into the current directory:
- scores.json  (step 5)
- free.json    (step 6)
Do not write anything else and do not touch other files. Finish with one short line saying that both files were written.
"""


class Fail(Exception):
    pass


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def eng(*args):
    r = subprocess.run([sys.executable, ENGINE, *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        raise Fail(f"engine {args[0]} selhal: {(r.stderr or r.stdout).strip()[-400:]}")
    return r.stdout


def pinned_code():
    """Verify engine/config against the pins in the task prompt and MANIFEST; -> code dict (without source/commit)."""
    text = open(PROMPT).read()
    pin = lambda k: re.search(rf"^{k} = (.*)$", text, re.M).group(1).strip()
    tag, e_pin, c_pin = pin("TAG"), pin("ENGINE_SHA256"), pin("CONFIG_SHA256")
    e_sha, c_sha = sha256(ENGINE), sha256(CONFIG)
    if e_sha != e_pin or c_sha != c_pin:
        raise Fail(f"hash nesouhlasí s připnutím {tag}: engine {e_sha[:12]} (čekáno {e_pin[:12]}), config {c_sha[:12]} (čekáno {c_pin[:12]})")
    r = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True)
    commit = r.stdout.strip() if r.returncode == 0 else None
    return {"tag": tag, "commit": commit, "engine_sha256": e_sha, "config_sha256": c_sha, "source": "server"}


# ---------------------------------------------------------------- data: checks, splits, cross-check
def check_with_split(cfg, hist, prev, today, work, tag):
    """engine check; on split suspects confirm the split with Yahoo's split events and apply it ONCE. -> (check, hist, prev, notes)"""
    notes = []
    out = os.path.join(work, f"check_{tag}.json")
    eng("check", CONFIG, _w(work, f"hist_{tag}.json", hist), _w(work, f"today_{tag}.json", today), out)
    res = read_json(out)
    sus = res.get("split_suspects") or {}
    if res["errors"] and sus:
        applied = False
        for tk, ratio in sorted(sus.items()):
            try:
                real = prices.yahoo_splits(tk).get(today["date"])
            except prices.FetchError:
                real = None
            if real and abs(real / ratio - 1) < 0.02:
                hp = _w(work, f"hist_{tag}.json", hist)
                pp = _w(work, f"prev_{tag}.json", prev) if prev else "-"
                eng("apply-split", hp, pp, tk, f"{real:g}", hp, pp)
                hist = read_json(hp)
                prev = read_json(pp) if prev else prev
                notes.append(f"Split {tk} {real:g}:1 přepočten")
                applied = True
        if applied:
            eng("check", CONFIG, _w(work, f"hist_{tag}.json", hist), _w(work, f"today_{tag}.json", today), out)
            res = read_json(out)
    return res, hist, prev, notes


def _w(work, name, obj):
    p = os.path.join(work, name)
    write_json(p, obj)
    return p


def crosscheck(rows_primary, today, date, ordinal):
    """SPY + one rotating stock against Nasdaq. -> (spy, stock, warnings); raises Fail on a disagreement > 0.3 %."""
    out, warns = [], []
    for tk in ("SPY", STOCKS_ROTATION[ordinal % len(STOCKS_ROTATION)]):
        prim = (today.get("close") or {}).get(tk)
        rec = {"ticker": tk, "primary": prim, "other": None, "other_source": "api.nasdaq.com (historical)", "diff_pct": None}
        try:
            for attempt in range(2):
                other = prices.nasdaq_rows(tk).get(date)
                if other:
                    break
                time.sleep(20)
            rec["other"] = other[3] if other else None
        except prices.FetchError as e:
            warns.append(f"Křížová kontrola {tk} nedostupná: {e}")
        if rec["other"] and prim:
            rec["diff_pct"] = round((prim / rec["other"] - 1) * 100, 3)
            if abs(rec["diff_pct"]) > 0.3:
                raise Fail(f"Křížová kontrola {tk}: Yahoo {prim} vs Nasdaq {rec['other']} ({rec['diff_pct']:+.2f} %)")
        elif not warns or tk not in warns[-1]:
            warns.append(f"Křížová kontrola {tk}: Nasdaq zatím nemá řádek pro {date}")
        out.append(rec)
    return out[0], out[1], warns


# ---------------------------------------------------------------- saving
def save_day(st, out, today, scores, hist, date):
    """Collections in the order of the original task: trades, prices/scores/history, days LAST."""
    for t in out.get("trades") or []:
        st.put("trades", f"{t['variant']}-{t['ticker']}-{t['entry_date']}", t)
    st.put("prices", date, today)
    if scores is not None:
        st.put("scores", date, scores)
    st.put_history(hist, date)
    st.put("days", date, out["day"])


# ---------------------------------------------------------------- the run
def run(data_dir, date=None, dry_run=False, force=False, scores_file=None, free_file=None, model=None, wait_s=600, llm_timeout=3000):
    started = now_utc()
    ny = started.astimezone(NY)
    date = date or ny.date().isoformat()
    day = dt.date.fromisoformat(date)
    model = model or os.environ.get("EVENING_MODEL") or DEFAULT_MODEL
    if dry_run:
        tmp = data_dir.rstrip("/") + ".dry"
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.isdir(data_dir):
            shutil.copytree(data_dir, tmp, ignore=shutil.ignore_patterns(".git"))
        data_dir = tmp
    st = Store(data_dir)
    work = os.path.join(data_dir, "work", date)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    run_rec = {"date": date, "started_at": iso(started)}
    notes = []

    def finish(status, message, **extra):
        run_rec.update({"status": status, "finished_at": iso(now_utc()), "message": message, **extra})
        st.put("runs", date, run_rec)
        if not dry_run:
            st.commit(f"run {date}: {status}")
        return run_rec

    try:
        code = pinned_code()
        run_rec["code"] = code
        cfg = read_json(CONFIG)
        write_json(st.p("config.json"), cfg)                                    # mirror for the dashboard
        if not cal.is_trading_day(day):
            return finish("closed", "Burza zavřená")
        if st.get("days", date):
            run_rec["status"] = "exists"
            return run_rec
        if date == ny.date().isoformat() and not force and (ny.hour, ny.minute) < (16, 10):
            raise Fail("Burza ještě neskončila (před 16:10 ET), nic se nezapisuje")
        prev = st.latest_day()
        if prev and prev["date"] >= date:
            raise Fail(f"poslední uložený den {prev['date']} není starší než {date}")
        hist = st.history()
        U = cfg["universe"]

        # --- data (Yahoo chart API, 6 months of rows: today, catch-up days, backfill)
        rows_all, errs = prices.fetch_all(U, "6mo")
        today = prices.day_prices(rows_all, U, date)
        if len(today["close"]) < len(U) and wait_s:
            time.sleep(wait_s)                                                  # the daily row may be late; one re-fetch
            rows_all, errs = prices.fetch_all(U, "6mo")
            today = prices.day_prices(rows_all, U, date)
        if errs:
            notes.append("Yahoo: " + "; ".join(errs[:3]))
        write_json(os.path.join(work, "yahoo_rows.json"), rows_all)

        # --- 1. catch-up of missed trading days (no scores, no new decisions)
        catchup = []
        if prev:
            missed = cal.trading_days_between(dt.date.fromisoformat(prev["date"]), day - dt.timedelta(days=1))
            for md in missed:
                d = md.isoformat()
                t_d = prices.day_prices(rows_all, U, d)
                chk, hist, prev, nn = check_with_split(cfg, hist, prev, t_d, work, d)
                if chk["errors"]:
                    raise Fail(f"Doplnění {d}: {'; '.join(chk['errors'])[:300]}")
                hp = _w(work, f"hist_{d}.json", hist)
                eng("history-append", hp, _w(work, f"today_{d}.json", t_d), hp)
                hist = read_json(hp)
                eng("settle", CONFIG, _w(work, f"prev_{d}.json", prev), os.path.join(work, f"today_{d}.json"), os.path.join(work, f"settled_{d}.json"))
                eng("plan", CONFIG, os.path.join(work, f"settled_{d}.json"), _w(work, "empty.json", {"scores": {}}), "-", os.path.join(work, f"out_{d}.json"))
                o = read_json(os.path.join(work, f"out_{d}.json"))
                o["day"]["regime"] = {"label": "Doplněno zpětně", "summary": "Večerní běh chyběl; den dopočítán z historických cen bez nových rozhodnutí."}
                o["day"]["notes"] = "Doplněno zpětně"
                o["day"]["code"] = code
                save_day(st, o, t_d, None, hist, d)
                st.put("runs", d, {"date": d, "status": "catchup", "message": f"Doplněno zpětně během běhu {date}", "checks": chk,
                                   "finished_at": iso(now_utc()), "code": code, "source": "yahoo"})
                prev = o["day"]
                catchup.append(d)

        # --- 2. today's data
        if len(today["close"]) < len(U):
            miss = [t for t in U if t not in today["close"]]
            if len(miss) > 5 or "SPY" in miss:
                raise Fail(f"Chybí řádek pro {date} u {len(miss)} titulů: {', '.join(miss)} (Yahoo ještě nemá denní svíčku?)")
            notes.append(f"Chybí ceny: {', '.join(miss)}")
        # backfill: history missing or a ticker has < 50 rows before today
        short = [t for t in U if len([r for r in hist.get(t, []) if r[0] < date]) < 50]
        for t in short:
            rows = [[d, *v] for d, v in sorted((rows_all.get(t) or {}).items()) if d < date][-60:]
            if rows:
                hist[t] = rows
        spy, stock, cc_warn = crosscheck(rows_all, today, date, len(st.ids("days")))
        chk, hist, prev, nn = check_with_split(cfg, hist, prev, today, work, "today")
        notes += nn
        if chk["errors"]:
            raise Fail("Kontrola dat: " + "; ".join(chk["errors"])[:400])
        hp = _w(work, "history.json", hist)
        eng("history-append", hp, os.path.join(work, "today_today.json"), hp)
        hist = read_json(hp)
        feat_txt = eng("features", hp, os.path.join(work, "features.json"))
        with open(os.path.join(work, "features.txt"), "w") as f:
            f.write(feat_txt)

        # --- 3. settle
        pprev = _w(work, "prev.json", prev) if prev else "-"
        eng("settle", CONFIG, pprev, os.path.join(work, "today_today.json"), os.path.join(work, "settled.json"))
        settled = read_json(os.path.join(work, "settled.json"))

        # --- 4-6. research, scores, free decisions (LLM) or injected files (replay / tests)
        llm_meta = None
        if scores_file:
            shutil.copy(scores_file, os.path.join(work, "scores.json"))
            if free_file:
                shutil.copy(free_file, os.path.join(work, "free.json"))
        else:
            write_json(os.path.join(work, "settled_volny.json"), (settled.get("variants") or {}).get("volny") or {})
            prompt = llm.build_prompt(open(PROMPT).read(), LLM_HEADER.format(date=date, universe=" ".join(U)))
            with open(os.path.join(work, "prompt.md"), "w") as f:
                f.write(prompt)
            llm_meta = llm.run_claude(model, work, prompt, llm_timeout, "transcript.jsonl")
            if llm_meta["exit_code"] != 0 or not os.path.exists(os.path.join(work, "scores.json")):
                raise Fail(f"LLM krok selhal (exit {llm_meta['exit_code']}, timeout {llm_meta['timed_out']}): {llm_meta['stderr_tail'][-300:]}")
            if not llm.model_ok(model, llm_meta):
                notes.append(f"Model v transcriptu ({llm_meta.get('model_init')}) ≠ požadovaný {model}")
        scores = read_json(os.path.join(work, "scores.json"))
        free_path = os.path.join(work, "free.json")
        free_arg = free_path if os.path.exists(free_path) else "-"

        # --- 7. validate (one repair attempt), plan
        val = _validate(cfg_path=CONFIG, work=work, free_arg=free_arg)
        if val["errors"] and llm_meta and llm_meta.get("session_id"):
            fix = ("Validation of your files found these errors; fix scores.json / free.json ONCE (score the missing tickers properly, do not fill "
                   "in defaults) and write both files again:\n- " + "\n- ".join(val["errors"][:30]))
            m2 = llm.run_claude(model, work, fix, 1200, "transcript_repair.jsonl", resume=llm_meta["session_id"])
            llm_meta["repair"] = {k: m2.get(k) for k in ("exit_code", "timed_out", "wall_s")}
            scores = read_json(os.path.join(work, "scores.json"))
            free_arg = free_path if os.path.exists(free_path) else "-"
            val = _validate(cfg_path=CONFIG, work=work, free_arg=free_arg)
        if val["errors"]:
            notes.append("Zbylé chyby skóre: " + "; ".join(val["errors"][:5]))
        eng("plan", CONFIG, os.path.join(work, "settled.json"), os.path.join(work, "scores.json"), free_arg, os.path.join(work, "out.json"))
        out = read_json(os.path.join(work, "out.json"))
        d = out["day"]
        d["regime"] = scores.get("regime") or {"label": "—", "summary": ""}
        d["code"] = code
        d["notes"] = "; ".join(notes + nn) or "—"
        save_day(st, out, read_json(os.path.join(work, "today_today.json")), scores, hist, date)

        # --- 8. run record
        warn = bool(chk["warnings"] or catchup or cc_warn or val["errors"] or errs or any("Chybí" in n or "Model" in n or "Split" in n for n in notes))
        msg = (f"Běh v pořádku: ceny {chk['tickers_ok']} titulů z Yahoo (křížová kontrola Nasdaq), "
               f"{sum(len(v.get('fills') or []) for v in d['variants'].values())} plnění.") if not warn else \
            "Běh dokončen s upozorněním: " + "; ".join((chk["warnings"] + cc_warn + notes)[:4])
        rec = finish("warning" if warn else "ok", msg, source="yahoo finance chart API (OHLC) + api.nasdaq.com (křížová kontrola)",
                     checks=chk, spy_crosscheck=spy, stock_crosscheck=stock, catchup_days=catchup, llm=llm_meta,
                     model=model if llm_meta else None)
        return rec
    except Fail as e:
        return finish("failed", str(e))
    except Exception as e:                                                       # never leave a day without a run record
        return finish("failed", f"{type(e).__name__}: {e}")


def _validate(cfg_path, work, free_arg):
    p = os.path.join(work, "validate.json")
    eng("validate", cfg_path, os.path.join(work, "scores.json"), free_arg, p)
    return read_json(p)


# ---------------------------------------------------------------- notification text (step 9)
def notification(data_dir, date):
    st = Store(data_dir)
    run_rec, day, cfg = st.get("runs", date), st.get("days", date), st.config() or {}
    status = (run_rec or {}).get("status")
    head = {"ok": "OK", "warning": "UPOZORNĚNÍ", "failed": "CHYBA", "closed": "OK", "catchup": "OK"}.get(status, "CHYBA")
    if status == "closed":
        return f"{head} – {date}\nBurza zavřená."
    if status != "ok" and status != "warning" or not day:
        return f"{head} – {date}\n{(run_rec or {}).get('message', 'běh nenalezen')}"
    start = cfg.get("start_capital", 10000)
    names = {v["id"]: v["name"] for v in cfg.get("variants", [])}
    names["nahoda"] = (cfg.get("random_baseline") or {}).get("name", "Náhoda")
    rets = {k: (v["equity"] / start - 1) * 100 for k, v in day["variants"].items() if k != "nahoda"}
    order = sorted(rets, key=rets.get, reverse=True)
    fmt = lambda ks: ", ".join(f"{names.get(k, k)} {rets[k]:+.1f} %" for k in ks)
    spy = (day.get("spy_equity", start) / start - 1) * 100
    rnd = (day["variants"].get("nahoda", {}).get("equity", start) / start - 1) * 100
    fills = sum(len(v.get("fills") or []) for v in day["variants"].values())
    lines = [f"{head} – {date}", f"Nejlepší: {fmt(order[:3])}", f"Nejhorší: {fmt(order[-2:])}",
             f"SPY {spy:+.1f} %, náhoda {rnd:+.1f} %; plnění dnes: {fills}"]
    if status == "warning":
        lines.append((run_rec.get("message") or "")[:200])
    return "\n".join(lines)
