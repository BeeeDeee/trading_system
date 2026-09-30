"""Run lifecycle around pipeline.run_day: pin check, lock, idempotence, finalize (run.json + hash chain)."""
import fcntl
import glob
import os
import shutil
import subprocess
import traceback

from . import canon, chain, pipeline

PINNED_PATHS = ("engine/engine.py", "engine/cpb/*.py", "config/config.json", "task/daily_prompt.md",
                "task/scores.schema.json", "run_daily.sh", "dashboard/template.html")


def manifest_files(repo):
    out = []
    for pat in PINNED_PATHS:
        out += sorted(os.path.relpath(p, repo) for p in glob.glob(os.path.join(repo, pat)))
    return out


def manifest_text(repo):
    return "".join(f"{canon.sha256_file(os.path.join(repo, p))}  {p}\n" for p in manifest_files(repo))


def pin_file():
    return os.environ.get("CPB_PIN_FILE", os.path.expanduser("~/.config/cpb/pin.json"))


def version_info(repo, require_pin=True):
    """Verify MANIFEST.sha256 against the files and against the pin stored OUTSIDE the repo."""
    problems = []
    mpath = os.path.join(repo, "MANIFEST.sha256")
    if not os.path.exists(mpath):
        problems.append("MANIFEST.sha256 chybí")
        man = ""
    else:
        man = open(mpath, encoding="utf-8").read()
        if man != manifest_text(repo):
            problems.append("soubory enginu/configu/promptu neodpovídají MANIFEST.sha256")
    tag = "v" + open(os.path.join(repo, "VERSION")).read().strip() if os.path.exists(os.path.join(repo, "VERSION")) else None
    msha = canon.sha256_bytes(man.encode())
    pin = canon.read_json(pin_file())
    if require_pin:
        if pin is None:
            problems.append(f"připnutí {pin_file()} chybí (tools/pin.py --install)")
        elif pin.get("manifest_sha256") != msha or pin.get("tag") != tag:
            problems.append(f"připnutí nesedí: pin {pin.get('tag')} {str(pin.get('manifest_sha256'))[:12]} vs repo {tag} {msha[:12]}")
    try:
        commit = subprocess.run(["git", "-C", repo, "log", "-1", "--format=%H", "--"] + list(PINNED_PATHS),
                                capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    sha = lambda p: canon.sha256_file(os.path.join(repo, p)) if os.path.exists(os.path.join(repo, p)) else None
    return {"tag": tag, "code_commit": commit, "manifest_sha256": msha, "engine_sha256": sha("engine/engine.py"),
            "config_sha256": sha("config/config.json"), "prompt_sha256": sha("task/daily_prompt.md"),
            "pin_ok": not problems}, problems


class Lock:
    def __init__(self, repo):
        os.makedirs(os.path.join(repo, "data"), exist_ok=True)
        self.f = open(os.path.join(repo, "data", ".lock"), "w")

    def __enter__(self):
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("jiný běh právě probíhá (lock)")
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def _move_contents(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        shutil.move(os.path.join(src, name), os.path.join(dst, name))
    os.rmdir(src)


def _summary(cfg, marks_after, trades):
    names = {d["id"]: d["name"] for d in cfg["variants"] + cfg["benchmarks"]}
    start = cfg["start_capital"]
    eq = {pid: m["base"]["equity"] for pid, m in (marks_after or {}).items()}
    var = sorted((pid for pid in eq if pid in {v["id"] for v in cfg["variants"]}), key=lambda p: (-eq[p], p))
    ret = lambda p: round((eq[p] / start - 1) * 100, 2)
    return {"n_portfolios": len(eq), "n_variants": len(cfg["variants"]),
            "n_trades_base": sum(1 for t in trades if t["scenario"] == "base"),
            "best": [{"id": p, "name": names[p], "ret_pct": ret(p)} for p in var[:3]],
            "worst": [{"id": p, "name": names[p], "ret_pct": ret(p)} for p in var[-2:]],
            "btc_ret_pct": ret("b_btc") if "b_btc" in eq else None}


def _write_record(repo, run_dir, rec):
    rec = dict(rec, files=chain.files_digest(run_dir))
    rec = chain.seal(rec, chain.last_hash(repo))
    canon.write_json(os.path.join(run_dir, "run.json"), rec)
    chain.append(repo, rec["type"], rec["date"], os.path.relpath(os.path.join(run_dir, "run.json"), repo), rec["this_hash"])
    return rec


def execute_run(repo, cfg, D, market_factory, llm_runner, clock, version, dry_run=False):
    """Returns (status, run_record). Never raises for run failures: they are recorded."""
    run_dir = os.path.join(repo, "runs", D)
    if os.path.exists(os.path.join(run_dir, "run.json")):
        return "exists", canon.read_json(os.path.join(run_dir, "run.json"))
    stamp = canon.ms_iso(clock.now_ms()).replace("-", "").replace(":", "")[9:15]
    work = os.path.join(run_dir, f".work-{stamp}")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    rc = pipeline.RunCtx(repo, cfg, D, None, llm_runner, clock, version, work, dry_run)
    rc.market_factory = market_factory
    base = {"type": "run", "date": D, "asof": rc.asof, "version": version, "llm_model_pinned": cfg["llm"]["model"]}
    try:
        res = pipeline.run_day(rc)
    except Exception as e:  # FailRun or unexpected crash: record precisely, change no state
        err = str(e) if isinstance(e, pipeline.FailRun) else f"{type(e).__name__}: {e}"
        if getattr(rc, "rec", None) is not None:
            canon.write_json(os.path.join(work, "raw_manifest.json"), rc.rec.manifest())
        fdir = os.path.join(run_dir, f"failed-{stamp}")
        _move_contents(work, fdir)
        rec = dict(base, status="failed", error=err, traceback=traceback.format_exc()[-3000:] if not isinstance(e, pipeline.FailRun) else None,
                   timing=dict(rc.timing, finished_at=canon.ms_iso(clock.now_ms())), checks=rc.checks, warnings=rc.warnings,
                   llm=_llm_brief(rc.llm_meta))
        return "failed", _write_record(repo, fdir, rec)
    # ---- success: catch-up records first (chronological), then today
    for d, sub, started, ntr in res["catchups"]:
        ddir = os.path.join(repo, "runs", d)
        _move_contents(sub.work, ddir)
        _write_record(repo, ddir, {"type": "run", "date": d, "asof": sub.asof, "status": "catchup", "version": version,
                                   "timing": {"started_at": canon.ms_iso(started), "finished_at": canon.ms_iso(clock.now_ms())},
                                   "checks": sub.checks, "warnings": sub.warnings, "summary": {"n_trades_base": ntr},
                                   "note": f"Doplněno zpětně během běhu {D}"})
    if res["new_universe"]:
        canon.write_json(os.path.join(repo, "universe", f"{D}.json"), res["new_universe"])
    pipeline.save_history(repo, res["hist"])
    canon.write_json(os.path.join(repo, "data", "state.json"), res["state"])
    _move_contents(work, run_dir)
    status = "warning" if rc.warnings else "ok"
    rec = dict(base, status=status, mode="decision",
               timing=dict(rc.timing, finished_at=canon.ms_iso(clock.now_ms())),
               checks={k: rc.checks.get(k) for k in ("errors", "warnings", "redenomination_suspects", "close_crosscheck", "sources")},
               snapshot_crosscheck=canon.read_json(os.path.join(run_dir, "snapshot.json"))["crosscheck"],
               warnings=rc.warnings, catchup_days=rc.catchup_days, llm=_llm_brief(rc.llm_meta),
               summary=_summary(cfg, res["marks_after"], res["trades"]), delisted=res["delisted"],
               universe_date=canon.read_json(os.path.join(run_dir, "universe.json"))["date"])
    return status, _write_record(repo, run_dir, rec)


def _llm_brief(m):
    if not m:
        return None
    r = m.get("result") or {}
    return {"model_pinned_ok": m.get("model_ok"), "model_init": m.get("model_init"), "models_used": m.get("models_used"),
            "session_id": m.get("session_id"), "exit_code": m.get("exit_code"), "timed_out": m.get("timed_out"),
            "wall_s": m.get("wall_s"), "web_search_requests": m.get("web_search_requests"),
            "web_fetch_requests": m.get("web_fetch_requests"), "num_turns": r.get("num_turns"),
            "repaired": m.get("repair") is not None}


def notify_text(rec):
    """First line OK/UPOZORNĚNÍ/CHYBA – date, then at most 4 lines."""
    head = {"ok": "OK", "warning": "UPOZORNĚNÍ", "catchup": "UPOZORNĚNÍ", "failed": "CHYBA"}.get(rec["status"], "CHYBA")
    lines = [f"{head} – {rec['date']}"]
    s = rec.get("summary") or {}
    if s.get("best"):
        lines.append("Nejlepší: " + ", ".join(f"{b['id']} {b['ret_pct']:+.1f} %" for b in s["best"]))
        lines.append("Nejhorší: " + ", ".join(f"{b['id']} {b['ret_pct']:+.1f} %" for b in s["worst"]) +
                     (f" | BTC {s['btc_ret_pct']:+.1f} %" if s.get("btc_ret_pct") is not None else ""))
        lines.append(f"Obchodů dnes: {s.get('n_trades_base', 0)}")
    if rec["status"] == "failed":
        lines.append(f"Chyba: {rec.get('error', '')[:200]}")
        lines.append("Co dělat: zkontroluj journalctl -u cpb-daily a spusť ručně run_daily.sh (--date dnes)")
    elif rec.get("warnings"):
        lines.append("Pozor: " + rec["warnings"][0][:200])
    return "\n".join(lines[:5])
