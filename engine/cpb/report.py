"""Dashboard data (public/data.json) + page (public/index.html) and the atomic publish to the web root.

The public directory only ever contains index.html and data.json (tests enforce it). Publishing copies them
into a fresh release directory and swaps a symlink with rename(2), so a visitor never sees a half-written file.
"""
import glob
import os
import shutil
import time

from . import analytics as A, canon, chain, pipeline

PUBLIC_FILES = ("index.html", "data.json")
REPO_URL = "https://github.com/BeeeDeee/trading_system"


def _runs(repo):
    recs = []
    for p in sorted(glob.glob(os.path.join(repo, "runs", "*", "run.json")) + glob.glob(os.path.join(repo, "runs", "*", "failed-*", "run.json"))):
        r = canon.read_json(p)
        r["_dir"] = os.path.dirname(p)
        recs.append(r)
    return sorted(recs, key=lambda r: (r["date"], r.get("timing", {}).get("started_at") or ""))


def collect(repo, cfg, null_paths=1000, full_replay=True, fast=False):
    runs = _runs(repo)
    good = [r for r in runs if r["status"] in ("ok", "warning", "catchup")]
    ledgers = [canon.read_json(os.path.join(r["_dir"], "ledger.json")) for r in good]
    hist = pipeline.load_history(repo)
    px = A.closes_by_date(hist)
    score_days, universes = [], {}
    for r in good:
        inp = canon.read_json(os.path.join(r["_dir"], "inputs.json"))
        universes[r["date"]] = inp["universe"]["coins"]
        if inp.get("scores"):
            score_days.append((r["date"], inp["asof"], inp["scores"]))
    defs = {d["id"]: d for d in cfg["variants"] + cfg["benchmarks"]}
    hvars = cfg.get("hourly", {}).get("variants", [])
    kinds = {d["id"]: ("variant" if d in cfg["variants"] else "benchmark") for d in cfg["variants"] + cfg["benchmarks"]}
    kinds.update({d["id"]: "hourly" for d in hvars})
    data = {"generated_at": canon.ms_iso(int(time.time() * 1000)), "repo_url": REPO_URL,
            "start_capital": cfg["start_capital"], "rules_text": _rules_text(cfg),
            "portfolios": [{"id": d["id"], "name": d["name"], "kind": kinds[d["id"]], "question": d.get("question", ""),
                            "uses_llm": d.get("uses_llm", False)} for d in cfg["variants"] + hvars + cfg["benchmarks"]]}
    hruns = _hourly_runs(repo)
    hgood = [r for r in hruns if r["status"] in ("ok", "warning")]
    # chain status
    probs = chain.verify(repo)
    full_replay = full_replay and not fast
    if full_replay and not probs:
        from . import replay
        rp, st, hi, _ = replay.replay(repo)
        probs += rp + replay.compare_final(repo, st, hi)
    data["chain"] = {"ok": not probs, "records": len(chain.entries(repo)), "problems": probs[:10],
                     "replayed": full_replay}
    data["runs"] = [{"date": r["date"], "status": r["status"], "error": r.get("error"), "warnings": (r.get("warnings") or [])[:6],
                     "tag": (r.get("version") or {}).get("tag"), "model": r.get("llm_model_pinned"),
                     "hash": r["this_hash"][:12]} for r in runs][-120:]
    if not ledgers:
        data["empty"] = True
        return data
    last_r = good[-1]
    last_any = runs[-1]
    S = A.series_from_ledgers(ledgers)
    dates = [L["date"] for L in ledgers]
    data["series"] = {"dates": dates,
                      "equity": {pid: {scn: [round(e, 2) for _, e in S[pid][scn]] for scn in S[pid]} for pid in S},
                      "segments": _segments(good)}
    met = A.metrics(ledgers, cfg["start_capital"])
    # hourly variants on the same daily axis: valued by the first successful hourly run of each day (normally 00:02)
    hl = _hourly_daily(hgood)
    if hl:
        by_date = {x["date"]: x for x in hl}
        for v in hvars:
            if v["id"] in hl[-1]["close"]:
                data["series"]["equity"][v["id"]] = {scn: [round(by_date[d]["close"][v["id"]][scn]["equity"], 2)
                                                           if d in by_date and v["id"] in by_date[d]["close"] else None
                                                           for d in dates] for scn in ("base", "stress", "gross")}
        hmet = A.metrics(hl, cfg["start_capital"], btc_ret=_btc_ret_at_hour(ledgers, hgood, cfg))
        hnull = _hourly_nulls(repo, cfg, hgood, max(100, null_paths // 5), cached_only=fast)
        for vid, m in hmet.items():
            if vid in hnull:
                m["null"] = {k: hnull[vid][k] for k in ("p05", "p50", "p95", "percentile")}
            met[vid] = m
    _append_live_point(data["series"], ledgers, hl[-1] if hl else None)
    data["hourly"] = {"runs_total": len(hruns), "runs_ok": len(hgood), "last": _hourly_last(hruns),
                      "today": [{"label": r["label"], "status": r["status"], "error": r.get("error"),
                                 "quarantine": sorted((r.get("checks") or {}).get("quarantine") or {})}
                                for r in hruns if r["date"] == (hruns[-1]["date"] if hruns else None)]}
    nulls = _nulls(repo, cfg, good, hist, null_paths, cached_only=fast)
    data["null_info"] = {"as_of": next(iter(nulls.values()), {}).get("as_of"), "paths": next(iter(nulls.values()), {}).get("paths")}
    for pid, m in met.items():
        if pid in nulls:
            m["null"] = {k: nulls[pid][k] for k in ("p05", "p50", "p95", "percentile")}
    data["metrics"] = met
    after = ledgers[-1].get("after") or ledgers[-1]["close"]
    data["positions"] = {pid: {"equity": m["base"]["equity"], "cash": m["base"]["cash"], "exposure": m["base"]["exposure"],
                               "positions": [{"coin": c, "value": p["value"], "weight": p["value"] / m["base"]["equity"],
                                              "entry_date": p["entry_date"], "stop": p.get("stop")}
                                             for c, p in sorted(m["base"]["positions"].items(), key=lambda kv: -kv[1]["value"])]}
                         for pid, m in after.items()}
    if hgood:
        la = canon.read_json(os.path.join(hgood[-1]["_dir"], "ledger.json"))["after"]
        for vid, m in la.items():
            data["positions"][vid] = {"equity": m["base"]["equity"], "cash": m["base"]["cash"], "exposure": m["base"]["exposure"],
                                      "as_of": hgood[-1]["label"],
                                      "positions": [{"coin": c, "value": p["value"], "weight": p["value"] / m["base"]["equity"],
                                                     "entry_date": p["entry_date"], "stop": p.get("stop")}
                                                    for c, p in sorted(m["base"]["positions"].items(), key=lambda kv: -kv[1]["value"])]}
    ldir = last_r["_dir"]
    dec = canon.read_json(os.path.join(ldir, "decisions.json")) or {}
    data["decisions_info"] = {pid: d.get("info") for pid, d in (dec.get("portfolios") or {}).items()}
    inp = canon.read_json(os.path.join(ldir, "inputs.json"))
    sc = inp.get("scores")
    feats = canon.read_json(os.path.join(ldir, "features.json"))
    keep = ("close", "ret1", "ret7", "ret30", "rel30", "rsi14", "atr14_pct", "sma20_dist")
    data["today"] = {"date": last_r["date"], "asof": inp["asof"], "universe": inp["universe"]["coins"],
                     "scores": sc, "features": {c: ({k: f.get(k) for k in keep} if f else None) for c, f in feats.items()
                                                if c in inp["universe"]["coins"]}}
    fills = canon.read_json(os.path.join(ldir, "fills.json")) or []
    data["today"]["fills"] = [t for t in fills if t["scenario"] == "base"]
    data["analytics"] = {"ic": A.ic_table(score_days, px, cfg["rules"]),
                         "calibration": [A.calibration(score_days, px, "p_outperform_btc_7d"), A.calibration(score_days, px, "p_up_7d")],
                         "score_days": len(score_days)}
    data["last_run"] = {k: last_any.get(k) for k in ("date", "status", "error", "warnings", "timing", "llm", "checks",
                                                     "snapshot_crosscheck", "catchup_days", "summary", "version", "llm_model_pinned",
                                                     "this_hash", "prev_hash", "universe_date", "delisted")}
    return data


def _append_live_point(series, ledgers, hl_last):
    """Chart's last point = the same latest (after-fill) mark the metrics table uses, so the chart and the table agree.
    The daily close marks (00:00 UTC) stay as the earlier points; the live point is labelled "<date> aktuálně"."""
    after = ledgers[-1].get("after")
    if not after:
        return
    hafter = (hl_last or {}).get("after") or {}
    for pid, scns in series["equity"].items():
        src = hafter.get(pid) or after.get(pid)
        for scn, vals in scns.items():
            m = (src or {}).get(scn)
            vals.append(round(m["equity"], 2) if m else None)
    series["dates"].append(ledgers[-1]["date"] + " aktuálně")


def _hourly_runs(repo):
    out = []
    for p in glob.glob(os.path.join(repo, "runs", "*", "hourly", "*", "run.json")):
        r = canon.read_json(p)
        r["_dir"] = os.path.dirname(p)
        out.append(r)
    return sorted(out, key=lambda r: (r["label"], r.get("timing", {}).get("started_at") or ""))


def _btc_ret_at_hour(ledgers, hgood, cfg):
    """BTC HOLD revalued at the BTC mid of the latest hourly run, so the hourly variants are compared with BTC at the
    same moment they are valued (the daily ledgers are valued at a different time)."""
    last = ledgers[-1].get("after") or ledgers[-1]["close"]
    b = last.get("b_btc", {}).get("base")
    if not b:
        return None
    px = canon.read_json(os.path.join(hgood[-1]["_dir"], "inputs.json"))["snapshot"]["prices"].get(cfg["btc"])
    pos = b["positions"].get(cfg["btc"])
    if px is None:
        return None
    return (b["cash"] + (pos["qty"] * px if pos else 0)) / cfg["start_capital"] - 1


def _hourly_daily(hgood):
    """Pseudo daily ledgers of the hourly variants: close marks of the first hourly run of each day, after-marks of the
    last one (exposure)."""
    days = {}
    for r in hgood:
        days.setdefault(r["date"], []).append(r)
    out = []
    for d in sorted(days):
        first = canon.read_json(os.path.join(days[d][0]["_dir"], "ledger.json"))
        last = canon.read_json(os.path.join(days[d][-1]["_dir"], "ledger.json"))
        out.append({"date": d, "close": first["close"], "after": last["after"]})
    return out


def _hourly_last(hruns):
    if not hruns:
        return None
    r = hruns[-1]
    return {"label": r["label"], "status": r["status"], "error": r.get("error"), "warnings": (r.get("warnings") or [])[:5]}


def _hourly_nulls(repo, cfg, hgood, n_paths, cached_only=False):
    """Engine-based null of the hourly variants (weekly on Mondays, cached in data/hourly/null_cache.json)."""
    from . import null
    if len(hgood) < 48:
        return {}
    cache_p = os.path.join(repo, "data", "hourly", "null_cache.json")
    cache = canon.read_json(cache_p)
    last = hgood[-1]["label"]
    if cache and cache.get("paths") == n_paths and (cache.get("as_of", "")[:10] == last[:10] or
                                                     (canon.weekday(last[:10]) != 0 and canon.days_between(cache["as_of"][:10], last[:10]) < 7)):
        return cache["results"]
    if cached_only:
        return (cache or {}).get("results", {})
    res = null.hourly_null_percentiles(repo, hgood, cfg, n_paths)
    canon.write_json(cache_p, {"as_of": last, "paths": n_paths, "results": res})
    return res


def _nulls(repo, cfg, good, hist, n_paths, cached_only=False):
    """Engine-based null distribution. Expensive, so it is recomputed on Mondays, when missing, or during the first
    4 weeks; otherwise the cached result (data/null_cache.json, labelled with its date) is shown."""
    from . import null
    cache_p = os.path.join(repo, "data", "null_cache.json")
    cache = canon.read_json(cache_p)
    last = good[-1]["date"]
    n_dec = sum(1 for r in good if r["status"] != "catchup")
    fresh = cache and cache.get("paths") == n_paths and (cache.get("as_of") == last or
                                                         (canon.weekday(last) != 0 and n_dec > 28 and canon.days_between(cache["as_of"], last) < 7))
    if fresh:
        return cache["results"]
    if cached_only:
        return (cache or {}).get("results", {})
    days = null.load_days(repo, [(r, r["_dir"]) for r in good])
    res = null.null_percentiles(days, hist, cfg["variants"], n_paths=n_paths)
    canon.write_json(cache_p, {"as_of": last, "paths": n_paths, "results": res})
    return res


def _segments(good):
    """Version/model per valuation date; catch-up records inherit the previous decision run's segment."""
    out, tag, model = [], None, None
    for r in good:
        if r["status"] != "catchup":
            tag, model = (r.get("version") or {}).get("tag"), r.get("llm_model_pinned")
        out.append({"date": r["date"], "tag": tag, "model": model})
    return out


def _rules_text(cfg):
    r = cfg["rules"]
    return (f"composite = trend + news + 0,5·mr + (conviction − 3); brána = close > SMA20 ∧ RSI14 < {r['gate']['rsi_max']} ∧ bez risk_flag; "
            f"váhy < {r['min_weight'] * 100:.0f} % pryč, max {r['max_weight'] * 100:.0f} %/coin, rebalanc při odchylce > {r['band'] * 100:.0f} % kapitálu. "
            f"Náklady: poplatek {cfg['costs']['scenarios']['base']['fee_bps']} bps + slippage dle objemu; stres {cfg['costs']['scenarios']['stress']['fee_bps']} bps a 2× slippage.")


def build(repo, cfg, out_dir, label=None, inline=False, null_paths=1000, fast=False):
    data = collect(repo, cfg, null_paths=null_paths, fast=fast)
    if label:
        data["label"] = label
    tpl = open(os.path.join(repo, "dashboard", "template.html"), encoding="utf-8").read()
    os.makedirs(out_dir, exist_ok=True)
    js = canon.dumps(data).replace("</", "<\\/")
    html = tpl.replace("/*__INLINE_DATA__*/null", js if inline else "null")
    html = html.replace("<!--__CSP__-->", csp_meta(html))
    canon.write_text(os.path.join(out_dir, "index.html"), html)
    canon.write_text(os.path.join(out_dir, "data.json"), canon.dumps(data))
    for f in os.listdir(out_dir):
        if f not in PUBLIC_FILES:
            raise RuntimeError(f"nepovolený soubor v public/: {f}")
    return os.path.join(out_dir, "index.html")


def csp_meta(html):
    """Content-Security-Policy pinned to the hash of the one inline script: no external code can run."""
    import base64
    import hashlib
    import re
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    hashes = " ".join("'sha256-" + base64.b64encode(hashlib.sha256(x.encode("utf-8")).digest()).decode() + "'" for x in scripts)
    return ('<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src ' + hashes +
            '; style-src \'unsafe-inline\'; connect-src \'self\'; img-src \'self\' data:; base-uri \'none\'; form-action \'none\'">')


def publish(src_dir, web_root):
    """web_root/current -> web_root/releases/<stamp>/ swapped atomically; keeps the last 5 releases."""
    rel_root = os.path.join(web_root, "releases")
    os.makedirs(rel_root, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"-{os.getpid()}"
    dst = os.path.join(rel_root, stamp)
    os.makedirs(dst)
    for f in PUBLIC_FILES:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst, f))
        os.chmod(os.path.join(dst, f), 0o644)
    os.chmod(dst, 0o755)
    tmp = os.path.join(web_root, f".current-{stamp}")
    os.symlink(os.path.join("releases", stamp), tmp)
    os.replace(tmp, os.path.join(web_root, "current"))
    for old in sorted(os.listdir(rel_root))[:-5]:
        shutil.rmtree(os.path.join(rel_root, old), ignore_errors=True)
