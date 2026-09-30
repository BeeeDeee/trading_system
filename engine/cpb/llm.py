"""LLM step: validation of llm/scores.json (untrusted input) and the headless `claude -p` runner.

Validation never invents a value. Out-of-range numbers are clamped (warning); a coin with a missing or
non-numeric required field is dropped entirely (error -> one repair attempt). Text is truncated and
stripped of control characters; only http(s) URLs are kept.
"""
import json
import os
import re
import subprocess
import time

from . import canon

LENS = {"trend": (-2, 2, True), "mr": (-2, 2, True), "news": (-2, 2, True), "conviction": (1, 5, True),
        "p_outperform_btc_7d": (0.0, 1.0, False), "p_up_7d": (0.0, 1.0, False), "expected_move_7d_pct": (-90.0, 300.0, False)}
EVENT_TYPES = {"unlock", "upgrade", "listing", "hack", "regulace", "jine", "jiné", "none", "zadna", "žádná"}
CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f‪-‮⁦-⁩]")


def _text(x, n=400):
    if not isinstance(x, str):
        return ""
    return CTRL.sub(" ", x).strip()[:n]


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return float(x)


def validate(raw, universe_coins, date):
    """-> (clean, report). clean is None when nothing usable came back."""
    errors, warnings = [], []
    if not isinstance(raw, dict):
        return None, {"errors": ["výstup není JSON objekt"], "warnings": [], "dropped": [], "clamped": []}
    if raw.get("date") != date:
        warnings.append(f"datum ve výstupu {raw.get('date')!r} ≠ {date}")
    U = set(universe_coins)
    coins_in = raw.get("coins") if isinstance(raw.get("coins"), dict) else {}
    clean_coins, dropped, clamped = {}, [], []
    for sym in sorted(coins_in):
        s = coins_in[sym]
        if sym not in U:
            warnings.append(f"{sym}: není v univerzu, ignorováno")
            continue
        if not isinstance(s, dict):
            dropped.append(sym)
            errors.append(f"{sym}: skóre není objekt")
            continue
        c, bad = {}, []
        for k, (lo, hi, integer) in LENS.items():
            v = _num(s.get(k))
            if v is None:
                bad.append(k)
                continue
            if integer and v != round(v):
                clamped.append(f"{sym}.{k}={v} zaokrouhleno")
                v = float(round(v))
            if v < lo or v > hi:
                clamped.append(f"{sym}.{k}={v} oříznuto do [{lo}, {hi}]")
                v = min(max(v, lo), hi)
            c[k] = int(v) if integer else v
        if bad:
            dropped.append(sym)
            errors.append(f"{sym}: chybí nebo nečíselné {', '.join(bad)} – coin vyřazen (nedoplňuje se)")
            continue
        et = _text(s.get("event_type"), 20).lower()
        c["event"] = bool(s.get("event")) if isinstance(s.get("event"), bool) else False
        c["event_type"] = et if et in EVENT_TYPES else ("jine" if et else "none")
        c["risk_flag"] = s.get("risk_flag") is True
        c["risk_reason"] = _text(s.get("risk_reason"), 200)
        c["note"] = _text(s.get("note"))
        c["sources"] = [u[:300] for u in (s.get("sources") or []) if isinstance(u, str) and re.match(r"^https?://\S+$", u)][:10]
        clean_coins[sym] = c
    missing = sorted(U - set(clean_coins))
    for sym in missing:
        if sym not in dropped:
            errors.append(f"{sym}: chybí skóre")
    top, seen = [], set()
    for t in raw.get("top10") or []:
        sym = t.get("coin") if isinstance(t, dict) else t
        if sym in clean_coins and sym not in seen:
            seen.add(sym)
            top.append({"coin": sym, "reason": _text(t.get("reason") if isinstance(t, dict) else "")})
    if len(top) < 10 and len(clean_coins) >= 10:
        errors.append(f"top10 obsahuje jen {len(top)} platných coinů")
    top = top[:10]
    fv = raw.get("claude_volne") if isinstance(raw.get("claude_volne"), dict) else {}
    pos, seen = [], set()
    for p in fv.get("positions") or []:
        if not isinstance(p, dict) or p.get("coin") not in clean_coins or p.get("coin") in seen:
            warnings.append(f"claude_volne: neplatná pozice {str(p)[:80]} ignorována")
            continue
        w = _num(p.get("weight_pct"))
        if w is None or w <= 0:
            warnings.append(f"claude_volne: {p.get('coin')} bez platné váhy, ignorováno")
            continue
        if w > 25:
            clamped.append(f"claude_volne.{p['coin']} {w} % oříznuto na 25 %")
            w = 25.0
        seen.add(p["coin"])
        pos.append({"coin": p["coin"], "weight_pct": w, "reason": _text(p.get("reason")), "invalidation": _text(p.get("invalidation"))})
    tot = sum(p["weight_pct"] for p in pos)
    if tot > 100:
        clamped.append(f"claude_volne: součet {tot:.1f} % zmenšen na 100 %")
        for p in pos:
            p["weight_pct"] = p["weight_pct"] * 100 / tot
    reg = raw.get("regime") if isinstance(raw.get("regime"), dict) else {}
    events = [{"date": _text(e.get("date"), 20), "title": _text(e.get("title"), 200), "type": _text(e.get("type"), 20)}
              for e in (raw.get("events_7d") or []) if isinstance(e, dict)][:20]
    clean = {"date": date, "regime": {"label": _text(reg.get("label"), 80), "summary": _text(reg.get("summary"), 600)},
             "events_7d": events, "coins": clean_coins, "top10": top,
             "claude_volne": {"positions": sorted(pos, key=lambda p: p["coin"]), "comment": _text(fv.get("comment"), 600)}}
    if not clean_coins:
        return None, {"errors": errors or ["žádné platné skóre"], "warnings": warnings, "dropped": dropped, "clamped": clamped}
    return clean, {"errors": errors, "warnings": warnings, "dropped": dropped, "clamped": clamped}


# ---------------------------------------------------------------- headless claude


def claude_cmd(cfg, prompt, resume=None):
    m = cfg["llm"]
    cmd = ["claude", "-p", prompt, "--model", m["model"], "--tools", m["tools"],
           "--allowedTools", m["tools"].replace(",", " "), "--permission-mode", "dontAsk",
           "--restricted", "--strict-mcp-config", "--disable-slash-commands",
           "--output-format", "stream-json", "--verbose"]
    if resume:
        cmd += ["--resume", resume]
    return cmd


def parse_stream(lines):
    """Extract init/result metadata from stream-json output."""
    meta = {"model_init": None, "tools": None, "session_id": None, "result": None, "tool_calls": {}}
    for ln in lines:
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        if ev.get("type") == "assistant":
            # usage.server_tool_use under-reports; count the tool calls actually made in the transcript
            for blk in (ev.get("message") or {}).get("content") or []:
                if isinstance(blk, dict) and blk.get("type") in ("tool_use", "server_tool_use"):
                    n = blk.get("name")
                    meta["tool_calls"][n] = meta["tool_calls"].get(n, 0) + 1
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            meta["model_init"] = ev.get("model")
            meta["tools"] = ev.get("tools")
            meta["session_id"] = ev.get("session_id")
        elif ev.get("type") == "result":
            meta["result"] = {k: ev.get(k) for k in ("subtype", "is_error", "duration_ms", "num_turns", "session_id",
                                                     "total_cost_usd", "usage", "modelUsage", "terminal_reason", "stop_reason")}
            meta["session_id"] = ev.get("session_id") or meta["session_id"]
    return meta


def run_claude(cfg, workdir, prompt, timeout_s, transcript_name, resume=None):
    """Run claude -p in workdir; the full stream is saved verbatim as the transcript."""
    t0 = time.time()
    cmd = claude_cmd(cfg, prompt, resume)
    env = dict(os.environ)
    try:
        p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout_s, env=env, stdin=subprocess.DEVNULL)
        out, err, code, timed_out = p.stdout, p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        code, timed_out = None, True
    canon.write_text(os.path.join(workdir, transcript_name), out)
    meta = parse_stream(out.splitlines())
    res = meta.get("result") or {}
    usage = res.get("usage") or {}
    stu = usage.get("server_tool_use") or {}
    models = sorted((res.get("modelUsage") or {}).keys())
    meta.update({"exit_code": code, "timed_out": timed_out, "wall_s": round(time.time() - t0, 1),
                 "stderr_tail": err[-2000:],
                 "web_search_requests": max(stu.get("web_search_requests") or 0, meta["tool_calls"].get("WebSearch", 0) + meta["tool_calls"].get("web_search", 0)),
                 "web_fetch_requests": max(stu.get("web_fetch_requests") or 0, meta["tool_calls"].get("WebFetch", 0)), "models_used": models,
                 "command": [c if i != 2 else "<prompt.md>" for i, c in enumerate(cmd)]})
    return meta


def model_ok(cfg, meta):
    want = cfg["llm"]["model"]
    used = meta.get("models_used") or []
    return meta.get("model_init") == want and want in used
