"""Replay the whole experiment from stored point-in-time inputs (runs/*/inputs.json + corrections).

With the recorded configs, the recomputed features, decisions, fills and ledgers must be byte-identical to
the stored files (tools/verify_chain.py). With an alternative config it produces POST-HOC results
(tools/replay.py), which are in-sample and never evidence.
"""
import os

from . import canon, chain, corrections, llm, pipeline


def _same(path, obj):
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        return f.read() == canon.dumps_pretty(obj).encode("utf-8")


def replay(repo, cfg_override=None, check=True):
    """Returns (problems, state, hist, series). series = [{date, marks_close, marks_after}]."""
    problems, series = [], []
    state = {"portfolios": {}, "last_date": None, "last_fill_ms": None}
    hist = {}
    for e in chain.entries(repo):
        path = os.path.join(repo, e["path"])
        rec = canon.read_json(path)
        if e["type"] == "correction":
            if rec.get("kind") == "redenomination":
                p = rec["params"]
                hist = corrections.rebase_history(hist, p["coin"], p["ratio"], p["effective_date"], p.get("new_coin"))
                state = corrections.apply_state(state, p["coin"], p["ratio"], p["effective_date"], p.get("new_coin"))
            continue
        if rec["status"] not in ("ok", "warning", "catchup"):
            continue
        rd = os.path.dirname(path)
        inp = canon.read_json(os.path.join(rd, "inputs.json"))
        cfg = cfg_override or canon.read_json(os.path.join(rd, "config.json"))
        D = inp["date"]
        tag = f"{D} ({rec['status']})"
        state, hist, tr_a, marks_close, feats, _ = pipeline.phase_a(cfg, state, hist, inp)
        if inp["mode"] == "catchup":
            if check:
                for fn, obj in (("features.json", feats), ("fills.json", tr_a), ("ledger.json", {"date": D, "close": marks_close, "after": None})):
                    if not _same(os.path.join(rd, fn), obj):
                        problems.append(f"{tag}: {fn} se nereprodukuje")
            series.append({"date": D, "mode": "catchup", "close": marks_close, "after": None})
            continue
        if check and inp.get("scores") is not None:
            raw = canon.read_json(os.path.join(rd, "llm", "scores.json"))
            clean, _ = llm.validate(raw, inp["universe"]["coins"], D)
            if canon.dumps(clean) != canon.dumps(inp["scores"]):
                problems.append(f"{tag}: validace llm/scores.json nedává uložené skóre")
        uv = {"coins": inp["universe"]["coins"], "coins_full": inp["universe_full"]}
        decisions = pipeline.plan_all(cfg, state, hist, feats, inp["scores"], uv, D, inp["first_run"], inp["status"])
        snap = inp["snapshot"]
        if check and not (inp["locked_ms"] < snap["fetched_ms"]):
            problems.append(f"{tag}: snímek není po zamčení rozhodnutí")
        state, tr_c, marks_after = pipeline.execute(cfg, state, hist, decisions, snap, inp["intraday_c"], inp["asof"], D)
        if check:
            dec = canon.read_json(os.path.join(rd, "decisions.json"))
            if canon.dumps(dec["portfolios"]) != canon.dumps(decisions):
                problems.append(f"{tag}: decisions.json se nereprodukuje")
            for fn, obj in (("features.json", feats), ("fills.json", tr_a + tr_c),
                            ("ledger.json", {"date": D, "close": marks_close, "after": marks_after})):
                if not _same(os.path.join(rd, fn), obj):
                    problems.append(f"{tag}: {fn} se nereprodukuje")
        series.append({"date": D, "mode": "decision", "close": marks_close, "after": marks_after})
    return problems, state, hist, series


def compare_final(repo, state, hist):
    probs = []
    stored = pipeline.load_state(repo)
    a = {k: v for k, v in stored.items() if k != "pairs"}
    b = {k: v for k, v in state.items() if k != "pairs"}
    if canon.dumps(a) != canon.dumps(b):
        probs.append("data/state.json se nereprodukuje z uložených vstupů")
    stored_h = pipeline.load_history(repo)
    if canon.dumps(stored_h) != canon.dumps(hist):
        probs.append("data/history se nereprodukuje z uložených vstupů")
    return probs
