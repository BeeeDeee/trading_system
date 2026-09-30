"""Corrections are new chained records, never edits of old runs.

  redenomination  token re-base / migration (like a stock split): history, positions and stops re-based so that
                  portfolio value is unchanged; optional symbol change. Only after research confirms it.
  note            a documented correction of an earlier record (e.g. a data problem found later); no state effect.
"""
import os

from . import canon, chain, ledger as L, pipeline


def rebase_history(hist, coin, ratio, before_date, new_coin=None):
    """Divide prices of rows dated < before_date by ratio (base volume x ratio). Rows from before_date on are
    already in new units. Symbol rename moves the whole series."""
    rows = []
    for r in hist.get(coin, []):
        if r[0] < before_date:
            r = [r[0], r[1] / ratio, r[2] / ratio, r[3] / ratio, r[4] / ratio, r[5] * ratio, r[6]]
        rows.append(r)
    out = dict(hist)
    out.pop(coin, None)
    tgt = new_coin or coin
    if new_coin and new_coin in hist:
        rows = sorted({r[0]: r for r in rows + hist[new_coin]}.values())
    out[tgt] = rows
    return out


def rebase_hourly(hh, coin, ratio, effective_date, new_coin=None):
    """Hourly rows opening before 00:00 of effective_date: prices / ratio (quote volumes unchanged)."""
    cut = canon.date_ms(effective_date)
    rows = [r if r[0] >= cut else [r[0], r[1] / ratio, r[2] / ratio, r[3] / ratio, r[4] / ratio, r[5], r[6]] for r in hh.get(coin, [])]
    out = dict(hh)
    out.pop(coin, None)
    if rows or new_coin:
        out[new_coin or coin] = rows
    return out


def apply_state(state, coin, ratio, effective_date, new_coin=None):
    for scns in state.get("portfolios", {}).values():
        for led in scns.values():
            L.redenominate(led, coin, ratio, new_coin)
    for meta in state.get("meta", {}).values():          # hourly state: entry meta (ref_open is a price)
        m = meta.pop(coin, None)
        if m:
            meta[new_coin or coin] = dict(m, ref_open=m["ref_open"] / ratio)
    if new_coin and coin in state.get("pairs", {}):
        state["pairs"].pop(coin)
    state.setdefault("fetch_from", {})[new_coin or coin] = effective_date
    return state


def value_check(state, hist, coin, ratio, new_coin, date):
    """Portfolio values before/after at the last stored close (must be equal)."""
    asof = max(r[0] for r in hist[coin]) if hist.get(coin) else None
    out = {}
    for pid, scns in state.get("portfolios", {}).items():
        led = scns["base"]
        if coin in led["positions"]:
            out[pid] = led["positions"][coin]["qty"] * hist[coin][-1][4]
    return asof, out


def redenomination(repo, clock, coin, ratio, effective_date, new_coin=None, evidence=None, reason=""):
    if ratio <= 0:
        raise SystemExit("ratio musí být kladné")
    if not evidence:
        raise SystemExit("--evidence URL je povinné (potvrzení rešerší)")
    state, hist = pipeline.load_state(repo), pipeline.load_history(repo)
    if coin not in hist:
        raise SystemExit(f"{coin} není v historii")
    _, before = value_check(state, hist, coin, ratio, new_coin, effective_date)
    hist2 = rebase_history(hist, coin, ratio, effective_date, new_coin)
    state2 = apply_state(state, coin, ratio, effective_date, new_coin)
    from . import hourly as hr
    hh = hr.load_hh(repo)
    if coin in hh:
        hh2 = rebase_hourly(hh, coin, ratio, effective_date, new_coin)
        tgt = new_coin or coin
        shutil_rm = os.path.join(hr.hdir(repo), "history", coin)
        hr.save_hh(repo, hh2, {tgt: [r[0] for r in hh2[tgt]]})
        if new_coin and os.path.isdir(shutil_rm):
            import shutil
            shutil.rmtree(shutil_rm)
    hs = hr.load_hstate(repo)
    if hs:
        canon.write_json(os.path.join(hr.hdir(repo), "state.json"), apply_state(hs, coin, ratio, effective_date, new_coin))
    tgt = new_coin or coin
    after = {pid: scns["base"]["positions"][tgt]["qty"] * [r for r in hist2[tgt] if r[0] < effective_date][-1][4]
             for pid, scns in state2.get("portfolios", {}).items() if tgt in scns["base"]["positions"]}
    rec = {"type": "correction", "kind": "redenomination", "date": effective_date, "created_at": canon.ms_iso(clock.now_ms()),
           "params": {"coin": coin, "ratio": ratio, "new_coin": new_coin, "effective_date": effective_date},
           "evidence": evidence, "reason": reason, "value_before": before, "value_after": after, "files": None}
    return _write(repo, rec, state2, hist2, coin if new_coin else None)


def note(repo, clock, date, message, refers_to=None):
    rec = {"type": "correction", "kind": "note", "date": date, "created_at": canon.ms_iso(clock.now_ms()),
           "message": message, "refers_to": refers_to, "files": None}
    return _write(repo, rec, None, None, None)


def _write(repo, rec, state, hist, drop_coin):
    with chain.lock(repo):
        n = len([e for e in chain.entries(repo) if e["type"] == "correction"]) + 1
        rel = os.path.join("corrections", f"{n:04d}-{rec['kind']}-{rec['date']}.json")
        rec = chain.seal(rec, chain.last_hash(repo))
        canon.write_json(os.path.join(repo, rel), rec)
        chain.append(repo, "correction", rec["date"], rel, rec["this_hash"])
    if state is not None:
        pipeline.save_history(repo, hist)
        if drop_coin:
            p = os.path.join(repo, "data", "history", f"{drop_coin}.json")
            if os.path.exists(p):
                os.remove(p)
        canon.write_json(os.path.join(repo, "data", "state.json"), state)
    return rec
