#!/usr/bin/env python3
"""Point-in-time panel dataset: date x coin x (indicators, all LLM scores, forward returns). Main output of the experiment.

  python tools/export_dataset.py [--repo DIR] [--out build/dataset] [--parquet]
  python tools/export_dataset.py --hourly    hodinový panel: hodina × coin × (výnosy, z-skóre, objem, agresivní nákupy,
                                             kniha, funding, ΔOI, L/S, bid/ask) + forward výnosy 1/4/24 h abs i vs BTC

One row per (run date, coin) for every coin that had features that day. Indicators are as of the close before
the run date (the information the decision had). Forward returns are from that close: fwd_{1,3,7,14}d (absolute)
and fwd_rel_{h}d (minus BTC); empty until the window has closed. Columns in_universe / in_top10 / llm_ok and the
version tag + model identify the segment (results of different versions/models must not be mixed).
"""
import argparse
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
from cpb import analytics as A, canon, pipeline, portfolio  # noqa: E402
from cpb.features import FIELDS  # noqa: E402

SCORE_FIELDS = ("trend", "mr", "news", "conviction", "p_outperform_btc_7d", "p_up_7d", "expected_move_7d_pct",
                "event", "event_type", "risk_flag")


def rows(repo):
    cfg_now = canon.read_json(os.path.join(repo, "config", "config.json"))
    px = A.closes_by_date(pipeline.load_history(repo))
    out = []
    for p in sorted(os.listdir(os.path.join(repo, "runs"))):
        rec = canon.read_json(os.path.join(repo, "runs", p, "run.json"))
        if not rec or rec["status"] not in ("ok", "warning", "catchup"):
            continue
        rd = os.path.join(repo, "runs", p)
        inp = canon.read_json(os.path.join(rd, "inputs.json"))
        feats = canon.read_json(os.path.join(rd, "features.json"))
        cfg = canon.read_json(os.path.join(rd, "config.json")) or cfg_now
        sc = inp.get("scores") or {}
        top = [t["coin"] for t in sc.get("top10", [])]
        U = set(inp["universe"]["coins"])
        for c in sorted(feats):
            f = feats[c]
            if not f:
                continue
            r = {"date": inp["date"], "asof": inp["asof"], "coin": c, "mode": inp["mode"], "in_universe": c in U,
                 "in_top10": c in top, "top10_rank": top.index(c) + 1 if c in top else None, "llm_ok": bool(sc),
                 "version": (rec.get("version") or {}).get("tag"), "model": rec.get("llm_model_pinned")}
            r.update({k: f.get(k) for k in FIELDS})
            s = (sc.get("coins") or {}).get(c)
            r.update({k: (s or {}).get(k) for k in SCORE_FIELDS})
            r["composite"] = portfolio.composite(s, cfg["rules"]) if s else None
            for h in (1, 3, 7, 14):
                ab, rel = A.fwd_rel(px, c, inp["asof"], h)
                r[f"fwd_{h}d"], r[f"fwd_rel_{h}d"] = ab, rel
            out.append(r)
    return out


HOURLY_FIELDS = ("close", "ret1h", "ret4h", "ret24h", "rel4h", "btc_ret1h", "z1h", "vol_ratio", "breakout", "rsi14h", "taker_ratio",
                 "vol24_usd", "book_imbalance", "spread_bps", "funding", "oi_chg_1h", "ls_ratio")


def hourly_rows(repo):
    from cpb import hourly as hr
    pxh = A.hourly_closes(hr.load_hh(repo, days=100000))
    out = []
    for p in sorted(glob.glob(os.path.join(repo, "runs", "*", "hourly", "[0-9][0-9]", "run.json"))):
        rec = canon.read_json(p)
        if rec["status"] not in ("ok", "warning"):
            continue
        rd = os.path.dirname(p)
        inp = canon.read_json(os.path.join(rd, "inputs.json"))
        feats = canon.read_json(os.path.join(rd, "features.json"))
        snap = inp["snapshot"]
        for c in sorted(feats):
            f = feats[c]
            if not f:
                continue
            r = {"label": inp["label"], "date": inp["date"], "hour": inp["hour"], "coin": c, "in_universe": c in inp["universe"],
                 "version": (rec.get("version") or {}).get("tag")}
            r.update({k: f.get(k) for k in HOURLY_FIELDS})
            q = (snap.get("quotes") or {}).get(c)
            r["snap_bid"], r["snap_ask"] = (q or [None, None])
            for h in A.HOURLY_HORIZONS:
                ab, rel = A.fwd_rel_h(pxh, c, inp["t_end"], h)
                r[f"fwd_{h}h"], r[f"fwd_rel_{h}h"] = ab, rel
            out.append(r)
    return out


def write_csv(path, data):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=ROOT)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "dataset"))
    ap.add_argument("--parquet", action="store_true")
    ap.add_argument("--hourly", action="store_true", help="hodinový panel (hodina × coin) místo denního")
    a = ap.parse_args()
    data = hourly_rows(a.repo) if a.hourly else rows(a.repo)
    name = "panel_hourly" if a.hourly else "panel"
    os.makedirs(a.out, exist_ok=True)
    if not data:
        print("žádná data")
        return
    p = os.path.join(a.out, f"{name}.csv")
    write_csv(p, data)
    print(f"{len(data)} řádků -> {p}")
    if a.parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise SystemExit("pro Parquet: pip install -r requirements.txt (pyarrow)")
        pq.write_table(pa.Table.from_pylist(data), os.path.join(a.out, f"{name}.parquet"))
        print(f"-> {name}.parquet")


if __name__ == "__main__":
    main()
