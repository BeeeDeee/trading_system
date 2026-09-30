#!/usr/bin/env python3
"""Point-in-time panel dataset: date x coin x (indicators, all LLM scores, forward returns). Main output of the experiment.

  python tools/export_dataset.py [--repo DIR] [--out build/dataset] [--parquet]

One row per (run date, coin) for every coin that had features that day. Indicators are as of the close before
the run date (the information the decision had). Forward returns are from that close: fwd_{1,3,7,14}d (absolute)
and fwd_rel_{h}d (minus BTC); empty until the window has closed. Columns in_universe / in_top10 / llm_ok and the
version tag + model identify the segment (results of different versions/models must not be mixed).
"""
import argparse
import csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=ROOT)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "dataset"))
    ap.add_argument("--parquet", action="store_true")
    a = ap.parse_args()
    data = rows(a.repo)
    os.makedirs(a.out, exist_ok=True)
    if not data:
        print("žádná data")
        return
    cols = list(data[0].keys())
    p = os.path.join(a.out, "panel.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(data)
    print(f"{len(data)} řádků -> {p}")
    if a.parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise SystemExit("pro Parquet: pip install -r requirements.txt (pyarrow)")
        pq.write_table(pa.Table.from_pylist(data), os.path.join(a.out, "panel.parquet"))
        print("-> panel.parquet")


if __name__ == "__main__":
    main()
