#!/usr/bin/env python3
"""POST-HOC: replay an alternative config over the stored point-in-time inputs.

  python tools/replay.py --config alt.json [--repo DIR] [--out build/posthoc/NAME]

The LLM scores, prices and snapshots are the ones actually recorded; only the rules change. Results are
IN-SAMPLE (the rules were chosen after seeing the data), so they are never evidence of an edge and are never
mixed into the dashboard. The output folder and every file are labelled "post-hoc".
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
from cpb import analytics as A, canon, replay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo", default=ROOT)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = canon.read_json(a.config)
    name = os.path.splitext(os.path.basename(a.config))[0]
    out = a.out or os.path.join(ROOT, "build", "posthoc", name)
    probs, state, hist, series = replay.replay(a.repo, cfg_override=cfg, check=False)
    ledgers = [{"date": s["date"], "close": s["close"], "after": s["after"]} for s in series if s["close"]]
    met = A.metrics(ledgers, cfg["start_capital"])
    canon.write_json(os.path.join(out, "POSTHOC_results.json"),
                     {"label": "POST-HOC (in-sample) – nevyhodnocovat jako důkaz", "config": a.config,
                      "config_sha256": canon.sha256_file(a.config), "metrics": met, "ledgers": ledgers})
    print("POST-HOC (in-sample) – nevyhodnocovat jako důkaz")
    for pid, m in sorted(met.items(), key=lambda kv: -kv[1]["ret"]):
        print(f"  {pid:<16}{m['ret'] * 100:>+8.1f} %   stres {m['ret_stress'] * 100:>+7.1f} %")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
