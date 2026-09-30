#!/usr/bin/env python3
"""Verify the hash chain and that the ledger is reproducible from stored inputs.

  python tools/verify_chain.py [REPO]

1. every run/correction record: this_hash, prev_hash link, sha256 of every file of the run;
2. replay of all runs (phase A, decisions, fills) from runs/*/inputs.json with each run's own config copy:
   features, decisions, fills and ledgers must be byte-identical, and so must data/state.json and data/history.
Exit code 0 = OK.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
from cpb import chain, replay  # noqa: E402


def main(repo):
    probs = chain.verify(repo)
    n = len(chain.entries(repo))
    rp, state, hist, series = replay.replay(repo)
    probs += rp
    probs += replay.compare_final(repo, state, hist)
    if probs:
        print("\n".join("CHYBA: " + p for p in probs))
        return 1
    nh = len(replay.LAST_HOURLY[2])
    print(f"OK: řetězec {n} záznamů neporušen, replay {len(series)} denních a {nh} hodinových běhů reprodukuje ledger bajt po bajtu")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ROOT))
