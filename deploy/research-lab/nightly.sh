#!/usr/bin/env bash
# Nightly: JSON export of lab.db, then commit what the lab produced (never framework code) and push.
set -euo pipefail
cd /srv/research-lab/app
lab export docs/research-lab/export
git add lab/hypotheses lab/strategies lab/knowledge lab/data docs/research-lab/export
if git diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
git commit -q -m "Research lab nightly $(date -u +%Y-%m-%d): cards, strategies, knowledge, export"
git push -q
