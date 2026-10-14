#!/usr/bin/env bash
# Weekly forward update of research 17 (docs/research17/FORWARD.md §6): download + descriptive status.
# Does NOT commit; review and commit docs/research17/results/forward_<asof>.json by hand.
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs
exec 9>runs/r17_forward.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) already running" >> runs/r17_forward_weekly.log; exit 0; }
{
  echo "=== $(date -u +%FT%TZ) download"
  /home/kapo/.local/bin/uv run python scripts/r17_forward.py download || echo "download FAILED (exit $?)"
  echo "=== $(date -u +%FT%TZ) status"
  /home/kapo/.local/bin/uv run python scripts/r17_forward.py status | head -40 || echo "status FAILED (exit $?)"
} >> runs/r17_forward_weekly.log 2>&1
