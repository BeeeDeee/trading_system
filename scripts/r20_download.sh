#!/usr/bin/env bash
# Daily raw snapshot of the research 20 ETFs (docs/research20/PREREGISTRATION.md §3.1), after the US close.
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs
exec 9>runs/r20_download.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) already running" >> runs/r20_download.log; exit 0; }
{
  echo "=== $(date -u +%FT%TZ) r20 download"
  /home/kapo/.local/bin/uv run python scripts/r20_download.py
  echo "exit $?"
} >> runs/r20_download.log 2>&1
