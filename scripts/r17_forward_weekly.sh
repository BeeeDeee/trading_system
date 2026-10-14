#!/usr/bin/env bash
# Weekly forward update of research 17 (docs/research17/FORWARD.md §6): see scripts/r17_forward_weekly.py.
# Optional alert settings: ~/.config/r17-forward/env with NTFY_TOPIC=... (and NTFY_SERVER=...).
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs
exec 9>runs/r17_forward.lock
flock -n 9 || { echo "$(date -u +%FT%TZ) already running" >> runs/r17_forward_weekly.log; exit 0; }
ENVF="$HOME/.config/r17-forward/env"
[ -r "$ENVF" ] && { set -a; . "$ENVF"; set +a; }
{
  echo "=== $(date -u +%FT%TZ) weekly run"
  UV=/home/kapo/.local/bin/uv /home/kapo/.local/bin/uv run python scripts/r17_forward_weekly.py
  echo "exit $?"
} >> runs/r17_forward_weekly.log 2>&1
