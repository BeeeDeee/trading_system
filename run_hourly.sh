#!/usr/bin/env bash
# Hourly data run (cpb-hourly.timer, HH:02 UTC): hourly candles, order book, futures sentiment and the 7 hourly
# strategies for the hour that just closed. No LLM call. Missed hours are not caught up.
#   ./run_hourly.sh             run the current hour + fast dashboard refresh
#   ./run_hourly.sh --dry-run   in a temp copy, nothing written to the repo or the web root
# Commits happen in run_daily.sh (once a day); this script never pushes.
set -uo pipefail
export TZ=UTC LC_ALL=C.UTF-8
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${CPB_ENV:-$HOME/.config/cpb/env}"
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
PY="$REPO/.venv/bin/python"; [ -x "$PY" ] || PY=python3
cd "$REPO"
mkdir -p data
if [ "${1:-}" = "--dry-run" ]; then
  exec timeout 10m "$PY" engine/engine.py hourly --dry-run --allow-unpinned
fi
# the daily commit takes this lock too, so git never sees a half-finished hourly run
exec 8>"$REPO/data/.hourly.flock"
flock -w 120 8 || { echo "hourly: zámek obsazený"; exit 0; }
timeout --kill-after=30 10m "$PY" engine/engine.py hourly
RC=$?
flock -u 8
exec 7>"$REPO/data/.report.flock"
flock -w 300 7 && "$PY" engine/engine.py report --fast ${CPB_PUBLISH_DIR:+--publish "$CPB_PUBLISH_DIR"} >/dev/null || echo "UPOZORNĚNÍ: rychlý dashboard se nevygeneroval" >&2
exit "$RC"
