#!/usr/bin/env bash
# Daily run of the evening paper-trading bot (started by evening-daily.timer after the US close; the retries are
# no-ops once the day is done). Manual use:
#   ./run_daily.sh              today's run + dashboard publish + commit + notification
#   ./run_daily.sh --dry-run    everything on a copy of the data dir: no commit, no publish, no notification
#   ./run_daily.sh --date 2026-10-09 --dry-run --force
set -uo pipefail
export LC_ALL=C.UTF-8
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${EVENING_ENV:-$HOME/.config/evening/env}"
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
export EVENING_DATA="${EVENING_DATA:-$HOME/evening-data}"
WEB="${EVENING_PUBLISH_DIR:-}"
DRY=0; for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done

mkdir -p "$EVENING_DATA"
exec 9>"$EVENING_DATA/.run.flock"
flock -n 9 || { echo "jiný běh právě probíhá"; exit 0; }
cd "$HERE"

TEXT="$(timeout --signal=TERM --kill-after=60 "${EVENING_RUN_TIMEOUT:-75m}" python3 -m evening run "$@")"
RC=$?
echo "$TEXT"
[ "$RC" = 124 ] && echo "CHYBA: celkový timeout běhu" >&2
[ "$DRY" = 1 ] && exit "$RC"

if [ -n "$WEB" ]; then
  python3 -m evening report --publish "$WEB" || echo "UPOZORNĚNÍ: dashboard se nevygeneroval" >&2
fi
# notify only when this invocation did something (not for "už existuje" nor a planned retry while waiting for Nasdaq)
if [ -n "${NTFY_TOPIC:-}" ] && ! grep -qE "už existuje|další pokus" <<<"$TEXT"; then
  curl -fsS -m 20 -H "Title: $(head -n1 <<<"$TEXT")" -d "$(tail -n +2 <<<"$TEXT")" "${NTFY_SERVER:-https://ntfy.sh}/$NTFY_TOPIC" >/dev/null || true
fi
exit "$RC"
