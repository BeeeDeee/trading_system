#!/usr/bin/env bash
# Daily run of crypto-paper-bot (started by cpb-daily.timer at 00:20 UTC; retries at 06:20 and 12:20 are no-ops
# when the day is already done). Manual use:
#   ./run_daily.sh              today's run + report + commit + push + notification
#   ./run_daily.sh --dry-run    everything in a temp copy: no commit, no publish, no push
# Missed days are caught up automatically (settle + mark only, status "catchup").
set -uo pipefail
export TZ=UTC LC_ALL=C.UTF-8
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${CPB_ENV:-$HOME/.config/cpb/env}"
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
PY="$REPO/.venv/bin/python"; [ -x "$PY" ] || PY=python3
BRANCH="${CPB_BRANCH:-crypto-paper-bot}"
DRY=0; ARGS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1; ARGS+=(--dry-run --allow-unpinned) ;;
    *) ARGS+=("$a") ;;
  esac
done
TODAY="$(date -u +%F)"

mkdir -p "$REPO/data"
exec 9>"$REPO/data/.run.flock"
flock -n 9 || { echo "jiný běh právě probíhá"; exit 0; }
cd "$REPO"

if [ "$DRY" = 0 ] && [ -f "runs/$TODAY/run.json" ]; then
  echo "běh $TODAY už existuje, nic se nedělá"; exit 0
fi

timeout --signal=TERM --kill-after=60 "${CPB_RUN_TIMEOUT:-75m}" "$PY" engine/engine.py run "${ARGS[@]}"
RC=$?
[ "$RC" = 124 ] && echo "CHYBA: celkový timeout běhu" >&2
[ "$DRY" = 1 ] && exit "$RC"

"$PY" engine/engine.py report ${CPB_PUBLISH_DIR:+--publish "$CPB_PUBLISH_DIR"} || echo "UPOZORNĚNÍ: dashboard se nevygeneroval" >&2

# one commit per run record (catch-up days first), never force-push
msg() { "$PY" - "$1" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
s = {"ok": "OK", "warning": "UPOZORNĚNÍ", "catchup": "DOPLNĚNO", "failed": "CHYBA"}[r["status"]]
sm = r.get("summary") or {}
extra = f", {sm.get('n_variants', 0)} variant, {sm.get('n_trades_base', 0)} obchodů" if r["status"] in ("ok", "warning") else ""
if r["status"] == "failed":
    extra = ": " + (r.get("error") or "")[:120]
print(f"run {r['date']}: {s}{extra}")
PYEOF
}
git add -A runs data universe corrections public 2>/dev/null
mapfile -t RECS < <(git diff --cached --name-only --diff-filter=A -- 'runs/*run.json' | sort)
for i in "${!RECS[@]}"; do
  f="${RECS[$i]}"; d="$(dirname "$f")"
  if [ "$i" -lt $(( ${#RECS[@]} - 1 )) ]; then
    git commit -q -m "$(msg "$f")" -- "$d" || true
  else
    git commit -q -m "$(msg "$f")" || true
  fi
done
git diff --cached --quiet || git commit -q -m "data $(date -u +%FT%TZ)"
for t in 1 2 3; do git push -q origin "HEAD:$BRANCH" && break; sleep $((t * 20)); done || echo "UPOZORNĚNÍ: push selhal" >&2

TEXT="$("$PY" engine/engine.py notify --date "$TODAY")"
echo "$TEXT"
if [ -n "${NTFY_TOPIC:-}" ]; then
  curl -fsS -m 20 -H "Title: $(head -n1 <<<"$TEXT")" -d "$(tail -n +2 <<<"$TEXT")" "${NTFY_SERVER:-https://ntfy.sh}/$NTFY_TOPIC" >/dev/null || true
fi
exit "$RC"
