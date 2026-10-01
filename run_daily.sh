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

exec 7>"$REPO/data/.report.flock"
flock -w 600 7 && "$PY" engine/engine.py report ${CPB_PUBLISH_DIR:+--publish "$CPB_PUBLISH_DIR"} || echo "UPOZORNĚNÍ: dashboard se nevygeneroval" >&2
flock -u 7

# commit while no hourly run is writing (it holds data/.hourly.flock for ~10 s)
exec 8>"$REPO/data/.hourly.flock"
flock -w 300 8 || echo "UPOZORNĚNÍ: hodinový zámek obsazený, commit i tak" >&2

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
# only paths that exist (git add aborts entirely on a missing pathspec, e.g. corrections/ before the first correction)
for p in runs data universe corrections public; do
  [ -e "$p" ] && { git add -A -- "$p" || echo "UPOZORNĚNÍ: git add $p selhal" >&2; }
done
# hourly records first, as one commit per day of hourly runs
mapfile -t HDAYS < <(git diff --cached --name-only --diff-filter=A -- 'runs/*/hourly/*/run.json' | cut -d/ -f2 | sort -u)
for d in "${HDAYS[@]}"; do
  n=$(git diff --cached --name-only --diff-filter=A -- "runs/$d/hourly/*/run.json" | wc -l)
  git commit -q -m "hourly $d: $n hodinových záznamů" -- "runs/$d/hourly" || true
done
mapfile -t RECS < <(git diff --cached --name-only --diff-filter=A -- 'runs/*run.json' | grep -v '/hourly/' | sort)
for i in "${!RECS[@]}"; do
  f="${RECS[$i]}"; d="$(dirname "$f")"
  if [ "$i" -lt $(( ${#RECS[@]} - 1 )) ]; then
    git commit -q -m "$(msg "$f")" -- "$d" || true
  else
    git commit -q -m "$(msg "$f")" || true
  fi
done
git diff --cached --quiet || git commit -q -m "data $(date -u +%FT%TZ)"
# push (still holding the hourly lock: a rejected push is followed by a rebase that touches the work tree).
# If the branch moved on GitHub (e.g. a release commit), replay our data commits on top of it; data and code never
# touch the same files. New code pulled this way only runs after it is pinned (pin check), never silently.
PUSHED=0
for t in 1 2 3; do
  git push -q origin "HEAD:$BRANCH" && { PUSHED=1; break; }
  git pull -q --rebase --autostash origin "$BRANCH" || { git rebase --abort 2>/dev/null; echo "UPOZORNĚNÍ: rebase selhal" >&2; }
  sleep $((t * 10))
done
flock -u 8
[ "$PUSHED" = 1 ] && echo "push: $(git log --oneline -1)" || echo "UPOZORNĚNÍ: push selhal" >&2

TEXT="$("$PY" engine/engine.py notify --date "$TODAY")"   # the run itself already printed it to the journal
if [ -n "${NTFY_TOPIC:-}" ]; then
  curl -fsS -m 20 -H "Title: $(head -n1 <<<"$TEXT")" -d "$(tail -n +2 <<<"$TEXT")" "${NTFY_SERVER:-https://ntfy.sh}/$NTFY_TOPIC" >/dev/null || true
fi
exit "$RC"
