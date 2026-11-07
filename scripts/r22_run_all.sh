#!/usr/bin/env bash
# Full-budget search queue for one run of research 22 (needs docs/research22/run_lock.json).
#   scripts/r22_run_all.sh A [parallel=3] [cache_mb=450]      DRY_RUN=1 prints the queue only
# Queue: GA seeds 1-10, random search 3 x 110 000 (merged later, N2 needs 300 000 distinct), nulls 0-49.
# A job whose output already exists is skipped, so the script can be restarted after a crash or a reboot.
set -u
run=${1:?run A or B}; par=${2:-3}; cache=${3:-450}
root=runs/r22/$run
mkdir -p "$root/logs"
queue() {
  for s in $(seq 1 10); do echo "ga $s"; done
  for s in 1 2 3; do echo "random $s"; done
  for i in $(seq 0 49); do echo "null $i"; done
}
if [ "${DRY_RUN:-0}" = "1" ]; then queue; exit 0; fi
job() {
  mode=$1; id=$2
  if [ -f "$root/${mode}_${id}.json" ]; then echo "skip $mode $id"; return 0; fi
  case $mode in
    ga) extra="--seed $id" ;;
    random) extra="--seed $id --n 110000" ;;
    null) extra="--index $id" ;;
  esac
  echo "$(date -u +%FT%TZ) start $mode $id"
  nice -n 10 uv run python scripts/r22_search.py "$mode" --run "$run" $extra --cache-mb "$cache" \
    > "$root/logs/${mode}_${id}.log" 2>&1 \
    && echo "$(date -u +%FT%TZ) done  $mode $id" || echo "$(date -u +%FT%TZ) FAILED $mode $id"
}
export -f job; export root run cache
queue | xargs -P "$par" -L1 bash -c 'job "$@"' _
echo "$(date -u +%FT%TZ) queue finished for run $run"
