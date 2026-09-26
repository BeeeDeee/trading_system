#!/usr/bin/env bash
# Download full-history Sharadar bulk zips into an immutable raw snapshot.
# Key: $SHARADAR_API_KEY or ~/.config/sharadar/api_key (never commit it).
# Usage: scripts/download_sharadar.sh [snapshot_dir] [table ...]
set -uo pipefail

API=https://api.sharadar.com/v1.0
KEY=${SHARADAR_API_KEY:-$(cat ~/.config/sharadar/api_key 2>/dev/null)}
[ -n "$KEY" ] || { echo "missing API key" >&2; exit 1; }

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:-$ROOT/data/raw/sharadar_$(date -u +%Y-%m-%d)}
shift || true
TABLES=${*:-"descriptions tickers actions sp500 events metrics stocks funds daily fundamentals \
holdings_ticker holdings_investor insiders holdings"}
mkdir -p "$DEST"

zip_ok() { python3 -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).namelist()" "$1" >/dev/null 2>&1; }

fetch() {  # $1 = table; resumes partial downloads, refreshes the pre-signed URL on every attempt
  local t=$1 out="$DEST/$1.csv.zip" url
  for attempt in $(seq 1 40); do
    if [ -f "$out" ] && zip_ok "$out"; then
      echo "$(date -u +%T) $t ok ($(du -h "$out" | cut -f1))"; return 0
    fi
    url=$(curl -sS -o /dev/null -w '%{redirect_url}' --max-time 60 \
          -H "x-api-key: $KEY" "$API/data/$t?years=full" 2>/dev/null)
    if [ -n "$url" ]; then
      curl -sS -L -C - --retry 5 --retry-delay 5 --connect-timeout 30 \
           --speed-limit 1000 --speed-time 120 -o "$out.part" "$url" 2>/dev/null \
        && mv "$out.part" "$out"
    fi
    [ -f "$out" ] && ! zip_ok "$out" && { echo "$t corrupt, retrying"; rm -f "$out"; }
    sleep $((attempt < 10 ? 5 : 30))
  done
  echo "$(date -u +%T) $t FAILED"; return 1
}

rc=0
for t in $TABLES; do fetch "$t" || rc=1; done
(cd "$DEST" && sha256sum ./*.csv.zip > SHA256SUMS)
echo "done rc=$rc -> $DEST"
exit $rc
