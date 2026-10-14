#!/usr/bin/env bash
# Install the dashboards portal on this server (needs sudo; idempotent).
#   - /srv/portal/releases/<ts>/index.html + symlink /srv/portal/current
#   - patches /etc/caddy/Caddyfile (backup /etc/caddy/Caddyfile.bak-portal), validates, reloads Caddy
# Layout afterwards:  /  portal shell | /crypto/ cpb | /lab/ research lab | /evening/ evening bot (404 until it exists)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF=/etc/caddy/Caddyfile
OWNER="${SUDO_USER:-$USER}"

TS="$(date -u +%Y%m%dT%H%M%S)"
sudo install -d -o "$OWNER" -g "$OWNER" -m 755 /srv/portal /srv/portal/releases "/srv/portal/releases/$TS"
install -m 644 "$HERE/index.html" "/srv/portal/releases/$TS/index.html"
ln -sfn "releases/$TS" /srv/portal/current.new && mv -T /srv/portal/current.new /srv/portal/current

TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
set +e; sudo cat "$CF" | python3 -I "$HERE/patch_caddyfile.py" /dev/stdin "$TMP"; rc=${PIPESTATUS[1]}; set -e
if [ "$rc" = 3 ]; then
  echo "Caddyfile už je upravený, mění se jen stránka portalu."
elif [ "$rc" = 0 ]; then
  sudo cp "$CF" "$CF.bak-portal"
  sudo install -m 644 -o root -g root "$TMP" "$CF"
  if ! sudo caddy validate --config "$CF" --adapter caddyfile; then
    echo "Validace selhala, vracím zálohu."; sudo cp "$CF.bak-portal" "$CF"; exit 1
  fi
  sudo systemctl reload caddy
else
  echo "Úprava Caddyfile selhala (rc=$rc), nic se nezměnilo."; exit 1
fi
echo "Hotovo: https://$(curl -fsS -m 10 https://api.ipify.org || echo '<server ip>')/"
