#!/usr/bin/env bash
# Nasazení arena5m na server (sudo, idempotentní). SPEC §15, D7–D9.
#   - uživatel `arena`, /srv/arena{,/site,/site/data,/raw,/log}, /etc/arena/env (mode 600, šablona bez klíčů)
#   - kód do /opt/arena5m, venv se systémovým python3.12 (uživatel arena nemá přístup do home)
#   - systemd: arena.service (Restart=always), arena-book-validate.timer (denně 03:30 UTC)
#   - Caddy: /arena/* s basic_auth (záloha /etc/caddy/Caddyfile.bak-arena, validace, reload)
#   - záložka „Arena“ v portálu (PORTAL_DIR, default ~/ccode/portal), když portál existuje
# Proměnné: PORTAL_DIR, ARENA_AUTH_USER + ARENA_AUTH_HASH (jen když web nemá vlastní basic_auth), NO_START=1
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
CF=/etc/caddy/Caddyfile
OWNER="${SUDO_USER:-$USER}"
PORTAL_DIR="${PORTAL_DIR:-/home/$OWNER/ccode/portal}"
UV="$(command -v uv || echo "/home/$OWNER/.local/bin/uv")"
PY=/usr/bin/python3.12

id arena >/dev/null 2>&1 || sudo useradd --system --home-dir /srv/arena --shell /usr/sbin/nologin arena
sudo install -d -o arena -g arena -m 751 /srv/arena
sudo install -d -o arena -g arena -m 755 /srv/arena/site /srv/arena/site/data
sudo install -d -o arena -g arena -m 750 /srv/arena/raw /srv/arena/log
sudo install -d -o root -g arena -m 750 /etc/arena
if ! sudo test -f /etc/arena/env; then
  printf '%s\n' "# arena5m klíče (D9): Ed25519, jen čtení + futures, bez výběrů, IP whitelist" \
    "#BINANCE_API_KEY=" "#BINANCE_PRIVATE_KEY_PATH=/etc/arena/binance_ed25519.pem" \
    "#BINANCE_TESTNET_API_KEY=" "#BINANCE_TESTNET_PRIVATE_KEY_PATH=/etc/arena/testnet_ed25519.pem" \
    "#NTFY_TOPIC=" | sudo tee /etc/arena/env >/dev/null
fi
sudo chown arena:arena /etc/arena/env && sudo chmod 600 /etc/arena/env

# kód + venv (systémový python, aby k němu měl uživatel arena přístup)
sudo install -d -m 755 /opt/arena5m
sudo chown -R "$OWNER:$OWNER" /opt/arena5m      # opakovaný běh: na konci se vrací root
rsync -a --delete --exclude .venv --exclude .git --exclude __pycache__ --exclude .pytest_cache \
  --exclude 'tests/data' "$REPO/" /opt/arena5m/
"$UV" venv --quiet --allow-existing --python "$PY" /opt/arena5m/.venv
"$UV" pip install --quiet --python /opt/arena5m/.venv/bin/python /opt/arena5m
sudo chown -R root:root /opt/arena5m && sudo chmod -R go-w /opt/arena5m

# dashboard (statické soubory; data/ zapisuje proces)
for f in index.html app.js style.css; do sudo install -o arena -g arena -m 644 "$REPO/dashboard/$f" "/srv/arena/site/$f"; done

# systemd
for u in arena.service arena-book-validate.service arena-book-validate.timer; do
  sudo install -m 644 -o root -g root "$HERE/$u" "/etc/systemd/system/$u"
done
sudo systemctl daemon-reload
sudo systemctl enable --now arena-book-validate.timer
sudo systemctl enable arena.service
if [ "${NO_START:-0}" != 1 ]; then sudo systemctl restart arena.service; fi

# Caddy
TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
set +e; sudo cat "$CF" | sudo -E python3 -I "$HERE/patch_caddyfile_arena.py" /dev/stdin "$TMP"; rc=${PIPESTATUS[1]}; set -e
if [ "$rc" = 3 ]; then
  echo "Caddyfile už má /arena/."
elif [ "$rc" = 0 ]; then
  sudo cp "$CF" "$CF.bak-arena"
  sudo install -m 644 -o root -g root "$TMP" "$CF"
  if ! sudo caddy validate --config "$CF" --adapter caddyfile; then
    echo "Validace selhala, vracím zálohu."; sudo cp "$CF.bak-arena" "$CF"; exit 1
  fi
  sudo systemctl reload caddy
else
  echo "Úprava Caddyfile selhala (rc=$rc), nic se nezměnilo."; exit 1
fi

# záložka v portálu
if [ -f "$PORTAL_DIR/index.html" ]; then
  set +e; python3 -I "$HERE/patch_portal_arena.py" "$PORTAL_DIR/index.html"; rc=$?; set -e
  if [ "$rc" = 0 ] && [ -x "$PORTAL_DIR/install.sh" ]; then "$PORTAL_DIR/install.sh"; fi
  [ "$rc" = 3 ] && echo "Portál už záložku Arena má."
else
  echo "Portál ($PORTAL_DIR) nenalezen, záložka se nepřidala."
fi
echo "Hotovo. Stav: systemctl status arena; log: /srv/arena/log/arena.log; web: /arena/"
