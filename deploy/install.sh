#!/usr/bin/env bash
# crypto-paper-bot: installation on the VPS, step by step. Every step prints what it will do and waits for "y".
# Run as your admin user (with sudo):   bash deploy/install.sh [step ...]
# Steps: audit user repo env claude venv pin web tls fail2ban ufw timer   (no argument = all, in this order)
# Nothing is changed without confirmation. Re-running a step is safe (idempotent where possible).
set -euo pipefail

REPO_URL="git@github.com:BeeeDeee/trading_system.git"
BRANCH="crypto-paper-bot"
SVC_USER="cpb"
SVC_HOME="/home/$SVC_USER"
APP="$SVC_HOME/crypto-paper-bot"
WEB="/srv/cpb"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_IP="$(curl -fsS -m 10 https://api.ipify.org || echo 38.109.11.51)"

ask() { printf '\n\033[1m== %s\033[0m\n%s\n' "$1" "$2"; read -r -p "Provést? [y/N] " a; [ "$a" = "y" ] || [ "$a" = "Y" ]; }
as_svc() { sudo -u "$SVC_USER" -H bash -lc "$1"; }

step_audit() {
  echo "== Audit (jen čtení)"
  . /etc/os-release; echo "OS: $PRETTY_NAME"; free -h | head -2; df -h / | tail -1
  timedatectl | grep -E "Time zone|synchronized|NTP"
  echo "-- ufw:"; sudo ufw status verbose || true
  echo "-- sshd:"; sudo sshd -T | grep -Ei '^(port|passwordauthentication|permitrootlogin|kbdinteractiveauthentication|pubkeyauthentication) ' || true
  echo "-- naslouchající porty:"; sudo ss -tlnp | sed 1d
  echo "-- Docker publikované porty (obchází ufw!):"; docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null || true
  echo "-- API burz z této IP ($PUBLIC_IP):"
  for u in https://api.binance.com/api/v3/ping https://www.okx.com/api/v5/public/time https://api.kraken.com/0/public/Time https://api.exchange.coinbase.com/time https://api.coingecko.com/api/v3/ping; do
    printf '  %-45s %s\n' "$u" "$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$u")"; done
}

step_user() {
  ask "Uživatel služby" "Vytvořím systémového uživatele '$SVC_USER' (bez sudo, bez hesla, login jen pro sudo -u)." || return 0
  id "$SVC_USER" >/dev/null 2>&1 || sudo useradd -m -s /bin/bash "$SVC_USER"
  sudo passwd -l "$SVC_USER" >/dev/null
  sudo install -d -o "$SVC_USER" -g "$SVC_USER" -m 700 "$SVC_HOME/.ssh" "$SVC_HOME/.config" "$SVC_HOME/.config/cpb"
}

step_repo() {
  ask "Deploy klíč a klon repa" "Vygeneruji SSH klíč pro '$SVC_USER', vypíšu veřejnou část (přidejte ji v GitHubu: repo → Settings → Deploy keys → Allow write access), pak naklonuji větev $BRANCH do $APP." || return 0
  [ -f "$SVC_HOME/.ssh/id_ed25519" ] || as_svc "ssh-keygen -t ed25519 -N '' -C 'cpb-deploy@$(hostname)' -f ~/.ssh/id_ed25519 >/dev/null"
  as_svc "ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null; sort -u -o ~/.ssh/known_hosts ~/.ssh/known_hosts"
  echo; sudo cat "$SVC_HOME/.ssh/id_ed25519.pub"; echo
  read -r -p "Přidal(a) jste klíč jako deploy key s právem zápisu? [y/N] " a; [ "$a" = "y" ] || return 1
  as_svc "ssh -T -o BatchMode=yes git@github.com 2>&1 | head -1" || true
  [ -d "$APP/.git" ] || as_svc "git clone --branch $BRANCH --single-branch $REPO_URL $APP"
  as_svc "cd $APP && git config user.name 'crypto-paper-bot' && git config user.email 'cpb@$(hostname)' && git config push.default current"
}

step_env() {
  local F="$SVC_HOME/.config/cpb/env"
  ask "Tajné hodnoty" "Zapíšu $F (vlastník $SVC_USER, práva 600): COINGECKO_DEMO_KEY, CLAUDE_CODE_OAUTH_TOKEN, volitelně NTFY_TOPIC.
Enter = ponechat uloženou hodnotu, takže jednotlivé položky jde doplnit i později (stačí krok spustit znovu)." || return 0
  sudo install -d -o "$SVC_USER" -g "$SVC_USER" -m 700 "$SVC_HOME/.config" "$SVC_HOME/.config/cpb"
  cur() { sudo sh -c "[ -f '$F' ] && grep -E '^$1=' '$F' | tail -n1 | cut -d= -f2-" || true; }
  state() { [ -n "$(cur "$1")" ] && echo "uloženo" || echo "zatím prázdné"; }
  local CG TOK NT PTH
  read -r -s -p "CoinGecko Demo API klíč [$(state COINGECKO_DEMO_KEY)]: " CG; echo
  [ -z "$CG" ] && CG="$(cur COINGECKO_DEMO_KEY)"
  echo "Token Claude: jako SVŮJ uživatel spusťte v jiném terminálu 'claude setup-token' a výsledný token vložte sem."
  read -r -s -p "CLAUDE_CODE_OAUTH_TOKEN [$(state CLAUDE_CODE_OAUTH_TOKEN)]: " TOK; echo
  [ -z "$TOK" ] && TOK="$(cur CLAUDE_CODE_OAUTH_TOKEN)"
  read -r -p "ntfy téma [$(cur NTFY_TOPIC || true)] (prázdné = ponechat, '-' = vypnout): " NT
  if [ "$NT" = "-" ]; then NT=""; elif [ -z "$NT" ]; then NT="$(cur NTFY_TOPIC)"; fi
  PTH="$(cur PATH)"; [ -z "$PTH" ] && PTH="$SVC_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
  local tmp; tmp="$(mktemp)"; chmod 600 "$tmp"
  { echo "COINGECKO_DEMO_KEY=$CG"; echo "CLAUDE_CODE_OAUTH_TOKEN=$TOK"; [ -n "$NT" ] && echo "NTFY_TOPIC=$NT"; echo "PATH=$PTH"; } > "$tmp"
  sudo install -o "$SVC_USER" -g "$SVC_USER" -m 600 "$tmp" "$F"; rm -f "$tmp"
  echo "Uloženo. Stav: CoinGecko $(state COINGECKO_DEMO_KEY), Claude token $(state CLAUDE_CODE_OAUTH_TOKEN), ntfy $( [ -n "$NT" ] && echo zapnuto || echo vypnuto )."
  if [ -n "$CG" ]; then
    printf 'Test CoinGecko klíče: '
    curl -s -o /dev/null -m 15 -w '%{http_code}\n' -H "x-cg-demo-api-key: $CG" "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&per_page=1&page=1"
  fi
}

step_claude() {
  ask "Claude Code pro uživatele služby" "Nainstaluji Claude Code CLI do $SVC_HOME/.local/bin (oficiální instalátor) a ověřím neinteraktivní běh s tokenem." || return 0
  as_svc "command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash"
  as_svc "set -a; . ~/.config/cpb/env; set +a; export PATH=\$HOME/.local/bin:\$PATH; cd /tmp && claude -p 'Reply with exactly: pong' --model claude-opus-5-5 --tools '' --output-format json --no-session-persistence | head -c 300; echo"
}

step_venv() {
  ask "Python venv" "Vytvořím $APP/.venv (engine je bez závislostí; pyarrow jen pro export do Parquetu)." || return 0
  sudo apt-get install -y python3-venv >/dev/null
  as_svc "cd $APP && python3 -m venv .venv && .venv/bin/pip -q install -r requirements.txt || echo 'pyarrow se nenainstaloval – export jen CSV'"
  as_svc "cd $APP && .venv/bin/python -m unittest discover -s tests -q"
}

step_pin() {
  ask "Připnutí verze" "Ověřím MANIFEST a zapíšu připnutí $SVC_HOME/.config/cpb/pin.json pro aktuální release (tag z VERSION)." || return 0
  as_svc "cd $APP && .venv/bin/python tools/verify_manifest.py --no-pin && .venv/bin/python tools/pin.py --install"
}

step_web() {
  ask "Webserver Caddy + heslo" "Nainstaluji Caddy (oficiální apt repo), web root $WEB (vlastník $SVC_USER, Caddy jen čte), basic auth s bcrypt hashem." || return 0
  if ! command -v caddy >/dev/null; then
    sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gnupg >/dev/null
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
    sudo apt-get update >/dev/null && sudo apt-get install -y caddy >/dev/null
  fi
  caddy version
  sudo install -d -o "$SVC_USER" -g "$SVC_USER" -m 755 "$WEB" "$WEB/releases"
  read -r -p "Uživatelské jméno pro dashboard: " WU
  echo "Heslo (min. 16 znaků; doporučuji z generátoru):"; HASH="$(caddy hash-password)"
  [ -n "$HASH" ] || return 1
  echo "$WU" | sudo tee /etc/caddy/cpb-user >/dev/null; echo "$HASH" | sudo tee /etc/caddy/cpb-hash >/dev/null
  sudo chmod 640 /etc/caddy/cpb-user /etc/caddy/cpb-hash; sudo chown root:caddy /etc/caddy/cpb-user /etc/caddy/cpb-hash
  as_svc "cd $APP && .venv/bin/python engine/engine.py report --publish $WEB"
}

render_caddy() {  # $1 site address, $2 tls block
  local email; read -r -p "E-mail pro Let's Encrypt (upozornění na expiraci): " email
  sed -e "s|{{SITE}}|$1|" -e "s|{{TLS}}|$2|" -e "s|{{ACME_EMAIL}}|$email|" \
      -e "s|{{USER}}|$(sudo cat /etc/caddy/cpb-user)|" -e "s|{{BCRYPT_HASH}}|$(sudo cat /etc/caddy/cpb-hash)|" \
      "$HERE/Caddyfile.tmpl" | sudo tee /etc/caddy/Caddyfile >/dev/null
  sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
  sudo systemctl reload caddy 2>/dev/null || sudo systemctl restart caddy
  if ! systemctl is-active --quiet caddy; then
    echo "Caddy neběží, poslední hlášky:"; sudo journalctl -u caddy -n 20 --no-pager; return 1
  fi
}

step_tls() {
  echo; echo "HTTPS bez domény – varianty:"
  echo "  1) Let's Encrypt certifikát přímo na IP $PUBLIC_IP (krátkodobý profil, Caddy obnovuje sám) – doporučeno"
  echo "  2) $(echo "$PUBLIC_IP" | tr . -).sslip.io – veřejný DNS záznam ukazující na IP, klasický LE certifikát"
  echo "  3) samopodepsaný certifikát (prohlížeč varuje; otisk si ověřte ručně) – jen nouzově"
  read -r -p "Volba [1/2/3, jiné = přeskočit]: " c
  case "$c" in
    1) render_caddy "https://$PUBLIC_IP" "tls {\n\t\tissuer acme {\n\t\t\tdir https://acme-v02.api.letsencrypt.org/directory\n\t\t\tprofile shortlived\n\t\t}\n\t}" ;;
    2) render_caddy "https://$(echo "$PUBLIC_IP" | tr . -).sslip.io" "" ;;
    3) render_caddy "https://$PUBLIC_IP" "tls internal" ;;
    *) return 0 ;;
  esac
  sleep 15
  echo "Test (má být 401 bez hesla):"; curl -sS -o /dev/null -w '%{http_code} %{ssl_verify_result}\n' -m 20 "https://$PUBLIC_IP/" || true
}

step_fail2ban() {
  ask "fail2ban" "Nainstaluji fail2ban s jailem na neúspěšná přihlášení k dashboardu (5 pokusů / 10 min → ban 1 h) a ponechám výchozí jail sshd." || return 0
  sudo apt-get install -y fail2ban >/dev/null
  sudo install -m 644 "$HERE/fail2ban/caddy-cpb-auth.conf" /etc/fail2ban/filter.d/caddy-cpb-auth.conf
  sudo install -m 644 "$HERE/fail2ban/cpb.local" /etc/fail2ban/jail.d/cpb.local
  sudo systemctl enable --now fail2ban && sudo systemctl restart fail2ban
  for i in 1 2 3 4 5 6 7 8 9 10; do sudo fail2ban-client ping >/dev/null 2>&1 && break; sleep 1; done
  sudo fail2ban-client status
  sudo fail2ban-client status caddy-cpb-auth
}

step_ufw() {
  echo "Současný stav:"; sudo ufw status verbose || true
  ask "Firewall ufw" "Povolím jen 22/tcp (SSH), 80/tcp a 443/tcp, výchozí příchozí = deny, a zapnu ufw. POZOR: porty publikované Dockerem (n8n :5678) ufw neblokuje – řeší se zvlášť." || return 0
  sudo ufw default deny incoming; sudo ufw default allow outgoing
  sudo ufw allow 22/tcp; sudo ufw allow 80/tcp; sudo ufw allow 443/tcp
  sudo ufw --force enable; sudo ufw status verbose
}

step_timer() {
  ask "systemd timery" "Nainstaluji cpb-daily (00:20 UTC, Persistent=true, opakování 06:20 a 12:20 jen když den chybí) a cpb-hourly (každou hodinu v HH:02, bez doplňování zmeškaných hodin) a oba zapnu." || return 0
  sudo install -m 644 "$HERE/cpb-daily.service" "$HERE/cpb-daily.timer" "$HERE/cpb-hourly.service" "$HERE/cpb-hourly.timer" /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now cpb-daily.timer cpb-hourly.timer
  systemctl list-timers 'cpb-*'
}

step_hourly_only() {
  echo "Jen hodinový timer (např. po prvním denním běhu):"
  sudo install -m 644 "$HERE/cpb-hourly.service" "$HERE/cpb-hourly.timer" /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now cpb-hourly.timer
}

STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(audit user repo env claude venv pin web tls fail2ban ufw timer)
for s in "${STEPS[@]}"; do "step_$s"; done
