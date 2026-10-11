#!/usr/bin/env bash
# evening paper-trading bot on this server, step by step. Every step prints what it will do and waits for "y".
# Run as your admin user (with sudo):   bash server/deploy/install.sh [step ...]
# Steps: user repo env claude seed test web timer   (no argument = all, in this order)
# Nothing is changed without confirmation. Re-running a step is safe (idempotent where possible).
#
# Environment overrides:  REPO_URL (default: this checkout, no keys needed; switch to GitHub after pushing the branch)
#                         BRANCH   (default evening-bot-server)   SEED (default ~/ccode/evening_seed_*.tgz, newest)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_REPO="$(cd "$HERE/../.." && pwd)"
REPO_URL="${REPO_URL:-$SRC_REPO}"
BRANCH="${BRANCH:-evening-bot-server}"
SVC_USER="evb"
SVC_HOME="/home/$SVC_USER"
APP="$SVC_HOME/evening-bot"
DATA="$SVC_HOME/evening-data"
WEB="/srv/evening"
SEED="${SEED:-$(ls -1t "$HOME"/ccode/evening_seed_*.tgz 2>/dev/null | head -1 || true)}"

ask() { printf '\n\033[1m== %s\033[0m\n%s\n' "$1" "$2"; read -r -p "Provést? [y/N] " a; [ "$a" = "y" ] || [ "$a" = "Y" ]; }
as_svc() { (cd / && sudo -u "$SVC_USER" -H bash -lc "$1"); }

step_user() {
  ask "Uživatel služby" "Vytvořím systémového uživatele '$SVC_USER' (bez sudo, bez hesla) a adresáře $WEB (web root) a $DATA." || return 0
  id "$SVC_USER" >/dev/null 2>&1 || sudo useradd -m -s /bin/bash "$SVC_USER"
  sudo passwd -l "$SVC_USER" >/dev/null
  sudo install -d -o "$SVC_USER" -g "$SVC_USER" -m 700 "$SVC_HOME/.config" "$SVC_HOME/.config/evening"
  sudo install -d -o "$SVC_USER" -g "$SVC_USER" -m 755 "$WEB" "$WEB/releases"
}

step_repo() {
  ask "Klon repa" "Naklonuji větev $BRANCH z $REPO_URL do $APP (uživatel $SVC_USER). Pozdější aktualizace: sudo -u $SVC_USER git -C $APP pull." || return 0
  # evb cannot read /home/<admin>; for a local path, publish a world-readable bare copy of the committed branch
  if [ -d "$REPO_URL" ]; then
    git -C "$REPO_URL" diff --quiet HEAD -- server 2>/dev/null && [ -z "$(git -C "$REPO_URL" status --porcelain server)" ] || { echo "V $REPO_URL jsou necommitnuté změny v server/ – nejdřív commit."; return 1; }
    local MIRROR=/var/tmp/evening-bot-src.git
    sudo rm -rf "$MIRROR" && git clone -q --bare --branch "$BRANCH" --single-branch "$REPO_URL" "$MIRROR" && sudo chmod -R a+rX "$MIRROR"
    REPO_URL="$MIRROR"
    as_svc "git config --global --add safe.directory '$MIRROR'"
  fi
  if [ -d "$APP/.git" ]; then as_svc "git -C $APP pull -q '$REPO_URL' $BRANCH"; else as_svc "git clone --branch $BRANCH --single-branch '$REPO_URL' $APP"; fi
  as_svc "cd $APP && git log --oneline -1 && python3 tools/verify_manifest.py"
}

step_env() {
  local F="$SVC_HOME/.config/evening/env"
  ask "Tajné hodnoty" "Zapíšu $F (vlastník $SVC_USER, práva 600): CLAUDE_CODE_OAUTH_TOKEN, volitelně NTFY_TOPIC a EVENING_MODEL.
Můžu je zkopírovat z crypto bota (/home/cpb/.config/cpb/env, stejné předplatné), nebo zadat ručně." || return 0
  local TOK="" NT="" MODEL=""
  read -r -p "Zkopírovat token a ntfy téma z crypto bota? [Y/n] " c
  if [ "$c" != "n" ] && [ "$c" != "N" ]; then
    TOK="$(sudo grep -E '^CLAUDE_CODE_OAUTH_TOKEN=' /home/cpb/.config/cpb/env | tail -n1 | cut -d= -f2- || true)"
    NT="$(sudo grep -E '^NTFY_TOPIC=' /home/cpb/.config/cpb/env | tail -n1 | cut -d= -f2- || true)"
  fi
  if [ -z "$TOK" ]; then
    echo "Token Claude: jako SVŮJ uživatel spusťte v jiném terminálu 'claude setup-token' a výsledek vložte sem."
    read -r -s -p "CLAUDE_CODE_OAUTH_TOKEN: " TOK; echo; TOK="$(printf '%s' "$TOK" | tr -d '[:space:]')"
  fi
  [ "${TOK#sk-ant-}" != "$TOK" ] || { echo "Token nezačíná 'sk-ant-', neukládám."; return 1; }
  read -r -p "Model pro skórování [claude-opus-5-5] (Enter = výchozí; musí odpovídat modelu naplánované úlohy v Claude): " MODEL
  local tmp; tmp="$(mktemp)"; chmod 600 "$tmp"
  { echo "CLAUDE_CODE_OAUTH_TOKEN=$TOK"; [ -n "$NT" ] && echo "NTFY_TOPIC=$NT"; [ -n "$MODEL" ] && echo "EVENING_MODEL=$MODEL"
    echo "PATH=$SVC_HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"; } > "$tmp"
  sudo install -o "$SVC_USER" -g "$SVC_USER" -m 600 "$tmp" "$F"; rm -f "$tmp"
  echo "Uloženo."
}

step_claude() {
  ask "Claude Code pro uživatele služby" "Nainstaluji Claude Code CLI do $SVC_HOME/.local/bin a ověřím neinteraktivní běh s tokenem." || return 0
  as_svc "command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash"
  as_svc "set -a; . ~/.config/evening/env; set +a; export PATH=\$HOME/.local/bin:\$PATH; cd /tmp && claude -p 'Reply with exactly: pong' --model \${EVENING_MODEL:-claude-opus-5-5} --tools '' --output-format json --no-session-persistence | head -c 300; echo"
}

step_seed() {
  [ -n "$SEED" ] && [ -f "$SEED" ] || { echo "Seed archiv nenalezen (SEED=...)."; return 1; }
  ask "Počáteční stav" "Rozbalím $SEED do $DATA (days, trades, scores, prices, runs, history.json = export databáze artefaktu) a ověřím, že engine
na serveru reprodukuje všechny uložené dny (tools/replay_check.py). Existující data se nepřepisují (krok se přeskočí, když $DATA/days existuje)." || return 0
  if sudo test -d "$DATA/days"; then echo "$DATA/days už existuje, přeskakuji."; else
    sudo install -d -o "$SVC_USER" -g "$SVC_USER" -m 755 "$DATA"
    sudo tar -xzf "$SEED" -C "$DATA" --no-same-owner && sudo chown -R "$SVC_USER:$SVC_USER" "$DATA"
  fi
  as_svc "cd $APP/server && python3 tools/replay_check.py $DATA"
}

step_test() {
  ask "Zkušební běh (dry-run)" "Spustí celý běh pro poslední obchodní den na KOPII dat (nic se nezapíše ani nepublikuje, nepošle se notifikace). Použije skutečný claude -p (spotřebuje trochu limitu), trvá ~20 min.
Datum: ${TEST_DATE:-poslední obchodní den bez uloženého záznamu je 'dnes'; zadejte TEST_DATE=YYYY-MM-DD pro jiný}." || return 0
  as_svc "cd $APP/server && ./run_daily.sh --dry-run --force ${TEST_DATE:+--date $TEST_DATE}"
}

step_web() {
  ask "Dashboard" "Publikuji dashboard do $WEB (Caddy ho už servíruje na /evening/ za stejným heslem, pokud jste pustili portal/install.sh)." || return 0
  as_svc "cd $APP/server && EVENING_DATA=$DATA python3 -m evening report --publish $WEB"
  echo "Hotovo: https://$(curl -fsS -m 10 https://api.ipify.org || echo '<server ip>')/#evening"
}

step_timer() {
  ask "systemd timery" "Nainstaluji evening-daily (po–pá 16:30 New York + opakování 17:30 a 19:00, Persistent=true) a zapnu ho.
POZOR: dokud běží naplánovaná úloha v Claude aplikaci, server NEMÁ běžet naostro (dvě instance by se rozešly). Tento krok spusťte až při přechodu." || return 0
  sudo install -m 644 "$HERE/evening-daily.service" "$HERE/evening-daily.timer" /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now evening-daily.timer
  systemctl list-timers 'evening-*'
}

STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(user repo env claude seed test web)
for s in "${STEPS[@]}"; do "step_$s"; done
