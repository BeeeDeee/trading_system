#!/usr/bin/env bash
# Research lab: production install on this VPS (PLAN §5, step 4). Every step says what it will do and waits
# for "y". Run as the admin user (kapo, with sudo):   bash deploy/research-lab/install.sh [step ...]
# Steps (no argument = all, in this order):
#   audit users dirs uv repo venv acl migrate claude token sudoers check canaries timers
# Later steps (run by name): update (pull, sync, canaries, units), ops (lab-prod for the owner), web (dashboard)
# Re-running a step is safe. Nothing outside the lab is changed except: two users, /srv/research-lab,
# /etc/research-lab, /etc/sudoers.d/research-lab, two systemd timers, and read ACLs on the data directories.
set -euo pipefail

REPO_URL="git@github.com:BeeeDeee/trading_system.git"
BRANCH="research-lab"
CORE="labcore"           # the framework: lab.db, data, gates, orchestrator
AGENT="labagent"         # the LLM agents: their workspaces and the app code, nothing else
GROUP="labwork"          # shared group of the two, only for the workspaces
SRV="/srv/research-lab"
APP="$SRV/app"
HOME_DIR="$SRV/home"
WS="$SRV/workspaces"
DEV="/home/kapo/ccode/research_lab"
DATA_DIRS=("/home/kapo/ccode/strategy_backtester_2026_sep/data" "/home/kapo/ccode/market_relations_2026_oct/data")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ask() { printf '\n\033[1m== %s\033[0m\n%s\n' "$1" "$2"; read -r -p "Proceed? [y/N] " a; [ "$a" = "y" ] || [ "$a" = "Y" ]; }
as_core() { sudo -u "$CORE" -H bash -lc "$1"; }
as_agent() { sudo -u "$AGENT" -H bash -lc "$1"; }

step_audit() {
  echo "== Audit (read only)"; free -h | head -2; df -h / | tail -1
  for u in "$CORE" "$AGENT"; do id "$u" 2>/dev/null || echo "user $u: missing"; done
  command -v bwrap setfacl uv || true
  systemctl list-timers 'cpb-*' 'research-lab-*' --no-pager || true
}

step_users() {
  ask "Users" "Create system users '$CORE' (framework) and '$AGENT' (agents), no passwords, no sudo, and the group
'$GROUP' with both as members (only the workspaces belong to it)." || return 0
  getent group "$GROUP" >/dev/null || sudo groupadd "$GROUP"
  for u in "$CORE" "$AGENT"; do
    id "$u" >/dev/null 2>&1 || sudo useradd -m -s /bin/bash "$u"
    sudo passwd -l "$u" >/dev/null; sudo usermod -aG "$GROUP" "$u"; sudo chmod 750 "/home/$u"
  done
}

step_dirs() {
  ask "Directories" "Create $SRV: app (labcore, readable by all: code only), home (labcore 0700: lab.db, data,
transcripts), workspaces (labcore:$GROUP 2770). Create /etc/research-lab (root:labcore 0750)." || return 0
  sudo install -d -o "$CORE" -g "$CORE" -m 755 "$SRV" "$APP"
  sudo install -d -o "$CORE" -g "$CORE" -m 700 "$HOME_DIR"
  sudo install -d -o "$CORE" -g "$GROUP" -m 2770 "$WS"
  sudo install -d -o root -g "$CORE" -m 750 /etc/research-lab
}

step_uv() {
  ask "uv" "Install the uv binary to /usr/local/bin (copy of $(command -v uv || echo '?'))." || return 0
  [ -x /usr/local/bin/uv ] || sudo install -m 755 "$(command -v uv)" /usr/local/bin/uv
  sudo apt-get install -y acl bubblewrap >/dev/null
}

step_repo() {
  ask "Deploy key and clone" "Generate an SSH key for '$CORE', print its public part (add it in GitHub: repo ->
Settings -> Deploy keys -> Allow write access; the nightly job pushes cards and exports), then clone branch
$BRANCH into $APP." || return 0
  as_core "mkdir -p ~/.ssh && chmod 700 ~/.ssh; [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -C 'labcore@$(hostname)' -f ~/.ssh/id_ed25519 >/dev/null"
  as_core "ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null; sort -u -o ~/.ssh/known_hosts ~/.ssh/known_hosts"
  echo; sudo cat "/home/$CORE/.ssh/id_ed25519.pub"; echo
  read -r -p "Deploy key added with write access? [y/N] " a; [ "$a" = "y" ] || return 1
  [ -d "$APP/.git" ] || as_core "git clone --branch $BRANCH --single-branch $REPO_URL $APP"
  as_core "cd $APP && git config user.name 'research-lab' && git config user.email 'labcore@$(hostname)' && git pull -q"
}

step_venv() {
  ask "Python environment" "uv sync in $APP (Python 3.12, as $CORE), then the lab test suite." || return 0
  as_core "cd $APP && UV_PYTHON_INSTALL_DIR=/srv/research-lab/app/.python uv sync -q && uv run pytest lab/tests -q -x 2>&1 | tail -3"
  sudo chmod -R a+rX "$APP"
}

step_acl() {
  ask "Read access to the data" "Give '$CORE' (not '$AGENT') traverse rights on /home/kapo and read rights on:
${DATA_DIRS[*]}
(ACLs, nothing is copied or moved; the sibling projects keep working)." || return 0
  sudo setfacl -m "u:$CORE:x" /home/kapo /home/kapo/ccode
  for d in "${DATA_DIRS[@]}"; do
    sudo setfacl -m "u:$CORE:x" "$(dirname "$d")"
    sudo setfacl -R -m "u:$CORE:rX" "$d"; sudo setfacl -R -d -m "u:$CORE:rX" "$d"
  done
  as_core "ls ${DATA_DIRS[0]} >/dev/null && echo 'labcore reads the data'"
  as_agent "ls ${DATA_DIRS[0]} >/dev/null 2>&1 && echo 'PROBLEM: labagent reads the data' || echo 'labagent cannot read the data (good)'"
}

step_migrate() {
  ask "Move the lab state" "Copy $DEV/var/lab (lab.db, data, cache, transcripts) to $HOME_DIR, owned by $CORE.
The development copy stays where it is; from now on the production state lives in $HOME_DIR." || return 0
  sudo rsync -a --exclude workspaces "$DEV/var/lab/" "$HOME_DIR/"
  sudo chown -R "$CORE:$CORE" "$HOME_DIR"; sudo chmod 700 "$HOME_DIR"
}

step_claude() {
  ask "Claude Code for the agent user" "Install Claude Code into /home/$AGENT/.local/bin (official installer)." || return 0
  as_agent "command -v ~/.local/bin/claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash"
}

step_token() {
  local F=/etc/research-lab/env
  ask "Token" "Write $F (root:$CORE 0640) with CLAUDE_CODE_OAUTH_TOKEN. Create the token yourself first with
'claude setup-token' (it uses your Claude subscription; the cpb bot's token can be reused)." || return 0
  local TOK; read -r -s -p "CLAUDE_CODE_OAUTH_TOKEN: " TOK; echo
  printf 'CLAUDE_CODE_OAUTH_TOKEN=%s\n' "$TOK" | sudo tee "$F" >/dev/null
  sudo chown root:"$CORE" "$F"; sudo chmod 640 "$F"
}

step_sudoers() {
  ask "sudoers" "Install /etc/sudoers.d/research-lab: '$CORE' may run /home/$AGENT/.local/bin/claude as '$AGENT', nothing else." || return 0
  sudo visudo -cf "$HERE/sudoers"
  sudo install -m 440 -o root -g root "$HERE/sudoers" /etc/sudoers.d/research-lab
}

step_check() {
  ask "End-to-end check" "As $CORE: dry-run cycle, a sandbox probe, and one tiny agent call as $AGENT." || return 0
  as_core "set -a; . /etc/research-lab/env; set +a; cd $APP && LAB_HOME=$HOME_DIR LAB_WORKSPACES=$WS .venv/bin/lab cycle --dry-run"
  as_core "cd $APP && .venv/bin/python -m pytest lab/tests/test_sandbox.py -q 2>&1 | tail -1"
  as_core "set -a; . /etc/research-lab/env; set +a; cd /tmp && sudo -n -u $AGENT --preserve-env=CLAUDE_CODE_OAUTH_TOKEN -- /home/$AGENT/.local/bin/claude -p 'Reply with exactly: pong' --tools '' --output-format text --no-session-persistence"
  as_agent "ls $HOME_DIR >/dev/null 2>&1 && echo 'PROBLEM: labagent lists LAB_HOME' || echo 'labagent cannot see LAB_HOME (good)'"
}

step_canaries() {
  ask "Canaries" "Run the canaries as $CORE (the judge refuses to run gates until they pass; ~10 min)." || return 0
  as_core "cd $APP && LAB_HOME=$HOME_DIR .venv/bin/lab canaries | tail -2"
}

step_timers() {
  ask "systemd timers" "Install research-lab-cycle (every 30 min at :05/:35) and research-lab-nightly (03:30 UTC:
export, commit, push) and enable them. Stop any time with: sudo systemctl disable --now research-lab-cycle.timer,
or pause agents only with: sudo -u $CORE touch $HOME_DIR/PAUSE" || return 0
  sudo install -m 644 "$HERE"/research-lab-{cycle,nightly}.{service,timer} /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now research-lab-cycle.timer research-lab-nightly.timer
  systemctl list-timers 'research-lab-*' --no-pager
}

step_update() {
  ask "Update the app" "As $CORE: git pull in $APP, uv sync, canaries; then reinstall the systemd units and the sudoers
file from the new checkout. The cycle timer is stopped meanwhile and restarted at the end." || return 0
  sudo systemctl stop research-lab-cycle.timer
  as_core "cd $APP && git pull --no-rebase -q && UV_PYTHON_INSTALL_DIR=$APP/.python uv sync -q && LAB_HOME=$HOME_DIR .venv/bin/lab canaries | tail -1"
  sudo chmod -R a+rX "$APP"
  sudo install -m 644 "$APP"/deploy/research-lab/research-lab-{cycle,nightly}.{service,timer} /etc/systemd/system/
  sudo visudo -cf "$APP/deploy/research-lab/sudoers" && sudo install -m 440 -o root -g root "$APP/deploy/research-lab/sudoers" /etc/sudoers.d/research-lab
  sudo systemctl daemon-reload && sudo systemctl start research-lab-cycle.timer
}

step_ops() {
  ask "Owner CLI" "Install /usr/local/sbin/research-lab-cli (root-owned, production environment) and /usr/local/bin/lab-prod,
so 'lab-prod status', 'lab-prod inbox', 'lab-prod ack 12 --reply ...' work for kapo without a password (sudo rule
already in the sudoers file: kapo may run only that wrapper, as $CORE)." || return 0
  sudo install -m 755 -o root -g root "$HERE/research-lab-cli" /usr/local/sbin/research-lab-cli
  sudo install -m 755 -o root -g root "$HERE/lab-prod" /usr/local/bin/lab-prod
  lab-prod status | head -5
}

step_web() {
  local CF=/etc/caddy/Caddyfile
  ask "Dashboard" "Create /srv/research-lab/web (labcore, readable by Caddy), build the page once, and add to $CF (inside the
existing cpb site, same password) a path /lab/ serving /srv/research-lab/web/current. A backup of the Caddyfile is
kept as $CF.bak-research-lab; the config is validated before Caddy reloads." || return 0
  sudo install -d -o "$CORE" -g "$CORE" -m 755 /srv/research-lab/web
  lab-prod dashboard /srv/research-lab/web
  if ! sudo grep -q "research-lab" "$CF"; then
    sudo cp "$CF" "$CF.bak-research-lab"
    sudo python3 - "$CF" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
block = """	# research-lab dashboard (static, generated by labcore after every cycle)
	redir /lab /lab/
	handle_path /lab/* {
		root * /srv/research-lab/web/current
		file_server
	}
"""
anchor = "\t@allowed path"
i = s.index(anchor)
open(p, "w").write(s[:i] + block + s[i:])
PY
  fi
  sudo caddy validate --config "$CF" --adapter caddyfile && sudo systemctl reload caddy
  echo "Dashboard: https://$(curl -fsS -m 10 https://api.ipify.org || echo '<server ip>')/lab/"
}

STEPS=(audit users dirs uv repo venv acl migrate claude token sudoers check canaries timers)
for s in "${@:-${STEPS[@]}}"; do "step_$s"; done
