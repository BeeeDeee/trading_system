"""Weekly forward update of research 17 (docs/research17/FORWARD.md §6): download, status, health, dashboard, alert.

    uv run python scripts/r17_forward_weekly.py [--test-alert]

Called by scripts/r17_forward_weekly.sh (cron). Does NOT commit: review and commit
docs/research17/results/forward_<asof>.json by hand. Alerts go to ntfy when NTFY_TOPIC is set (optional, env file
~/.config/r17-forward/env); the dashboard and the portal banner (status.json) always show them.
Env: LEAD_DASH_DIR (default /srv/lead-forward/site, falls back to runs/lead_dashboard when not writable).
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from qlab.research17 import dashboard, forward as F, ops

STATE = Path("runs/r17_forward_state.json")
RESULTS = Path("docs/research17/results")
UV = os.environ.get("UV", "uv")


def run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run([UV, "run", "python", "scripts/r17_forward.py", cmd], capture_output=True, text=True)


def dash_dir() -> Path:
    d = Path(os.environ.get("LEAD_DASH_DIR", "/srv/lead-forward/site"))
    if d.is_dir() and os.access(d, os.W_OK):
        return d
    d = Path("runs/lead_dashboard")
    d.mkdir(parents=True, exist_ok=True)
    return d


if "--test-alert" in sys.argv:                       # check that notifications arrive
    ok = ops.notify(os.environ.get("NTFY_TOPIC"), "LEAD forward: test", "Testovací upozornění z týdenního běhu.",
                    server=os.environ.get("NTFY_SERVER", "https://ntfy.sh"), priority="default")
    print("sent" if ok else "NOT sent (NTFY_TOPIC missing or the request failed)")
    sys.exit(0 if ok else 1)

now = datetime.now(timezone.utc)
prev = json.loads(STATE.read_text()) if STATE.exists() else {}
d = run("download")
print(d.stdout[-3000:], d.stderr[-1500:], sep="\n")
s = run("status")
print(s.stdout[-1500:], s.stderr[-1500:], sep="\n")
files = sorted(RESULTS.glob("forward_2*.json"))
forward = json.loads(files[-1].read_text()) if files and s.returncode == 0 else None
state = {"last_run": now.isoformat(timespec="seconds"),
         "download": {"exit": d.returncode, "errors": ops.parse_download(d.stdout + d.stderr, d.returncode)},
         "status": {"exit": s.returncode, "error": (s.stderr or s.stdout)[-300:] if s.returncode else None,
                    "asof": forward["asof"] if forward else None},
         "last_ok": prev.get("last_ok")}
state["alerts"] = ops.problems(state, forward, now.date())
if not state["alerts"]:
    state["last_ok"] = state["last_run"]
STATE.write_text(json.dumps(state, indent=1))

page = dashboard.render(forward, state, now, F.FORWARD_START, F.FORWARD_END, F.MIN_EVENTS)
out = dash_dir()
tmp = out / ".index.html.tmp"
tmp.write_text(page)
tmp.replace(out / "index.html")
print(f"dashboard: {out / 'index.html'}")
# machine-readable status for the portal's warning banner (shown above every tab)
st = out / ".status.json.tmp"
st.write_text(json.dumps({"title": "LEAD forward", "last_run": state["last_run"], "stale_days": ops.STALE_RUN_DAYS,
                          "alerts": state["alerts"]}, ensure_ascii=False))
st.replace(out / "status.json")

if state["alerts"]:
    topic = os.environ.get("NTFY_TOPIC")
    sent = ops.notify(topic, "LEAD forward: problem", "\n".join(state["alerts"]),
                      server=os.environ.get("NTFY_SERVER", "https://ntfy.sh"))
    print(f"ALERT ({'sent' if sent else 'not sent: no NTFY_TOPIC or send failed'}):", *state["alerts"], sep="\n  ")
    sys.exit(1)
