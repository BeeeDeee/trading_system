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
from qlab.research18 import dashboard as dash18, setup as T18

STATE = Path("runs/r17_forward_state.json")
RESULTS = Path("docs/research17/results")
UV = os.environ.get("UV", "uv")


def run(cmd: str, script: str = "scripts/r17_forward.py") -> subprocess.CompletedProcess:
    return subprocess.run([UV, "run", "python", script, cmd], capture_output=True, text=True)


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
r18 = run("status", "scripts/r18_forward.py")             # research 18 (docs/research18): computed after the r17 status
print(r18.stdout[-800:], r18.stderr[-800:], sep="\n")
r18_waiting = "nothing to evaluate yet" in r18.stdout + r18.stderr
r18_error = None if r18.returncode == 0 or r18_waiting else (r18.stderr or r18.stdout)[-300:]
files18 = sorted(T18.RESULTS.glob("status_2*.json"))
status18 = json.loads(files18[-1].read_text()) if files18 else None
mix = run("status", "scripts/r18_mix_forward.py") if r18.returncode == 0 else None
if mix:
    print(mix.stdout[-1000:], mix.stderr[-800:], sep="\n")
mix_error = None if not mix or mix.returncode == 0 else (mix.stderr or mix.stdout)[-300:]
mix_files = sorted(T18.RESULTS.glob("mix_status_2*.json"))
mix_status = json.loads(mix_files[-1].read_text()) if mix_files else None
files = sorted(RESULTS.glob("forward_2*.json"))
forward = json.loads(files[-1].read_text()) if files and s.returncode == 0 else None
state = {"last_run": now.isoformat(timespec="seconds"),
         "download": {"exit": d.returncode, "errors": ops.parse_download(d.stdout + d.stderr, d.returncode)},
         "status": {"exit": s.returncode, "error": (s.stderr or s.stdout)[-300:] if s.returncode else None,
                    "asof": forward["asof"] if forward else None},
         "last_ok": prev.get("last_ok")}
state["alerts"] = ops.problems(state, forward, now.date())
state["r18"] = {"exit": r18.returncode, "waiting": r18_waiting, "error": r18_error,
                "asof": status18["asof"] if status18 else None}
state["r18_mix"] = {"exit": mix.returncode if mix else None, "error": mix_error,
                    "asof": mix_status["asof"] if mix_status else None}
if r18_error:
    state["alerts"].append(f"research 18: status selhal (kód {r18.returncode}): {r18_error[:200]}")
if mix_error:
    state["alerts"].append(f"research 18 BTC/H2 mix: status selhal (kód {mix.returncode}): {mix_error[:200]}")
if not state["alerts"]:
    state["last_ok"] = state["last_run"]
STATE.write_text(json.dumps(state, indent=1))

page = dashboard.render(forward, state, now, F.FORWARD_START, F.FORWARD_END, F.MIN_EVENTS)
out = dash_dir()
tmp = out / ".index.html.tmp"
tmp.write_text(page)
tmp.replace(out / "index.html")
print(f"dashboard: {out / 'index.html'}")
tmp = out / ".research18.html.tmp"
tmp.write_text(dash18.render(status18, {"error": r18_error, "last_run": state["last_run"]}, now,
                             mix_status, {"error": mix_error, "last_run": state["last_run"]}))
tmp.replace(out / "research18.html")
print(f"dashboard: {out / 'research18.html'}")
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
