"""Operations of the research 17 forward test: health of a weekly run, alerts (ntfy), run state.

A weekly run = `scripts/r17_forward.py download` + `status`. `problems` turns their outcome into the list of
things worth a notification; the same list is shown on the dashboard.
"""

import re
import urllib.request
from datetime import date, datetime, timedelta

MAX_DATA_LAG_DAYS = 3          # the newest daily ZIP should be at most this many days old (weekly run + publication lag)
MEMBERSHIP_GRACE_DAY = 6       # monthly 1d ZIPs of the previous month are normally published by then
STALE_RUN_DAYS = 9             # the dashboard flags a weekly run that did not happen for this long


def parse_download(output: str, exit_code: int) -> list[str]:
    """Error lines of a `download` run: the explicit error records plus a generic one for a non-zero exit."""
    errs = [ln.strip() for ln in output.splitlines() if re.match(r"\{.*'error':", ln)]
    if exit_code != 0 and not errs:
        errs = [f"download skončil s kódem {exit_code}"]
    return errs[:20]


def problems(state: dict, forward: dict | None, today: date) -> list[str]:
    """Alerts for the latest weekly run. `state` as written by the weekly script, `forward` the latest status
    JSON (None if `status` failed)."""
    out = []
    dl = state.get("download", {})
    if dl.get("errors"):
        out.append(f"stahování: {len(dl['errors'])} chyb (např. {dl['errors'][0][:160]})")
    st = state.get("status", {})
    if st.get("exit", 0) != 0:
        out.append(f"status selhal (kód {st['exit']}): {str(st.get('error', ''))[:200]}")
    if forward is None:
        if st.get("exit", 0) == 0:
            out.append("chybí výsledek statusu")
        return out
    asof = date.fromisoformat(forward["asof"])
    if (today - asof).days > MAX_DATA_LAG_DAYS:
        out.append(f"data zaostávají: poslední společný den {asof}, dnes {today}")
    if forward.get("stale_symbols"):
        out.append("zastaralá data u: " + ", ".join(sorted(forward["stale_symbols"])))
    late = [m for m in forward.get("pending_months", []) if today >= date(int(m[:4]), int(m[5:]), MEMBERSHIP_GRACE_DAY)]
    if late:
        out.append("chybí členství pro měsíc " + ", ".join(late) + " (obchody se nepočítají)")
    return out


def notify(topic: str | None, title: str, body: str, server: str = "https://ntfy.sh", priority: str = "high",
           post=None) -> bool:
    """Send a ntfy notification; never raises (a failed alert must not break the run). Title is ASCII only."""
    if not topic:
        return False
    title = title.encode("ascii", "replace").decode()
    req = urllib.request.Request(f"{server.rstrip('/')}/{topic}", data=body.encode(), method="POST",
                                 headers={"Title": title, "Priority": priority, "Tags": "warning"})
    try:
        (post or (lambda r: urllib.request.urlopen(r, timeout=20)))(req)
        return True
    except Exception:
        return False


def run_age_days(state: dict, now: datetime) -> float | None:
    t = state.get("last_run")
    return (now - datetime.fromisoformat(t)).total_seconds() / 86400 if t else None
