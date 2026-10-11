"""Headless `claude -p` for the research/scoring/free-decision part (steps 4-6 of the original task prompt)."""
import json
import os
import re
import subprocess
import time

TOOLS = "WebSearch,WebFetch,Read,Write"


def build_prompt(task_prompt_text, header):
    """Steps 4-6 are cut out of the pinned task prompt VERBATIM (so the LLM instructions stay exactly the v1.1.x ones)."""
    m = re.search(r"^## 4\. Research.*?(?=^## 7\. Plan and save)", task_prompt_text, re.S | re.M)
    if not m:
        raise RuntimeError("steps 4-6 not found in task/task_prompt.md")
    return header.rstrip() + "\n\n" + m.group(0).rstrip() + "\n"


def parse_stream(lines):
    meta = {"model_init": None, "session_id": None, "result": None, "tool_calls": {}}
    for ln in lines:
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        if ev.get("type") == "assistant":
            for blk in (ev.get("message") or {}).get("content") or []:
                if isinstance(blk, dict) and blk.get("type") in ("tool_use", "server_tool_use"):
                    meta["tool_calls"][blk.get("name")] = meta["tool_calls"].get(blk.get("name"), 0) + 1
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            meta["model_init"], meta["session_id"] = ev.get("model"), ev.get("session_id")
        elif ev.get("type") == "result":
            meta["result"] = {k: ev.get(k) for k in ("subtype", "is_error", "duration_ms", "num_turns", "total_cost_usd", "modelUsage", "terminal_reason")}
            meta["session_id"] = ev.get("session_id") or meta["session_id"]
    return meta


def run_claude(model, workdir, prompt, timeout_s, transcript, resume=None):
    cmd = ["claude", "-p", prompt, "--model", model, "--tools", TOOLS, "--allowedTools", TOOLS.replace(",", " "),
           "--permission-mode", "dontAsk", "--restricted", "--strict-mcp-config", "--disable-slash-commands",
           "--output-format", "stream-json", "--verbose",
           "--effort", os.environ.get("EVENING_EFFORT", "high")]
    if resume:
        cmd += ["--resume", resume]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
        out, err, code, timed_out = p.stdout, p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        code, timed_out = None, True
    with open(os.path.join(workdir, transcript), "w") as f:
        f.write(out)
    meta = parse_stream(out.splitlines())
    meta.update({"exit_code": code, "timed_out": timed_out, "wall_s": round(time.time() - t0, 1), "stderr_tail": err[-1500:],
                 "models_used": sorted(((meta.get("result") or {}).get("modelUsage") or {}).keys())})
    return meta


def model_ok(model, meta):
    return meta.get("model_init") == model and model in (meta.get("models_used") or [])
