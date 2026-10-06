"""Headless agent runs: `claude -p` in the agent's workspace, transcript kept, audited, usage extracted.

The run itself has no authority. What it produces is the workspace (files + staged outbox) and a transcript.
`invocations.finish()` applies the outbox only if the transcript audit is clean:
- Bash: only `lab <verb>` for the verbs the agent definition allows, one command, no shell operators,
- file tools (Read, Write, Edit, Glob, Grep, NotebookEdit): only paths inside the workspace,
- no tool outside the agent definition.
The audit counts attempts, not successes: a call that the permission layer denied is still a violation,
because it shows the agent trying to leave its box.

The transcript is written outside the workspace (`LAB_HOME/transcripts`), so the agent cannot rewrite it.
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from lab.framework import agents
from lab.framework.blackboard import Lab
from lab.framework.invocations import Invocation

BIN_DIR = Path(sys.executable).parent           # the venv's bin, where the `lab` entry point lives
FILE_TOOLS = {"Read": "file_path", "Write": "file_path", "Edit": "file_path", "NotebookEdit": "notebook_path",
              "Glob": "path", "Grep": "path"}
HARMLESS_TOOLS = {"TodoWrite"}                  # bookkeeping inside the model, no side effects
SHELL_OPERATORS = set(";&|<>()")
DROP_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "LAB_HOME", "LAB_DATA_ROOT", "VIRTUAL_ENV")


@dataclass
class RunResult:
    exit_code: int
    transcript_path: str | None = None
    usage: dict = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    timed_out: bool = False
    note: str | None = None


def command(spec: agents.AgentSpec, prompt: str, claude_bin: str = "claude") -> list[str]:
    allowed = [t for t in spec.tools if t != "Bash"]
    if "Bash" in spec.tools:
        allowed += [f"Bash(lab {v}:*)" for v in spec.lab_verbs]
    settings = json.loads((agents.AGENTS_DIR / "settings.json").read_text())
    cmd = [claude_bin, "-p", prompt, "--model", spec.model, "--system-prompt", spec.prompt,
           "--tools", ",".join(spec.tools), "--allowedTools", *allowed,
           "--permission-mode", "dontAsk", "--restricted", "--strict-mcp-config", "--disable-slash-commands",
           "--no-session-persistence", "--settings", json.dumps(settings),
           "--output-format", "stream-json", "--verbose"]
    if spec.effort:
        cmd += ["--effort", spec.effort]
    return cmd


def task_prompt(inv: Invocation) -> str:
    ctx = json.loads((inv.workspace / "context.json").read_text())
    task = ctx.get("task") or ("answer" if inv.hypothesis_id else "propose")
    target = f" for {inv.hypothesis_id}" if inv.hypothesis_id else ""
    return (f"Task: {task}{target}. Your workspace is the current directory; start with context.json and "
            f"inbox.json. Follow your instructions and stop when the message is staged.")


def environment(ws: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in DROP_ENV}
    env["LAB_WORKSPACE"] = str(ws)
    env["PATH"] = f"{BIN_DIR}{os.pathsep}{env.get('PATH', '')}"
    return env


# Production: the agent runs as its own OS user (`LAB_AGENT_USER`, e.g. labagent) through a narrow sudoers rule.
# That user can write its workspace and read the app code, nothing else of the lab (no lab.db, no data).
AGENT_ENV = ("LAB_WORKSPACE", "PATH", "CLAUDE_CODE_OAUTH_TOKEN", "LANG")


def as_agent_user(cmd: list[str], ws: Path) -> list[str]:
    user = os.environ.get("LAB_AGENT_USER")
    if not user:
        return cmd
    for p in [ws, *ws.rglob("*")]:            # the agent user writes through the shared group
        os.chmod(p, 0o2770 if p.is_dir() else 0o660)
    return ["sudo", "-n", "-u", user, f"--preserve-env={','.join(AGENT_ENV)}", "--", *cmd]


# ---------------------------------------------------------------------------- transcript

def events(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def tool_calls(evs: list[dict]) -> list[tuple[str, dict]]:
    calls = []
    for ev in evs:
        if ev.get("type") != "assistant":
            continue
        for blk in (ev.get("message") or {}).get("content") or []:
            if isinstance(blk, dict) and blk.get("type") in ("tool_use", "server_tool_use"):
                calls.append((blk.get("name", "?"), blk.get("input") or {}))
    return calls


def usage(evs: list[dict]) -> dict:
    init = next((e for e in evs if e.get("type") == "system" and e.get("subtype") == "init"), {})
    res = next((e for e in reversed(evs) if e.get("type") == "result"), {})
    u = res.get("usage") or {}
    return {"n_turns": res.get("num_turns"),
            "tokens_in": sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                     "cache_read_input_tokens")) or None,
            "tokens_out": u.get("output_tokens"),
            "model": init.get("model"), "tools": init.get("tools"), "result_subtype": res.get("subtype"),
            "is_error": res.get("is_error"), "result_text": (res.get("result") or "")[:4000],
            "duration_ms": res.get("duration_ms")}


def audit(calls: list[tuple[str, dict]], ws: Path, spec: agents.AgentSpec) -> list[str]:
    root = ws.resolve()
    out = []
    for name, inp in calls:
        if name == "Bash":
            problem = _bash_problem(str(inp.get("command", "")), spec.lab_verbs) if "Bash" in spec.tools \
                else "Bash is not allowed"
            if problem:
                out.append(f"Bash {inp.get('command', '')[:200]!r}: {problem}")
        elif name in FILE_TOOLS:
            if name not in spec.tools:
                out.append(f"{name} is not allowed")
                continue
            raw = inp.get(FILE_TOOLS[name])
            if raw is None:
                continue   # Glob/Grep default to the cwd
            path = Path(raw).expanduser()
            path = (path if path.is_absolute() else root / path).resolve()
            if not path.is_relative_to(root):
                out.append(f"{name} outside the workspace: {raw}")
        elif name not in spec.tools and name not in HARMLESS_TOOLS:
            out.append(f"tool {name} is not allowed")
    return out


def _bash_problem(cmd: str, verbs: tuple[str, ...]) -> str | None:
    if "\n" in cmd or "`" in cmd or "$(" in cmd or "${" in cmd:
        return "newlines and substitutions are not allowed (put content in a file)"
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError as e:
        return f"cannot parse: {e}"
    if any(t and set(t) <= SHELL_OPERATORS for t in tokens):
        return "shell operators are not allowed"
    if len(tokens) < 2 or tokens[0] != "lab" or tokens[1] not in verbs:
        return f"only `lab {{{','.join(verbs)}}}` may run"
    return None


# ---------------------------------------------------------------------------- the runner

class ClaudeRunner:
    """Orchestrator runner for one agent: `runner(lab, invocation) -> RunResult`."""

    def __init__(self, agent: str, claude_bin: str | None = None, spec: agents.AgentSpec | None = None):
        self.spec = spec or agents.spec(agent)
        self.model = self.spec.model
        self.claude_bin = claude_bin or os.environ.get("LAB_CLAUDE_BIN", "claude")

    def __call__(self, lab: Lab, inv: Invocation) -> RunResult:
        if inv.agent != self.spec.name:
            raise ValueError(f"runner for {self.spec.name} called for {inv.agent}")
        tdir = lab.paths.home / "transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        transcript, stderr = tdir / f"{inv.id}.jsonl", tdir / f"{inv.id}.stderr"
        cmd = as_agent_user(command(self.spec, task_prompt(inv), self.claude_bin), inv.workspace)
        t0, timed_out = time.monotonic(), False
        with transcript.open("w") as out, stderr.open("w") as err:
            p = subprocess.Popen(cmd, cwd=inv.workspace, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                 env=environment(inv.workspace), start_new_session=True)
            try:
                code = p.wait(timeout=self.spec.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill(p)
                code = p.returncode if p.returncode is not None else -9
        evs = events(transcript)
        use = usage(evs) | {"wall_s": round(time.monotonic() - t0, 1)}
        note = None
        if use["is_error"] or (code != 0 and not timed_out):
            note = (f"{use['result_subtype'] or 'no result'}: {use['result_text'][:300]}"
                    f" | stderr: {stderr.read_text()[-300:]}")
        return RunResult(code, str(transcript), use, audit(tool_calls(evs), inv.workspace, self.spec),
                         timed_out, note)


def _kill(p: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(p.pid, sig)
            p.wait(timeout=wait)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue
