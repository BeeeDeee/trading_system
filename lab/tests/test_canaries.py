"""The judge on trial: every canary must die where expected and the positive control must live.
Runs on every pytest run, i.e. on every change of the framework (about a minute and a half)."""

from lab.framework import canaries as C


def test_canaries(tmp_path):
    results = C.run_all(quick=True)
    failed = [f"{r.name}: {r.detail}" for r in results if not r.ok]
    assert not failed, "the judge is broken:\n" + "\n".join(failed)
