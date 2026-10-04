"""Walking-skeleton demo: two hypotheses through the whole pipeline with stub agents and stub gates.

H-0001 (entered by hand): needs a dataset that does not exist -> BLOCKED_DATA -> Archivist -> DATA_READY ->
Builder -> G0..G3 -> Skeptic objects once -> Builder fixes -> G0..G3 again -> no objection -> G4 ->
Sentinel -> PAPER -> G5 -> LIVE_CANDIDATE.
H-0002 (from the stub Scout): data ready at once -> Builder -> G0, G1 -> killed at G2 (isolated peak).

Everything runs in a throwaway lab under `root`; the real lab.db and cards are not touched.
"""

from pathlib import Path

from lab import stubs
from lab.framework import invocations, report
from lab.framework.blackboard import Lab
from lab.framework.gates import StubEvaluator
from lab.framework.invocations import stage
from lab.framework.paths import sandbox_paths
from lab.orchestrator import run_until_idle


class DemoEvaluator(StubEvaluator):
    def paper_ready(self, lab, hid) -> bool:
        return True  # the stub "paper period" is over at once


def run_demo(root: Path) -> Lab:
    lab = Lab.open(sandbox_paths(root))
    evaluator = DemoEvaluator({("H-0002", "G2"): (False, "param_isolated_peak")}, allow=True)

    # The owner enters a hypothesis by hand (direct mode, actor = human).
    lab.send("NEW_HYPOTHESIS", "human", "system", None, {"card": stubs.load_example("btc_dvol_vrp.yaml")})

    # The Scout proposes one (staged mode, through an invocation like every agent).
    inv = invocations.start(lab, "scout")
    stage(inv.workspace, "NEW_HYPOTHESIS", "system", None, {"card": stubs.load_example("etf_sector_momentum.yaml")})
    invocations.finish(lab, inv.id)

    runners = {"archivist": stubs.archivist, "builder": stubs.builder, "scout": stubs.scout,
               "skeptic": stubs.make_skeptic({"H-0001": 1}), "librarian": stubs.librarian}
    run_until_idle(lab, evaluator, runners)
    return lab


def print_report(lab: Lab) -> None:
    print("Funnel:", {k: v for k, v in report.funnel(lab).items() if v})
    print("Rejections:", report.rejection_reasons(lab))
    for h in lab.hypotheses():
        print(f"\n=== {h['id']} v{h['version']} {h['status']}: {h['title']}")
        for e in report.trace(lab, h["id"]):
            print(f"  {e['ts'][11:23]}  {e['kind']:<10} {e['actor']:<10} {e['text']}")
    print(f"\nCards: {lab.paths.hypotheses}\nDatabase: {lab.paths.db}")
