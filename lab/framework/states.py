"""State machine of a hypothesis and the permission tables of the roles.

`GATE_n` and `HOLDOUT` mean "passed that gate". Every failure goes to REJECTED with a stage and a reason
code. The tables here are the whole authority model: an action that is not listed is refused.

Agents (LLM) only send messages. Transitions are performed by framework handlers in reaction to those
messages or to gate results, and each transition records the actor whose message or verdict caused it.
"""

from enum import StrEnum


class S(StrEnum):
    IDEA = "IDEA"
    SPECIFIED = "SPECIFIED"
    BLOCKED_DATA = "BLOCKED_DATA"
    DATA_READY = "DATA_READY"
    IMPLEMENTED = "IMPLEMENTED"
    GATE_1 = "GATE_1"
    GATE_2 = "GATE_2"
    GATE_3 = "GATE_3"
    SKEPTIC_REVIEW = "SKEPTIC_REVIEW"
    HOLDOUT = "HOLDOUT"
    PAPER = "PAPER"
    LIVE_CANDIDATE = "LIVE_CANDIDATE"
    REJECTED = "REJECTED"
    PARKED = "PARKED"
    RETIRED = "RETIRED"


TERMINAL = {S.REJECTED, S.PARKED, S.RETIRED}
FUNNEL = [S.IDEA, S.SPECIFIED, S.BLOCKED_DATA, S.DATA_READY, S.IMPLEMENTED, S.GATE_1, S.GATE_2, S.GATE_3,
          S.SKEPTIC_REVIEW, S.HOLDOUT, S.PAPER, S.LIVE_CANDIDATE, S.REJECTED, S.PARKED, S.RETIRED]

LLM_AGENTS = ("chair", "scout", "archivist", "builder", "skeptic", "steward", "librarian")
SYSTEM_ACTORS = ("system", "gatekeeper", "sentinel")   # deterministic framework code
HUMAN = "human"                                        # the owner, via the CLI in direct mode
ACTORS = LLM_AGENTS + SYSTEM_ACTORS + (HUMAN,)

# Gate that must pass (for the current version) to enter a state.
GATE_FOR = {S.IMPLEMENTED: "G0", S.GATE_1: "G1", S.GATE_2: "G2", S.GATE_3: "G3", S.HOLDOUT: "G4",
            S.LIVE_CANDIDATE: "G5"}
# State a hypothesis is in when a gate runs, i.e. the gate's stage for rejection reports.
GATE_FROM = {"G0": S.DATA_READY, "G1": S.IMPLEMENTED, "G2": S.GATE_1, "G3": S.GATE_2,
             "G4": S.SKEPTIC_REVIEW, "G5": S.PAPER}

_NON_TERMINAL = [s for s in S if s not in TERMINAL]

# (from, to) -> actors allowed to cause it. `None` as from = creation. Rows are unioned, not overwritten.
_ROWS: list[tuple[S | None, S, set[str]]] = [
    (None, S.IDEA, {"scout", HUMAN}),
    (S.IDEA, S.SPECIFIED, {"scout", HUMAN}),           # spec check by `system` on their message
    (S.SPECIFIED, S.DATA_READY, {"system"}),            # catalog resolves every requirement
    (S.SPECIFIED, S.BLOCKED_DATA, {"system"}),
    (S.BLOCKED_DATA, S.DATA_READY, {"system"}),         # after an ingest
    (S.BLOCKED_DATA, S.PARKED, {"archivist"}),          # infeasible data
    (S.DATA_READY, S.IMPLEMENTED, {"gatekeeper"}),      # G0 integrity passed
    (S.IMPLEMENTED, S.GATE_1, {"gatekeeper"}),
    (S.GATE_1, S.GATE_2, {"gatekeeper"}),
    (S.GATE_2, S.GATE_3, {"gatekeeper"}),
    (S.GATE_3, S.SKEPTIC_REVIEW, {"system"}),
    (S.SKEPTIC_REVIEW, S.DATA_READY, {"skeptic"}),      # OBJECTION to the builder
    (S.SKEPTIC_REVIEW, S.IDEA, {"skeptic"}),            # OBJECTION to the scout (new version)
    (S.SKEPTIC_REVIEW, S.HOLDOUT, {"gatekeeper"}),      # no objection + G4 passed
    (S.HOLDOUT, S.PAPER, {"sentinel"}),
    (S.HOLDOUT, S.PARKED, {"sentinel"}),                # no forward data / no capacity
    (S.PAPER, S.LIVE_CANDIDATE, {"gatekeeper"}),        # G5 passed (+ sentinel check)
    (S.PAPER, S.RETIRED, {"sentinel", HUMAN}),
    (S.LIVE_CANDIDATE, S.RETIRED, {"sentinel", HUMAN}),
    (S.PARKED, S.IDEA, {"chair", HUMAN}),               # reopen as a new version (Q7)
    (S.REJECTED, S.DATA_READY, {HUMAN}),                # void a gate rejection caused by a framework bug
    (S.SKEPTIC_REVIEW, S.REJECTED, {"skeptic"}),        # the Skeptic can only kill what it reviews
    *((s, S.REJECTED, {"gatekeeper", "system", HUMAN}) for s in _NON_TERMINAL),
    *((s, S.PARKED, {"chair", HUMAN}) for s in _NON_TERMINAL),
]
TRANSITIONS: dict[tuple[S | None, S], frozenset[str]] = {}
for _f, _t, _a in _ROWS:
    TRANSITIONS[(_f, _t)] = TRANSITIONS.get((_f, _t), frozenset()) | frozenset(_a)


def allowed(frm: S | None, to: S, actor: str) -> bool:
    return actor in TRANSITIONS.get((frm, to), frozenset())


# Message types: who may send them, to whom, and (for VERDICT) which decisions each sender may take.
MESSAGE_RULES: dict[str, dict] = {
    "NEW_HYPOTHESIS": {"from": {"scout", HUMAN}, "to": {"system"}},
    "REVISION":       {"from": {"scout", HUMAN}, "to": {"system"}},
    "DATA_REQUEST":   {"from": {"system", "builder", "scout"}, "to": {"archivist"}},
    "DATA_READY":     {"from": {"archivist"}, "to": {"system"}},
    "IMPL_DONE":      {"from": {"builder"}, "to": {"gatekeeper"}},
    "GATE_RESULT":    {"from": {"gatekeeper"}, "to": {"builder", "scout", "skeptic", "librarian", "steward"}},
    "OBJECTION":      {"from": {"skeptic"}, "to": {"builder", "scout"}},
    "VERDICT":        {"from": {"skeptic", "archivist", "chair", "sentinel", "system"},
                       "to": {"system", "librarian", "chair", HUMAN}},
    "QUESTION":       {"from": set(LLM_AGENTS) | {"system", HUMAN}, "to": set(LLM_AGENTS) | {HUMAN}},
    "LESSON":         {"from": {"librarian"}, "to": {"system"}},
    "ALERT":          {"from": {"sentinel", "steward", "system", "gatekeeper"},
                       "to": {HUMAN, "chair", "librarian"}},
}
VERDICT_DECISIONS = {
    "skeptic": {"no_objection", "reject"},
    "archivist": {"infeasible"},
    "chair": {"park", "reopen"},
    "sentinel": {"admit", "park", "retire"},
    "system": {"rejected", "parked"},
}
