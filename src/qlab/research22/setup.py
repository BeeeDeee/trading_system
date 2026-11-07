"""Constants and stage definitions of research 22 (spec: `reports/Experiment 1 specifikace`, v1.2 -> docs/research22).

Research 22 declares the whole Sharadar history (to 2026-09-25) already seen, like research 19/21: the README
says the 2020-2026 holdout is consumed. Its "late validation" stage is therefore not a clean holdout, and the
OOS vault is not used. Stages are only a protocol for the search, so a result here is at best a forward candidate.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from qlab.data.panel import load_panel
from qlab.engine.costs import CostModel
from qlab.research22.compact import Compact, build_compact
from qlab.validation.vault import Vault

SHARADAR = "sharadar_2026-09-25"
PANEL_NAME = "panel_liq1000"
SEED = 22
REGISTRY = Path("runs/registry/research22.sqlite")
SCRATCH_REGISTRY = Path("runs/scratch/research22_smoke.sqlite")

TOPK = 100                    # equal-weight top decile of the ~1000-name universe
N_FOLDS = {"A": 8, "B": 7}    # dev folds for the fitness quantile (two years each)
FITNESS_QUANTILE = 0.25
FITNESS_COST_MULTIPLIER = 2.0  # fitness is computed at 2x costs, reporting at 1x/2x/3x
EMBARGO_DAYS = 21             # trading days skipped at the start of validation / late stages
MIN_DISPERSION_SHARE = 0.9    # a score must rank (>=2 distinct values) on >= 90 % of decision weeks
MAX_DEPTH, MAX_NODES = 5, 15
ANN_WEEKS = 52
# search budget (spec §6/§7); pop_size must be divisible by 4 for DEAP's tournament
POP_SIZE, N_GEN, N_SEEDS = 500, 60, 10
P_CX, P_MUT = 0.7, 0.3        # per offspring pair / per individual; not in the spec text, frozen here
N_NULL, N_RANDOM = 50, 300_000
LOCK = Path("docs/research22/run_lock.json")   # full-budget runs refuse to start without it
RESULTS = Path("docs/research22/results")
FINAL_ENV = "QLAB_R22_METHODOLOGY"
COST_MODEL = CostModel()      # frozen defaults of the project (spec §7.3), multiplier applied at evaluation time


@dataclass(frozen=True)
class Stage:
    name: str
    start: date
    end: date
    embargo: int              # trading days skipped after `start` before the first decision


# Run A (primary) and run B (mirror). Warm-up rows precede `dev` and may be read by causal features only.
RUNS: dict[str, dict[str, Stage]] = {
    "A": {"dev": Stage("dev", date(1999, 1, 4), date(2014, 12, 31), 0),
          "validation": Stage("validation", date(2015, 1, 1), date(2019, 12, 31), EMBARGO_DAYS),
          "late": Stage("late", date(2020, 1, 1), date(2025, 12, 31), EMBARGO_DAYS)},
    "B": {"dev": Stage("dev", date(2006, 1, 3), date(2019, 12, 31), 0),
          "validation": Stage("validation", date(2020, 1, 1), date(2022, 12, 31), EMBARGO_DAYS),
          "late": Stage("late", date(2023, 1, 1), date(2025, 12, 31), EMBARGO_DAYS)},
}


def panel_dir(snapshot: str = SHARADAR) -> Path:
    return Path("data/derived") / snapshot / PANEL_NAME


def stage_rows(dates: np.ndarray, stage: Stage) -> tuple[int, int]:
    """First row inside the stage and last row inside it (inclusive)."""
    d = np.asarray(dates, dtype="datetime64[D]")
    first = int(np.searchsorted(d, np.datetime64(stage.start, "D"), side="left"))
    last = int(np.searchsorted(d, np.datetime64(stage.end, "D"), side="right")) - 1
    return first, last


def load_compact(run: str, stage: str, snapshot: str = SHARADAR, directory: Path | None = None) -> Compact:
    """Compact weekly panel of one stage; rows after the stage end are never read."""
    panel, extra = load_panel(directory or panel_dir(snapshot), mmap=True)
    st = RUNS[run][stage]
    first, last = stage_rows(panel.dates, st)
    compact = build_compact(panel, extra, first, last, st.embargo, COST_MODEL)
    compact.n_folds = N_FOLDS[run]
    return compact


def vault(run: str) -> Vault:
    """One vault per run: it freezes the methodology hash and allows the late stage to be opened once."""
    return Vault(RUNS[run]["validation"].end, Path(f"runs/vault_r22_{run}/frozen.lock"),
                 Path(f"runs/vault_r22_{run}/vault.log"))
