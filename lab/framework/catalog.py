"""The data catalog (`lab/data/catalog.yaml`): what exists, from when, with which biases.

Only the ingest step writes it. In the walking skeleton the ingest is a stub that validates the entry and
appends it; the real ingest (fetch in a sandbox, validation, dev/holdout split) comes in step 3.
"""

import textwrap
from dataclasses import dataclass
from pathlib import Path

import yaml

ENTRY_REQUIRED = ("id", "loader", "asset_class", "instruments", "frequency", "range", "clock", "source", "quality",
                  "holdout_from", "known_biases", "forward_source")


@dataclass(frozen=True)
class Resolution:
    ok: bool
    missing: list[dict]      # requirements whose dataset is not in the catalog
    problems: list[str]      # dataset exists but does not cover the request


def load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return {d["id"]: d for d in data["datasets"]}


def resolve(path: Path, requirements: list[dict]) -> Resolution:
    datasets = load(path)
    missing, problems = [], []
    for req in requirements:
        ds = datasets.get(req["dataset"])
        if ds is None:
            missing.append(req)
            continue
        if not ds.get("loader", False):
            problems.append(f"{req['dataset']}: in the catalog, but the gate runner has no loader for it yet")
            continue
        period = req.get("period")
        if period and (str(period[0]) < str(ds["range"][0]) or str(period[1]) > str(ds["range"][1])):
            problems.append(f"{req['dataset']}: requested {period[0]}..{period[1]} outside "
                            f"{ds['range'][0]}..{ds['range'][1]}")
    return Resolution(not missing and not problems, missing, problems)


def entry_errors(entry: dict) -> list[str]:
    errors = [f"missing field {f!r}" for f in ENTRY_REQUIRED if f not in entry]
    if not entry.get("known_biases"):
        errors.append("known_biases must list at least one bias (there is always one)")
    rng = entry.get("range") or []
    if len(rng) != 2 or str(rng[0]) >= str(rng[1]):
        errors.append("range must be [start, end] with start < end")
    elif not str(rng[0]) < str(entry.get("holdout_from", "")) <= str(rng[1]):
        errors.append("holdout_from must lie inside range")
    return errors


def add(path: Path, entry: dict) -> None:
    """Append a validated entry. Existing entries are immutable (their holdout boundary is fixed)."""
    errors = entry_errors(entry)
    if errors:
        raise ValueError("; ".join(errors))
    data = yaml.safe_load(path.read_text())
    if any(d["id"] == entry["id"] for d in data["datasets"]):
        raise ValueError(f"dataset {entry['id']!r} already in the catalog")
    with path.open("a") as f:
        f.write("\n" + textwrap.indent(yaml.safe_dump([entry], sort_keys=False, allow_unicode=True), "  "))
    load(path)  # the file must still parse


def missing_forward_source(path: Path, dataset_ids: list[str]) -> list[str]:
    """Datasets among `dataset_ids` without a forward source (they block paper trading, decision Q3)."""
    datasets = load(path)
    return [d for d in dataset_ids if not datasets.get(d, {}).get("forward_source")]
