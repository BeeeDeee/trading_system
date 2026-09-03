from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from scout.config.hashing import config_hash
from scout.config.loader import load_config
from scout.config.schema import ScoutConfig
from scout.utils.errors import ScoutConfigError

ROOT = Path(__file__).resolve().parents[2]
BASE_YAML = ROOT / "config" / "base.yaml"


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("SCOUT_"):
            monkeypatch.delenv(key, raising=False)


def _overlay(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "overlay.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_key_rejected(tmp_path: Path, isolated_env: None) -> None:
    path = _overlay(
        tmp_path,
        "portfolio:\n  not_a_real_field: 1\n",
    )
    with pytest.raises(ScoutConfigError, match="not_a_real_field"):
        load_config(path)


def test_defaults_complete() -> None:
    empty = ScoutConfig.model_validate({})
    from_models = ScoutConfig()
    raw = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    from_yaml = ScoutConfig.model_validate(raw)
    assert empty.model_dump(mode="json") == from_models.model_dump(mode="json")
    assert from_models.model_dump(mode="json") == from_yaml.model_dump(mode="json")


def test_load_base_yaml_fully_populated(isolated_env: None) -> None:
    cfg = load_config(BASE_YAML)
    dumped = cfg.model_dump(mode="json")
    assert dumped["run"]["seed"] == 20260827
    assert dumped["strategies"]
    assert dumped["universe"]["universe_size"] == 1000
    assert dumped["scoring"]["min_ev_net_r"] == 0.05
    assert "rank_by" not in dumped["scoring"]


def test_config_hash_stable(tmp_path: Path, isolated_env: None) -> None:
    a = _overlay(
        tmp_path,
        "run:\n  seed: 42\n  strategy_slug: hash-check\n",
    )
    b_dir = tmp_path / "reordered"
    b_dir.mkdir()
    b = b_dir / "overlay.yaml"
    b.write_text(
        "run:\n  strategy_slug: hash-check\n  seed: 42\n",
        encoding="utf-8",
    )
    assert config_hash(load_config(a)) == config_hash(load_config(b))


def test_config_hash_changes_on_value_change(tmp_path: Path, isolated_env: None) -> None:
    a = _overlay(tmp_path, "run:\n  seed: 1\n")
    b = tmp_path / "other.yaml"
    b.write_text("run:\n  seed: 2\n", encoding="utf-8")
    assert config_hash(load_config(a)) != config_hash(load_config(b))


def test_warmup_at_least_six_months(tmp_path: Path, isolated_env: None) -> None:
    path = _overlay(
        tmp_path,
        """
period:
  start: 2000-01-01T00:00:00Z
  warmup_end: 2000-03-01T00:00:00Z
  end: 2001-01-01T00:00:00Z
  split: DEVELOPMENT
""",
    )
    with pytest.raises(ScoutConfigError, match="period.warmup_end"):
        load_config(path)


def test_unmapped_symbol_warns(tmp_path: Path, isolated_env: None) -> None:
    candidates = tmp_path / "universe_candidates.txt"
    candidates.write_text("PERM1,ZZZX\n", encoding="utf-8")
    clusters = tmp_path / "clusters.yaml"
    clusters.write_text("OTHER: []\nINFO_TECH: []\n", encoding="utf-8")
    path = _overlay(
        tmp_path,
        f"""
universe:
  candidates_file: {candidates.as_posix()}
  clusters_file: {clusters.as_posix()}
""",
    )
    with pytest.warns(UserWarning, match="ZZZX"):
        load_config(path)


def test_unknown_strategy_id_rejected(tmp_path: Path, isolated_env: None) -> None:
    path = _overlay(
        tmp_path,
        """
strategies:
  - strategy_id: not_a_real_strategy_v1
    enabled: true
    params: {}
""",
    )
    with pytest.raises(ScoutConfigError, match="not_a_real_strategy_v1"):
        load_config(path)


def test_holdout_exhausted_refused(
    tmp_path: Path, isolated_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scout.config.loader.git_tree_is_clean", lambda: True)
    lock = tmp_path / "holdout_lockbox.json"
    lock.write_text('{"budget": 3, "used": 3, "evaluations": []}\n', encoding="utf-8")
    path = _overlay(
        tmp_path,
        f"""
period:
  start: 1998-01-01T00:00:00Z
  warmup_end: 2018-01-01T00:00:00Z
  end: 2026-08-01T00:00:00Z
  split: HOLDOUT
research:
  lockbox_path: {lock.as_posix()}
""",
    )
    with pytest.raises(ScoutConfigError, match="lockbox"):
        load_config(path)
    load_config(path, force_holdout=True)


def test_holdout_dirty_git_refused_even_when_forced(
    tmp_path: Path, isolated_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scout.config.loader.git_tree_is_clean", lambda: False)
    lock = tmp_path / "holdout_lockbox.json"
    lock.write_text('{"budget": 3, "used": 0, "evaluations": []}\n', encoding="utf-8")
    path = _overlay(
        tmp_path,
        f"""
period:
  start: 1998-01-01T00:00:00Z
  warmup_end: 2018-01-01T00:00:00Z
  end: 2026-08-01T00:00:00Z
  split: HOLDOUT
research:
  lockbox_path: {lock.as_posix()}
""",
    )
    with pytest.raises(ScoutConfigError, match="clean git"):
        load_config(path, force_holdout=True)
