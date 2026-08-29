from __future__ import annotations

import json
import logging
import os
import subprocess
import warnings
from calendar import monthrange
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from scout.config.schema import (
    KNOWN_STRATEGY_PARAMS,
    AssetClass,
    PeriodSplit,
    ScoutConfig,
)
from scout.domain.enums import RunMode
from scout.utils.errors import ScoutConfigError

_LOG = logging.getLogger("scout.config")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_BASE = _REPO_ROOT / "config" / "base.yaml"

_SECRET_ENV = frozenset(
    {
        "SCOUT_NORGATE_USER",
        "SCOUT_NORGATE_PASSWORD",
        "SCOUT_SHARADAR_API_KEY",
        "SCOUT_BROKER_API_KEY",
        "SCOUT_BROKER_API_SECRET",
    }
)

_TOP_LEVEL_SECTIONS = frozenset(
    {
        "run",
        "period",
        "data",
        "universe",
        "features",
        "regime",
        "market_regime",
        "gates",
        "strategies",
        "labeling",
        "edge",
        "scoring",
        "costs",
        "portfolio",
        "risk",
        "sentiment",
        "audit",
        "logging",
        "research",
    }
)

# M7 stay locked until an equity holdout write-up exists.
_M7_UNLOCKED = False


def load_config(path: Path) -> ScoutConfig:
    """Deep-merge, validate, then run cross-field validation (§8).
    Raises ScoutConfigError with the offending key path on any failure.
    """
    resolved = path.expanduser()
    if not resolved.is_file():
        raise ScoutConfigError(f"config file not found: {path}")
    data = _load_layered_yaml(resolved)
    applied = _apply_env_overrides(data)
    for key in applied:
        _LOG.warning("env override applied: %s", key)
    cfg = _parse_config(data)
    validate_config(cfg)
    return cfg


def _load_layered_yaml(path: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    base = _DEFAULT_BASE
    if base.is_file():
        merged = _deep_merge(merged, _read_yaml(base))
    resolved_path = path.resolve()
    if not base.is_file() or resolved_path != base.resolve():
        merged = _deep_merge(merged, _read_yaml(resolved_path))
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScoutConfigError(f"{path}: invalid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ScoutConfigError(f"{path}: root must be a mapping")
    return loaded


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = value
    return out


def _apply_env_overrides(data: dict[str, Any]) -> list[str]:
    applied: list[str] = []
    trading = os.environ.get("SCOUT_TRADING_ENABLED")
    if trading is not None:
        _assign(data, ["risk", "trading_enabled"], _parse_bool(trading))
        applied.append("SCOUT_TRADING_ENABLED")
    skip = _SECRET_ENV | {"SCOUT_TRADING_ENABLED"}
    for raw_key, raw_val in os.environ.items():
        if not raw_key.startswith("SCOUT_") or raw_key in skip:
            continue
        parts = [p.lower() for p in raw_key.removeprefix("SCOUT_").split("__")]
        if not parts or parts[0] not in _TOP_LEVEL_SECTIONS:
            continue
        _assign(data, parts, raw_val)
        applied.append(raw_key)
    return applied


def _assign(data: dict[str, Any], path: list[str], value: object) -> None:
    current: dict[str, Any] = data
    for key in path[:-1]:
        nested = current.get(key)
        if not isinstance(nested, dict):
            nested = {}
            current[key] = nested
        current = nested
    current[path[-1]] = value


def _parse_bool(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _parse_config(data: dict[str, Any]) -> ScoutConfig:
    try:
        return ScoutConfig.model_validate(data)
    except ValidationError as exc:
        raise _wrap_validation(exc) from exc


def _wrap_validation(exc: ValidationError) -> ScoutConfigError:
    parts: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error["loc"])
        parts.append(f"{loc}: {error['msg']}")
    return ScoutConfigError("; ".join(parts))


def validate_config(cfg: ScoutConfig) -> None:
    _validate_period(cfg)
    _validate_costs(cfg)
    _validate_portfolio(cfg)
    _validate_edge(cfg)
    _validate_regime(cfg)
    _validate_features(cfg)
    _validate_strategies(cfg)
    _validate_sentiment(cfg)
    _warn_unmapped_symbols(cfg)
    _validate_live(cfg)
    _validate_holdout(cfg)
    _validate_crypto_holdout(cfg)


def _validate_period(cfg: ScoutConfig) -> None:
    start = cfg.period.start
    warmup_end = cfg.period.warmup_end
    end = cfg.period.end
    floor = _add_months(start, 6)
    if warmup_end < floor:
        raise ScoutConfigError(
            "period.warmup_end: must be at least 6 months after period.start"
        )
    if end <= warmup_end:
        raise ScoutConfigError("period.end: must be after period.warmup_end")


def _add_months(ts: datetime, months: int) -> datetime:
    month_index = ts.month - 1 + months
    year = ts.year + month_index // 12
    month = month_index % 12 + 1
    day = min(ts.day, monthrange(year, month)[1])
    return ts.replace(year=year, month=month, day=day)


def _validate_costs(cfg: ScoutConfig) -> None:
    if cfg.costs.cost_multiplier < 1.0:
        raise ScoutConfigError("costs.cost_multiplier: must be >= 1.0")


def _validate_portfolio(cfg: ScoutConfig) -> None:
    heat = cfg.portfolio.max_portfolio_heat_pct
    cluster = cfg.portfolio.max_cluster_risk_pct
    if cluster > heat:
        raise ScoutConfigError(
            "portfolio.max_cluster_risk_pct: must be <= portfolio.max_portfolio_heat_pct"
        )
    reachable = cfg.portfolio.risk_fraction_per_trade * cfg.portfolio.max_positions
    if reachable < heat:
        warnings.warn(
            "portfolio heat cap is unreachable: "
            "risk_fraction_per_trade * max_positions < max_portfolio_heat_pct",
            UserWarning,
            stacklevel=2,
        )


def _validate_edge(cfg: ScoutConfig) -> None:
    if cfg.edge.z < 0:
        raise ScoutConfigError("edge.z: must be >= 0")


def _validate_regime(cfg: ScoutConfig) -> None:
    if cfg.regime.er_range_max >= cfg.regime.er_trend_min:
        raise ScoutConfigError(
            "regime.er_range_max: must be < regime.er_trend_min"
        )
    if cfg.regime.vol_low_pct >= cfg.regime.vol_high_pct:
        raise ScoutConfigError("regime.vol_low_pct: must be < regime.vol_high_pct")


def _validate_features(cfg: ScoutConfig) -> None:
    if cfg.features.ema_fast >= cfg.features.ema_slow:
        raise ScoutConfigError("features.ema_fast: must be < features.ema_slow")


def _validate_strategies(cfg: ScoutConfig) -> None:
    known = sorted(KNOWN_STRATEGY_PARAMS)
    for index, strategy in enumerate(cfg.strategies):
        allowed = KNOWN_STRATEGY_PARAMS.get(strategy.strategy_id)
        if allowed is None:
            raise ScoutConfigError(
                f"strategies.{index}.strategy_id: unknown strategy_id "
                f"{strategy.strategy_id!r}; known: {known}"
            )
        extra = sorted(set(strategy.params) - allowed)
        if extra:
            raise ScoutConfigError(
                f"strategies.{index}.params.{extra[0]}: unknown param for "
                f"{strategy.strategy_id}; extra: {extra}"
            )


def _validate_sentiment(cfg: ScoutConfig) -> None:
    mult = cfg.sentiment.min_multiplier
    if not (0.0 < mult < 1.0):
        raise ScoutConfigError("sentiment.min_multiplier: must be in (0, 1)")
    if cfg.sentiment.enabled:
        ids = cfg.sentiment.source_ids
        if ids == [None] or all(item is None for item in ids):
            raise ScoutConfigError(
                "sentiment.source_ids: cannot be [null] when sentiment.enabled is true"
            )


def _warn_unmapped_symbols(cfg: ScoutConfig) -> None:
    candidates = _read_candidate_symbols(Path(cfg.universe.candidates_file))
    if not candidates:
        return
    mapped = _read_cluster_symbols(Path(cfg.universe.clusters_file))
    missing = sorted(sym for sym in candidates if sym not in mapped)
    if missing:
        warnings.warn(
            f"symbols absent from every cluster: {', '.join(missing)}",
            UserWarning,
            stacklevel=2,
        )


def _read_candidate_symbols(path: Path) -> list[str]:
    if not path.is_file():
        return []
    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [p.strip() for p in stripped.split(",")]
        symbols.append(parts[-1] if parts else stripped)
    return symbols


def _read_cluster_symbols(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return set()
    mapped: set[str] = set()
    for members in loaded.values():
        if isinstance(members, list):
            mapped.update(str(item) for item in members)
    return mapped


def _validate_live(cfg: ScoutConfig) -> None:
    if cfg.run.mode is not RunMode.LIVE:
        return
    if "trading_enabled" not in cfg.risk.model_fields_set:
        raise ScoutConfigError(
            "risk.trading_enabled: must be set explicitly when run.mode is LIVE"
        )
    key = os.environ.get("SCOUT_BROKER_API_KEY", "").strip()
    secret = os.environ.get("SCOUT_BROKER_API_SECRET", "").strip()
    if not key or not secret:
        raise ScoutConfigError(
            "run.mode: LIVE requires SCOUT_BROKER_API_KEY and SCOUT_BROKER_API_SECRET"
        )
    if not git_tree_is_clean():
        raise ScoutConfigError("run.mode: LIVE requires a clean git tree")


def _validate_holdout(cfg: ScoutConfig) -> None:
    if cfg.period.split is not PeriodSplit.HOLDOUT:
        return
    if not lockbox_budget_available(Path(cfg.research.lockbox_path)):
        raise ScoutConfigError(
            "period.split: HOLDOUT lockbox budget is exhausted"
        )
    if not git_tree_is_clean():
        raise ScoutConfigError("period.split: HOLDOUT requires a clean git tree")


def _validate_crypto_holdout(cfg: ScoutConfig) -> None:
    if (
        cfg.run.asset_class is AssetClass.CRYPTO
        and cfg.period.split is PeriodSplit.HOLDOUT
        and not _M7_UNLOCKED
    ):
        raise ScoutConfigError(
            "run.asset_class: crypto HOLDOUT is refused until M7 is unlocked"
        )


def git_tree_is_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.returncode == 0 and result.stdout.strip() == ""


def lockbox_budget_available(path: Path) -> bool:
    if not path.is_file():
        return False
    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        used = int(raw["used"])
        budget = int(raw["budget"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoutConfigError(f"{path}: invalid lockbox file") from exc
    return used < budget


__all__ = [
    "git_tree_is_clean",
    "load_config",
    "lockbox_budget_available",
    "validate_config",
]
