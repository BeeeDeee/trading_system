"""Candidate grid (spec §6.3): full cartesian grid from configs/frozen_defaults.yaml."""

import itertools
from dataclasses import asdict, dataclass

from qlab.validation.registry import config_hash


@dataclass(frozen=True)
class StrategyConfig:
    family: str
    params: tuple[tuple[str, int], ...]   # sorted (name, value) pairs
    top_n: int = 0                        # 0 for regime_market
    weighting: str = "equal"
    rebalance: str = "weekly"
    hysteresis: float = 1.0
    overlay: str = "none"
    max_weight: float = 0.10

    @property
    def param_dict(self) -> dict[str, int]:
        return dict(self.params)

    @property
    def signal_key(self) -> str:
        return self.family + "".join(f"_{k}{v}" for k, v in self.params)

    @property
    def candidate_id(self) -> str:
        d = asdict(self)
        d["params"] = dict(self.params)
        return config_hash(d)


def _param_sets(ranges: dict[str, list]) -> list[tuple[tuple[str, int], ...]]:
    names = sorted(ranges)
    return [tuple(zip(names, values)) for values in itertools.product(*(ranges[n] for n in names))]


def build_grid(grid_cfg: dict) -> list[StrategyConfig]:
    families = grid_cfg["families"]
    port = grid_cfg["portfolio"]
    out = []
    for family, ranges in families.items():
        if family == "regime_market":
            out += [StrategyConfig(family, p, rebalance="weekly") for p in _param_sets(ranges)]
            continue
        for params in _param_sets(ranges):
            for top_n, weighting, rebalance, hyst, overlay in itertools.product(
                    port["top_n"], port["weighting"], port["rebalance"], port["hysteresis"],
                    port["overlay"]):
                out.append(StrategyConfig(family, params, top_n, weighting, rebalance, hyst,
                                          overlay, grid_cfg["max_weight"]))
    ids = [c.candidate_id for c in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate ids in grid")
    return out


def neighbours(a: StrategyConfig, b: StrategyConfig, grid_cfg: dict) -> bool:
    """True if b differs from a by at most one grid step in every ordered dimension (spec §9.5).

    Categorical dimensions (weighting, rebalance, overlay) must be equal.
    """
    if (a.family, a.weighting, a.rebalance, a.overlay) != (b.family, b.weighting, b.rebalance,
                                                           b.overlay):
        return False
    ranges = dict(grid_cfg["families"][a.family])
    if a.family != "regime_market":
        ranges.update(top_n=grid_cfg["portfolio"]["top_n"],
                      hysteresis=grid_cfg["portfolio"]["hysteresis"])
    va = {**a.param_dict, "top_n": a.top_n, "hysteresis": a.hysteresis}
    vb = {**b.param_dict, "top_n": b.top_n, "hysteresis": b.hysteresis}
    return all(abs(ranges[k].index(va[k]) - ranges[k].index(vb[k])) <= 1 for k in ranges)
