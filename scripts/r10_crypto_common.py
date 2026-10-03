"""Shared runner of research 10 part C (prereg §5.2) for the dev and the final script."""

import numpy as np

from qlab.data.panel import Panel
from qlab.engine.vector import simulate
from qlab.research9 import panel as P9, strategy as S9
from qlab.research10 import crypto as C
from qlab.research10.sim import monthly_mix, simulate_signed
from qlab.validation.metrics import max_drawdown
from qlab.validation.stats import sharpe

ANN = np.sqrt(365)


def stats(r: np.ndarray) -> dict:
    nav = np.cumprod(1 + r)
    return {"cagr": float(nav[-1] ** (365 / len(r)) - 1), "sharpe": float(sharpe(r) * ANN),
            "vol": float(r.std(ddof=1) * ANN), "max_dd": float(max_drawdown(nav)[0])}


class Runner:
    def __init__(self, p: Panel, extra: dict, syms: list[str], fund: np.ndarray, first: int):
        self.p, self.first = p, first
        self.btc, self.eth = syms.index("BTCUSDT"), syms.index("ETHUSDT")
        assert not p.delisting[:, [self.btc, self.eth]].any()
        self.qv = np.asarray(extra["qv"])
        self.cost = {1.0: P9.cost_rate(self.qv), 2.0: P9.cost_rate(self.qv, 2.0)}
        self.fund = fund
        self.fund_stress = np.full_like(fund, C.FUNDING_STRESS)

    def signals(self, n: int) -> dict[int, np.ndarray]:
        return {j: C.signal(self.p.close_u[:, j], n) for j in (self.btc, self.eth)}

    def hold(self, mult: float = 1.0) -> np.ndarray:
        return simulate(self.p, S9.monthly_5050(self.p, self.btc, self.eth, self.first), self.cost[mult]).returns

    def lc(self, n: int, mult: float = 1.0) -> np.ndarray:
        cfg = S9.Config("T", "sma", n)
        return S9.run(self.p, self.qv, cfg, self.cost[mult], self.first, self.btc, self.eth).returns

    def signed(self, n: int, short_only: bool, mult: float, stress_funding: bool):
        d, w = C.decisions(self.signals(n), self.p.shape[1], self.first, short_only=short_only)
        carry = self.fund_stress if stress_funding else self.fund
        return simulate_signed(self.p.ret_co, self.p.ret_oc, d, w, self.cost[mult], None, carry, self.p.tradable)

    def run(self, cfg: C.Config, mult: float = 1.0, stress_funding: bool = False) -> np.ndarray:
        if cfg.kind == "LS":
            return self.signed(cfg.n, False, mult, stress_funding).returns
        short = self.signed(cfg.n, True, mult, stress_funding).returns
        return monthly_mix(np.column_stack([self.hold(mult), short]), np.array([0.5, 0.5]), self.p.dates)
