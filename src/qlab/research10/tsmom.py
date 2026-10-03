"""Time-series momentum on 9 ETFs, long/short or long/cash (prereg §5.1).

Point in time: the decision after the close of the first trading day of a month uses rows <= t only.
"""

from dataclasses import asdict, dataclass

import numpy as np

from qlab.data.panel import Panel
from qlab.schedule import period_starts

UNIVERSE = ("SPY", "EFA", "EEM", "IEF", "TLT", "TIP", "GLD", "DBC", "VNQ")
MIN_HISTORY = 252
VOL_WINDOW = 63
COST = 0.0005
COST_STRESS = 0.0010
BORROW = 0.005 / 252     # per trading day, on the short value


@dataclass(frozen=True)
class Config:
    kind: str        # LS | LO
    L: int           # lookback in months (21 trading days each)
    weighting: str   # eq | iv

    @property
    def id(self) -> str:
        return f"{self.kind}_L{self.L}_{self.weighting}"

    def as_dict(self) -> dict:
        return {**asdict(self), "study": "research10", "part": "B"}


def grid() -> list[Config]:
    return [Config(k, L, w) for k in ("LS", "LO") for L in (3, 6, 12) for w in ("eq", "iv")]


def decisions(p: Panel, cash_ret: np.ndarray, cols: list[int], cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Monthly decision days and signed weights (T_dec x N) over the columns `cols`."""
    T, N = p.shape
    r = (1 + p.ret_co[:, cols]) * (1 + p.ret_oc[:, cols]) - 1
    idx = np.cumprod(1 + r, axis=0)
    cash_idx = np.cumprod(1 + np.asarray(cash_ret, dtype=float))
    seen = np.cumsum(p.listed[:, cols], axis=0)
    look = 21 * cfg.L
    days, rows = [], []
    for t in np.flatnonzero(period_starts(p.dates, "M")):
        if t < look or t + 1 >= T:
            continue
        ok = seen[t] >= max(MIN_HISTORY, look + 1)
        excess = idx[t] / idx[t - look] - cash_idx[t] / cash_idx[t - look]
        sign = np.where(ok, np.sign(excess), 0.0)
        if cfg.kind == "LO":
            sign = np.maximum(sign, 0.0)
        if cfg.weighting == "eq":
            base = ok.astype(float)
        else:
            vol = r[t - VOL_WINDOW + 1:t + 1].std(axis=0, ddof=1)
            base = np.where(ok & (vol > 0), 1.0 / np.where(vol > 0, vol, 1.0), 0.0)
        w = np.zeros(N)
        if base.sum() > 0:
            w[cols] = sign * base / base.sum()      # LS: sum |w| = 1; LO: the short part stays in cash
        days.append(t)
        rows.append(w)
    return np.array(days, dtype=int), np.array(rows).reshape(len(days), N)
