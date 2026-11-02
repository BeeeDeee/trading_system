"""H1b: ETF Donchian ensemble, inverse-volatility sizing, fixed denominator (prereg §4.2)."""

from dataclasses import asdict, dataclass

import numpy as np

from qlab.data.panel import Panel
from qlab.engine.costs import CostModel
from qlab.engine.vector import Decisions, SimResult, simulate
from qlab.features.basic import tr_index
from qlab.research19 import setup as S
from qlab.research19.signals import ensemble, realized_vol


@dataclass(frozen=True)
class Config:
    name: str = "h1b"
    tickers: tuple[str, ...] = tuple(S.ETF_TREND)
    lookbacks: tuple[int, ...] = S.LOOKBACKS_ETF
    vol_target: float = S.ETF_VOL_TARGET
    leverage_cap: float = S.LEVERAGE_CAP
    band: float | None = S.BAND
    cost_mult: float = 1.0

    def as_dict(self) -> dict:
        return {**asdict(self), "study": "research19"}


def etf_cost(dates: np.ndarray, shape: tuple[int, int], mult: float = 1.0) -> np.ndarray:
    """3 bps per side (x2.5 before 2001-04-09) as a (T, N) matrix."""
    model = CostModel(floor_bps=S.ETF_COST_BPS, tiers=(), default_bps=S.ETF_COST_BPS,
                      pre_decimal_multiplier=S.PRE_DECIMAL_MULT, multiplier=mult)
    return model.rate(np.zeros(shape), dates)


def signals(p: Panel, cols: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    P = tr_index(p)
    valid = np.asarray(p.listed & p.tradable)
    E, sig = np.zeros(p.shape), np.full(p.shape, np.nan)
    E[:, cols] = ensemble(P[:, cols], valid[:, cols], cfg.lookbacks)
    sig[:, cols] = realized_vol(P[:, cols], valid[:, cols], S.VOL_WINDOW, S.ETF_YEAR)
    return E, sig


def target_weights(E: np.ndarray, sig: np.ndarray, cols: np.ndarray, cfg: Config,
                   scale: np.ndarray | None = None) -> np.ndarray:
    """w_i = E_i * min(cap, vol_target / sigma_i) / len(basket) (fixed denominator); scaled to sum 1 if above."""
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.minimum(cfg.leverage_cap, cfg.vol_target / sig)
    w = np.zeros(E.shape)
    w[:, cols] = np.where(np.isfinite(size[:, cols]), E[:, cols] * size[:, cols], 0.0) / len(cols)
    if scale is not None:
        w = w * np.asarray(scale, dtype=float)[:, None]
    tot = w.sum(axis=1, keepdims=True)
    return np.where(tot > 1.0, w / np.maximum(tot, 1e-300), w)


def columns(panel_tickers: list[str], wanted) -> np.ndarray:
    return np.array([panel_tickers.index(t) for t in wanted], dtype=int)


def run_panel(p: Panel, panel_tickers: list[str], cfg: Config, first: int, cash_ret: np.ndarray,
              E_override: np.ndarray | None = None, scale: np.ndarray | None = None,
              signal_cache: tuple | None = None) -> SimResult:
    cols = columns(panel_tickers, cfg.tickers)
    E, sig = signal_cache or signals(p, cols, cfg)
    w = target_weights(E if E_override is None else E_override, sig, cols, cfg, scale)
    days = np.arange(first, p.shape[0])
    return simulate(p, Decisions(days, w[days]), etf_cost(p.dates, p.shape, cfg.cost_mult), cash_ret, cfg.band)


def run(cfg: Config, scale: np.ndarray | None = None) -> SimResult:
    p, ex, tickers = S.load_etf()
    first = S.index_of(p, S.PERIOD_H1B[0]) - 1
    return run_panel(p, tickers, cfg, first, np.asarray(ex["cash_ret"]), scale=scale)
