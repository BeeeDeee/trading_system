"""Cost per unit of traded value, per instrument, by asset class. The agent never chooses its costs.

- US ETF: 5 bps per side (qlab cost model floor; liquid ETFs), x2.5 before decimalization (2001-04-09).
- US equity: qlab liquidity tiers (`CostModel`: 5 bps floor, 3/6/12 bps half-spread by liquidity rank
  <= 200/500/1000, 25 bps beyond, x2.5 before decimalization), rank from `extras["liq_rank"]` (PIT).
- Crypto spot: 10 bps taker fee + slippage by trailing 30-day median quote volume, tiers of the crypto
  paper bot (>= 500 M USD: 5 bps, >= 100 M: 10, >= 20 M: 25, else 50). Uses volume up to day t-1 only.
- Synthetic: 5 bps.
`multiplier` scales everything for the G2 cost stress.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from qlab.engine.costs import DECIMALIZATION, CostModel

from lab.framework.data import DataView

CRYPTO_FEE_BPS = 10.0
CRYPTO_TIERS = ((500e6, 5.0), (100e6, 10.0), (20e6, 25.0), (0.0, 50.0))
FLAT_BPS = {"us_etf": 5.0, "synthetic": 5.0}


def cost_rates(view: DataView, multiplier: float = 1.0) -> np.ndarray:
    T, N = view.shape
    bps = np.zeros((T, N))
    early = (view.dates < DECIMALIZATION)[:, None]
    for j, cls in enumerate(view.asset_class):
        if cls in FLAT_BPS:
            bps[:, j] = FLAT_BPS[cls] * np.where(early[:, 0] & (cls == "us_etf"), 2.5, 1.0)
        elif cls == "us_equity":
            rank = view.extras.get("liq_rank")
            bps[:, j] = 1e4 * CostModel().rate(np.full(T, np.nan) if rank is None else rank[:, j], view.dates)
        elif cls in ("crypto_spot", "crypto_perp"):
            bps[:, j] = CRYPTO_FEE_BPS + _crypto_slippage(view.dollar_volume[:, j])
        else:
            raise NotImplementedError(f"no cost model for asset class {cls!r} yet")
    return multiplier * bps / 1e4


def _crypto_slippage(qv: np.ndarray, window: int = 30) -> np.ndarray:
    v = np.nan_to_num(np.asarray(qv, dtype=float))
    pad = np.concatenate([np.zeros(window), v[:-1]])           # volume up to t-1
    med = np.median(sliding_window_view(pad, window), axis=-1)[: len(v)]
    out = np.full(len(v), CRYPTO_TIERS[-1][1])
    for floor, bps in reversed(CRYPTO_TIERS):
        out = np.where(med >= floor, bps, out)
    return out
