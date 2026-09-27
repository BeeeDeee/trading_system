"""STR-TF universe features (research 5 pre-registration §3), per security, point in time.

A security is in the universe at t if `base_ok` (unadjusted close >= 5, >= 252 bars, not a SPAC at
t) and `adv20 >= min_adv`, both from bars up to and including t.
"""

import numpy as np
import polars as pl

from qlab.universe import eligibility_features

MIN_PRICE = 5.0
MIN_HISTORY = 252
ADV_WINDOW = 20
MIN_ADV_PANEL = 5e6  # widest grid point; the panel holds every security that ever passed it


def r5_features(bars: pl.DataFrame, sic: pl.DataFrame) -> pl.DataFrame:
    """(permaticker, date, adv20, base_ok, sic) for every non-delisting bar."""
    f = (eligibility_features(bars, sic)
         .with_columns(adv20=pl.col("dollar_volume").fill_null(0.0)
                       .rolling_mean(ADV_WINDOW, min_samples=ADV_WINDOW).over("permaticker"),
                       base_ok=(pl.col("close_u") >= MIN_PRICE) & (pl.col("n_bars") >= MIN_HISTORY)
                       & ~pl.col("spac"))
         .with_columns(pl.col("base_ok").fill_null(False)))
    iv = sic.select("permaticker", "valid_from", "valid_to", "sic").sort("valid_from")
    f = (f.sort("date")
         .join_asof(iv, left_on="date", right_on="valid_from", by="permaticker", strategy="backward")
         .with_columns(sic=pl.when(pl.col("date") < pl.col("valid_to")).then(pl.col("sic")))
         .sort("permaticker", "date"))
    return f.select("permaticker", "date", "adv20", "base_ok", pl.col("sic").cast(pl.Int16))


def universe_mask(base_ok: np.ndarray, adv20: np.ndarray, min_adv: float) -> np.ndarray:
    """(T, N) bool universe for one liquidity threshold."""
    with np.errstate(invalid="ignore"):
        return np.asarray(base_ok, bool) & (np.asarray(adv20) >= min_adv)


# SIC divisions (sector breakdown), by the first two digits of the SIC code.
SIC_DIVISIONS = ((1, 9, "Agriculture"), (10, 14, "Mining"), (15, 17, "Construction"),
                 (20, 39, "Manufacturing"), (40, 49, "Transport & Utilities"),
                 (50, 51, "Wholesale"), (52, 59, "Retail"), (60, 67, "Finance"),
                 (70, 89, "Services"), (91, 99, "Public Administration"))


def sic_division(sic: np.ndarray) -> np.ndarray:
    two = np.asarray(sic) // 100
    out = np.full(two.shape, "Unknown", dtype=object)
    for lo, hi, name in SIC_DIVISIONS:
        out[(two >= lo) & (two <= hi)] = name
    return out
