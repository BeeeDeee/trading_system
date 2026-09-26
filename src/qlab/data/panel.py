"""Dense date x asset matrices built from `bars`; the input format of the engines."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from qlab.data.schema import BARS_SCHEMA, DELISTED, validate


@dataclass(frozen=True)
class Panel:
    dates: np.ndarray          # datetime64[D], shape (T,)
    assets: np.ndarray         # int64 permatickers, shape (N,)
    ret_co: np.ndarray         # (T, N) overnight total return; 0 where not listed
    ret_oc: np.ndarray         # (T, N) intraday total return; 0 where not listed
    tradable: np.ndarray       # (T, N) bool; orders can fill at the open
    listed: np.ndarray         # (T, N) bool; a bars row exists (incl. halted and delisting rows)
    delisting: np.ndarray      # (T, N) bool; DELISTED row: position is paid out at this open
    close_u: np.ndarray        # (T, N) unadjusted close; NaN where no price
    dollar_volume: np.ndarray  # (T, N); NaN where no price

    @property
    def shape(self) -> tuple[int, int]:
        return self.ret_co.shape

    def slice(self, end: int) -> "Panel":
        """Rows [0, end) – the view of the world up to (excluding) day `end`."""
        return Panel(self.dates[:end], self.assets, *(getattr(self, f)[:end] for f in _MATRICES))


_MATRICES = ("ret_co", "ret_oc", "tradable", "listed", "delisting", "close_u", "dollar_volume")


def panel_from_bars(bars: pl.DataFrame, calendar) -> Panel:
    validate(bars, BARS_SCHEMA, "bars")
    dates = np.asarray(pl.Series(calendar, dtype=pl.Date).unique().sort().to_numpy(),
                       dtype="datetime64[D]")
    assets = np.sort(bars["permaticker"].unique().to_numpy())
    t = np.searchsorted(dates, bars["date"].to_numpy().astype("datetime64[D]"))
    n = np.searchsorted(assets, bars["permaticker"].to_numpy())
    if (t >= len(dates)).any() or (dates[np.minimum(t, len(dates) - 1)]
                                   != bars["date"].to_numpy().astype("datetime64[D]")).any():
        raise ValueError("bars contain dates outside the calendar")

    shape = (len(dates), len(assets))

    def dense(values, fill, dtype):
        out = np.full(shape, fill, dtype=dtype)
        out[t, n] = values
        return out

    return Panel(
        dates=dates,
        assets=assets,
        ret_co=dense(bars["ret_co"].to_numpy(), 0.0, np.float64),
        ret_oc=dense(bars["ret_oc"].to_numpy(), 0.0, np.float64),
        tradable=dense(bars["tradable"].to_numpy(), False, bool),
        listed=dense(True, False, bool),
        delisting=dense((bars["status"] == DELISTED).to_numpy(), False, bool),
        close_u=dense(bars["close_u"].to_numpy(), np.nan, np.float64),
        dollar_volume=dense(bars["dollar_volume"].to_numpy(), np.nan, np.float64),
    )
