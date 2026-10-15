"""Intraday crypto panel (15m, later 5m) for the scalping branch: raw Binance kline zips -> (T, N) float32 matrices.

    build(raw_root, out_dir, tf)      one directory of .npy matrices on a common UTC grid + meta.json
    load(out_dir, end=None)           -> IntradayView (dev stage: rows before HOLDOUT_FROM only)

Fields (all perpetual futures, `real` = a candle exists): o h l c, qv (quote volume), nt (number of trades),
tbq (taker-buy quote volume), funding (rate a long pays, on the candle that contains the settlement, else 0).
A missing candle repeats the previous close (o = h = l = c, volume 0) and is not tradable.

Row t is the candle that OPENS at `times[t]` and closes at `times[t] + tf`; it is known only after it closed. The
universe is point-in-time: the `top_n` perpetuals by the 30-day quote volume up to the previous UTC day.
"""

import io
import json
from functools import cached_property
import zipfile
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

import numpy as np

TF_MINUTES = {"15m": 15, "5m": 5}
GRID_START = date(2020, 1, 1)
GRID_END = date(2026, 9, 30)               # last day with candles in the snapshot
HOLDOUT_FROM = date(2023, 1, 1)            # lab-wide boundary of crypto (gates.yaml / PLAN §6.1)
FIELDS = ("o", "h", "l", "c", "qv", "nt", "tbq", "funding")
VOLUME_WINDOW_DAYS = 30
FUNDING_STEP_H = 8


@dataclass(frozen=True)
class IntradayView:
    """The data a scalping strategy sees. All matrices are (T, N) over `instruments`."""
    times: np.ndarray                       # datetime64[m] (T,), candle open time (UTC)
    tf: str                                 # "15m" | "5m"
    instruments: tuple[str, ...]
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    qv: np.ndarray                          # quote volume (USD) of the candle, 0 where no candle
    nt: np.ndarray                          # number of trades
    tbq: np.ndarray                         # taker-buy quote volume (USD)
    funding: np.ndarray                     # rate a long pays at the settlement candle (0 elsewhere)
    real: np.ndarray                        # bool: a candle exists
    listed: np.ndarray                      # bool: the instrument has traded at least once up to this row
    universe: np.ndarray                    # bool: PIT liquidity membership
    extras: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.c.shape

    @property
    def bars_per_day(self) -> int:
        return 1440 // TF_MINUTES[self.tf]

    @cached_property
    def tradable(self) -> np.ndarray:
        return self.real & self.listed

    def col(self, name: str) -> int:
        return self.instruments.index(name)

    def truncate(self, n: int) -> "IntradayView":
        return replace(self, **{k: getattr(self, k)[:n] for k in
                                ("times", "o", "h", "l", "c", "qv", "nt", "tbq", "funding", "real", "listed",
                                 "universe")})

    @cached_property
    def ret_co(self) -> np.ndarray:         # previous close -> open (a 24/7 market: ~0, the funding bar excluded)
        r = np.zeros(self.shape)
        r[1:] = self.o[1:].astype(float) / self.c[:-1] - 1.0
        return np.where(self.tradable, np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    @cached_property
    def ret_oc(self) -> np.ndarray:
        return np.where(self.tradable, np.nan_to_num(self.c.astype(float) / self.o - 1.0, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    def periods_per_year(self) -> float:
        return 365.0 * self.bars_per_day


# ---------------------------------------------------------------------------- build from the raw zips

def _read_zip(f: Path) -> np.ndarray:
    with zipfile.ZipFile(f) as z:
        txt = z.read(z.namelist()[0]).decode()
    lines = [ln for ln in txt.splitlines() if ln and ln[0].isdigit()]
    if not lines:
        return np.empty((0, 11))
    return np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",", ndmin=2, usecols=range(11))


def _open_ms(t: np.ndarray) -> np.ndarray:
    return np.where(t > 1e14, t / 1000, t).astype(np.int64)         # microseconds since 2025


def _funding(files: list[Path], t0_ms: int, step_ms: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    for f in files:
        with zipfile.ZipFile(f) as z:
            lines = z.read(z.namelist()[0]).decode().splitlines()[1:]
        for ln in lines:
            p = ln.split(",") if ln else []
            if len(p) >= 3:
                i = (int(p[0]) - t0_ms) // step_ms
                if 0 <= i < n:
                    out[i] += float(p[2])
    return out


def build(raw_root: Path, out_dir: Path, tf: str = "15m", symbols: list[str] | None = None,
          start: date = GRID_START, end: date = GRID_END) -> dict:
    """`raw_root` = the Binance snapshot directory (contains perp_<tf>/<SYM>/*.zip and funding/<SYM>/*.zip)."""
    step_min = TF_MINUTES[tf]
    step_ms = step_min * 60_000
    t0 = int(np.datetime64(start, "m").astype("int64")) * 60_000
    n = (int(np.datetime64(end, "D").astype("int64")) + 1 - int(np.datetime64(start, "D").astype("int64"))) * 1440 // step_min
    src = raw_root / f"perp_{tf}"
    symbols = sorted(symbols or [p.name for p in src.iterdir() if p.is_dir()])
    out_dir.mkdir(parents=True, exist_ok=True)
    mm = {k: np.lib.format.open_memmap(out_dir / f"{k}.npy", mode="w+", dtype=np.float32, shape=(n, len(symbols)))
          for k in FIELDS}
    real = np.lib.format.open_memmap(out_dir / "real.npy", mode="w+", dtype=bool, shape=(n, len(symbols)))
    audit = {}
    for j, sym in enumerate(symbols):
        rows = [_read_zip(f) for f in sorted((src / sym).glob(f"{sym}-{tf}-*.zip"))]
        rows = [r for r in rows if len(r)]
        if not rows:
            audit[sym] = {"candles": 0}
            continue
        a = np.vstack(rows)
        i = (_open_ms(a[:, 0]) - t0) // step_ms
        ok = (i >= 0) & (i < n)
        a, i = a[ok], i[ok].astype(np.int64)
        _, first = np.unique(i, return_index=True)                      # duplicates: keep the first
        a, i = a[first], i[first]
        r = np.zeros(n, dtype=bool)
        r[i] = True
        c = np.full(n, np.nan)
        c[i] = a[:, 4]
        c = _ffill(c)
        for k, col in (("o", 1), ("h", 2), ("l", 3)):
            x = c.copy()
            x[i] = a[:, col]
            mm[k][:, j] = x
        mm["c"][:, j] = c
        for k, col in (("qv", 7), ("nt", 8), ("tbq", 10)):
            x = np.zeros(n)
            x[i] = a[:, col]
            mm[k][:, j] = x
        mm["funding"][:, j] = _funding(sorted((raw_root / "funding" / sym).glob(f"{sym}-fundingRate-*.zip")),
                                       t0, step_ms, n)
        real[:, j] = r
        audit[sym] = {"candles": int(r.sum()), "first": str((np.datetime64(t0 // 60_000 + int(i[0]) * step_min, "m"))),
                      "gaps": int(((np.diff(i) > 1)).sum())}
    for m in (*mm.values(), real):
        m.flush()
    meta = {"tf": tf, "symbols": symbols, "start": str(start), "end": str(end), "n_rows": n, "audit": audit}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _ffill(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[: int(np.argmax(ok)) if ok.any() else len(x)] = np.nan
    return out


# ---------------------------------------------------------------------------- load

def pit_universe(qv: np.ndarray, real: np.ndarray, bars_per_day: int, top_n: int,
                 window_days: int = VOLUME_WINDOW_DAYS) -> np.ndarray:
    """(T, N) bool: the `top_n` instruments by quote volume of the `window_days` days before the current UTC day
    (so a row never uses its own day). Rows before `window_days` full days have no members."""
    T, N = qv.shape
    days = T // bars_per_day
    daily = qv[: days * bars_per_day].reshape(days, bars_per_day, N).sum(axis=1)
    csum = np.vstack([np.zeros((1, N)), np.cumsum(daily, axis=0)])
    trail = np.full((days, N), np.nan)
    for d in range(window_days, days):
        trail[d] = csum[d] - csum[d - window_days]                      # days d-window .. d-1
    rank = np.argsort(np.argsort(-np.nan_to_num(trail, nan=-1.0), axis=1), axis=1)
    member_day = (rank < top_n) & np.isfinite(trail) & (np.nan_to_num(trail) > 0)
    member = np.repeat(member_day, bars_per_day, axis=0)
    out = np.zeros((T, N), dtype=bool)
    out[: len(member)] = member
    return out & real


def load(out_dir: Path, top_n: int = 20, stage: str = "dev", symbols: list[str] | None = None) -> IntradayView:
    """`stage`: "dev" rows before HOLDOUT_FROM, "full" everything (G4 only)."""
    meta = json.loads((out_dir / "meta.json").read_text())
    tf = meta["tf"]
    t0 = np.datetime64(meta["start"], "m")
    times = t0 + np.arange(meta["n_rows"]) * np.timedelta64(TF_MINUTES[tf], "m")
    n = meta["n_rows"] if stage == "full" else int((times < np.datetime64(HOLDOUT_FROM, "m")).sum())
    keep = [meta["symbols"].index(s) for s in symbols] if symbols else list(range(len(meta["symbols"])))
    names = tuple(meta["symbols"][k] for k in keep)
    bpd = 1440 // TF_MINUTES[tf]
    rd = lambda k, cols: np.load(out_dir / f"{k}.npy", mmap_mode="r")[:n][:, cols].astype(np.float32)
    real_all = rd("real", keep).astype(bool)
    uni_all = pit_universe(rd("qv", keep), real_all, bpd, top_n)
    if not symbols:                                                    # a symbol that is never a member is never traded:
        keep = [k for k, m in zip(keep, uni_all.any(axis=0)) if m]     # leave it out (the 5m panel would not fit the judge)
    names = tuple(meta["symbols"][k] for k in keep)
    a = {k: rd(k, keep) for k in FIELDS}
    real = rd("real", keep).astype(bool)
    listed = np.maximum.accumulate(real, axis=0)
    uni = pit_universe(a["qv"], real, bpd, top_n)
    return IntradayView(times=times[:n], tf=tf, instruments=names, real=real, listed=listed, universe=uni, **a)
