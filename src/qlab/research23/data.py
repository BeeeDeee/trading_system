"""5m BTC/ETH perp data, stages, vault and registry of research 23 (prereg §3).

A candle index is `epoch_ms // 300_000`, so candle t covers [t, t+1) and its close is known at t+1. The UTC
session of t is `t // 288` and its position in the session `t % 288`. Missing candles carry the previous close,
have zero volume and `real = False`. `fund[t]` is the perp funding rate settled at the start of candle t.
"""

import io
import os
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from qlab.validation.registry import TrialRegistry, current_commit
from qlab.validation.vault import Vault

SYMBOLS = ("BTCUSDT", "ETHUSDT")
STEP_MS = 300_000
PER_DAY = 288
GRID_START, GRID_END = date(2020, 1, 1), date(2026, 9, 30)
DEV = (date(2020, 1, 1), date(2022, 12, 31))
HOLDOUT = (date(2023, 1, 1), date(2026, 9, 30))
SUBPERIODS = {"2023-01..2024-12": (date(2023, 1, 1), date(2024, 12, 31)),
              "2025-01..2026-09": (date(2025, 1, 1), date(2026, 9, 30))}
FINAL_ENV = "QLAB_R23_METHODOLOGY"
REGISTRY = Path("runs/registry/research23.sqlite")


def epoch_5m(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()) * 1000 // STEP_MS


def to_dates(t: np.ndarray) -> np.ndarray:
    return (np.asarray(t, np.int64) * 300).astype("datetime64[s]").astype("datetime64[D]")


def cache_path(snapshot: str, sym: str) -> Path:
    return Path("data/derived") / snapshot / "r23_5m" / f"{sym}.npz"


def _read_zip(f: Path) -> np.ndarray:
    with zipfile.ZipFile(f) as z:
        txt = z.read(z.namelist()[0]).decode()
    lines = [ln for ln in txt.splitlines() if ln and ln[0].isdigit()]
    return np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",", ndmin=2, usecols=range(8)) if lines \
        else np.empty((0, 8))


def build(snapshot: str, sym: str) -> Path:
    """Raw kline + funding zips -> one npz on the full grid (t, o, h, l, c, v, qv, real, fund)."""
    raw = Path("data/raw") / snapshot
    t0 = epoch_5m(GRID_START)
    n = epoch_5m(GRID_END) + PER_DAY - t0
    a = np.vstack([r for r in (_read_zip(f) for f in sorted((raw / "perp_5m" / sym).glob(f"{sym}-5m-*.zip")))
                   if len(r)])
    ms = np.where(a[:, 0] > 1e14, a[:, 0] / 1000, a[:, 0]).astype(np.int64)
    i = ms // STEP_MS - t0
    ok = (i >= 0) & (i < n)
    a, i = a[ok], i[ok]
    i, first = np.unique(i, return_index=True)
    a = a[first]
    real = np.zeros(n, bool)
    real[i] = True
    c = np.full(n, np.nan)
    c[i] = a[:, 4]
    idx = np.maximum.accumulate(np.where(np.isfinite(c), np.arange(n), 0))
    c = c[idx]
    out = {"t": t0 + np.arange(n, dtype=np.int64), "c": c, "real": real}
    for k, col in (("o", 1), ("h", 2), ("l", 3)):
        x = c.copy()
        x[i] = a[:, col]
        out[k] = x
    for k, col in (("v", 5), ("qv", 7)):
        x = np.zeros(n)
        x[i] = a[:, col]
        out[k] = x
    fund = np.zeros(n)
    for f in sorted((raw / "funding" / sym).glob(f"{sym}-fundingRate-*.zip")):
        with zipfile.ZipFile(f) as z:
            for ln in z.read(z.namelist()[0]).decode().splitlines():
                p = ln.split(",")
                if len(p) >= 3 and p[0].isdigit():
                    j = int(p[0]) // STEP_MS - t0
                    if 0 <= j < n:
                        fund[j] += float(p[2])
    out["fund"] = fund
    p = cache_path(snapshot, sym)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **out)
    return p


def _load(snapshot: str, sym: str, end_date: date | None) -> dict[str, np.ndarray]:
    with np.load(cache_path(snapshot, sym)) as z:
        d = {k: z[k] for k in z.files}
    if end_date is not None:                                    # the dev stage never sees later candles
        n = int(epoch_5m(end_date) + PER_DAY - d["t"][0])
        d = {k: v[:n].copy() for k, v in d.items()}
    return d


def vault() -> Vault:
    return Vault(DEV[1], Path("runs/vault_r23/frozen.lock"), Path("runs/vault_r23/vault.log"))


def load_dev(snapshot: str, syms=SYMBOLS) -> dict[str, dict[str, np.ndarray]]:
    v = vault()
    out = {s: _load(snapshot, s, DEV[1]) for s in syms}
    for d in out.values():
        v.check_dev_only(to_dates(d["t"]))
    return out


def load_final(snapshot: str, syms=SYMBOLS) -> dict[str, dict[str, np.ndarray]]:
    vault().check_final_session(os.environ.get(FINAL_ENV, ""), snapshot, current_commit())
    return {s: _load(snapshot, s, None) for s in syms}


def registry() -> TrialRegistry:
    return TrialRegistry(REGISTRY)
