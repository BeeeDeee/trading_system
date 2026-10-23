"""Featury na uzavřeném baru (SPEC §5.1). Čisté funkce nad numpy poli: hodnota v čase t závisí jen na
barech ≤ t (look-ahead test v tests/test_features.py).

Konvence (kde SPEC nechává volnost, zapsáno v CHANGELOG P2):
- sd a z-skóre s ddof=0 a ignorováním NaN, stejně jako `research/obi_prior/analyze.py` (`np.nanstd`).
- `sigma288` a `beta288` jsou přes 288 barů **bez** aktuálního (stejně jako z-skóre), aby skok aktuálního baru
  nezkreslil vlastní normalizaci.
- `lvz288` je z ln(quote volume).
- ATR a EMA mají pevné okno se seedem na jeho začátku (ATR 288 barů, EMA 2n barů), takže výsledek nezávisí
  na tom, kolik historie je zrovna v paměti, a live i replay dají totéž."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

W = 288
OBI_MIN_VALID = 200
BOOK_PCTS = (0.2, 1.0, 2.0)


@dataclass(frozen=True)
class BarArrays:
    """Sloupce barů jednoho symbolu, vzestupně podle času, souvislé (bez děr)."""

    open_time: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray  # noqa: E741
    c: np.ndarray
    qv: np.ndarray
    tbqv: np.ndarray

    def __len__(self) -> int:
        return len(self.c)

    def upto(self, i: int) -> BarArrays:
        """Pohled na bary 0..i včetně (pro look-ahead test a replay)."""
        return BarArrays(*(a[: i + 1] for a in (self.open_time, self.o, self.h, self.l, self.c, self.qv, self.tbqv)))

    @classmethod
    def from_rows(cls, rows: Sequence[object]) -> BarArrays:
        cols = ("open_time", "o", "h", "l", "c", "qv", "tbqv")
        arr = {k: np.array([getattr(r, k) for r in rows], dtype=np.int64 if k == "open_time" else np.float64)
               for k in cols}
        return cls(**arr)


@dataclass(frozen=True)
class BookArrays:
    """Bar-level book featury zarovnané s BarArrays: obi[p][i], ok[i] (book_ok pro p=1 % definuje platnost)."""

    obi: Mapping[float, np.ndarray]
    ok: np.ndarray

    def upto(self, i: int) -> BookArrays:
        return BookArrays({p: a[: i + 1] for p, a in self.obi.items()}, self.ok[: i + 1])


# --- primitiva (všechna jen z minulosti) ----------------------------------------------------------

def zscore_last(x: np.ndarray, w: int = W, min_valid: int = 1) -> float:
    """z-skóre poslední hodnoty vůči předchozím `w` hodnotám (bez aktuální), NaN se ignorují.

    Jako research `rz`: bez plného okna `w` je výsledek NaN."""
    if len(x) < w + 1:
        return math.nan
    cur = x[-1]
    hist = x[-w - 1:-1]
    hist = hist[np.isfinite(hist)]
    if len(hist) < max(min_valid, 2) or not math.isfinite(cur):
        return math.nan
    sd = float(np.std(hist))
    return (float(cur) - float(np.mean(hist))) / sd if sd > 0 else math.nan


def ret1(c: np.ndarray) -> np.ndarray:
    out = np.full(len(c), np.nan)
    if len(c) > 1:
        out[1:] = np.log(c[1:] / c[:-1])
    return out


def wilder_atr_last(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14, window: int = W) -> float:  # noqa: E741
    """Wilder ATR(n) posledního baru, počítaný přes posledních `window` barů se seedem SMA prvních n TR."""
    k = min(len(c), window + 1)
    if k < n + 1:
        return math.nan
    hh, ll, cc = h[-k:], l[-k:], c[-k:]
    tr = np.maximum(hh[1:] - ll[1:], np.maximum(np.abs(hh[1:] - cc[:-1]), np.abs(ll[1:] - cc[:-1])))
    atr = float(np.mean(tr[:n]))
    for v in tr[n:]:
        atr = (atr * (n - 1) + float(v)) / n
    return atr


def ema_last(c: np.ndarray, n: int) -> float:
    """EMA(n) posledního baru přes okno 2n barů se seedem SMA prvních n."""
    if len(c) < 2 * n:
        return math.nan
    x = c[-2 * n:]
    a = 2.0 / (n + 1)
    e = float(np.mean(x[:n]))
    for v in x[n:]:
        e = a * float(v) + (1 - a) * e
    return e


def donchian(h: np.ndarray, l: np.ndarray, n: int) -> tuple[float, float]:  # noqa: E741
    """max high / min low barů t−n … t−1 (bez aktuálního)."""
    if len(h) < n + 1:
        return math.nan, math.nan
    return float(np.max(h[-n - 1:-1])), float(np.min(l[-n - 1:-1]))


def cvd_series(qv: np.ndarray, tbqv: np.ndarray, m: int = 12) -> np.ndarray:
    """cvd12[i] = Σ(2·tbqv − qv) / Σ qv přes bary i−11 … i."""
    out = np.full(len(qv), np.nan)
    if len(qv) >= m:
        dq = np.convolve(2 * tbqv - qv, np.ones(m), "valid")
        sq = np.convolve(qv, np.ones(m), "valid")
        with np.errstate(divide="ignore", invalid="ignore"):
            out[m - 1:] = np.where(sq > 0, dq / sq, np.nan)
    return out


def beta_last(y: np.ndarray, x: np.ndarray, w: int = W) -> float:
    """OLS beta y na x přes `w` barů bez aktuálního."""
    yy, xx = y[-w - 1:-1], x[-w - 1:-1]
    ok = np.isfinite(yy) & np.isfinite(xx)
    if ok.sum() < w // 2:
        return math.nan
    xv = xx[ok] - xx[ok].mean()
    var = float(np.dot(xv, xv))
    return float(np.dot(xv, yy[ok] - yy[ok].mean()) / var) if var > 0 else math.nan


def sd_last(x: np.ndarray, w: int = W) -> float:
    hist = x[-w - 1:-1]
    hist = hist[np.isfinite(hist)]
    return float(np.std(hist)) if len(hist) >= w // 2 else math.nan


# --- featury baru --------------------------------------------------------------------------------

def compute(bars: BarArrays, book: BookArrays | None = None, btc: BarArrays | None = None,
            donchian_ns: Sequence[int] = (12, 24, 48), ema_ns: Sequence[int] = (576,)) -> dict[str, float]:
    """Featury posledního baru `bars` (čas t = bars.open_time[-1]). `btc` musí končit stejným časem."""
    f: dict[str, float] = {"open_time": float(bars.open_time[-1]), "c": float(bars.c[-1])}
    r = ret1(bars.c)
    f["ret1"] = float(r[-1])
    f["ret1_prev"] = float(r[-2]) if len(r) > 1 else math.nan
    f["sigma288"] = sd_last(r)
    atr = wilder_atr_last(bars.h, bars.l, bars.c)
    f["atr14_pct"] = atr / float(bars.c[-1]) * 100 if math.isfinite(atr) else math.nan
    with np.errstate(divide="ignore"):
        lv = np.log(np.where(bars.qv > 0, bars.qv, np.nan))
    f["lvz288"] = zscore_last(lv)
    for n in donchian_ns:
        f[f"don_hi_{n}"], f[f"don_lo_{n}"] = donchian(bars.h, bars.l, n)
    for n in ema_ns:
        f[f"ema_{n}"] = ema_last(bars.c, n)
    cvd = cvd_series(bars.qv, bars.tbqv)
    f["cvd12"] = float(cvd[-1])
    f["cvd12_z"] = zscore_last(cvd)

    if book is not None:
        ok = book.ok.astype(bool)
        f["book_ok"] = 1.0 if ok[-1] else 0.0
        for p in BOOK_PCTS:
            x = book.obi.get(p)
            if x is None:
                continue
            f[f"obi{p:g}"] = float(x[-1]) if ok[-1] else math.nan
            f[f"obi{p:g}_z"] = zscore_last(np.where(ok, x, np.nan), W, OBI_MIN_VALID) if ok[-1] else math.nan
    else:
        f["book_ok"] = 0.0

    if btc is not None:
        if len(btc) == 0 or btc.open_time[-1] != bars.open_time[-1]:
            raise ValueError("BTC bary musí končit stejným časem jako symbol")
        rb = ret1(btc.c)
        f["ret1_btc"] = float(rb[-1])
        f["ret1_btc_prev"] = float(rb[-2]) if len(rb) > 1 else math.nan
        f["sigma288_btc"] = sd_last(rb)
        f["c_btc"] = float(btc.c[-1])
        for n in ema_ns:
            f[f"ema_{n}_btc"] = ema_last(btc.c, n)
        # beta alt na BTC: zarovnání podle času (oba souvislé a končí stejně → stačí ořez na kratší)
        k = min(len(r), len(rb))
        f["beta288"] = beta_last(r[-k:], rb[-k:])
    return f


def book_arrays(open_time: np.ndarray, rows: Sequence[Mapping[str, Any]],
                pcts: Sequence[float] = BOOK_PCTS) -> BookArrays:
    """Řádky `bar_book` (open_time, pct, obi, book_ok) → BookArrays zarovnané s `open_time`.

    `book_ok` baru se bere z pásma 1 % (všechna pásma jednoho vzorku mají stejný počet vzorků)."""
    idx = {int(t): i for i, t in enumerate(open_time)}
    obi = {p: np.full(len(open_time), np.nan) for p in pcts}
    ok = np.zeros(len(open_time), dtype=np.int8)
    for r in rows:   # dict nebo sqlite3.Row
        i = idx.get(int(r["open_time"]))
        if i is None:
            continue
        p = float(r["pct"])
        if p in obi and r["obi"] is not None:
            obi[p][i] = float(r["obi"])
        if abs(p - 1.0) < 1e-9:
            ok[i] = int(r["book_ok"])
    return BookArrays(obi, ok)
