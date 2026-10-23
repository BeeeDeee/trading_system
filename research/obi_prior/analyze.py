"""Explorativní test: má order book / taker flow / OI na 5m prediktivní sílu větší než náklady (~12 bps round trip)?"""
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

D = np.load(sys.argv[1])
O = np.load(sys.argv[2])
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
H = [1, 12, 48]
W = 288


def rz(x, w=W):
    """z-skóre vůči předchozím w hodnotám (bez aktuální)."""
    out = np.full_like(x, np.nan)
    if len(x) <= w:
        return out
    win = swv(x[:-1], w)  # win[i] = x[i : i+w] → pro index i+w
    mu = np.nanmean(win, axis=1)
    sd = np.nanstd(win, axis=1)
    out[w:] = (x[w:] - mu) / np.where(sd > 0, sd, np.nan)
    return out


def rsum(x, m):
    out = np.full_like(x, np.nan)
    out[m - 1:] = swv(x, m).sum(axis=1)
    return out


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return np.corrcoef(ra, rb)[0, 1], ok.sum()


def features(sym):
    k = D[f"{sym}_k"]
    ot, o, h, l, c, qv, tbqv = k.T
    n = len(c)
    lc = np.log(c)
    ret1 = np.r_[np.nan, np.diff(lc)] * 1e4
    f = {}
    f["ret1"] = ret1
    dq = 2 * tbqv - qv
    f["dlt"] = dq / qv  # taker imbalance baru
    f["cvd12"] = rsum(dq, 12) / rsum(qv, 12)
    f["ret12"] = np.r_[np.full(12, np.nan), lc[12:] - lc[:-12]] * 1e4
    # book: průměr snapshotů uvnitř baru, imbalance na pásmech
    bd = D[f"{sym}_bd"]
    close_t = ot + 300_000
    for band in (0.2, 1.0, 2.0, 5.0):
        bid = bd[bd[:, 1] == -band]
        ask = bd[bd[:, 1] == band]
        tb, ta = bid[:, 0], ask[:, 0]
        # párování podle timestampu
        common, ib, ia = np.intersect1d(tb, ta, return_indices=True)
        imb = (bid[ib, 2] - ask[ia, 2]) / (bid[ib, 2] + ask[ia, 2])
        idx = np.searchsorted(ot, common, side="right") - 1  # bar, do kterého snapshot patří
        ok = (idx >= 0) & (common < close_t[np.clip(idx, 0, n - 1)])
        s = np.bincount(idx[ok], weights=imb[ok], minlength=n)
        cnt = np.bincount(idx[ok], minlength=n)
        f[f"obi{band:g}"] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        # změna celkové hloubky (stažení likvidity) přes bar
        tot = np.log(bid[ib, 2] + ask[ia, 2])
        s2 = np.bincount(idx[ok], weights=tot[ok], minlength=n)
        f[f"dep{band:g}"] = np.where(cnt > 0, s2 / np.maximum(cnt, 1), np.nan)
    # OI z metrics (5m), hodnota k času close baru
    m = O[sym]
    m = m[(m[:, 1] > 0)]
    m = m[np.argsort(m[:, 0])]
    # label metrics řádku = začátek 5m periody, hodnota platí na jejím konci → OI k close baru i má label ot[i]
    j = np.searchsorted(m[:, 0], ot)
    jj = np.clip(j, 0, len(m) - 1)
    oi = np.where(m[jj, 0] == ot, m[jj, 1], np.nan)
    f["doi"] = np.r_[np.nan, np.diff(np.log(oi))] * 1e4
    f["doi12"] = np.r_[np.full(12, np.nan), np.log(oi[12:] / oi[:-12])] * 1e4
    fwd = {hh: np.r_[lc[hh:] - lc[:-hh], np.full(hh, np.nan)] * 1e4 for hh in H}
    return f, fwd, c


print("=== IC (Spearman) feature_t vs fwd return; t ~ IC*sqrt(N/h); pooled přes 5 symbolů ===")
allf = {s: features(s) for s in SYMS}
names = ["dlt", "cvd12", "obi0.2", "obi1", "obi2", "obi5", "doi", "doi12"]
print(f"{'feature':8s} {'contemp':>8s} " + " ".join(f"{'h='+str(hh):>14s}" for hh in H))
for nm in names:
    row = []
    ics = [spearman(allf[s][0][nm], allf[s][0]["ret1"])[0] for s in SYMS]
    row.append(f"{np.mean(ics):8.3f}")
    for hh in H:
        r = [spearman(allf[s][0][nm], allf[s][1][hh]) for s in SYMS]
        ic = np.mean([x[0] for x in r]); N = sum(x[1] for x in r)
        row.append(f"{ic:7.3f} (t{ic*np.sqrt(N/hh):5.1f})")
    print(f"{nm:8s} " + " ".join(row))

print("\n=== Decily: (top − bottom)/2 průměrný fwd výnos v bps = hrubá hrana na obchod long-top/short-bottom ===")
print("náklad ~12 bps round trip. Hodnoty po symbolech BTC ETH SOL XRP DOGE")
for nm in names + ["ret1", "ret12"]:
    for hh in (12, 48):
        vals = []
        for s in SYMS:
            x, y = allf[s][0][nm], allf[s][1][hh]
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            q1, q9 = np.quantile(x, [0.1, 0.9])
            vals.append((y[x >= q9].mean() - y[x <= q1].mean()) / 2)
        print(f"{nm:8s} h={hh:3d}: " + " ".join(f"{v:7.1f}" for v in vals))

print("\n=== Podmíněné signály (z-skóre vůči 288 barům), průměr výnosu ve směru signálu, bps; n = počet signálů ===")


NS = []


def cond(name, fn, hh):
    tot_n, tot_s, per = 0, 0.0, []
    NS.clear()
    for s in SYMS:
        f, fwd, _ = allf[s]
        side = fn(f)  # +1 long, -1 short, 0 nic
        y = fwd[hh]
        ok = (side != 0) & np.isfinite(y)
        # neprekryvajici: po signalu preskoc hh baru
        idxs, last = [], -10**9
        for i in np.where(ok)[0]:
            if i - last >= hh:
                idxs.append(i); last = i
        idxs = np.array(idxs, dtype=int)
        r = side[idxs] * y[idxs] if len(idxs) else np.array([])
        NS.append(((side[idxs] > 0).sum(), (side[idxs] < 0).sum()))
        per.append(f"{r.mean() if len(r) else np.nan:6.1f}({len(r)})")
        tot_n += len(r); tot_s += r.sum()
    # permutační null: stejné počty L/S na symbol, náhodné časy (stejné napříč symboly kvůli korelaci) 
    mean = tot_s / max(tot_n, 1)
    nulls = []
    rng = np.random.default_rng(7)
    for _ in range(300):
        tot = 0.0; cnt = 0
        for s, k_s in zip(SYMS, NS):
            y = allf[s][1][hh]
            nl, ns = k_s
            idx = rng.choice(np.where(np.isfinite(y))[0], nl + ns, replace=False)
            sd = np.r_[np.ones(nl), -np.ones(ns)]
            tot += (sd * y[idx]).sum(); cnt += nl + ns
        nulls.append(tot / max(cnt, 1))
    nulls = np.array(nulls)
    p = (np.sum(np.abs(nulls) >= abs(mean)) + 1) / (len(nulls) + 1)
    print(f"{name:34s} h={hh:3d}: pooled {mean:6.1f} bps (null sd {nulls.std():4.1f}, p={p:.2f}) n={tot_n:5d} | " + " ".join(per))


Z = {}
for s in SYMS:
    f = allf[s][0]
    for nm in ("cvd12", "ret12", "obi1", "obi2", "obi5", "doi", "doi12", "dlt", "ret1"):
        f[nm + "_z"] = rz(f[nm])
    f["dep1_ch"] = np.r_[np.full(12, np.nan), f["dep1"][12:] - f["dep1"][:-12]]
    f["dep1_ch_z"] = rz(f["dep1_ch"])


def sgn(cond_long, cond_short):
    return np.where(cond_long, 1, np.where(cond_short, -1, 0))


sigs = {
    "absorpce: cvd z<=-2, cena drží": lambda f: sgn((f["cvd12_z"] <= -2) & (f["ret12_z"] >= 0), (f["cvd12_z"] >= 2) & (f["ret12_z"] <= 0)),
    "pokračování: cvd z>=2 & ret z>=2": lambda f: sgn((f["cvd12_z"] >= 2) & (f["ret12_z"] >= 2), (f["cvd12_z"] <= -2) & (f["ret12_z"] <= -2)),
    "vyčerpání: ret z>=2.5, fade": lambda f: sgn(f["ret12_z"] <= -2.5, f["ret12_z"] >= 2.5),
    "OI flush: doi12 z<=-2.5, fade ret12": lambda f: sgn((f["doi12_z"] <= -2.5) & (f["ret12"] < 0), (f["doi12_z"] <= -2.5) & (f["ret12"] > 0)),
    "kniha 1%: obi z>=2 → směr": lambda f: sgn(f["obi1_z"] >= 2, f["obi1_z"] <= -2),
    "kniha 2%: obi z>=2 → směr": lambda f: sgn(f["obi2_z"] >= 2, f["obi2_z"] <= -2),
    "OI build + cena: doi12 z>=2.5, follow": lambda f: sgn((f["doi12_z"] >= 2.5) & (f["ret12_z"] >= 1), (f["doi12_z"] >= 2.5) & (f["ret12_z"] <= -1)),
    "vyčerp.+OI pokles: ret z>=2.5, doi z<=-1.5": lambda f: sgn((f["ret12_z"] <= -2.5) & (f["doi12_z"] <= -1.5), (f["ret12_z"] >= 2.5) & (f["doi12_z"] <= -1.5)),
    "kniha 1%: obi z>=2.5 → směr": lambda f: sgn(f["obi1_z"] >= 2.5, f["obi1_z"] <= -2.5),
    "kniha 1%, lag 1 bar": lambda f: sgn(np.r_[False, f["obi1_z"][:-1] >= 2], np.r_[False, f["obi1_z"][:-1] <= -2]),
    "kniha 1% bez absolutni hodnoty (obi1>0.2)": lambda f: sgn(f["obi1"] >= 0.2, f["obi1"] <= -0.2),
    "kniha 1% + flow nesouhlasí (cvd z<=-1)": lambda f: sgn((f["obi1_z"] >= 2) & (f["cvd12_z"] <= -1), (f["obi1_z"] <= -2) & (f["cvd12_z"] >= 1)),
    "kniha 1% + 2% obě z>=1.5": lambda f: sgn((f["obi1_z"] >= 1.5) & (f["obi2_z"] >= 1.5), (f["obi1_z"] <= -1.5) & (f["obi2_z"] <= -1.5)),
    "kniha 5%: obi z>=2 → směr": lambda f: sgn(f["obi5_z"] >= 2, f["obi5_z"] <= -2),
    "absorpce + kniha1 souhlasí": lambda f: sgn((f["cvd12_z"] <= -2) & (f["ret12_z"] >= 0) & (f["obi1_z"] >= 1), (f["cvd12_z"] >= 2) & (f["ret12_z"] <= 0) & (f["obi1_z"] <= -1)),
    "vyčerp. + kniha1 proti pohybu": lambda f: sgn((f["ret12_z"] <= -2.5) & (f["obi1_z"] >= 1), (f["ret12_z"] >= 2.5) & (f["obi1_z"] <= -1)),
    "náhoda (kontrola, p=1%)": lambda f: sgn(np.random.default_rng(1).random(len(f["ret1"])) < 0.005, np.random.default_rng(2).random(len(f["ret1"])) < 0.005),
}
for nm, fn in sigs.items():
    for hh in (12, 48):
        cond(nm, fn, hh)

print("\nUnconditional sd fwd (bps):", " ".join(f"{s[:3]} h12={np.nanstd(allf[s][1][12]):.0f} h48={np.nanstd(allf[s][1][48]):.0f}" for s in SYMS))
print("bars:", {s: len(allf[s][2]) for s in SYMS})
print("book coverage (frac bars with obi1):", {s: round(float(np.isfinite(allf[s][0]['obi1']).mean()), 3) for s in SYMS})
print("oi coverage:", {s: round(float(np.isfinite(allf[s][0]['doi']).mean()), 3) for s in SYMS})


print("\n=== Robustnost obi1/obi2: IC po polovinách období a parciální IC (rezidua po ret12, ret48, cvd12) ===")
for nm in ("obi1", "obi2", "doi12"):
    for hh in (12, 48):
        halves = []
        for part in (0, 1):
            ics = []
            for s in SYMS:
                x, y = allf[s][0][nm], allf[s][1][hh]
                n = len(x); sl = slice(0, n // 2) if part == 0 else slice(n // 2, n)
                ics.append(spearman(x[sl], y[sl])[0])
            halves.append(np.mean(ics))
        part_ics = []
        for s in SYMS:
            f, fwd, c = allf[s]
            lc = np.log(c)
            r48 = np.r_[np.full(48, np.nan), lc[48:] - lc[:-48]]
            X = np.c_[np.ones(len(c)), f["ret12"], r48, f["cvd12"]]
            x, y = f[nm], fwd[hh]
            ok = np.isfinite(X).all(1) & np.isfinite(x) & np.isfinite(y)
            bx = np.linalg.lstsq(X[ok], x[ok], rcond=None)[0]
            by = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
            part_ics.append(spearman(x[ok] - X[ok] @ bx, y[ok] - X[ok] @ by)[0])
        print(f"{nm:6s} h={hh:3d}: 1. pol {halves[0]:.3f}  2. pol {halves[1]:.3f}  parciální {np.mean(part_ics):.3f}  po symbolech " + " ".join(f"{v:.3f}" for v in part_ics))
print("korelace obi1 s ret12 / ret48 (průměr přes symboly):",
      np.mean([spearman(allf[s][0]["obi1"], allf[s][0]["ret12"])[0] for s in SYMS]).round(3))
