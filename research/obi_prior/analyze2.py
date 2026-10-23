import sys, numpy as np
sys.argv = [sys.argv[0], "data90.npz", "oi90.npz"]
src = open("analyze.py").read()
cut = src.index('print("=== IC (Spearman)')
exec(src[:cut])
allf = {s: features(s) for s in SYMS}
for s in SYMS:
    f = allf[s][0]
    for nm in ("cvd12", "ret12", "obi1", "obi2", "obi5", "doi12"):
        f[nm + "_z"] = rz(f[nm])
    lc = np.log(allf[s][2])
    allf[s][1][96] = np.r_[lc[96:] - lc[:-96], np.full(96, np.nan)] * 1e4
OT = {s: D[f"{s}_k"][:, 0] for s in SYMS}

def sgn(a, b): return np.where(a, 1, np.where(b, -1, 0))

def cond(name, fn, hh):
    R, DAY = [], []
    per = []
    for s in SYMS:
        f, fwd, _ = allf[s]; side = fn(f); y = fwd[hh]
        ok = np.where((side != 0) & np.isfinite(y))[0]
        idxs, last = [], -10**9
        for i in ok:
            if i - last >= hh: idxs.append(i); last = i
        idxs = np.array(idxs, int)
        r = side[idxs] * y[idxs]; R.append(r); DAY.append(OT[s][idxs] // 86_400_000)
        per.append(f"{r.mean() if len(r) else np.nan:6.1f}")
    r = np.concatenate(R); d = np.concatenate(DAY)
    # cluster SE po dnech UTC (pooled přes symboly)
    ud, inv = np.unique(d, return_inverse=True)
    dsum = np.bincount(inv, weights=r - r.mean())
    se = np.sqrt((dsum ** 2).sum()) / len(r)
    print(f"{name:32s} h={hh:3d}: {r.mean():6.1f} ± {se:4.1f} (t {r.mean()/se:4.1f}) net@12 {r.mean()-12:6.1f}  n={len(r):5d} /den {len(r)/90:4.1f} | " + " ".join(per))

sigs = {
    "náhoda sync (kontrola)": lambda f: sgn(np.random.default_rng(1).random(len(f["ret1"])) < 0.005, np.random.default_rng(2).random(len(f["ret1"])) < 0.005),
    "absorpce cvd z<=-2, cena drží": lambda f: sgn((f["cvd12_z"] <= -2) & (f["ret12_z"] >= 0), (f["cvd12_z"] >= 2) & (f["ret12_z"] <= 0)),
    "pokračování cvd&ret z>=2": lambda f: sgn((f["cvd12_z"] >= 2) & (f["ret12_z"] >= 2), (f["cvd12_z"] <= -2) & (f["ret12_z"] <= -2)),
    "fade ret z>=2.5 (≈ mr)": lambda f: sgn(f["ret12_z"] <= -2.5, f["ret12_z"] >= 2.5),
    "OI flush fade": lambda f: sgn((f["doi12_z"] <= -2.5) & (f["ret12"] < 0), (f["doi12_z"] <= -2.5) & (f["ret12"] > 0)),
    "kniha 1% z>=2": lambda f: sgn(f["obi1_z"] >= 2, f["obi1_z"] <= -2),
    "kniha 1% z>=2.5": lambda f: sgn(f["obi1_z"] >= 2.5, f["obi1_z"] <= -2.5),
    "kniha 1% z>=3": lambda f: sgn(f["obi1_z"] >= 3, f["obi1_z"] <= -3),
    "kniha 2% z>=2": lambda f: sgn(f["obi2_z"] >= 2, f["obi2_z"] <= -2),
    "kniha 2% z>=2.5": lambda f: sgn(f["obi2_z"] >= 2.5, f["obi2_z"] <= -2.5),
    "kniha 0.2% z>=2.5": lambda f: sgn(rz(f["obi0.2"]) >= 2.5, rz(f["obi0.2"]) <= -2.5),
    "kniha 5% z>=2.5": lambda f: sgn(f["obi5_z"] >= 2.5, f["obi5_z"] <= -2.5),
    "kniha 1% z>=2.5 jen LONG": lambda f: sgn(f["obi1_z"] >= 2.5, np.zeros(len(f["ret1"]), bool)),
    "kniha 1% z>=2.5 jen SHORT": lambda f: sgn(np.zeros(len(f["ret1"]), bool), f["obi1_z"] <= -2.5),
}
for nm, fn in sigs.items():
    for hh in (12, 48, 96):
        cond(nm, fn, hh)
# stabilita kniha 1% z>=2.5 po třetinách
print("\nkniha 1% z>=2.5, h=48, po třetinách období:")
for part in range(3):
    R = []
    for s in SYMS:
        f, fwd, _ = allf[s]; n = len(f["ret1"]); side = sgn(f["obi1_z"] >= 2.5, f["obi1_z"] <= -2.5); y = fwd[48]
        lo, hi = part * n // 3, (part + 1) * n // 3
        ok = [i for i in np.where((side != 0) & np.isfinite(y))[0] if lo <= i < hi]
        idxs, last = [], -10**9
        for i in ok:
            if i - last >= 48: idxs.append(i); last = i
        R.append(side[idxs] * y[idxs])
    r = np.concatenate(R); print(f"  třetina {part+1}: {r.mean():6.1f} bps n={len(r)}")
