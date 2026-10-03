# POST-HOC data-artefact check: same frozen methodology, holdout ends 2026-08-31 (spot data for 2026-09 missing).
import json
from datetime import date
from pathlib import Path
import numpy as np
from qlab.research8 import panel as P, report as R, strategy as S
from qlab.research8.setup import DEV, HOLDOUT, index_of, panel_path
from qlab.research8.sim import Costs, simulate
from qlab.validation.stats import deflated_sharpe, probabilistic_sharpe, stationary_bootstrap_indices
END = date(2026, 8, 31)
p = P.load(panel_path("binance_2026-10-03"), END)          # deliberate vault bypass, logged in PREREGISTRATION §10
dev = json.loads(Path("docs/research8/results/dev.json").read_text())
by = {c.id: c for c in S.grid()}
d0, h0 = index_of(p, DEV[0]), index_of(p, HOLDOUT[0]); H = slice(h0, None)
out, pv = {}, {}
for name, cid in (("P1", "S0"), ("P2", dev["P2"])):
    cfg = by[cid]
    tg = {t: w for t, w in S.targets(p, cfg).items() if t >= d0}
    b, st = simulate(p, tg, costs=Costs()), simulate(p, tg, costs=Costs(mult=2.0))
    tb = p.tbill[H]; ex = (b.ret[H] - tb) * 365
    rng = np.random.default_rng(0)
    bs = np.array([ex[stationary_bootstrap_indices(len(ex), 21.0, rng)].mean() for _ in range(1000)])
    pv[name] = float((bs <= 0).mean())
    est, lo, hi = R.excess_ci(b.ret[H], tb)
    subs = {}
    for k, (a, e) in (("2023-01..2024-06", (date(2023, 1, 1), date(2024, 6, 30))), ("2024-07..2026-08", (date(2024, 7, 1), END))):
        sl = slice(index_of(p, a), index_of(p, e) + 1); subs[k] = R.summary(b.ret[sl], p.tbill[sl])["excess_ann"]
    stat = probabilistic_sharpe(b.ret[H] - tb) if name == "P1" else deflated_sharpe(b.ret[H] - tb, dev["n_trials"], dev["sr_variance_daily"])
    s = R.summary(b.ret[H], tb, b.funding[H], b.costs[H])
    out[name] = {"id": cid, "base": s, "stress_excess": R.summary(st.ret[H], tb)["excess_ann"], "ci90": [est, lo, hi],
                 "subperiods_excess": subs, "psr_or_dsr": stat, "delistings": len([1 for t, j in b.delistings if t >= h0])}
order = sorted(pv, key=pv.get)
holm = {k: min(1.0, max(pv[order[i]] * (len(order) - i) for i in range(order.index(k) + 1))) for k in pv}
for n, c in out.items():
    c["criteria"] = {"1": holm[n] < 0.05 and c["ci90"][1] > 0, "2": all(v > 0 for v in c["subperiods_excess"].values()),
                     "3": c["stress_excess"] > 0, "4": c["psr_or_dsr"] >= 0.95}
    b = c["base"]
    print(f"{n} {c['id']}: excess {b['excess_ann']*100:+.2f} % CI90 [{c['ci90'][1]*100:+.2f}, {c['ci90'][2]*100:+.2f}] SR {b['sharpe_excess']:.2f} "
          f"funding {b['funding_ann']*100:.2f} % costs {b['costs_ann']*100:.2f} % tbill {b['tbill_ann']*100:.2f} % MDD {b['max_dd']*100:.1f} % "
          f"sub {{{', '.join(f'{k}: {v*100:+.2f} %' for k, v in c['subperiods_excess'].items())}}} stress {c['stress_excess']*100:+.2f} % "
          f"psr/dsr {c['psr_or_dsr']:.3f} delist {c['delistings']} -> {c['criteria']}")
Path("docs/research8/results/final_posthoc_to_2026-08-31.json").write_text(json.dumps(out, indent=1))
