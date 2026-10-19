"""Evaluation and decision of research 18 (PREREGISTRATION.md §7-9): event bootstrap, Holm, status and decision.

`evaluate(panels, members, asof)` takes the panels of `research17.forward.load_panels` and returns the metrics of
both hypotheses up to `asof` plus descriptive extras; `decide` turns metrics into the pre-registered decision.
"""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import date

import numpy as np

from qlab.research15.evaluate import by_year, stats
from qlab.research15.setup import epoch_15m
from qlab.research16 import engine as G
from qlab.research17 import forward as F, setup as T, variants as V
from qlab.research18 import rules as H
from qlab.research18.setup import ALPHA, COST_MARGIN, MIN_EVENTS, N_BOOT, N_BOOT_DESC, T0
from qlab.validation.stats import stationary_bootstrap_indices


@contextmanager
def window(asof: date):
    """The grid constants of `research17.setup` (read at call time by the engine) cover T0 .. asof."""
    old = T.START, T.END
    T.START, T.END = T0, asof
    try:
        yield
    finally:
        T.START, T.END = old


# ---------------------------------------------------------------- statistics

def event_boot(events: np.ndarray, rets: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Mean trade return and bootstrap over whole events (every event with all its trades): 90 % interval and the
    one-sided p of H0 'mean <= 0', p = (1 + #{resampled mean <= 0}) / (1 + n_boot)."""
    if not len(rets):
        return {"mean": None, "lo": None, "hi": None, "p": None, "n_events": 0}
    u, inv = np.unique(events, return_inverse=True)
    s, c = np.bincount(inv, rets), np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(u), (n_boot, len(u)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return {"mean": float(rets.mean()), "lo": float(np.quantile(m, 0.05)), "hi": float(np.quantile(m, 0.95)),
            "p": float((1 + (m <= 0).sum()) / (1 + n_boot)), "n_events": int(len(u))}


def holm(p: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    """Holm step-down: the i-th smallest p (0-based) must be < alpha / (m - i); after the first failure all fail."""
    out, alive = {}, True
    for i, (h, v) in enumerate(sorted(p.items(), key=lambda kv: kv[1])):
        alive = alive and v < alpha / (len(p) - i)
        out[h] = bool(alive)
    return out


def decide(metrics: dict[str, dict], alive: list[str], look: int) -> dict[str, str]:
    """Pre-registered decision of the hypotheses in `alive` (§7): 'success', 'rejected', 'inconclusive'
    (at look 2 an inconclusive one becomes 'not_confirmed'). A rejected hypothesis leaves the Holm family."""
    out, rest = {}, []
    for h in alive:
        m = metrics[h]
        if m["n_events"] >= MIN_EVENTS[h] and (m["sharpe"] <= 0 or not m["mean_trade"] or m["mean_trade"] <= 0):
            out[h] = "rejected"
        else:
            rest.append(h)
    ok = holm({h: metrics[h]["p"] for h in rest if metrics[h]["p"] is not None})
    for h in rest:
        m = metrics[h]
        good = (ok.get(h, False) and m["n_events"] >= MIN_EVENTS[h] and m["sharpe"] > 0 and m["sharpe_cost15"] > 0)
        out[h] = "success" if good else ("inconclusive" if look == 1 else "not_confirmed")
    return out


def paired_sharpe_diff(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT_DESC, seed: int = 0) -> dict:
    """Sharpe(a) - Sharpe(b) with a paired stationary bootstrap of the days (mean block 10); descriptive."""
    sh = lambda x: x.mean() / x.std(ddof=1) * 365 ** 0.5 if x.std(ddof=1) > 0 else 0.0
    if len(a) <= 10:
        return {"diff": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = [stationary_bootstrap_indices(len(a), 10.0, rng) for _ in range(n_boot)]
    d = np.array([sh(a[i]) - sh(b[i]) for i in idx])
    return {"diff": float(sh(a) - sh(b)), "lo": float(np.quantile(d, 0.05)), "hi": float(np.quantile(d, 0.95))}


# ---------------------------------------------------------------- evaluation

def _summary(r: dict, boot: bool = True) -> dict:
    st = stats(r["daily"], r["rets"] if len(r["rets"]) else None)
    out = {"sharpe": st["sharpe"], "stats": st, "n_trades": int(len(r["rets"])),
           "mean_trade": st.get("mean_trade"), "n_events": int(len(np.unique(r["events"])))}
    if boot:
        eb = event_boot(r["events"], r["rets"])
        out |= {"p": eb["p"], "ci90_events": [eb["lo"], eb["hi"]]}
    return out


def evaluate(panels: dict, members: dict, asof: date) -> dict:
    """Metrics of H1 and H2 on T0 .. asof (§7) and the descriptive extras of §9. `members` {'YYYY-MM': [symbols]}."""
    ok_members = {m: v for m, v in members.items() if v}
    followers = sorted({s for v in ok_members.values() for s in v if s in panels})
    pan = {k: F.truncate(v, asof) for k, v in panels.items()}
    spot = pan["BTCUSDT@spot"]
    with window(asof):
        s0 = int(epoch_15m(T0))
        btc = V.leader_moves(spot)
        gate = H.ma_gate(spot["c"])
        assets = {s: G.Asset.build(pan[s]) for s in followers}
        nd = T.n_days()
        dates = (T.day0() + np.arange(nd)).astype("datetime64[D]")
        runs = {}
        for h, (sp, gated) in H.HYPOTHESES.items():
            g = gate if gated else None
            runs[h] = H.run(sp, assets, btc, ok_members, s0, g)
            runs[h + "_cost15"] = H.run(H.variant(sp, cost=COST_MARGIN), assets, btc, ok_members, s0, g)
        metrics = {}
        for h in H.HYPOTHESES:
            metrics[h] = _summary(runs[h]) | {"sharpe_cost15": stats(runs[h + "_cost15"]["daily"])["sharpe"],
                                              "min_events": MIN_EVENTS[h],
                                              "spec": asdict(H.HYPOTHESES[h][0]), "gated": H.HYPOTHESES[h][1]}
        # ---- descriptive (§9): never decides anything
        v01 = H.run(V.SPECS[0], assets, btc, ok_members, s0)                      # V01_base: H2 without the gate
        combo = H.run(H.H1, assets, btc, ok_members, s0, gate)                    # H1 + BTC > MA200
        desc = {"V01_base_no_gate": _summary(v01, boot=False), "H1_plus_gate": _summary(combo, boot=False),
                "H2_minus_V01": paired_sharpe_diff(runs["H2"]["daily"], v01["daily"]),
                "scenarios": {}}
        for h, (sp, gated) in H.HYPOTHESES.items():
            g = gate if gated else None
            for name, kw in (("cost75", {"cost": 7.5e-4}), ("cost200", {"cost": 2e-3}), ("delay1", {"delay": 1}),
                             ("delay2", {"delay": 2})):
                desc["scenarios"][f"{h}_{name}"] = _summary(H.run(H.variant(sp, **kw), assets, btc, ok_members, s0, g),
                                                            boot=False)
        sig_ok = gate[(spot["t"] >= s0)]
        desc["btc_above_ma200_share"] = float(sig_ok.mean()) if len(sig_ok) else None
        desc["h2_trades_dropped_by_gate_vs_v01"] = int(len(v01["rets"]) - len(runs["H2"]["rets"]))
        per_sym = {}
        for h in H.HYPOTHESES:
            r = runs[h]
            per_sym[h] = {s: {"n": int(sum(1 for x in r["syms"] if x == s)),
                              "mean": float(np.mean([x for x, y in zip(r["rets"], r["syms"]) if y == s]))}
                          for s in sorted(set(r["syms"]))}
        out = {"asof": str(asof), "days": int(nd), "period": [str(T0), str(asof)], "followers": followers,
               "metrics": metrics, "descriptive": desc, "per_symbol": per_sym,
               "daily": {"dates": [str(d) for d in dates], **{h: [float(x) for x in runs[h]["daily"]] for h in H.HYPOTHESES}},
               "trades": {h: sorted(({"sym": s, "entry": str(np.datetime64(int(t) * 900, "s")), "ret": float(x)}
                                     for s, t, x in zip(runs[h]["syms"], runs[h]["events"], runs[h]["rets"])),
                                    key=lambda x: x["entry"]) for h in H.HYPOTHESES},
               "by_year": {h: by_year(dates, runs[h]["daily"]) for h in H.HYPOTHESES} if nd > 1 else {}}
    out["provisional_decision"] = decide(metrics, list(H.HYPOTHESES), 1)       # what the decision would be now
    return out
