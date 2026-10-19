"""Descriptive parameter variants of the LEAD forward test (docs/research17/FORWARD.md §10).

The primary rule (`LEAD_K3_LONG`, the files of `forward.RULE_FILES`) is not touched: this module only re-implements the
signal with free parameters on the same panels, members and daily grid and reports the results next to it.
Variants never decide anything, are not selected from, and do not change the test (FORWARD.md §4, §10). The first
variant `V01_base` must reproduce the primary portfolio exactly (tests/test_research17_variants.py).
"""

from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np

from qlab.research15 import strategy as S
from qlab.research15.evaluate import stats
from qlab.research16 import engine as G, families as F
from qlab.research17 import engine as R
from qlab.research17.setup import COST, HOLD, K, WEIGHT, n_days


@dataclass(frozen=True)
class Spec:
    name: str
    group: str
    K: float = K                       # leader move >= K * sqrt(window) * sigma15m
    ratio: float | None = 0.5          # follower move / leader move < ratio; None = no lag condition
    window: int = 4                    # candles of the move (4 = 1 h)
    hold: int = HOLD                   # candles held (16 = 4 h)
    stop: float | None = None          # stop distance below the signal close, fraction
    target: float | None = None        # profit target above the signal close, fraction
    delay: int = 0                     # fill this many candles after the signal
    cost: float = COST                 # per side
    weight: float = WEIGHT             # capital share of one member sleeve
    top: int = 10                      # only the first `top` members of the month (ranked by volume)
    desc: str = ""


GROUPS = {"base": "Základ", "cons": "Konzervativnější: přísnější signál", "aggr": "Agresivnější: volnější signál",
          "hold": "Doba držení", "exit": "Výstupy (stop / cíl)", "window": "Okno skoku BTC",
          "exec": "Plnění a náklady", "size": "Velikost pozice"}

SPECS = [
    Spec("V01_base", "base", desc="primární pravidlo (kontrola shody)"),
    Spec("V02_K3.5", "cons", K=3.5, desc="skok BTC ≥ 3,5σ"),
    Spec("V03_K4", "cons", K=4.0, desc="skok BTC ≥ 4σ"),
    Spec("V04_lag0.25", "cons", ratio=0.25, desc="altcoin udělal < 25 % pohybu BTC"),
    Spec("V05_lag0", "cons", ratio=0.0, desc="altcoin stál nebo klesal"),
    Spec("V06_K3.5_lag0.25", "cons", K=3.5, ratio=0.25, desc="3,5σ a < 25 %"),
    Spec("V07_top5", "cons", top=5, desc="jen 5 největších členů měsíce"),
    Spec("V08_K2.5", "aggr", K=2.5, desc="skok BTC ≥ 2,5σ"),
    Spec("V09_K2", "aggr", K=2.0, desc="skok BTC ≥ 2σ"),
    Spec("V10_lag0.75", "aggr", ratio=0.75, desc="altcoin udělal < 75 % pohybu BTC"),
    Spec("V11_lag1", "aggr", ratio=1.0, desc="altcoin zaostal za BTC (< 100 %)"),
    Spec("V12_nolag", "aggr", ratio=None, desc="bez podmínky zaostávání"),
    Spec("V13_K2.5_lag0.75", "aggr", K=2.5, ratio=0.75, desc="2,5σ a < 75 %"),
    Spec("V14_K2_nolag", "aggr", K=2.0, ratio=None, desc="2σ bez podmínky zaostávání"),
    Spec("V15_hold2h", "hold", hold=8, desc="držení 2 h"),
    Spec("V16_hold6h", "hold", hold=24, desc="držení 6 h"),
    Spec("V17_hold12h", "hold", hold=48, desc="držení 12 h"),
    Spec("V18_hold24h", "hold", hold=96, desc="držení 24 h"),
    Spec("V19_stop2", "exit", stop=0.02, desc="stop −2 %"),
    Spec("V20_stop1.5_tgt3", "exit", stop=0.015, target=0.03, desc="stop −1,5 %, cíl +3 %"),
    Spec("V21_tgt2", "exit", target=0.02, desc="cíl +2 %"),
    Spec("V22_stop2_hold12h", "exit", stop=0.02, hold=48, desc="stop −2 %, držení 12 h"),
    Spec("V23_stop3_tgt6_hold12h", "exit", stop=0.03, target=0.06, hold=48, desc="stop −3 %, cíl +6 %, 12 h"),
    Spec("V24_win30m", "window", window=2, desc="skok BTC za 30 min"),
    Spec("V25_win2h", "window", window=8, desc="skok BTC za 2 h"),
    Spec("V26_delay1", "exec", delay=1, desc="plnění o svíčku později (15 min)"),
    Spec("V27_delay2", "exec", delay=2, desc="plnění o 2 svíčky později (30 min)"),
    Spec("V28_cost5", "exec", cost=5e-4, desc="náklady 5 bps/stranu (limitní plnění)"),
    Spec("V29_size0.5x", "size", weight=0.05, desc="poloviční velikost rukávu (5 %)"),
    Spec("V30_size2x", "size", weight=0.2, desc="dvojnásobná velikost rukávu (20 %)"),
]
assert len(SPECS) == 30 and len({s.name for s in SPECS}) == 30


def move(c: np.ndarray, real: np.ndarray, w: int) -> np.ndarray:
    """log(c[k] / c[k-w]) where candles k-w..k are all real, else NaN (w = 4 is `families.move_1h`)."""
    n = len(c)
    out = np.full(n, np.nan)
    if n > w:
        cs = np.concatenate([[0], np.cumsum(real.astype(np.int64))])
        ok = cs[w + 1:] - cs[:n - w] == w + 1
        out[w:] = np.where(ok, np.log(c[w:] / c[:-w]), np.nan)
    return out


@dataclass
class LeaderMoves:
    t0: int
    c: np.ndarray
    real: np.ndarray
    sig: np.ndarray                    # sigma of 15m returns (not scaled)
    cache: dict


def leader_moves(d: dict) -> LeaderMoves:
    a = SimpleNamespace(c=d["c"], real=d["real"])
    return LeaderMoves(int(d["t"][0]), d["c"], d["real"], F.sigma(a), {})


def candidates(a: G.Asset, ld: LeaderMoves, sp: Spec, own: dict) -> G.Cand:
    """Long candidates of follower `a`: same logic as `engine.signals` with free parameters."""
    if sp.window not in ld.cache:
        ld.cache[sp.window] = move(ld.c, ld.real, sp.window)
    if sp.window not in own:
        own[sp.window] = move(a.c, a.real, sp.window)
    idx = a.t - ld.t0
    ok = (idx >= 0) & (idx < len(ld.c))
    idx = np.where(ok, idx, 0)
    rb = np.where(ok, ld.cache[sp.window][idx], np.nan)
    sg = np.where(ok, ld.sig[idx], np.nan)
    ra = own[sp.window]
    with np.errstate(invalid="ignore", divide="ignore"):
        big = (rb >= sp.K * np.sqrt(sp.window) * sg) & a.real
        if sp.ratio is not None:
            big &= (ra / rb < sp.ratio)
    k = np.flatnonzero(big)
    p = a.c[k]
    return G.Cand.make(k, np.ones(len(k)), k + sp.hold,
                       None if sp.stop is None else p * (1 - sp.stop),
                       None if sp.target is None else p * (1 + sp.target))


def run_spec(sp: Spec, assets: dict[str, G.Asset], btc: LeaderMoves, members: dict[str, list[str]], s0f: int,
             own: dict[str, dict]) -> dict:
    """Portfolio of one variant over the followers; `members` {'YYYY-MM': [symbols ranked by volume]}."""
    top = {m: v[:sp.top] for m, v in members.items()}
    daily, rets, entries = np.zeros(n_days()), [], []
    for sym, a in assets.items():
        n, start = len(a.c), max(s0f - int(a.t[0]), 0)
        b = G.trade_book(a, candidates(a, btc, sp, own.setdefault(sym, {})), start, n, delay=sp.delay, cost=sp.cost,
                         allowed=R.member_mask(a, sym, top))
        R._to_grid(a.t, *S.daily(a.t, b.net, start, n), sp.weight, daily)
        rets.append(b.ret)
        entries.append(a.t[b.entries] if len(b.entries) else np.zeros(0, np.int64))
    r = np.concatenate(rets) if rets else np.zeros(0)
    ent = np.concatenate(entries) if entries else np.zeros(0, np.int64)
    st = stats(daily, r if len(r) else None)
    return {"daily": daily, "stats": st, "n_events": int(len(np.unique(ent - sp.delay))), "n_trades": int(len(r))}


def run_all(pan: dict, followers: list[str], members: dict[str, list[str]], s0f: int, specs: list[Spec] = SPECS) -> dict:
    """All variants; call inside `forward.forward_window`. `pan` as in `forward.evaluate` (already truncated)."""
    btc = leader_moves(pan["BTCUSDT@spot"])
    assets = {s: G.Asset.build(pan[s]) for s in followers}
    own: dict[str, dict] = {}
    out = {}
    for sp in specs:
        r = run_spec(sp, assets, btc, members, s0f, own)
        st = r["stats"]
        out[sp.name] = {"spec": asdict(sp), "sharpe": st["sharpe"], "cagr": st["cagr"], "max_dd": st["max_dd"],
                        "mean_trade": st.get("mean_trade"), "hit_rate": st.get("hit_rate"),
                        "n_events": r["n_events"], "n_trades": r["n_trades"],
                        "total_return": float(np.prod(1 + r["daily"]) - 1),
                        "meets": bool(r["n_trades"] and st["sharpe"] > 0 and st.get("mean_trade", 0) > 0)}
    return out
