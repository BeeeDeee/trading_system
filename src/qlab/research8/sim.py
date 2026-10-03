"""Delta-neutral hedge simulator (prereg §4.1, §4.4): long spot + short perp of equal notional.

Day loop, order of events for day t:
  1. open:  mark both legs to the open; trades decided before (weekly targets, maintenance, delisting,
            invalid perp/spot pair at yesterday's close) are filled at the open with fee + slippage per leg
  2. day:   funding of events in (t 00:00, t+1 00:00] paid to the short on q x perp open
  3. day:   liquidation if the loss to the day's MARK-price high (Binance liquidates on mark price;
            last-price high where no mark kline exists) wipes the perp account (minus 0.5 % maintenance);
            the spot leg is then sold at the close
  4. close: mark both legs to the close; maintenance flag if the perp account left 50-150 % of target
Units: q is in perp units on both legs (spot prices are pre-multiplied by the contract multiplier).
"""

from dataclasses import dataclass, field

import numpy as np

from .panel import Panel, pair_ok

MAINT_MARGIN = 0.005
DELIST_SLIP = 0.02


@dataclass(frozen=True)
class Costs:
    spot_fee: float = 0.0010
    perp_fee: float = 0.0005
    tiers: tuple = ((1e9, 0.0002), (2e8, 0.0005), (5e7, 0.0010), (0.0, 0.0025))   # 30d avg qv -> slippage
    mult: float = 1.0                                                                # 2.0 = stress

    def slip(self, qv30: float) -> float:
        for lo, s in self.tiers:
            if qv30 >= lo:
                return s * self.mult
        return self.tiers[-1][1] * self.mult


@dataclass
class Leg:
    q: float          # units short on perp = units long on spot
    m: float          # perp account equity (USD)
    target_m: float   # collateral target at the last (re)balance
    w: float          # target weight of the last weekly decision


@dataclass
class Result:
    nav: np.ndarray
    ret: np.ndarray
    funding: np.ndarray        # daily funding income / previous NAV
    costs: np.ndarray          # daily costs / previous NAV
    turnover: np.ndarray
    liquidations: list = field(default_factory=list)
    delistings: list = field(default_factory=list)
    mismatches: list = field(default_factory=list)


def qv30(p: Panel) -> np.ndarray:
    """30-day mean perp quote volume known at the open of day t (days t-30..t-1)."""
    qv = np.nan_to_num(p.m["qv"])
    cs = np.vstack([np.zeros((1, qv.shape[1])), np.cumsum(qv, axis=0)])
    T = qv.shape[0]
    out = np.full_like(qv, np.nan)
    for t in range(1, T):
        lo = max(0, t - 30)
        out[t] = (cs[t] - cs[lo]) / (t - lo)
    return out


def simulate(p: Panel, targets: dict[int, dict[int, float]], s: float = 2 / 3, costs: Costs = Costs(),
             nav0: float = 1.0, liq_price: str = "mark") -> Result:
    """`targets`: day index -> {symbol index: weight}; applied at that day's open (weights sum <= 1)."""
    po, ph, pc, so, sc = (p.m[k] for k in ("po", "ph", "pc", "so", "sc"))
    fh = p.m["fund_hold"]
    liq_hi = np.where(np.isnan(p.m["mh"]), ph, p.m["mh"]) if liq_price == "mark" else ph
    T = len(p.dates)
    vol = qv30(p)
    cash, legs = nav0, {}
    last_px: dict[int, tuple[float, float]] = {}      # j -> last valid (perp close, spot close)
    nav = np.empty(T)
    fund_d, cost_d, turn_d = np.zeros(T), np.zeros(T), np.zeros(T)
    res = Result(nav, np.zeros(T), fund_d, cost_d, turn_d)
    pending_maint: set[int] = set()
    prev_nav = nav0
    ok = pair_ok(p)

    def value(j: int, perp: float, spot: float) -> float:
        L = legs[j]
        return L.q * spot + L.m

    def trade(t: int, j: int, dq: float, perp: float, spot: float, extra: float = 0.0) -> float:
        """Change the hedge by dq units on both legs at the given prices; returns cost in USD."""
        sl = (costs.slip(vol[t, j]) if not np.isnan(vol[t, j]) else costs.tiers[-1][1] * costs.mult) + extra
        notional = abs(dq) * (spot + perp)
        c = abs(dq) * spot * (costs.spot_fee * costs.mult + sl) + abs(dq) * perp * (costs.perp_fee * costs.mult + sl)
        turn_d[t] += notional
        cost_d[t] += c
        return c

    def set_hedge(t: int, j: int, V: float, w: float, perp: float, spot: float) -> None:
        """Rebuild the hedge of j to total value V (spot s*V, perp account (1-s)*V), costs from cash."""
        nonlocal cash
        L = legs.get(j)
        q_old, cur = (L.q, value(j, perp, spot)) if L else (0.0, 0.0)
        q_new = s * V / spot
        c = trade(t, j, q_new - q_old, perp, spot)
        cash += cur - V - c
        legs[j] = Leg(q_new, (1 - s) * V, (1 - s) * V, w)

    def close(t: int, j: int, perp: float, spot: float, extra: float = 0.0) -> None:
        nonlocal cash
        L = legs.pop(j)
        c = trade(t, j, L.q, perp, spot, extra)
        cash += L.q * spot + L.m - c

    for t in range(T):
        # ---- 1. open: mark to open, delistings, maintenance and weekly targets
        for j in list(legs):
            if np.isnan(po[t, j]) or np.isnan(so[t, j]):
                lp, ls = last_px[j]
                close(t, j, lp, ls, DELIST_SLIP)
                res.delistings.append((t, j))
                continue
            L = legs[j]
            L.m += L.q * (last_px[j][0] - po[t, j])
            if t > 0 and not ok[t - 1, j]:                     # legs stopped being the same asset
                close(t, j, po[t, j], so[t, j], DELIST_SLIP)
                res.mismatches.append((t, j))
        open_nav = cash + sum(value(j, po[t, j], so[t, j]) for j in legs)
        tgt = targets.get(t)
        rebuilt: set[int] = set()
        if tgt is not None:
            for j in [j for j in legs if j not in tgt]:
                close(t, j, po[t, j], so[t, j])
            for j, w in sorted(tgt.items()):
                if np.isnan(po[t, j]) or np.isnan(so[t, j]):
                    continue
                if j in legs and legs[j].w == w:
                    continue                                   # unchanged weight: no trade (prereg §4.2)
                set_hedge(t, j, w * open_nav, w, po[t, j], so[t, j])
                rebuilt.add(j)
        for j in sorted(pending_maint - rebuilt):
            if j in legs:                                      # re-balance the legs, value unchanged
                set_hedge(t, j, value(j, po[t, j], so[t, j]), legs[j].w, po[t, j], so[t, j])
        pending_maint.clear()
        # ---- 2. funding over the day
        for j, L in legs.items():
            f = fh[t, j]
            if not np.isnan(f):
                inc = L.q * po[t, j] * f
                L.m += inc
                fund_d[t] += inc
        # ---- 3. liquidation at the day's high
        for j in list(legs):
            L = legs[j]
            hi = liq_hi[t, j]
            if not np.isnan(hi) and L.m - L.q * (hi - po[t, j]) <= MAINT_MARGIN * L.q * hi:
                L.m = 0.0
                q = L.q
                legs.pop(j)
                c = trade(t, j, q, 0.0, sc[t, j])          # only the spot leg is sold
                cash += q * sc[t, j] - c
                res.liquidations.append((t, j))
        # ---- 4. close: mark to close, maintenance flags
        for j, L in legs.items():
            L.m += L.q * (po[t, j] - pc[t, j])
            last_px[j] = (pc[t, j], sc[t, j])
            if not (0.5 * L.target_m <= L.m <= 1.5 * L.target_m):
                pending_maint.add(j)
        nav[t] = cash + sum(value(j, pc[t, j], sc[t, j]) for j in legs)
        res.ret[t] = nav[t] / prev_nav - 1
        fund_d[t] /= prev_nav
        cost_d[t] /= prev_nav
        turn_d[t] /= prev_nav
        prev_nav = nav[t]
    return res
