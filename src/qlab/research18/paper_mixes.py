"""Locked, descriptive forward paper portfolios combining BTC and H2."""

from datetime import date

import numpy as np

from qlab.research15.evaluate import stats
from qlab.research15.setup import epoch_15m

START = date(2026, 10, 20)
BTC_WEIGHT, H2_WEIGHT = 0.25, 0.75
RV_WINDOW, RV_TARGET = 14, 0.60
DD_WINDOW, DD_TRIGGER, DD_REDUCED = 30, 0.10, 0.25
DD_RESUME_SMA = 20
REBALANCE_COST = 1e-3
VARIANTS = {
    "mix25_75": {"label": "BTC/H2 25/75", "kind": "static",
                 "description": "Váhy 25 % BTC / 75 % H2; nevyužitá expozice je hotovost."},
    "mix25_75_rv14": {"label": "25/75 · RV14", "kind": "vol_target",
                       "description": "BTC váha × min(1, 60 % / anualizovaná RV14); signál po close platí od dalšího dne."},
    "mix25_75_dd10": {"label": "25/75 · brzda DD10", "kind": "drawdown",
                       "description": "Při BTC −10 % od 30denního maxima se BTC váha sníží na 6,25 %; obnoví se po close nad SMA20."},
}


def daily_closes(panel: dict) -> tuple[np.ndarray, np.ndarray]:
    """Last available BTC spot close per UTC day."""
    day = panel["t"] // 96
    _, rev = np.unique(day[::-1], return_index=True)
    idx = len(day) - 1 - rev
    idx.sort()
    return day[idx].astype(np.int64), panel["c"][idx].astype(float)


def signal_multipliers(day_ids: np.ndarray, closes: np.ndarray, target_days: np.ndarray) -> dict[str, np.ndarray]:
    close_map = {int(d): float(c) for d, c in zip(day_ids, closes)}
    returns = {int(day_ids[i]): float(closes[i] / closes[i - 1] - 1)
               for i in range(1, len(day_ids)) if closes[i - 1] > 0}
    out = {key: np.ones(len(target_days), float) for key in VARIANTS}
    # RV14 for target day d uses 14 daily returns ending at d-1.
    for i, d in enumerate(target_days):
        vals = [returns.get(int(x)) for x in range(int(d - RV_WINDOW), int(d))]
        if len(vals) == RV_WINDOW and all(v is not None for v in vals):
            rv = float(np.std(vals, ddof=1) * np.sqrt(365))
            out["mix25_75_rv14"][i] = min(1.0, RV_TARGET / rv) if rv > 0 else 1.0

    # DD10 is a stateful close-to-close rule. The state is warmed up on prior history;
    # every signal formed after close d is assigned to target day d+1.
    active, signals = False, {}
    for d in day_ids:
        d = int(d)
        c = close_map[d]
        prev20 = [close_map.get(x) for x in range(d - DD_RESUME_SMA + 1, d + 1)]
        prev30 = [close_map.get(x) for x in range(d - DD_WINDOW + 1, d + 1)]
        sma = float(np.mean(prev20)) if all(x is not None for x in prev20) else None
        peak = max(prev30) if all(x is not None for x in prev30) else None
        if active:
            if sma is not None and c > sma:
                active = False
        elif peak is not None and c <= peak * (1 - DD_TRIGGER):
            active = True
        signals[d + 1] = DD_REDUCED if active else 1.0
    out["mix25_75_dd10"] = np.array([signals.get(int(d), 1.0) for d in target_days], float)
    return out


def simulate(btc_returns: np.ndarray, h2_returns: np.ndarray, btc_multiplier: np.ndarray) -> tuple[np.ndarray, dict]:
    """Mix return with daily target rebalancing; signals are already shifted to the valid next day."""
    if not (len(btc_returns) == len(h2_returns) == len(btc_multiplier)):
        raise ValueError("BTC, H2 and multiplier arrays must align")
    net = np.zeros(len(btc_returns), float)
    wb, wh = BTC_WEIGHT * btc_multiplier[0], H2_WEIGHT
    turn_sum, fee_sum = 0.0, 0.0
    for i, (rb, rh) in enumerate(zip(btc_returns, h2_returns)):
        cash = 1 - wb - wh
        btc_value, h2_value = wb * (1 + rb), wh * (1 + rh)
        nav = btc_value + h2_value + cash
        # Rebalance at this close to the next date's known target; charge turnover costs.
        if i + 1 < len(btc_returns):
            next_b, next_h = BTC_WEIGHT * btc_multiplier[i + 1], H2_WEIGHT
            turnover = abs(btc_value / nav - next_b) + abs(h2_value / nav - next_h)
            fee = REBALANCE_COST * turnover
        else:
            next_b, next_h, turnover, fee = btc_value / nav, h2_value / nav, 0.0, 0.0
        net[i] = nav - 1 - fee
        turn_sum += turnover
        fee_sum += fee
        wb, wh = next_b, next_h
    st = stats(net)
    return net, {**st, "total_return": float(np.prod(1 + net) - 1),
                 "avg_btc_multiplier": float(np.mean(btc_multiplier)),
                 "pct_days_btc_reduced": float(np.mean(btc_multiplier < 0.999999)),
                 "turnover_sum_nav": float(turn_sum), "rebalance_cost_sum_nav": float(fee_sum)}


def run(status: dict, btc_panel: dict) -> dict:
    dates = [date.fromisoformat(x) for x in status["daily"]["dates"]]
    target_days = np.array([epoch_15m(d) // 96 for d in dates], dtype=np.int64)
    day_ids, closes = daily_closes(btc_panel)
    close_map = {int(d): float(c) for d, c in zip(day_ids, closes)}
    btc_returns = np.array([close_map[int(d)] / close_map[int(d - 1)] - 1 for d in target_days], float)
    h2_returns = np.asarray(status["daily"]["H2"], float)
    multipliers = signal_multipliers(day_ids, closes, target_days)

    results, series = {}, {"dates": [d.isoformat() for d in dates]}
    for key, spec in VARIANTS.items():
        daily, metrics = simulate(btc_returns, h2_returns, multipliers[key])
        results[key] = {**spec, **metrics}
        series[key] = [float(x) for x in daily]
    return {"asof": status["asof"], "period": [START.isoformat(), status["asof"]],
            "n_days": len(dates), "source_commit": status.get("commit"),
            "source_rule_hash": status.get("rule_hash"), "source_data_hash": status.get("data_hash"),
            "results": results, "daily": series,
            "note": "Descriptive paper simulation only; no orders; not part of H1/H2 decision family."}
