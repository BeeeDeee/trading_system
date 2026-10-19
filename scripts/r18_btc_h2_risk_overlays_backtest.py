"""Post-hoc descriptive BTC sleeve risk overlays on BTC/H2 mixes.

    uv run python scripts/r18_btc_h2_risk_overlays_backtest.py binance_2026-10-03 runs/r18_btc_h2_risk_overlays_history.json

The full overlay grid is registered before loading any market data.
"""

import json
import sys
from pathlib import Path

import numpy as np

from qlab.research15.setup import _load, epoch_15m
from qlab.research15.evaluate import by_year, stats
from qlab.research16 import engine as G
from qlab.research17 import variants as V
from qlab.research17.setup import END, START
from qlab.research18 import exits as X, rules as H
from qlab.validation.registry import TrialRegistry, current_commit


snapshot, out_path = sys.argv[1], Path(sys.argv[2])
mixes = [(0.5, 0.5), (0.25, 0.75)]
overlays = [{"name": f"rv{w}", "kind": "vol_target", "window": w, "target_vol": 0.60}
            for w in (7, 14, 30, 60)]
overlays += [{"name": f"dd{p}", "kind": "drawdown", "window": 30, "threshold": p,
              "reduced_multiplier": 0.25, "resume": "BTC close > SMA20"}
             for p in (0.10, 0.15, 0.20)]
config = {"study": "research18_btc_h2_risk_overlays", "prereg": "docs/research18/BTC_H2_RISK_OVERLAYS.md v0.1",
          "snapshot": snapshot, "period": [str(START), str(END)], "mixes_btc_h2": mixes,
          "overlays": overlays, "application": "daily close signal; next-day BTC sleeve only",
          "rebalance_cost_bps": 10}
TrialRegistry("runs/registry/research18.sqlite").record(
    "candidates", config, n_configs=len(mixes) * (len(overlays) + 1),
    note="Post-hoc BTC/H2 mix risk overlays: short RV windows and drawdown brakes")

manifest = json.loads((Path("data/raw") / snapshot / "r15_manifest.json").read_text())
members = {m: v for m, v in manifest["members"].items() if m >= f"{START.year}-{START.month:02d}"}
symbols = sorted({s for vals in members.values() for s in vals})
btc_raw = _load(snapshot, "spot", "BTCUSDT", None)
btc = V.leader_moves(btc_raw)
gate = H.ma_gate(btc.c)
assets = {s: G.Asset.build(_load(snapshot, "perp", s, None)) for s in symbols}
s0 = int(epoch_15m(START))
h2 = X.run(X.SPECS[0], assets, btc, members, s0, gate)["daily"]

# Daily BTC close series includes pre-test history so short lookbacks are warm at test start.
btc_days = btc_raw["t"] // 96
_, reverse_idx = np.unique(btc_days[::-1], return_index=True)
last_idx = len(btc_days) - 1 - reverse_idx
last_idx.sort()
day_ids = btc_days[last_idx].astype(np.int64)
day_close = btc_raw["c"][last_idx].astype(float)
close_map = {int(d): float(c) for d, c in zip(day_ids, day_close)}
hist_returns = {int(day_ids[i]): float(day_close[i] / day_close[i - 1] - 1)
                for i in range(1, len(day_ids)) if day_close[i - 1] > 0}
grid_start = int(s0 // 96)
grid_days = np.arange(grid_start, grid_start + len(h2), dtype=np.int64)
btc_daily = np.array([close_map[int(d)] / close_map[int(d - 1)] - 1 for d in grid_days])
dates = (grid_days * 86400).astype("datetime64[s]").astype("datetime64[D]")


def overlay_multipliers(ov: dict) -> np.ndarray:
    """Multiplier for BTC sleeve each day; uses only data through previous UTC close."""
    multipliers = np.ones(len(grid_days), dtype=float)
    if ov["kind"] == "vol_target":
        w, target = int(ov["window"]), float(ov["target_vol"])
        for i, d in enumerate(grid_days):
            vals = [hist_returns.get(int(x)) for x in range(int(d - w), int(d))]
            if len(vals) == w and all(v is not None for v in vals):
                rv = float(np.std(vals, ddof=1) * np.sqrt(365))
                multipliers[i] = min(1.0, target / rv) if rv > 0 else 1.0
        return multipliers

    threshold = float(ov["threshold"])
    active = False
    # Run the state machine from the beginning of available BTC history, including warmup.
    signal_by_day = {}
    for d in day_ids:
        d = int(d)
        close = close_map[d]
        prior20 = [close_map.get(x) for x in range(d - 19, d + 1)]
        prior30 = [close_map.get(x) for x in range(d - 29, d + 1)]
        sma20 = float(np.mean(prior20)) if all(x is not None for x in prior20) else None
        peak30 = max(prior30) if all(x is not None for x in prior30) else None
        if active:
            if sma20 is not None and close > sma20:
                active = False
        elif peak30 is not None and close <= peak30 * (1 - threshold):
            active = True
        signal_by_day[d + 1] = 0.25 if active else 1.0
    for i, d in enumerate(grid_days):
        multipliers[i] = signal_by_day.get(int(d), 1.0)
    return multipliers


def simulate(wb: float, wh: float, mult: np.ndarray) -> tuple[np.ndarray, dict]:
    """Daily target portfolio; BTC/H2 sleeve changes pay 10bp per dollar turnover."""
    result = np.zeros(len(grid_days), dtype=float)
    start_b, start_h = wb * mult[0], wh
    turnover_total = cost_total = throttled = 0.0
    for i, (rb, rh) in enumerate(zip(btc_daily, h2)):
        start_cash = 1.0 - start_b - start_h
        pre_b = start_b * (1 + rb)
        pre_h = start_h * (1 + rh)
        nav_pre = pre_b + pre_h + start_cash
        gross_ret = nav_pre - 1.0
        # Rebalance at today's close for tomorrow's signal; no end-of-sample liquidation.
        target_b = wb * mult[i + 1] if i + 1 < len(mult) else pre_b / nav_pre
        target_h = wh if i + 1 < len(mult) else pre_h / nav_pre
        turn = (abs(pre_b / nav_pre - target_b) + abs(pre_h / nav_pre - target_h)
                if i + 1 < len(mult) else 0.0)
        fee = 0.001 * turn
        result[i] = gross_ret - fee
        turnover_total += turn
        cost_total += fee
        throttled += float(mult[i] < 0.999999)
        start_b, start_h = target_b, target_h
    return result, {"avg_btc_multiplier": float(mult.mean()),
                    "pct_days_btc_throttled": float(throttled / len(mult)),
                    "turnover_sum_nav": float(turnover_total),
                    "rebalance_cost_sum_nav": float(cost_total)}


def worst_rolling(r: np.ndarray, window: int) -> dict:
    values = np.array([np.prod(1 + r[i - window + 1:i + 1]) - 1
                       for i in range(window - 1, len(r))])
    i = int(np.argmin(values)) + window - 1
    return {"return": float(values[i - window + 1]),
            "start": str(dates[i - window + 1]), "end": str(dates[i])}


def dd_episodes(r: np.ndarray) -> list[dict]:
    eq = np.cumprod(1 + r)
    peaks = np.maximum.accumulate(np.r_[1.0, eq])
    out, active = [], None
    peak_value, peak_i = 1.0, -1
    for i, value in enumerate(eq):
        if value >= peak_value:
            if active is not None:
                active["recovery_date"] = str(dates[i])
                active["underwater_days"] = i - active["peak_idx"]
                out.append(active)
                active = None
            peak_value, peak_i = float(value), i
        else:
            if active is None:
                active = {"peak_idx": peak_i, "peak_date": str(dates[peak_i]) if peak_i >= 0 else "before_start",
                          "peak_value": peak_value, "trough_idx": i, "trough_value": float(value)}
            elif value < active["trough_value"]:
                active["trough_idx"], active["trough_value"] = i, float(value)
    if active is not None:
        active["recovery_date"] = None
        active["underwater_days"] = None
        out.append(active)
    for e in out:
        e["depth"] = float(1 - e["trough_value"] / e["peak_value"])
        e["peak_to_trough_days"] = int(e["trough_idx"] - e["peak_idx"])
        e["trough_date"] = str(dates[e["trough_idx"]])
        del e["peak_idx"], e["trough_idx"], e["peak_value"], e["trough_value"]
    return sorted(out, key=lambda x: x["depth"], reverse=True)


rows, variants_for_btc = {}, {}
base_overlays = [{"name": "static", "kind": "static"}, *overlays]
for wb, wh in mixes:
    mix_key = f"BTC{int(wb*100)}_H2{int(wh*100)}"
    for ov in base_overlays:
        mult = np.ones(len(grid_days)) if ov["kind"] == "static" else overlay_multipliers(ov)
        daily, diagnostics = simulate(wb, wh, mult)
        annual = by_year(dates, daily)
        name = f"{mix_key}_{ov['name']}"
        rows[name] = {**stats(daily), "total_return": float(np.prod(1 + daily) - 1),
                      "worst_year_return": min(annual.values(), key=lambda x: x["ret"])["ret"],
                      "by_year": annual, "overlay": ov, **diagnostics}
        if wb == 0.5 and wh == 0.5:
            variants_for_btc[ov["name"]] = daily

btc_episodes = dd_episodes(btc_daily)
btc_diag = {"worst_drawdown_episodes": btc_episodes[:5],
            "worst_7d": worst_rolling(btc_daily, 7), "worst_14d": worst_rolling(btc_daily, 14)}

out = {"study": "research18_btc_h2_risk_overlays", "snapshot": snapshot,
       "commit": current_commit(), "period": [str(START), str(END)], "n_symbols": len(symbols),
       "n_variants": len(rows), "btc_drawdown_diagnostics": btc_diag,
       "note": "Post-hoc descriptive, selection-biased historical results; next-day close signals; 10bp per dollar sleeve turnover; no live/OOS confirmation.",
       "results": rows}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, separators=(",", ":"), default=float))
print("BTC drawdown episodes:")
for ep in btc_episodes[:5]:
    print(f"  peak={ep['peak_date']} trough={ep['trough_date']} depth={ep['depth']:.2%} peak-to-trough={ep['peak_to_trough_days']}d recovery={ep['underwater_days']}d")
print(f"BTC worst 7d: {btc_diag['worst_7d']}")
print(f"BTC worst 14d: {btc_diag['worst_14d']}")
for name, result in rows.items():
    print(f"{name}: CAGR={result['cagr']:.2%}, maxDD={result['max_dd']:.2%}, Sharpe={result['sharpe']:.3f}, total={result['total_return']:.2%}, avgBTCmult={result['avg_btc_multiplier']:.2f}")
print(f"saved {out_path}")
