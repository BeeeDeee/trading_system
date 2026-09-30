"""Accounting: one ledger per (portfolio, cost scenario). Fills, costs, stops, delisting, marking.

Cost scenarios share identical decisions (target weights); only execution prices and fees differ:
  base   = config fees and slippage
  stress = 40 bps fee, 2x slippage
  gross  = no costs
Buys fill at ask * (1 + slip), sells at bid * (1 - slip) (best bid/ask of the snapshot; the liquidity-tier slip
models impact beyond the top of the book). The gross scenario trades at the mid. Positions are valued at the mid
(snapshot) or at the daily close. Fees are charged on notional.
"""
from . import canon


def new_ledger(capital):
    return {"cash": canon.money(capital), "positions": {}, "blocked": {}, "fees": 0.0, "slippage": 0.0,
            "turnover": 0.0, "n_trades": 0, "last_rebalance": None, "started": None}


def scenario_costs(cfg, scn):
    s = cfg["costs"]["scenarios"][scn]
    return {"fee_bps": s["fee_bps"], "slip_mult": s["slip_mult"]}


def slip_bps(cfg, quote_volume, extra=0.0):
    tiers = cfg["costs"]["slippage_tiers"]        # [[min_volume_usd, bps], ...] descending
    for vmin, bps in tiers:
        if (quote_volume or 0) >= vmin:
            return bps + extra
    return tiers[-1][1] + extra


def equity(led, prices):
    return led["cash"] + sum(p["qty"] * prices[c] for c, p in led["positions"].items())


def _trade(led, pid, scn, coin, side, qty, snap_px, sbps, fee_bps, reason, date, ts_ms, source, trades, touch=None):
    """snap_px = reference (mid) price; touch = ask (BUY) / bid (SELL) the order executes against (default mid)."""
    touch = snap_px if touch is None else touch
    if side == "BUY":
        px = touch * (1 + sbps / 1e4)
        notional = qty * px
        fee = notional * fee_bps / 1e4
        led["cash"] = canon.money(led["cash"] - notional - fee)
        p = led["positions"].get(coin)
        if p:
            p["avg_px"] = (p["avg_px"] * p["qty"] + px * qty) / (p["qty"] + qty)
            p["ref_px"] = (p["ref_px"] * p["qty"] + snap_px * qty) / (p["qty"] + qty)
            p["qty"] = p["qty"] + qty
            p["high"] = max(p["high"], snap_px)
        else:
            # ref_px = snapshot (pre-cost) entry price: stop levels are identical in every cost scenario
            led["positions"][coin] = {"qty": qty, "avg_px": px, "ref_px": snap_px, "entry_date": date, "entry_ms": ts_ms,
                                      "high": snap_px, "stop_checked_ms": ts_ms}
    else:
        px = touch * (1 - sbps / 1e4)
        notional = qty * px
        fee = notional * fee_bps / 1e4
        led["cash"] = canon.money(led["cash"] + notional - fee)
        p = led["positions"][coin]
        p["qty"] -= qty
        if p["qty"] <= 1e-12 * max(1.0, qty):
            del led["positions"][coin]
    slip_usd = abs(px - snap_px) * qty
    led["fees"] = canon.money(led["fees"] + fee)
    led["slippage"] = canon.money(led["slippage"] + slip_usd)
    led["turnover"] = canon.money(led["turnover"] + notional)
    led["n_trades"] += 1
    if trades is None:              # null-distribution paths do not keep a trade log
        return
    trades.append({"portfolio": pid, "scenario": scn, "coin": coin, "side": side, "qty": qty,
                   "snapshot_price": snap_px, "touch_price": touch, "fill_price": px, "notional": canon.money(notional),
                   "fee_usd": canon.money(fee), "slippage_usd": canon.money(slip_usd), "slippage_bps": sbps,
                   "fee_bps": fee_bps, "reason": reason, "date": date, "price_time": canon.ms_iso(ts_ms), "source": source})


def stop_level(p, stop_cfg):
    lv = []
    if stop_cfg.get("stop_pct"):
        lv.append(p["ref_px"] * (1 - stop_cfg["stop_pct"]))
    if stop_cfg.get("trail_pct"):
        lv.append(p["high"] * (1 - stop_cfg["trail_pct"]))
    return max(lv) if lv else None


def apply_stops(led, pid, scn, cfg, stop_cfg, bars, vol, date, block_date, trades, source):
    """bars = {coin: [[open_ms, o, h, l, c], ...]} sorted. Only bars opening at/after entry are used.

    Order within a bar: check the stop against the level known before the bar, then update the high.
    Gap below the stop -> fill at the bar open. Extra stop slippage on top of the liquidity tier.
    """
    if not stop_cfg:
        return
    c = scenario_costs(cfg, scn)
    for coin in sorted(led["positions"]):
        p = led["positions"][coin]
        for t, o, h, l, cl in bars.get(coin, []):
            if t < p["stop_checked_ms"]:          # only bars opening after the entry / last check
                continue
            lvl = stop_level(p, stop_cfg)
            if lvl is not None and l <= lvl:
                px = min(lvl, o)
                sb = slip_bps(cfg, vol.get(coin), cfg["costs"]["stop_extra_bps"]) * c["slip_mult"]
                _trade(led, pid, scn, coin, "SELL", p["qty"], px, sb, c["fee_bps"], "stop", date, t, source, trades)
                led["blocked"][coin] = block_date
                break
            p["high"] = max(p["high"], h)
            p["stop_checked_ms"] = t + 1
        if coin in led["positions"]:
            led["positions"][coin]["stop"] = stop_level(led["positions"][coin], stop_cfg)


def force_sell(led, pid, scn, cfg, coin, px, vol, date, ts_ms, reason, extra_bps, trades, source):
    if coin not in led["positions"]:
        return
    c = scenario_costs(cfg, scn)
    sb = slip_bps(cfg, vol, extra_bps) * c["slip_mult"]
    _trade(led, pid, scn, coin, "SELL", led["positions"][coin]["qty"], px, sb, c["fee_bps"], reason, date, ts_ms, source, trades)


def rebalance(led, pid, scn, cfg, targets, exits, prices, vol, date, ts_ms, source, trades, quotes=None, keep=()):
    """Move the ledger towards target weights (fractions of equity at snapshot prices).

    targets: {coin: weight} or None (hold everything). exits: {coin: reason} to close regardless.
    keep: held coins left exactly as they are (hourly strategies with a minimum holding period).
    Full exits are always executed; otherwise a coin trades only when |target - current| > band of equity.
    New positions and buys below min_trade_usd are skipped. Sells first, then buys with available cash.
    """
    c = scenario_costs(cfg, scn)
    band = cfg["rules"]["band"]

    def touch(coin, side):          # gross scenario (no costs) trades at the mid
        if not quotes or coin not in quotes or c["slip_mult"] == 0:
            return prices[coin]
        return quotes[coin][1] if side == "BUY" else quotes[coin][0]

    min_trade = cfg["costs"]["min_trade_usd"]
    E = equity(led, prices)
    fee = c["fee_bps"] / 1e4
    for coin in sorted(exits):
        if coin in led["positions"]:
            sb = slip_bps(cfg, vol.get(coin)) * c["slip_mult"]
            _trade(led, pid, scn, coin, "SELL", led["positions"][coin]["qty"], prices[coin], sb, c["fee_bps"], exits[coin], date, ts_ms, source, trades,
                   touch(coin, "SELL"))
    if targets is None:
        return
    for coin in sorted(led["positions"]):
        if coin in keep:
            continue
        p = led["positions"][coin]
        cur = p["qty"] * prices[coin]
        tgt = targets.get(coin, 0.0) * E
        sb = slip_bps(cfg, vol.get(coin)) * c["slip_mult"]
        if tgt <= 0:
            if cur >= min_trade:
                _trade(led, pid, scn, coin, "SELL", p["qty"], prices[coin], sb, c["fee_bps"], "signal", date, ts_ms, source, trades,
                       touch(coin, "SELL"))
        elif cur - tgt > band * E and cur - tgt >= min_trade:
            _trade(led, pid, scn, coin, "SELL", (cur - tgt) / prices[coin], prices[coin], sb, c["fee_bps"], "rebalance", date, ts_ms, source, trades,
                   touch(coin, "SELL"))
    order = sorted(targets, key=lambda k: (-targets[k], k))
    for coin in order:
        w = targets[coin]
        if w <= 0:
            continue
        held = coin in led["positions"]
        cur = led["positions"][coin]["qty"] * prices[coin] if held else 0.0
        diff = w * E - cur
        if held and diff <= band * E:
            continue
        sb = slip_bps(cfg, vol.get(coin)) * c["slip_mult"]
        px = touch(coin, "BUY") * (1 + sb / 1e4)
        spend = min(diff, max(led["cash"], 0) / (1 + fee))        # notional at fill price
        if spend < min_trade:
            continue
        _trade(led, pid, scn, coin, "BUY", spend / px, prices[coin], sb, c["fee_bps"],
               "rebalance" if held else "signal", date, ts_ms, source, trades, touch(coin, "BUY"))


def mark(led, closes):
    pos = {}
    for c, p in sorted(led["positions"].items()):
        pos[c] = {"qty": p["qty"], "price": closes[c], "value": canon.money(p["qty"] * closes[c]),
                  "avg_px": p["avg_px"], "entry_date": p["entry_date"]}
        if p.get("stop") is not None:
            pos[c]["stop"] = p["stop"]
    eq = led["cash"] + sum(v["value"] for v in pos.values())
    return {"equity": canon.money(eq), "cash": led["cash"], "positions": pos,
            "exposure": canon.money(1 - led["cash"] / eq) if eq > 0 else 0.0,
            "fees": led["fees"], "slippage": led["slippage"], "turnover": led["turnover"], "n_trades": led["n_trades"]}


def weights(led, prices):
    E = equity(led, prices)
    return {c: p["qty"] * prices[c] / E for c, p in led["positions"].items()} if E > 0 else {}


def redenominate(led, coin, ratio, new_coin=None):
    """ratio = new units per old unit (1000 for a 1000x rebase). Value is unchanged."""
    p = led["positions"].pop(coin, None)
    if p:
        p = dict(p, qty=p["qty"] * ratio, avg_px=p["avg_px"] / ratio, ref_px=p["ref_px"] / ratio, high=p["high"] / ratio)
        if p.get("stop") is not None:
            p["stop"] = p["stop"] / ratio
        led["positions"][new_coin or coin] = p
    if coin in led["blocked"]:
        led["blocked"][new_coin or coin] = led["blocked"].pop(coin)
