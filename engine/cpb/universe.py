"""Weekly universe: 20 coins from the top ~50 by market cap, frozen point-in-time.

Excluded: stablecoins, wrapped / staked / bridged derivatives, commodity-backed tokens, and coins without a
TRADING Binance <SYM>USDT spot pair with >= min_quote_volume_usd of 24h volume.
"""
import re


def build(cfg, date, caps, cap_source, status, vol24):
    ucfg = cfg["universe"]
    quote = cfg["quote"]
    excl = set(ucfg["exclude_symbols"])
    pat = re.compile("|".join(ucfg["exclude_name_patterns"]), re.I)
    chosen, rejected = [], []
    seen = set()
    for c in caps[: ucfg["candidates"]]:
        sym = ucfg.get("symbol_overrides", {}).get(c["symbol"], c["symbol"])
        pair = sym + quote
        reason = None
        if sym in excl:
            reason = "vyloučen seznamem (stablecoin/wrapped/staked/komoditní)"
        elif pat.search(c["name"]) or pat.search(c["id"]):
            reason = f"vyloučen podle názvu ({c['name']})"
        elif sym in seen:
            reason = "duplicitní symbol"
        elif status.get(pair) != "TRADING":
            reason = f"bez obchodovatelného páru {pair} na Binance ({status.get(pair) or 'neexistuje'})"
        elif vol24.get(pair, 0) < ucfg["min_quote_volume_usd"]:
            reason = f"objem 24h ${vol24.get(pair, 0) / 1e6:.1f}M < ${ucfg['min_quote_volume_usd'] / 1e6:.0f}M"
        elif len(chosen) >= ucfg["size"]:
            reason = "mimo prvních 20"
        entry = {"coin": sym, "pair": pair, "name": c["name"], "id": c["id"], "rank": c["rank"],
                 "market_cap": c["market_cap"], "quote_volume_24h": vol24.get(pair)}
        if reason:
            rejected.append(dict(entry, reason=reason))
        else:
            seen.add(sym)
            chosen.append(entry)
    return {"date": date, "source": cap_source, "coins": chosen, "rejected": rejected,
            "rules": {k: ucfg[k] for k in ("size", "candidates", "min_quote_volume_usd")}}


def needs_rebuild(cfg, date, current):
    from . import canon
    return current is None or canon.weekday(date) == cfg["universe"]["rebuild_weekday"] and current["date"] != date


def coins(u):
    return [c["coin"] for c in u["coins"]]
