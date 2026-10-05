"""Scout briefs: steer each `propose` run toward the least explored horizon and asset group.

Left alone, the Scout converges on the cheapest instruments and one horizon (four of its first five cards were
SPY/IEF or monthly switches). The brief is chosen deterministically from the registry: the (horizon, asset
group) cell with the fewest hypotheses, ties broken in a fixed order, and a multi-timeframe requirement while
fewer than half of the cards combine timeframes. It is a direction for diversity, not a quality waiver.
"""

from lab.framework.blackboard import Lab

HORIZONS = (("short", "1-5 trading days", 0, 5), ("swing", "1-4 weeks", 5, 25),
            ("medium", "1-6 months", 25, 130), ("long", "6-24 months", 130, 10**6))
GROUPS = {"crypto": "crypto (Binance spot or perpetuals, optionally the hourly-derived daily features)",
          "us_equity": "US single stocks, cross-sectional (sharadar_sep, liq_n or sp500)",
          "us_etf": "US ETFs across asset classes (sharadar_sfp)",
          "cross_asset": "cross-asset (signals from one market, trades in another; FRED macro series allowed)"}


def horizon(card: dict) -> str:
    days = float((card.get("holding_period") or {}).get("typical_days") or 0)
    return next(h for h, _, lo, hi in HORIZONS if lo <= days < hi)


def group(card: dict) -> str:
    classes = set(card.get("asset_classes") or [])
    if len(classes) > 1 or "macro" in classes:
        return "cross_asset"
    if classes & {"crypto_spot", "crypto_perp"}:
        return "crypto"
    return "us_equity" if "us_equity" in classes else "us_etf"


def multi_timeframe(card: dict) -> bool:
    return len(set((card.get("signal") or {}).get("timeframes") or [])) >= 2


def choose(lab: Lab) -> dict:
    import json
    cards = [json.loads(h["card_json"]) for h in lab.hypotheses()]
    counts = {(h, g): 0 for h, *_ in HORIZONS for g in GROUPS}
    for c in cards:
        counts[(horizon(c), group(c))] += 1
    (h, g), _ = min(counts.items(), key=lambda kv: kv[1])         # dict order breaks ties deterministically
    mtf = sum(map(multi_timeframe, cards)) < len(cards) / 2
    return {"horizon": h, "group": g, "multi_timeframe": mtf}


def text(brief: dict) -> str:
    span = next(s for h, s, *_ in HORIZONS if h == brief["horizon"])
    out = (f"propose. Brief for this run (diversity of the lab's portfolio of ideas): holding horizon "
           f"{brief['horizon']} ({span}); asset group: {GROUPS[brief['group']]}.")
    if brief["multi_timeframe"]:
        out += (" Combine timeframes: a signal on one timeframe confirmed or filtered by another (e.g. a slow "
                "weekly/monthly regime confirming a faster daily entry, or hourly-derived features gating a "
                "daily signal); list them in signal.timeframes.")
    return out + (" The brief is a direction, not a waiver: if no idea with a real mechanism fits it, take the "
                  "closest fit and say why in notes.")
