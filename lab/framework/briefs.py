"""Scout briefs: steer each `propose` run toward the least explored horizon and asset group.

Left alone, the Scout converges on the cheapest instruments and one horizon (four of its first five cards were
SPY/IEF or monthly switches). The brief is chosen deterministically from the registry: the (horizon, asset
group) cell with the fewest hypotheses, ties broken in a fixed order, and a multi-timeframe requirement while
fewer than half of the cards combine timeframes. It is a direction for diversity, not a quality waiver.
"""

import json
import re
from pathlib import Path

import yaml

from lab.framework.blackboard import Lab

HORIZONS = (("scalp", "minutes to a day (5m / 15m candles)", 0, 1), ("short", "1-5 trading days", 1, 5), ("swing", "1-4 weeks", 5, 25),
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


def is_ml(card: dict) -> bool:
    return bool(card.get("ml"))


ML_SHARE = 0.25     # the Scout is asked for a learned-model card while fewer than this share of the cards have one
ML_TEXT = (
    " Make it a LEARNED-MODEL card (`ml` block): the signal is the output of a walk-forward model trained inside "
    "`lab.framework.ml.walk_forward` (ridge, logit, rf = random forest, gbm = LightGBM, mlp, or lstm / tcn on the last "
    "seq_len days of the feature vectors), not a hand-written rule. Cross-sectional scoring of US stocks or crypto pairs "
    "is where there is enough data; a model with 2-3 instruments learns nothing. The card needs a feature list with "
    "a reason for each feature, a label horizon that matches the holding period, a refit schedule, and `why_not_linear`: "
    "the interaction, nonlinearity or temporal pattern a ridge on the same features would miss, and who pays for it. "
    "Small models and few features beat large ones here; the judge's default for 'the network found something' is "
    "overfitting. Read the ML section of the instructions before writing the card.")


def choose(lab: Lab) -> dict:
    import json
    cards = [json.loads(h["card_json"]) for h in lab.hypotheses()]
    counts = {(h, g): 0 for h, *_ in HORIZONS for g in GROUPS if h != "scalp" or g == "crypto"}   # scalping: crypto only
    for c in cards:
        counts[(horizon(c), group(c))] += 1
    # the scalping cell counts double (owner, 2026-10-13): it is proposed about twice as often as any other cell
    (h, g), _ = min(counts.items(), key=lambda kv: kv[1] / (SCALP_WEIGHT if kv[0][0] == "scalp" else 1.0))   # dict order breaks ties
    mtf = sum(map(multi_timeframe, cards)) < len(cards) / 2
    ml = h != "scalp" and sum(map(is_ml, cards)) < ML_SHARE * max(len(cards), 1)   # scalping has no ml contract
    return {"horizon": h, "group": g, "multi_timeframe": mtf, "ml": ml}


SCALP_WEIGHT = 2.0
SCALP_TEXT = (
    "propose. Brief for this run (diversity of the lab's portfolio of ideas): SCALPING. A signal on 5m or 15m candles "
    "of Binance USD-M perpetuals (dataset binance_perp_15m; use '5m' only if the card truly needs it, 5m data may not "
    "exist yet), holding a few candles to a few hours. Fields: o h l c, quote volume, number of trades, taker-buy "
    "volume, funding. Universe: crypto_top_n (n 5-30) or a short instrument list; long and short allowed. List the "
    "timeframe ('15m') in signal.timeframes, put typical_days below 1 in holding_period, and do NOT declare a "
    "mechanism_test (the intraday judge has none). Think about who pays you: forced liquidations, funding "
    "settlement, session opens, order-flow imbalance (taker-buy share), range compression and expansion. A scalp "
    "pays two taker fees plus slippage per trade (at least 13 bps round trip even for BTC), so rare, high-"
    "conviction entries survive and frequent small ones do not.")


def text(brief: dict) -> str:
    if brief["horizon"] == "scalp":
        return SCALP_TEXT + (" The brief is a direction, not a waiver: if no idea with a real mechanism fits it, take "
                             "the closest fit and say why in notes.")
    span = next(s for h, s, *_ in HORIZONS if h == brief["horizon"])
    out = (f"propose. Brief for this run (diversity of the lab's portfolio of ideas): holding horizon "
           f"{brief['horizon']} ({span}); asset group: {GROUPS[brief['group']]}.")
    if brief["multi_timeframe"]:
        out += (" Combine timeframes: a signal on one timeframe confirmed or filtered by another (e.g. a slow "
                "weekly/monthly regime confirming a faster daily entry, or hourly-derived features gating a "
                "daily signal); list them in signal.timeframes.")
    if brief.get("ml"):
        out += ML_TEXT
    out += " Declare a mechanism_test (event + primary horizon) so the judge tests the mechanism before the strategy."
    return out + (" The brief is a direction, not a waiver: if no idea with a real mechanism fits it, take the "
                  "closest fit and say why in notes.")


# Owner queue (owner, 2026-11-10): ideas the owner wants tested go first, one per Scout `propose` run, before the
# diversity brief. An entry is consumed once a card cites it with `owner_seed: <id>` in its notes; `skip: true`
# takes an entry out by hand. The Scout still writes the whole card (mechanism, forecasts, falsification).
OWNER_QUEUE = Path(__file__).resolve().parents[1] / "briefs" / "owner_queue.yaml"
SEED_RE = re.compile(r"owner_seed:\s*([A-Za-z0-9_-]+)")


def owner_seed(lab: Lab, path: Path | None = None) -> dict | None:
    """The first queue entry that no card cites yet, or None."""
    path = path or OWNER_QUEUE
    if not path.exists():
        return None
    seeds = yaml.safe_load(path.read_text()) or []
    done = {m for h in lab.hypotheses() for m in SEED_RE.findall(json.loads(h["card_json"]).get("notes") or "")}
    return next((s for s in seeds if s["id"] not in done and not s.get("skip")), None)


def seed_text(seed: dict) -> str:
    return (f"propose. OWNER QUEUE item {seed['id']} ({seed['title']}): the owner asked for this idea to be tested. "
            f"Turn it into one full card. Write `owner_seed: {seed['id']}` as the first line of notes (this marks the "
            "queue item done). You own the details: mechanism, counterparty, exact rules, grids, forecasts and "
            "falsification criteria, within the lab's data and contracts. If the data the idea needs is not in the "
            "catalog, still write the card (it will wait in BLOCKED_DATA) and say what is missing in notes. If the "
            "idea is a duplicate or cannot be made honest, write the closest honest card and explain why in notes. "
            "Owner's description:\n" + seed["text"].strip())


def next_task(lab: Lab, path: Path | None = None) -> str:
    """The Scout's propose task: the owner's queue first, then the diversity brief."""
    seed = owner_seed(lab, path)
    return seed_text(seed) if seed else text(choose(lab))
