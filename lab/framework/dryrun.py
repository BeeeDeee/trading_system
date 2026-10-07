"""`lab try`: G0 on a synthetic market inside the Builder's workspace. No real data, no trial.

The Builder is data-blind like the Scout. It still has to know before IMPL_DONE whether its code passes G0,
so `lab try` builds a synthetic DataView with the card's instruments (names, asset classes, dev calendar),
random returns and some awkward cases (a late listing, a delisting for crypto, non-tradable days), and runs
the same `integrity()` check the gate runner runs on real data. It also runs every G2 grid neighbor once
(a crash there would fail G2), the Builder's own pytest file, and prints the trading statistics the card
promises (decision dates, exposure, turnover). It never prints returns: on random data they mean nothing,
and on real data the Builder must not see them.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from lab.framework import catalog, data, strategy
from lab.framework.data import DataView
from lab.framework.evaluator import TRADABLE, QlabEvaluator, integrity, strategy_instruments

DAILY_VOL = {"us_etf": 0.012, "us_equity": 0.02, "crypto_spot": 0.04, "crypto_perp": 0.04}
SYNTHETIC_PAIRS = 40
SYNTHETIC_STOCKS = 80
TEST_TIMEOUT_S = 300
TEST_EXTRA_IMPORTS = ("pytest", "strategy")


def instruments(card: dict, cat: dict) -> tuple[list[str], list[str], int | None]:
    """(names, asset classes, top-n) of the strategy's instruments, resolved exactly as the gate runner does.
    Universe-based datasets get synthetic names in the same format (SYNxxUSDT pairs, E<number> stocks)."""
    uni = card["universe"]
    if uni["kind"] not in ("instruments", "crypto_top_n", "liq_n", "sp500"):
        raise ValueError(f"universe kind {uni['kind']!r} has no loader yet")
    names, classes = [], []
    for ds, inst in strategy_instruments(card).items():
        if inst is None:
            inst = ([f"E{100000 + i}" for i in range(SYNTHETIC_STOCKS)] if ds == "sharadar_sep"
                    else [f"SYN{i:02d}USDT" for i in range(SYNTHETIC_PAIRS)])
        for i in inst:
            if i not in names:
                names.append(i)
                classes.append(cat[ds]["asset_class"])
    n = {"crypto_top_n": uni.get("n"), "liq_n": min(int(uni.get("n") or 0), SYNTHETIC_STOCKS // 2),
         "sp500": SYNTHETIC_STOCKS // 2}.get(uni["kind"])
    return names, classes, n


def synthetic_view(card: dict, cat: dict, seed: int) -> DataView:
    reqs = card["data_requirements"]
    if {r["dataset"] for r in reqs} == {"synthetic_market"}:   # canaries: the generator itself (with its series)
        names = instruments(card, cat)[0]
        end = data.as_date(cat["synthetic_market"]["holdout_from"]) - np.timedelta64(1, "D")
        return data.with_cash(data.synthetic().between(None, end).columns(names), None)
    names, classes, top_n = instruments(card, cat)
    start = max([data.as_date(cat[r["dataset"]]["range"][0]) for r in reqs]
                + [data.as_date(r["period"][0]) for r in reqs if r.get("period")])
    end = min(data.as_date(cat[r["dataset"]]["holdout_from"]) for r in reqs) - np.timedelta64(1, "D")
    days = np.arange(start, end + np.timedelta64(1, "D"))
    crypto = np.array([c.startswith("crypto") for c in classes])
    if not crypto.all():
        days = days[np.is_busday(days)] if not crypto.any() else days
    T, N = len(days), len(names)
    rng = np.random.default_rng(seed)
    vol = np.array([DAILY_VOL.get(c, 0.015) for c in classes])
    factor = rng.normal(0, 1, T)
    beta = rng.uniform(0.3, 1.2, N)
    ret = 0.6 * beta * factor[:, None] * vol + rng.normal(0, 1, (T, N)) * vol * 0.8
    ret_co, ret_oc = ret * 0.3, ret * 0.7
    listed = np.ones((T, N), dtype=bool)
    weekday = np.is_busday(days)
    listed[~weekday[:, None] & ~crypto[None, :]] = False         # ETFs do not trade on weekends
    if N > 1:
        late = int(rng.integers(0, N))
        listed[: T // 5, late] = False                             # one instrument lists late
    delisting = np.zeros((T, N), dtype=bool)
    stocks = np.array([c == "us_equity" for c in classes])
    for group, frac, terminal in ((crypto, 0.7, -0.02), (stocks, 0.5, -0.30), (stocks, 0.8, 0.10)):
        if group.sum() > 2:                                        # delistings (stocks: a bankruptcy-like and
            j = int(np.flatnonzero(group)[-1 if frac != 0.8 else -2])   # an acquisition-like terminal return)
            last = int(T * frac)
            delisting[last + 1, j], ret_co[last + 1, j] = True, terminal
            listed[last + 2:, j] = False
    tradable = listed & ~delisting
    tradable &= rng.random((T, N)) > 0.002                         # rare missing opens
    ret_co, ret_oc = np.where(listed | delisting, ret_co, 0.0), np.where(tradable, ret_oc, 0.0)
    close = 20 * np.cumprod((1 + ret_co) * (1 + ret_oc), axis=0)
    close = np.where(listed, close, np.nan)
    dv = np.where(listed, rng.lognormal(17, 1, (T, N)), np.nan)
    view = DataView(days, tuple(names), tuple(classes), ret_co, ret_oc, tradable, listed, delisting, close, dv,
                    cash_ret=np.zeros(T))
    from dataclasses import replace
    kind = card["universe"]["kind"]
    if top_n and kind == "crypto_top_n":
        view = replace(view, universe=data.crypto_top_n(view, top_n))
    elif top_n and kind in ("liq_n", "sp500"):                    # PIT membership by trailing liquidity
        member = data.crypto_top_n(view, top_n, window=63) & stocks[None, :]
        rank = np.where(stocks[None, :] & listed, np.argsort(np.argsort(-np.nan_to_num(dv), axis=1), axis=1) + 1.0,
                        np.nan)
        alt = data.crypto_top_n(view, min(N, int(top_n * 1.25)), window=63) & stocks[None, :]
        view = replace(view, universe=member, extras={"liq_rank": rank, "alt_universe": alt.astype(float)})
    if crypto.any() and any(c == "crypto_perp" for c in classes):
        perp = np.array([c == "crypto_perp" for c in classes])
        f = np.where(perp[None, :] & listed, rng.normal(3e-4, 4e-4, (T, N)), np.nan)
        view = replace(view, extras={**view.extras, "funding": f, "funding_paid": np.nan_to_num(f),
                                     "basis": np.where(perp[None, :] & listed, rng.normal(0, 2e-3, (T, N)), np.nan)})
    series, attached_extras = {}, {}
    for r in reqs:                                                 # signal-only datasets (Archivist ingest)
        if cat[r["dataset"]].get("loader") == "sep_attached":      # SF1 / insiders / 13F: (T, N) extras on the stocks
            for f in r.get("fields") or []:
                step = rng.random((T // 63 + 2, N)).astype(np.float32) * 10
                m = np.repeat(step, 63, axis=0)[:T].copy()          # a new value about every quarter
                m[: T // 8] = np.nan                                 # data starts later than the prices
                m[~stocks[None, :] | ~listed] = np.nan
                attached_extras[f] = m
            continue
        if r["dataset"] in TRADABLE:
            continue
        fields = cat[r["dataset"]].get("fields") or r.get("fields") or []   # what the gate runner will attach
        if cat[r["dataset"]].get("per_instrument"):        # keys "<PAIR>.<field>" for the card's pairs
            pairs = [n.removesuffix(".P") for n, c in zip(names, classes) if c.startswith("crypto")]
            fields = [f"{p}.{f}" for p in pairs for f in fields]
        for f in fields:
            x = 20 + np.cumsum(rng.normal(0, 1, T))
            x[: T // 10] = np.nan                                 # starts later than the prices
            series[f"{r['dataset']}.{f}"] = x
    view = replace(view, extras={**view.extras, **attached_extras}) if attached_extras else view
    return replace(view, series=series) if series else view


def stats(w: np.ndarray, view: DataView) -> dict:
    decided = ~np.isnan(w).all(axis=1)
    held = np.where(decided[:, None], np.nan_to_num(w), np.nan)
    held = np.asarray([np.nan_to_num(r) for r in _ffill(held)])
    years = max(len(view.dates) / (365.0 if any(c.startswith("crypto") for c in view.asset_class) else 252.0), 1e-9)
    turnover = np.abs(np.diff(held, axis=0)).sum() / 2
    entries = ((np.abs(held[1:]) > 1e-9) & (np.abs(held[:-1]) <= 1e-9)).sum()
    changes = (np.abs(np.diff(held, axis=0)).sum(axis=1) > 1e-9).sum()
    return {"rows": len(view.dates), "first_decision": str(view.dates[np.argmax(decided)]) if decided.any() else None,
            "decision_rows": int(decided.sum()), "rows_with_weight_change": int(changes),
            "mean_gross": round(float(np.abs(held).sum(axis=1).mean()), 3),
            "max_gross": round(float(np.abs(held).sum(axis=1).max()), 3),
            "one_way_turnover_per_year": round(float(turnover / years), 2),
            "position_entries": int(entries), "years": round(years, 1)}


def _ffill(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for t in range(1, len(out)):
        nan = np.isnan(out[t])
        out[t, nan] = out[t - 1, nan]
    return out


def run(ws: Path) -> tuple[bool, list[str]]:
    """Everything `lab try` checks. Returns (ok, report lines)."""
    card = yaml.safe_load((ws / "card.yaml").read_text())
    cat = catalog.load(ws / "catalog.yaml")
    gates = yaml.safe_load((ws / "gates.yaml").read_text())
    th = gates["G0"]
    th_events = gates["G1"]["mechanism"]["min_events"]
    th_dates = gates["G1"]["mechanism"]["min_event_dates"]
    path = ws / "strategy" / "strategy.py"
    content = {k: v for k, v in card.items() if k not in ("id", "version", "status", "history", "terminal")}
    seed = int(hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()[:8], 16)
    view = synthetic_view(card, cat, seed)
    cols = list(view.instruments)
    params = QlabEvaluator.primary(card)
    lines, ok = [f"synthetic market: {len(cols)} instruments, {view.dates[0]} .. {view.dates[-1]} "
                 f"({len(view.dates)} rows; random returns, a late listing, missing opens)"], True

    out = integrity(card, path, view, cols, params, len(view.dates), th, th["max_runtime_min"] * 60.0,
                    events_path=path.parent / "diagnostic.py")
    if out.passed:
        lines.append(f"G0 integrity on synthetic data: passed ({out.metrics.get('runtime_s')} s, "
                     f"{out.metrics['lookahead_failures']['cuts']} look-ahead cuts)")
    else:
        ok = False
        lines.append(f"G0 integrity on synthetic data: FAILED {out.reason_code}")
        lines += [f"  {k}: {v}" for k, v in out.metrics.items() if v]
    if out.passed:
        module = strategy.load(path)
        sv = view.columns(cols)
        w = np.asarray(module.target_weights(sv, dict(params)), dtype=float)
        lines.append("trading statistics (primary config, synthetic data; check them against the card):")
        lines += [f"  {k}: {v}" for k, v in stats(w, sv).items()]
        if card.get("mechanism_test"):
            from lab.framework import diagnostic
            from lab.framework.evaluator import load_events
            mt = card["mechanism_test"]
            mask, side = load_events(path.parent / "diagnostic.py", sv, params)
            st = diagnostic.event_study(sv, mask, side, int(mt["primary_horizon_days"]), placebo_runs=5)
            lines.append("mechanism test (events on the synthetic market; the judge's result on real data is not shown): "
                         f"{st['n_events']} events on {st['n_event_dates']} dates ({st['events_per_year']} per year); "
                         f"the gate needs >= {th_events} events on >= {th_dates} dates in the dev period")
        for cfg in QlabEvaluator().neighbors(card):
            full = params | cfg
            try:
                with strategy.time_limit(th["max_runtime_min"] * 60.0):
                    wn = np.asarray(module.target_weights(sv, dict(full)), dtype=float)
                from lab.framework.engine import validate_targets
                errs = validate_targets(wn, sv.shape)
                if errs:
                    raise ValueError("; ".join(errs))
            except Exception as e:  # noqa: BLE001
                ok = False
                lines.append(f"grid neighbor {cfg}: FAILED {type(e).__name__}: {e}")
        lines.append(f"grid neighbors: {len(QlabEvaluator().neighbors(card))} configurations run")
    ok_tests, test_lines = run_tests(ws, card, cat)
    return ok and ok_tests, lines + test_lines


def run_tests(ws: Path, card: dict, cat: dict) -> tuple[bool, list[str]]:
    tests = sorted((ws / "strategy").glob("test_*.py"))
    if not tests:
        return False, ["tests: none found; write strategy/test_strategy.py"]
    allowed = set(card["universe"].get("instruments") or [])
    for t in tests:
        problems = [p for p in strategy.scan(t.read_text(), allowed, set())
                    if not any(f"import {m}" in p or f"from {m} " in p for m in TEST_EXTRA_IMPORTS)]
        if problems:
            return False, [f"tests: {t.name} breaks the static rules: {problems}"]
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                            *[str(t.relative_to(ws)) for t in tests]], cwd=ws, capture_output=True, text=True,
                           timeout=TEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, [f"tests: timed out after {TEST_TIMEOUT_S} s"]
    tail = (p.stdout + p.stderr).strip().splitlines()[-25:]
    return p.returncode == 0, [f"tests: {'passed' if p.returncode == 0 else 'FAILED'}"] + [f"  {x}" for x in tail]
