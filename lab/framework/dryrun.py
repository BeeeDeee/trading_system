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
TEST_TIMEOUT_S = 300
TEST_EXTRA_IMPORTS = ("pytest", "strategy")


def instruments(card: dict, cat: dict) -> tuple[list[str], list[str], int | None]:
    """(names, asset classes, top-n) of the strategy's instruments, resolved exactly as the gate runner does."""
    uni = card["universe"]
    if uni["kind"] not in ("instruments", "crypto_top_n"):
        raise ValueError(f"universe kind {uni['kind']!r} has no loader yet")
    names, classes = [], []
    for ds, inst in strategy_instruments(card).items():
        inst = [f"SYN{i:02d}USDT" for i in range(SYNTHETIC_PAIRS)] if inst is None else inst
        for i in inst:
            if i not in names:
                names.append(i)
                classes.append(cat[ds]["asset_class"])
    return names, classes, uni.get("n") if uni["kind"] == "crypto_top_n" else None


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
    if crypto.sum() > 2:
        j = int(np.flatnonzero(crypto)[-1])
        last = int(T * 0.7)
        delisting[last + 1, j], ret_co[last + 1, j] = True, -0.02
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
    if top_n:
        view = replace(view, universe=data.crypto_top_n(view, top_n))
    series = {}
    for r in reqs:                                                 # signal-only datasets (Archivist ingest)
        if r["dataset"] in TRADABLE:
            continue
        for f in cat[r["dataset"]].get("fields") or r.get("fields") or []:   # what the gate runner will attach
            x = 20 + np.cumsum(rng.normal(0, 1, T))
            x[: T // 10] = np.nan                                 # starts later than the prices
            series[f"{r['dataset']}.{f}"] = x
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
    th = yaml.safe_load((ws / "gates.yaml").read_text())["G0"]
    path = ws / "strategy" / "strategy.py"
    seed = int(hashlib.sha256(json.dumps(card, sort_keys=True, default=str).encode()).hexdigest()[:8], 16)
    view = synthetic_view(card, cat, seed)
    cols = list(view.instruments)
    params = QlabEvaluator.primary(card)
    lines, ok = [f"synthetic market: {len(cols)} instruments, {view.dates[0]} .. {view.dates[-1]} "
                 f"({len(view.dates)} rows; random returns, a late listing, missing opens)"], True

    out = integrity(card, path, view, cols, params, len(view.dates), th, th["max_runtime_min"] * 60.0)
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
