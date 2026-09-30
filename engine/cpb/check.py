"""Data checks before anything is settled or traded. Any error = no trading today.

check_coin() is pure; the >60 % move confirmation is injected (a second exchange's close).
"""
import math

from . import canon


def redenomination_ratio(prev_close, close):
    """If the move looks like a clean 10^k re-base (k != 0), return the ratio new units per old unit."""
    if prev_close <= 0 or close <= 0:
        return None
    lg = math.log10(prev_close / close)
    k = round(lg)
    if k != 0 and abs(lg - k) < 0.03:
        return 10 ** k
    return None


def check_coin(coin, rows, stored, asof, cfg, confirm=None):
    """rows: freshly fetched candles (sorted); stored: history rows already in the repo.

    Returns (errors, warnings, suspect) lists of Czech messages.
    """
    dcfg = cfg["data"]
    errors, warnings, suspect = [], [], None
    if not rows:
        return [f"{coin}: žádné svíčky"], [], None
    dates = [r[0] for r in rows]
    if len(set(dates)) != len(dates):
        errors.append(f"{coin}: duplicitní svíčky")
    if dates != sorted(dates):
        errors.append(f"{coin}: svíčky nejsou seřazené")
    expected = canon.date_range(dates[0], asof)
    missing = sorted(set(expected) - set(dates))
    if dates[-1] != asof:
        errors.append(f"{coin}: zastaralá data (poslední svíčka {dates[-1]}, očekáváno {asof})")
    elif missing:
        errors.append(f"{coin}: chybějící svíčky {', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}")
    for r in rows:
        d, o, h, l, c, vb, vq = r
        if min(o, h, l, c) <= 0:
            errors.append(f"{coin} {d}: nekladná cena")
        elif h < l:
            errors.append(f"{coin} {d}: high < low")
        elif h < max(o, c) * (1 - 1e-9) or l > min(o, c) * (1 + 1e-9):
            errors.append(f"{coin} {d}: open/close mimo rozsah high–low")
        if vb <= 0 or vq <= 0:
            if d == asof:
                errors.append(f"{coin} {d}: nulový objem")
            else:
                warnings.append(f"{coin} {d}: nulový objem (starší svíčka)")
    old = {r[0]: r for r in stored}
    for r in rows:
        o = old.get(r[0])
        if o and abs(o[4] / r[4] - 1) > dcfg["revision_tol"]:
            errors.append(f"{coin} {r[0]}: close se liší od uložené historie ({o[4]} vs {r[4]})")
    # daily moves on the new part (including the join with stored history)
    seq = sorted({**old, **{r[0]: r for r in rows}}.values())
    new_dates = {r[0] for r in rows if r[0] not in old}
    for p, r in zip(seq, seq[1:]):
        if r[0] not in new_dates:
            continue
        mv = r[4] / p[4] - 1
        ratio = redenomination_ratio(p[4], r[4])
        if ratio:
            suspect = {"coin": coin, "date": r[0], "ratio": ratio, "prev_close": p[4], "close": r[4]}
            errors.append(f"{coin} {r[0]}: pohyb {mv * 100:+.1f} % vypadá jako redenominace 1:{ratio:g} – potvrdit rešerší a apply-redenomination")
        elif abs(mv) > dcfg["max_daily_move"]:
            ok = confirm(coin, r[0], r[4]) if confirm else None
            if ok is True:
                warnings.append(f"{coin} {r[0]}: pohyb {mv * 100:+.1f} % potvrzen druhým zdrojem")
            else:
                errors.append(f"{coin} {r[0]}: pohyb {mv * 100:+.1f} % bez potvrzení druhým zdrojem")
    return errors, warnings, suspect


def crosscheck(primary, other, tol):
    diff = primary / other - 1
    return {"primary": primary, "other": other, "diff_pct": round(diff * 100, 4), "ok": abs(diff) <= tol}


HOUR_MS = 3_600_000


def check_hourly(coin, rows, stored, t_end, cfg, confirm=None):
    """Hourly candles [open_ms, o, h, l, c, quote_vol, taker_buy_quote]: same rules as the daily check, one hour step.
    The last candle must be the one that closed at t_end; a move > max_hourly_move needs a second exchange."""
    hc = cfg["hourly"]
    errors, warnings = [], []
    if not rows:
        return [f"{coin}: žádné hodinové svíčky"], []
    ts = [r[0] for r in rows]
    if len(set(ts)) != len(ts) or ts != sorted(ts):
        errors.append(f"{coin}: duplicitní nebo neseřazené hodinové svíčky")
    if ts[-1] != t_end - HOUR_MS:
        errors.append(f"{coin}: zastaralá hodinová data (poslední svíčka {ts[-1]}, očekáváno {t_end - HOUR_MS})")
    elif (ts[-1] - ts[0]) // HOUR_MS + 1 != len(ts):
        errors.append(f"{coin}: chybějící hodinové svíčky")
    for r in rows:
        t, o, h, l, c, vq, tb = r
        if min(o, h, l, c) <= 0:
            errors.append(f"{coin} {t}: nekladná cena")
        elif h < l or h < max(o, c) * (1 - 1e-9) or l > min(o, c) * (1 + 1e-9):
            errors.append(f"{coin} {t}: nekonzistentní high/low")
        if vq <= 0:
            warnings.append(f"{coin} {t}: nulový hodinový objem")
    old = {r[0]: r for r in stored}
    for r in rows:
        o = old.get(r[0])
        if o and abs(o[4] / r[4] - 1) > cfg["data"]["revision_tol"]:
            errors.append(f"{coin} {r[0]}: hodinová close se liší od uložené historie")
    seq = sorted({**old, **{r[0]: r for r in rows}}.values())
    new = {r[0] for r in rows if r[0] not in old}
    for p, r in zip(seq, seq[1:]):
        if r[0] in new and abs(r[4] / p[4] - 1) > hc["max_hourly_move"]:
            ok = confirm(coin, r[0], r[4]) if confirm else None
            if ok is True:
                warnings.append(f"{coin} {r[0]}: hodinový pohyb {(r[4] / p[4] - 1) * 100:+.1f} % potvrzen druhým zdrojem")
            else:
                errors.append(f"{coin} {r[0]}: hodinový pohyb {(r[4] / p[4] - 1) * 100:+.1f} % bez potvrzení")
    return errors, warnings
