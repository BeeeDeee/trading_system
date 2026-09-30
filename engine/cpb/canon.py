"""Canonical JSON, hashing, atomic writes and UTC date helpers (stdlib only).

Canonical JSON = sorted keys, no whitespace, UTF-8, floats rounded to 10 decimals
(money is rounded to 1e-8 before it is stored). Same inputs -> byte-identical files.
"""
import datetime as _dt
import hashlib
import json
import math
import os

FLOAT_DECIMALS = 10


def _norm(x):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        r = round(x, FLOAT_DECIMALS)
        return 0.0 if r == 0 else r          # no "-0.0"
    if isinstance(x, dict):
        return {str(k): _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x


def dumps(obj):
    """Canonical compact JSON (the form that is hashed)."""
    return json.dumps(_norm(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def roundtrip(obj):
    """Exactly the values a later reader of the stored JSON will see (used at every phase boundary)."""
    return json.loads(dumps(obj))


def dumps_pretty(obj):
    """Canonical but human-readable JSON (the form that is written to disk)."""
    return json.dumps(_norm(obj), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def sha256_obj(obj):
    return sha256_bytes(dumps(obj).encode("utf-8"))


def write_bytes(path, data):
    """Atomic write: a crash never leaves a half-written file."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        if not os.environ.get("CPB_NO_FSYNC"):        # simulations and tests only
            os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, obj):
    write_bytes(path, dumps_pretty(obj).encode("utf-8"))


def write_text(path, text):
    write_bytes(path, text.encode("utf-8"))


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def money(x):
    return round(float(x), 8)


# ---------------------------------------------------------------- time (always UTC)

def parse_date(s):
    return _dt.date.fromisoformat(s)


def add_days(s, n):
    return (parse_date(s) + _dt.timedelta(days=n)).isoformat()


def days_between(a, b):
    """b - a in days (ISO dates)."""
    return (parse_date(b) - parse_date(a)).days


def date_range(a, b):
    """Dates a..b inclusive."""
    out, d = [], parse_date(a)
    while d <= parse_date(b):
        out.append(d.isoformat())
        d += _dt.timedelta(days=1)
    return out


def weekday(s):
    """0 = Monday."""
    return parse_date(s).weekday()


def date_ms(s):
    """00:00 UTC of date s in epoch milliseconds."""
    d = parse_date(s)
    return int(_dt.datetime(d.year, d.month, d.day, tzinfo=_dt.timezone.utc).timestamp() * 1000)


def ms_date(ms):
    return _dt.datetime.fromtimestamp(ms / 1000, tz=_dt.timezone.utc).date().isoformat()


def ms_iso(ms):
    return _dt.datetime.fromtimestamp(ms / 1000, tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def iso_ms(s):
    s = s.replace("Z", "+00:00")
    return int(_dt.datetime.fromisoformat(s).timestamp() * 1000)


def hash_rank(*parts):
    """Deterministic tie-break key: no alphabetical advantage, independent of dict order."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
