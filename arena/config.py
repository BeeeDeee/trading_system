"""Načtení a validace konfigurace (SPEC §11) + klíče z env (D9)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from arena.binance.signing import Ed25519Signer, HmacSigner, Signer
from arena.util import canonical_json, sha256_hex

FROZEN_UNIVERSE = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")  # D5
FAMILIES = ("brk", "obi", "ll", "ctl", "xp")   # xp = exec probe: jen testnet, mimo experiment
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
LOCK_FILE = "variants.lock.json"
ENV_FILE = Path("/etc/arena/env")


class ConfigError(ValueError):
    pass


# --- YAML bez duplicitních klíčů (2 live sloty na 1 symbol se jinak tiše přepíšou) ---------------

class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ConfigError(f"duplicitní klíč {key!r} (řádek {key_node.start_mark.line + 1})")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 (SafeLoader podtřída)


# --- dataclasses -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Endpoints:
    rest: str
    ws_market: str
    ws_public: str
    ws_private: str


@dataclass(frozen=True)
class PaperCfg:
    notional_usdt: float = 1000.0
    entry_latency_ms: int = 250
    trigger_latency_ms: int = 100
    stale_book_penalty_bps: float = 1.0
    outage_penalty_bps: float = 2.0
    extra_slippage_bps: Mapping[str, float] = field(default_factory=dict)
    cost_model_version: int = 1


@dataclass(frozen=True)
class LiveSlotCfg:
    symbol: str
    variant: str
    notional_usdt: float


@dataclass(frozen=True)
class LiveCfg:
    enabled: bool = False
    leverage: int = 2
    slots: Mapping[str, LiveSlotCfg] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskCfg:
    account_daily_loss_pct: float = 3.0
    slot_daily_loss_pct: float = 3.0
    slot_max_dd_pct: float = 15.0
    max_live_positions: int = 5


@dataclass(frozen=True)
class EngineCfg:
    cooldown_bars: int = 3
    bar_close_wait_s: float = 3.0
    snapshot_every_s: float = 15.0
    reconcile_every_s: float = 60.0


@dataclass(frozen=True)
class RecorderCfg:
    raw_dir: str = "/srv/arena/raw"
    raw_aggtrades: bool = True
    raw_depth: bool = False
    retention_days: int | None = None


@dataclass(frozen=True)
class BookCfg:
    bands_pct: tuple[float, ...] = (0.2, 1.0, 2.0, 5.0)
    sample_every_s: int = 30
    resnapshot_every_min: int = 30
    max_diff_age_s: float = 2.0


@dataclass(frozen=True)
class PathsCfg:
    db: str = "/srv/arena/arena.db"
    site: str = "/srv/arena/site"
    kill_file: str = "/srv/arena/KILL"


@dataclass(frozen=True)
class AlertsCfg:
    ntfy_topic_env: str = "NTFY_TOPIC"


@dataclass(frozen=True)
class VariantSpec:
    family: str
    variant_id: str          # "brk.v00"
    symbols: tuple[str, ...]
    params: Mapping[str, Any]
    params_hash: str
    is_control: bool = False


@dataclass(frozen=True)
class Config:
    env: str                  # obchodní účet (podepsané REST, private WS, live executor)
    data_env: str             # tržní data (bary, kniha, aggTrades); default mainnet, jen veřejná read-only volání
    universe: tuple[str, ...]
    endpoints_by_env: Mapping[str, Endpoints]
    paper: PaperCfg
    live: LiveCfg
    risk: RiskCfg
    engine: EngineCfg
    recorder: RecorderCfg
    book: BookCfg
    paths: PathsCfg
    alerts: AlertsCfg
    variants: Mapping[str, VariantSpec]
    config_hash: str
    source_text: str          # celé YAML (arena + strategie) pro tabulku config_versions
    config_dir: Path

    @property
    def endpoints(self) -> Endpoints:
        return self.endpoints_by_env[self.env]

    @property
    def data_endpoints(self) -> Endpoints:
        return self.endpoints_by_env[self.data_env]


# --- načtení -----------------------------------------------------------------------------------

def _build(cls: type, data: Mapping[str, Any] | None, section: str) -> Any:
    data = dict(data or {})
    known = set(cls.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{section}: neznámé klíče {sorted(unknown)}")
    return cls(**data)


def params_hash(family: str, symbols: tuple[str, ...], params: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json({"family": family, "symbols": list(symbols), "params": dict(params)}))[:16]


def _load_family(path: Path) -> tuple[list[VariantSpec], str]:
    text = path.read_text(encoding="utf-8")
    doc = load_yaml(text) or {}
    family = doc.get("family")
    if family not in FAMILIES:
        raise ConfigError(f"{path.name}: neznámá rodina {family!r}")
    if path.stem != family:
        raise ConfigError(f"{path.name}: soubor musí se jmenovat {family}.yaml")
    unknown = set(doc) - {"family", "symbols", "defaults", "variants"}
    if unknown:
        raise ConfigError(f"{path.name}: neznámé klíče {sorted(unknown)}")
    symbols = tuple(doc.get("symbols") or ())
    defaults = dict(doc.get("defaults") or {})
    out: list[VariantSpec] = []
    for key, override in (doc.get("variants") or {}).items():
        override = dict(override or {})
        v_symbols = tuple(override.pop("symbols", symbols))   # např. ctl.ll běží jen na 4 symbolech
        unknown_p = set(override) - set(defaults)
        if unknown_p:
            raise ConfigError(f"{family}.{key}: parametry {sorted(unknown_p)} nejsou v defaults")
        params = {**defaults, **override}
        vid = f"{family}.{key}"
        out.append(VariantSpec(family, vid, v_symbols, params, params_hash(family, v_symbols, params),
                               is_control=(family == "ctl")))
    return out, text


def load_config(config_dir: Path | str | None = None, *, env_override: str | None = None,
                check_lock: bool = True) -> Config:
    cdir = Path(config_dir or os.environ.get("ARENA_CONFIG_DIR") or DEFAULT_CONFIG_DIR)
    main_text = (cdir / "arena.yaml").read_text(encoding="utf-8")
    raw = load_yaml(main_text) or {}
    allowed = {"env", "data_env", "universe", "endpoints", "paper", "live", "risk", "engine", "recorder", "book", "paths", "alerts"}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"arena.yaml: neznámé klíče {sorted(unknown)}")

    variants: dict[str, VariantSpec] = {}
    texts = [f"# arena.yaml\n{main_text}"]
    for p in sorted((cdir / "strategies").glob("*.yaml")):
        specs, text = _load_family(p)
        texts.append(f"# strategies/{p.name}\n{text}")
        for v in specs:
            if v.variant_id in variants:
                raise ConfigError(f"duplicitní varianta {v.variant_id}")
            variants[v.variant_id] = v

    endpoints = {name: _build(Endpoints, e, f"endpoints.{name}") for name, e in (raw.get("endpoints") or {}).items()}
    live_raw = dict(raw.get("live") or {})
    slots_raw = live_raw.pop("slots", None) or {}
    slots = {}
    for sym, s in slots_raw.items():
        s = dict(s or {})
        unknown_s = set(s) - {"variant", "notional_usdt"}
        if unknown_s or "variant" not in s or "notional_usdt" not in s:
            raise ConfigError(f"live.slots.{sym}: očekávám přesně variant a notional_usdt")
        slots[sym] = LiveSlotCfg(sym, str(s["variant"]), float(s["notional_usdt"]))
    live = _build(LiveCfg, {**live_raw, "slots": slots}, "live")

    book_raw = dict(raw.get("book") or {})
    if "bands_pct" in book_raw:
        book_raw["bands_pct"] = tuple(float(x) for x in book_raw["bands_pct"])

    resolved = {
        "main": raw,
        "variants": {k: {"symbols": list(v.symbols), "params": dict(v.params)} for k, v in sorted(variants.items())},
    }
    cfg = Config(
        env=str(env_override or raw.get("env", "testnet")),
        data_env=str(raw.get("data_env", "mainnet")),
        universe=tuple(raw.get("universe") or ()),
        endpoints_by_env=endpoints,
        paper=_build(PaperCfg, raw.get("paper"), "paper"),
        live=live,
        risk=_build(RiskCfg, raw.get("risk"), "risk"),
        engine=_build(EngineCfg, raw.get("engine"), "engine"),
        recorder=_build(RecorderCfg, raw.get("recorder"), "recorder"),
        book=_build(BookCfg, book_raw, "book"),
        paths=_build(PathsCfg, raw.get("paths"), "paths"),
        alerts=_build(AlertsCfg, raw.get("alerts"), "alerts"),
        variants=variants,
        config_hash=sha256_hex(canonical_json(resolved))[:16],
        source_text="\n".join(texts),
        config_dir=cdir,
    )
    validate(cfg, check_lock=check_lock)
    return cfg


# --- validace ----------------------------------------------------------------------------------

def read_lock(config_dir: Path) -> dict[str, str]:
    p = config_dir / LOCK_FILE
    if not p.exists():
        return {}
    return dict(json.loads(p.read_text(encoding="utf-8")))


def write_lock(cfg: Config) -> list[str]:
    """Zaregistruje nové varianty do lock souboru. Existující ID se nikdy nepřepíší. Vrací přidaná ID."""
    lock = read_lock(cfg.config_dir)
    added = []
    for vid, v in sorted(cfg.variants.items()):
        if vid not in lock:
            lock[vid] = v.params_hash
            added.append(vid)
        elif lock[vid] != v.params_hash:
            raise ConfigError(f"{vid}: parametry se liší od zamčené verze; změna = nové ID (SPEC §11)")
    (cfg.config_dir / LOCK_FILE).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return added


def validate(cfg: Config, *, check_lock: bool = True) -> None:
    errs: list[str] = []
    if cfg.env not in ("testnet", "mainnet"):
        errs.append(f"env musí být testnet|mainnet, je {cfg.env!r}")
    if cfg.data_env not in ("testnet", "mainnet"):
        errs.append(f"data_env musí být testnet|mainnet, je {cfg.data_env!r}")
    if cfg.universe != FROZEN_UNIVERSE:
        errs.append(f"universe je zmrazené (D5) na {list(FROZEN_UNIVERSE)}, je {list(cfg.universe)}")
    for name in ("testnet", "mainnet"):
        if name not in cfg.endpoints_by_env:
            errs.append(f"chybí endpoints.{name}")
    for name in sorted({cfg.env, cfg.data_env} & set(cfg.endpoints_by_env)):
        e = cfg.endpoints_by_env[name]
        for fld in ("rest", "ws_market", "ws_public", "ws_private"):
            val = getattr(e, fld)
            if not val.startswith(("https://", "wss://")):
                errs.append(f"endpoints.{name}.{fld}: neplatná URL {val!r}")
        for fld, route in (("ws_market", "/market"), ("ws_public", "/public"), ("ws_private", "/private")):
            if not getattr(e, fld).rstrip("/").endswith(route):
                errs.append(f"endpoints.{name}.{fld} musí končit routou {route} (nástraha 2)")

    for fam in ("brk", "obi", "ll"):
        n = sum(1 for v in cfg.variants.values() if v.family == fam)
        if n == 0:
            errs.append(f"chybí varianty rodiny {fam}")
    for v in cfg.variants.values():
        bad = set(v.symbols) - set(FROZEN_UNIVERSE)
        if bad or not v.symbols:
            errs.append(f"{v.variant_id}: symboly {sorted(bad) or '[]'} mimo universe")

    if cfg.env != "testnet" and any(v.family == "xp" for v in cfg.variants.values()):
        errs.append("rodina xp (exec probe) smí existovat jen s env: testnet; před mainnetem smaž config/strategies/xp.yaml "
                    "a její záznam v variants.lock.json")
    # live sloty: max. 1 na symbol (D2) – klíč mapy je symbol a duplicity odmítá loader
    if len(cfg.live.slots) > cfg.risk.max_live_positions:
        errs.append(f"live.slots: {len(cfg.live.slots)} slotů > max_live_positions {cfg.risk.max_live_positions}")
    for sym, s in cfg.live.slots.items():
        if sym not in FROZEN_UNIVERSE:
            errs.append(f"live.slots.{sym}: symbol mimo universe")
        v = cfg.variants.get(s.variant)
        if v is None:
            errs.append(f"live.slots.{sym}: neznámá varianta {s.variant}")
        elif v.is_control:
            errs.append(f"live.slots.{sym}: kontrolní rodina nesmí běžet live")
        elif sym not in v.symbols:
            errs.append(f"live.slots.{sym}: varianta {s.variant} na symbolu neběží")
        if s.notional_usdt <= 0:
            errs.append(f"live.slots.{sym}: notional musí být > 0")
    if cfg.live.leverage < 1:
        errs.append("live.leverage musí být ≥ 1")
    if cfg.paper.notional_usdt <= 0:
        errs.append("paper.notional_usdt musí být > 0")

    if check_lock:
        lock = read_lock(cfg.config_dir)
        for vid, v in sorted(cfg.variants.items()):
            if vid not in lock:
                errs.append(f"{vid}: varianta není v {LOCK_FILE} (nová varianta → `arena lock-variants`)")
            elif lock[vid] != v.params_hash:
                errs.append(f"{vid}: parametry změněny proti {LOCK_FILE}; změna parametrů = nové ID (SPEC §11)")
    if errs:
        raise ConfigError("neplatná konfigurace:\n  - " + "\n  - ".join(errs))


# --- klíče (D9) --------------------------------------------------------------------------------

def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().removeprefix("export ").strip()] = v.strip().strip("'\"")
    return out


def load_signer(env: str, env_file: Path = ENV_FILE) -> Signer | None:
    """Klíče z proměnných prostředí, jinak z `/etc/arena/env`. Testnet má prefix `BINANCE_TESTNET_`.

    Ed25519: `*_API_KEY` + `*_PRIVATE_KEY_PATH`. HMAC fallback: `*_API_KEY` + `*_API_SECRET`."""
    prefix = "BINANCE_TESTNET_" if env == "testnet" else "BINANCE_"
    file_vals = _read_env_file(env_file)

    def get(name: str) -> str | None:
        return os.environ.get(prefix + name) or file_vals.get(prefix + name) or None

    api_key = get("API_KEY")
    if not api_key:
        return None
    key_path = get("PRIVATE_KEY_PATH")
    if key_path:
        pw = get("PRIVATE_KEY_PASSWORD")
        return Ed25519Signer.from_file(api_key, Path(key_path), pw.encode() if pw else None)
    secret = get("API_SECRET")
    if secret:
        return HmacSigner(api_key, secret)
    raise ConfigError(f"{prefix}API_KEY je nastavený, ale chybí {prefix}PRIVATE_KEY_PATH nebo {prefix}API_SECRET")
