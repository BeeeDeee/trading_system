import shutil
from pathlib import Path

import pytest

from arena.config import DEFAULT_CONFIG_DIR, ConfigError, load_config, load_signer


@pytest.fixture
def cdir(tmp_path: Path) -> Path:
    d = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, d)
    return d


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new, 1))


def test_repo_config_is_valid():
    cfg = load_config(DEFAULT_CONFIG_DIR)
    assert cfg.env == "testnet"   # live smí být zapnutý jen na testnetu (mainnet vyžaduje výslovný souhlas)
    assert cfg.live.enabled is False or cfg.env == "testnet"
    fams = {}
    for v in cfg.variants.values():
        fams[v.family] = fams.get(v.family, 0) + 1
    assert fams == {"brk": 10, "obi": 10, "ll": 10, "ctl": 3, "xp": 1}
    assert cfg.variants["ll.v00"].symbols == ("ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")
    assert cfg.variants["ctl.ll"].symbols == ("ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")
    assert cfg.endpoints.rest == "https://demo-fapi.binance.com"


def test_two_live_slots_on_one_symbol_rejected(cdir):
    _edit(cdir / "arena.yaml", "    XRPUSDT: {variant: xp.v00, notional_usdt: 100}",
          "    XRPUSDT: {variant: xp.v00, notional_usdt: 100}\n    XRPUSDT: {variant: obi.v00, notional_usdt: 150}")
    with pytest.raises(ConfigError, match="duplicitní klíč 'XRPUSDT'"):
        load_config(cdir)


def test_changed_params_of_existing_variant_rejected(cdir):
    _edit(cdir / "strategies" / "brk.yaml", "  v01: {N: 12}", "  v01: {N: 13}")
    with pytest.raises(ConfigError, match="brk.v01: parametry změněny"):
        load_config(cdir)


def test_changed_default_changes_all_variant_hashes(cdir):
    _edit(cdir / "strategies" / "obi.yaml", "  cb_max: 3.0", "  cb_max: 3.5")
    with pytest.raises(ConfigError) as ei:
        load_config(cdir)
    assert all(f"obi.v0{i}: parametry změněny" in str(ei.value) for i in range(10))


def test_new_variant_requires_lock(cdir):
    _edit(cdir / "strategies" / "brk.yaml", "  v09: {k_cb: 2.5, activate_mult: 1.5}",
          "  v09: {k_cb: 2.5, activate_mult: 1.5}\n  v10: {N: 36}")
    with pytest.raises(ConfigError, match="brk.v10: varianta není v variants.lock.json"):
        load_config(cdir)


def test_live_slot_rules(cdir):
    _edit(cdir / "arena.yaml", "BTCUSDT: {variant: xp.v00", "BTCUSDT: {variant: ll.v00")
    with pytest.raises(ConfigError, match="na symbolu neběží"):
        load_config(cdir)


def test_control_cannot_be_live(cdir):
    _edit(cdir / "arena.yaml", "BTCUSDT: {variant: xp.v00", "BTCUSDT: {variant: ctl.brk")
    with pytest.raises(ConfigError, match="kontrolní rodina"):
        load_config(cdir)


def test_universe_frozen(cdir):
    _edit(cdir / "arena.yaml", "universe: [BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT]",
          "universe: [BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, ZECUSDT]")
    with pytest.raises(ConfigError, match="zmrazené"):
        load_config(cdir)


def test_unrouted_ws_url_rejected(cdir):
    _edit(cdir / "arena.yaml", "ws_market: wss://demo-fstream.binance.com/market", "ws_market: wss://demo-fstream.binance.com")
    with pytest.raises(ConfigError, match="routou /market"):
        load_config(cdir)


def test_unknown_key_rejected(cdir):
    _edit(cdir / "arena.yaml", "risk: {", "risk: {max_leverage: 5, ")
    with pytest.raises(ConfigError, match="neznámé klíče"):
        load_config(cdir)


def test_config_hash_stable_and_sensitive(cdir):
    h1 = load_config(cdir).config_hash
    assert load_config(cdir).config_hash == h1
    _edit(cdir / "arena.yaml", "cooldown_bars: 3", "cooldown_bars: 4")
    assert load_config(cdir).config_hash != h1


def test_load_signer_from_env_file(tmp_path, monkeypatch):
    for k in ("BINANCE_API_KEY", "BINANCE_PRIVATE_KEY_PATH", "BINANCE_API_SECRET",
              "BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_API_SECRET", "BINANCE_TESTNET_PRIVATE_KEY_PATH"):
        monkeypatch.delenv(k, raising=False)
    from tests.test_signing import DOCS_ED25519_PEM
    pem = tmp_path / "k.pem"
    pem.write_bytes(DOCS_ED25519_PEM)
    envf = tmp_path / "env"
    envf.write_text(f"BINANCE_API_KEY=abc\nBINANCE_PRIVATE_KEY_PATH={pem}\nBINANCE_TESTNET_API_KEY=t\n"
                    "BINANCE_TESTNET_API_SECRET='s'\n")
    main = load_signer("mainnet", envf)
    test = load_signer("testnet", envf)
    assert type(main).__name__ == "Ed25519Signer" and main.api_key == "abc"
    assert type(test).__name__ == "HmacSigner" and test.api_key == "t"
    assert load_signer("mainnet", tmp_path / "missing") is None
