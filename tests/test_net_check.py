"""Živé read-only kontroly (spouští se jen `uv run pytest -m net`)."""

import pytest

from arena.check import run_check
from arena.config import DEFAULT_CONFIG_DIR, load_config


@pytest.mark.net
@pytest.mark.parametrize("env", ["mainnet", "testnet"])
async def test_check_public_part(env):
    cfg = load_config(DEFAULT_CONFIG_DIR, env_override=env)
    rep = await run_check(cfg, signer=None)
    assert not rep.failed, rep.render()
