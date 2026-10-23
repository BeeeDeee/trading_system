import csv
import gzip
from pathlib import Path

import numpy as np
import pytest

from arena.features import BarArrays

DATA = Path(__file__).parent / "data"


def load_klines(sym: str) -> BarArrays:
    with gzip.open(DATA / f"klines_{sym}_5m.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    col = lambda k, t=float: np.array([t(r[k]) for r in rows])  # noqa: E731
    return BarArrays(col("open_time", int).astype(np.int64), col("o"), col("h"), col("l"), col("c"), col("qv"),
                     col("tbqv"))


@pytest.fixture(scope="session")
def btc() -> BarArrays:
    return load_klines("BTCUSDT")


@pytest.fixture(scope="session")
def eth() -> BarArrays:
    return load_klines("ETHUSDT")
