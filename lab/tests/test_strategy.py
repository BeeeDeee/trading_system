import pytest

from lab.framework.strategy import scan

CLEAN = '''
import numpy as np
from lab.framework.api import only_on, period_starts

def target_weights(data, params):
    rng = np.random.default_rng(params["seed"])
    w = np.zeros(data.shape)
    w[:, data.instruments.index("SPY")] = 1.0
    return only_on(period_starts(data.dates, "M"), w)
'''


def test_clean_strategy_passes():
    assert scan(CLEAN, {"SPY"}, {"SPY", "IEF"}) == []


@pytest.mark.parametrize("snippet,match", [
    ("import os", "import os not allowed"),
    ("import subprocess", "import subprocess"),
    ("from pathlib import Path", "from pathlib import"),
    ("x = open('/srv/research-lab/lab.db').read()", "open not allowed"),
    ("x = eval('1')", "eval not allowed"),
    ("x = getattr(np, 'load')", "getattr not allowed"),
    ("x = np.load('holdout.npy')", ".load not allowed"),
    ("x = np.random.rand(3)", "random.rand not allowed"),
    ("x = np.datetime64('today')", ".datetime64 not allowed"),
    ("CRASH = '2020-03-16'", "date literal"),
    ("CRASH = '2008-09'", "date literal"),
    ("x = ().__class__", "dunder attribute"),
    ("BEST = 'IEF'", "instrument 'IEF' is not in the card's universe"),
])
def test_forbidden_constructs(snippet, match):
    problems = scan(CLEAN + "\n" + snippet + "\n", {"SPY"}, {"SPY", "IEF"})
    assert any(match in p for p in problems), problems
