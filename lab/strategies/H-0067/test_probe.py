import numpy as np

from strategy import _max_scale, GROSS_LIMIT


def test_max_scale_linear_pieces():
    # g_e = 0.5, net_e = 0.5, new: g_n = 1, net_n = 1 -> 0.5 + s + |0.5 + s| <= 1 -> s = 0
    assert _max_scale([(1.0, 2.0), (0.0, 0.0)]) == 0.0
    # empty book, 10 longs of 1/30: (0 + 0, 1/3 + 1/3) -> s = 1 fits (2/3 <= 1)
    assert _max_scale([(0.0, 2.0 / 3.0), (0.0, 0.0)]) == 1.0
    # kept hedge of 0.3 adds a piece: 0.3 + s * 1 <= 1 -> s = 0.7
    assert np.isclose(_max_scale([(0.0, 0.0), (0.0, 0.0), (0.3, 1.0)]), GROSS_LIMIT - 0.3)
