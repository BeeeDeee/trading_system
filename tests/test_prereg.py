"""Statistics of the pre-registered evaluation (tools/prereg_eval.py) against known values."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "engine"))
import prereg_eval as P  # noqa: E402


class TestPreregStats(unittest.TestCase):
    def test_t_pvalue_matches_tables(self):
        for df, t, p in ((60, 2.0, 0.0500), (5, 2.571, 0.0500), (10, 3.169, 0.0100), (30, 1.697, 0.1000)):
            self.assertAlmostEqual(P.betainc(df / 2, 0.5, df / (df + t * t)), p, places=3)

    def test_t_test_and_halves(self):
        r = P.t_test([0.1, 0.2, 0.15, 0.05, 0.12, 0.18])
        self.assertGreater(r["mean"], 0)
        self.assertLess(r["p"], 0.01)
        self.assertTrue(r["halves_agree"])
        self.assertEqual(P.verdict(r), "POTVRZENO")
        r = P.t_test([0.3, 0.2, 0.25, -0.2, -0.3, -0.1])
        self.assertFalse(r["halves_agree"])
        self.assertEqual(P.verdict(r), "nepotvrzeno")

    def test_holm_and_bh(self):
        self.assertEqual(P.holm({"a": 0.01, "b": 0.04, "c": 0.03}), {"a": 0.03, "c": 0.06, "b": 0.06})
        bh = P.bh({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(bh["a"], 0.03)
        self.assertAlmostEqual(bh["b"], 0.04)

    def test_block_means_non_overlapping(self):
        self.assertEqual(P.block_means([1, 2, 3, 4, 5, 6, 7], 3), [2, 5])


if __name__ == "__main__":
    unittest.main()
