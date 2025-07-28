import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from signal_pipeline.gex_parser import parse_gex_comment

class TestGEXParser(unittest.TestCase):
    def test_gamma_break_detected(self):
        text = "Massive snap zone indicates a gamma break ahead"
        res = parse_gex_comment(text)
        self.assertTrue(res["gamma_break_near"])
        self.assertFalse(res["fragile_containment"])
        self.assertFalse(res["macro_risk_overlay"])

    def test_cluster_weakness_detected(self):
        text = "Thin gamma means a very weak cluster forming"
        res = parse_gex_comment(text)
        self.assertTrue(res["fragile_containment"])

if __name__ == '__main__':
    unittest.main()
