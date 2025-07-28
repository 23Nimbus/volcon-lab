import unittest
from signal_pipeline.gex_parser import parse_gex_comment

class TestGEXParser(unittest.TestCase):
    def test_gamma_break_detected(self):
        text = "Massive snap zone indicates a gamma break ahead"
        res = parse_gex_comment(text)
        self.assertTrue(res["gamma_break_near"])
        self.assertFalse(res["support_discussed"])
        self.assertFalse(res["resistance_discussed"])
        self.assertFalse(res["fragile_containment"])
        self.assertFalse(res["macro_risk_overlay"])

    def test_cluster_weakness_detected(self):
        text = "Thin gamma means a very weak cluster forming"
        res = parse_gex_comment(text)
        self.assertTrue(res["fragile_containment"])

    def test_support_and_resistance_detection(self):
        support_text = "Expecting a bounce off the dip zone support"
        resistance_text = "There's a major sell wall acting as resistance"
        support_res = parse_gex_comment(support_text)
        resistance_res = parse_gex_comment(resistance_text)

        self.assertTrue(support_res["support_discussed"])
        self.assertFalse(support_res["resistance_discussed"])
        self.assertTrue(resistance_res["resistance_discussed"])
        self.assertFalse(resistance_res["support_discussed"])

if __name__ == '__main__':
    unittest.main()
