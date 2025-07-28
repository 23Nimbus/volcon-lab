import unittest
from signal_pipeline.alert_utils import classify_alert_level

class TestAlertLevel(unittest.TestCase):
    def test_fractional_classification(self):
        self.assertEqual(classify_alert_level(1.1), "🟢 Watch")
        self.assertEqual(classify_alert_level(2.0), "🟡 Tension")
        self.assertEqual(classify_alert_level(2.2), "🟡 Tension")
        self.assertEqual(classify_alert_level(2.6), "🔴 Breakout Potential")

if __name__ == '__main__':
    unittest.main()
