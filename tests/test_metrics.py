"""Regression tests for incremental-session metric boundaries."""
import unittest
from utils.metrics import accuracy, harmonic_accuracy


class MetricTests(unittest.TestCase):
    def test_mstar_first_increment_includes_last_label(self):
        labels = [0, 1, 2, 3, 4]
        grouped = accuracy(labels, labels, nb_old=4, increment=1)
        self.assertEqual(grouped['04-04'], 100.0)
        self.assertEqual(grouped['total'], 100.0)
        self.assertEqual(harmonic_accuracy(grouped), (100.0, 100.0, 100.0))

    def test_previous_increment_is_now_old(self):
        grouped = accuracy([0, 1, 2, 3, 0, 5], [0, 1, 2, 3, 4, 5], nb_old=5, increment=1)
        score, old, new = harmonic_accuracy(grouped)
        self.assertEqual(old, 80.0)
        self.assertEqual(new, 100.0)
        if score is None:
            self.fail('An incremental session must have a harmonic score')
        self.assertAlmostEqual(score, 88.88888888888889)

    def test_base_session_has_no_harmonic_score(self):
        grouped = accuracy([0, 1], [0, 1], nb_old=0, increment=1)
        self.assertEqual(harmonic_accuracy(grouped, has_old=False), (None, 0.0, 100.0))

    def test_zero_old_accuracy_is_not_treated_as_missing(self):
        grouped = accuracy([1, 0, 2], [0, 1, 2], nb_old=2, increment=1)
        self.assertEqual(harmonic_accuracy(grouped), (0.0, 0.0, 100.0))

    def test_empty_label_group(self):
        grouped = accuracy([0, 2], [0, 2], nb_old=2, increment=1)
        self.assertEqual(grouped['01-01'], 0.0)
        self.assertEqual(grouped['02-02'], 100.0)


if __name__ == '__main__':
    unittest.main()
